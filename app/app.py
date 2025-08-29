import os
import json
import tempfile
import csv
import pandas as pd
import io
import uuid
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from dotenv import load_dotenv
import secrets
import google.generativeai as genai
from .pdf_generator import EvaluationPDFGenerator

# Import analytics with error handling
try:
    from .analytics import AnalyticsTracker
    analytics = AnalyticsTracker()
except ImportError as e:
    print(f"Warning: Analytics disabled due to import error: {e}")
    analytics = None

load_dotenv()

app = FastAPI(title="EvalNow", description="AI evaluation made simple")

# Add a debug route to test if routes are being registered
@app.get("/debug/routes")
async def list_routes():
    """Debug endpoint to list all registered routes"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unnamed')
            })
    return {"routes": routes}

# Simple in-memory storage for results (in production, use Redis or database)
results_cache = {}

# Basic auth for admin endpoints
security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify admin credentials"""
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "changeme123")
    
    is_correct_username = secrets.compare_digest(credentials.username, admin_username)
    is_correct_password = secrets.compare_digest(credentials.password, admin_password)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Add custom filter for cleaning markdown
def clean_markdown_filter(text):
    """Clean markdown formatting from text"""
    import re
    if not text:
        return text
    # Remove **bold** formatting and keep the text
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove *italic* formatting
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = text.strip()
    return text

templates.env.filters['clean_markdown'] = clean_markdown_filter

# Translation system - Load translations from JSON files
TRANSLATIONS = {}

def load_translations():
    """Load translation files from the translations directory"""
    global TRANSLATIONS
    
    translation_dir = os.path.join(os.path.dirname(__file__), '..', 'translations')
    
    for lang_code in ['en', 'fr']:
        file_path = os.path.join(translation_dir, f'{lang_code}.json')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                TRANSLATIONS[lang_code] = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Translation file {file_path} not found")
            TRANSLATIONS[lang_code] = {}
        except json.JSONDecodeError as e:
            print(f"Error loading translation file {file_path}: {e}")
            TRANSLATIONS[lang_code] = {}

# Load translations on startup
load_translations()

def detect_language(request: Request) -> str:
    """Detect user language from Accept-Language header or URL param"""
    # Check URL parameter first (manual override)
    if hasattr(request, 'query_params') and 'lang' in request.query_params:
        lang = request.query_params['lang'].lower()
        if lang in ['fr', 'french']:
            return 'fr'
        else:
            return 'en'
    
    # Check Accept-Language header
    accept_language = request.headers.get('accept-language', '')
    if accept_language:
        # Parse accept-language header
        languages = []
        for lang_part in accept_language.split(','):
            if ';' in lang_part:
                lang = lang_part.split(';')[0].strip()
            else:
                lang = lang_part.strip()
            languages.append(lang.lower())
        
        # Check if French is preferred
        for lang in languages:
            if lang.startswith('fr'):
                return 'fr'
    
    return 'en'  # Default to English

def get_translation(key: str, language: str = 'en') -> str:
    """Get translated text for a key"""
    return TRANSLATIONS.get(language, {}).get(key, TRANSLATIONS['en'].get(key, key))

# Add translation filter to Jinja2
def translate_filter(key, language='en'):
    return get_translation(key, language)

templates.env.filters['t'] = translate_filter

# Configure Gemini (AI Vertex default)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL", "gemini-2.5-flash-lite")
GEMINI_ANALYSIS_MODEL = os.getenv("GEMINI_ANALYSIS_MODEL", "gemini-2.5-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class EvaluationResult:
    def __init__(
        self, question: str, reference: str, answer: str, score: float, reasoning: str, 
        input_tokens: int = 0, output_tokens: int = 0
    ):
        self.question = question
        self.reference = reference
        self.answer = answer
        self.score = score
        self.reasoning = reasoning
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class SimpleEvaluator:
    def __init__(self):
        self.judge_model = genai.GenerativeModel(GEMINI_JUDGE_MODEL) if GEMINI_API_KEY else None
        self.analysis_model = genai.GenerativeModel(GEMINI_ANALYSIS_MODEL) if GEMINI_API_KEY else None

    async def generate_summary_insights(self, results: List[EvaluationResult], filename: str, language: str = 'en') -> Dict[str, str]:
        """Generate intelligent summary and insights from all evaluation results"""
        if not self.analysis_model:
            return {
                "summary": "Your AI system was evaluated against the reference answers. Review individual scores for detailed feedback.",
                "key_insights": [
                    "No AI model configured for detailed analysis",
                    "Consider reviewing individual question results",
                    "Look for patterns in scoring to identify improvement areas"
                ],
                "recommendations": [
                    "Configure Gemini API key for detailed insights",
                    "Focus on questions with lower scores",
                    "Consider improving AI training data"
                ]
            }

        # Prepare data for LLM analysis
        evaluation_data = []
        for i, result in enumerate(results, 1):
            evaluation_data.append(f"""
Question {i}: {result.question}
Reference Answer: {result.reference}
AI Answer: {result.answer}
Score: {result.score:.2f}/1.0
AI Judgment: {result.reasoning}
---""")

        # Calculate basic stats
        total_questions = len(results)
        average_score = sum(r.score for r in results) / total_questions if total_questions > 0 else 0
        pass_rate = sum(1 for r in results if r.score >= 0.7) / total_questions if total_questions > 0 else 0
        
        # Create language-specific prompts
        if language == 'fr':
            analysis_prompt = f"""Vous êtes un consultant en affaires aidant un chef de produit à comprendre les performances de leur chatbot pour leurs parties prenantes et leur équipe.

RAPPORT DE PERFORMANCE DU CHATBOT:
- Fichier Analysé: {filename}
- Questions Testées: {total_questions}
- Taux de Réussite: {average_score:.1%}
- Taux de Satisfaction Client: {pass_rate:.1%} (réponses qui satisferaient les clients)

INTERACTIONS CLIENT DÉTAILLÉES:
{''.join(evaluation_data)}

Veuillez fournir une réponse JSON avec des insights de résumé exécutif:
{{
    "summary": "Écrivez 2-3 phrases expliquant à quel point ce chatbot sert bien les clients, dans un langage qu'un PDG comprendrait",
    "key_insights": [
        "Listez 3-5 insights sur ce que le chatbot fait bien ou avec quoi il a des difficultés",
        "Concentrez-vous sur l'impact client: 'Les clients obtiennent des réponses utiles pour X mais ont des difficultés avec Y'",
        "Utilisez des termes commerciaux, pas de jargon technique"
    ],
    "recommendations": [
        "Listez 3-5 actions spécifiques que l'équipe produit devrait entreprendre",
        "Concentrez-vous sur l'impact commercial: 'Améliorez X pour réduire la frustration client' ou 'Formez sur Y pour augmenter la satisfaction'"
    ]
}}

Pensez comme un consultant en affaires répondant à:
• Les clients obtiennent-ils l'aide dont ils ont besoin?
• Quels patterns frustreraient ou raviraient les clients?
• Sur quoi l'équipe produit devrait-elle se concentrer pour améliorer l'expérience client?
• Comment pouvons-nous rendre ce chatbot plus précieux pour l'entreprise?

IMPORTANT: Répondez entièrement en français."""
        else:
            analysis_prompt = f"""You are a business consultant helping a product manager understand their chatbot's performance for their stakeholders and team.

CHATBOT PERFORMANCE REPORT:
- File Analyzed: {filename}
- Questions Tested: {total_questions}
- Success Rate: {average_score:.1%}
- Customer Satisfaction Rate: {pass_rate:.1%} (answers that would satisfy customers)

DETAILED CUSTOMER INTERACTIONS:
{''.join(evaluation_data)}

Please provide a JSON response with executive summary insights:
{{
    "summary": "Write 2-3 sentences explaining how well this chatbot serves customers, in language a CEO would understand",
    "key_insights": [
        "List 3-5 insights about what the chatbot does well or struggles with",
        "Focus on customer impact: 'Customers get helpful answers for X but struggle with Y'",
        "Use business terms, not technical jargon"
    ],
    "recommendations": [
        "List 3-5 specific actions the product team should take",
        "Focus on business impact: 'Improve X to reduce customer frustration' or 'Train on Y to increase satisfaction'"
    ]
}}

Think like a business consultant answering:
• Are customers getting the help they need?
• What patterns would frustrate or delight customers?
• What should the product team prioritize to improve customer experience?
• How can we make this chatbot more valuable for the business?

Write insights that a product manager can share with executives or use to guide their team's priorities."""

        try:
            response = self.analysis_model.generate_content(analysis_prompt)
            analysis_text = response.text
            
            # Track token usage for analysis
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            
            # Try to parse JSON response
            import json
            # Clean the response to extract JSON
            if "```json" in analysis_text:
                json_start = analysis_text.find("```json") + 7
                json_end = analysis_text.find("```", json_start)
                analysis_text = analysis_text[json_start:json_end].strip()
            elif "{" in analysis_text:
                json_start = analysis_text.find("{")
                json_end = analysis_text.rfind("}") + 1
                analysis_text = analysis_text[json_start:json_end]
            
            analysis_data = json.loads(analysis_text)
            
            return {
                "summary": analysis_data.get("summary", "Analysis completed successfully."),
                "key_insights": analysis_data.get("key_insights", []),
                "recommendations": analysis_data.get("recommendations", [])
            }
            
        except Exception as e:
            # Fallback analysis based on scores
            high_scores = sum(1 for r in results if r.score >= 0.9)
            low_scores = sum(1 for r in results if r.score < 0.5)
            
            return {
                "summary": f"Your AI system achieved an average score of {average_score:.1%} across {total_questions} questions, with {pass_rate:.1%} meeting the quality threshold.",
                "key_insights": [
                    f"Strong performance on {high_scores} questions (≥90% score)" if high_scores > 0 else "No questions achieved excellent scores (≥90%)",
                    f"Significant issues with {low_scores} questions (<50% score)" if low_scores > 0 else "No questions had critical issues (<50% score)",
                    f"Pass rate of {pass_rate:.1%} indicates {'good' if pass_rate >= 0.7 else 'poor'} overall performance",
                    "Review individual question details for specific improvement areas"
                ],
                "recommendations": [
                    "Focus on improving responses for lowest-scoring questions",
                    "Analyze patterns in successful vs unsuccessful responses",
                    "Consider expanding training data for problematic question types",
                    f"Target improvement to reach ≥80% pass rate" if pass_rate < 0.8 else "Maintain current performance level"
                ]
            }

    async def evaluate_answer(
        self, question: str, reference: str, answer: str
    ) -> EvaluationResult:
        """Evaluate a single answer against the reference"""
        if not self.judge_model:
            # Fallback when no API key configured
            return EvaluationResult(
                question=question,
                reference=reference,
                answer=answer,
                score=0.0,
                reasoning="No AI model configured. Cannot evaluate without Gemini API key.",
            )

        prompt = f"""You are a business consultant helping a product manager understand if their chatbot is working well for customers.

CUSTOMER'S QUESTION: {question}
IDEAL ANSWER: {reference}
CHATBOT'S ACTUAL ANSWER: {answer}

Your job: Decide if a customer would be satisfied with this chatbot answer.

Ask yourself:
• Does this actually help the customer solve their problem?
• Is the information correct and useful?
• Would a customer feel their question was properly answered?

DECISION RULES:
✅ PASS (Score: 1) if the answer:
- Gives the customer the help they need
- Is factually correct (even if worded differently than the ideal answer)
- Actually addresses what the customer asked about
- Minor style differences are fine if the core help is there

❌ FAIL (Score: 0) if the answer:
- Contains wrong information that could mislead the customer
- Doesn't actually help with what the customer asked
- Is confusing, irrelevant, or unhelpful
- Misses critical information the customer needs

Format your response as:
SCORE: [either 0 or 1]
REASONING: [Explain in business terms why this would help or frustrate a customer]"""

        try:
            response = self.judge_model.generate_content(prompt)
            text = response.text

            # Get token usage from response
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0

            # Parse score and reasoning
            score = 0.0  # Default to fail if parsing fails
            reasoning = "Could not parse evaluation"

            lines = text.split("\n")
            for line in lines:
                if line.startswith("SCORE:"):
                    try:
                        parsed_score = float(line.split(":", 1)[1].strip())
                        # Force binary: round to nearest integer (0 or 1)
                        score = 1.0 if parsed_score >= 0.5 else 0.0
                    except:
                        score = 0.0  # Default to fail on parsing error
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            return EvaluationResult(
                question=question,
                reference=reference,
                answer=answer,
                score=score,
                reasoning=reasoning,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            return EvaluationResult(
                question=question,
                reference=reference,
                answer=answer,
                score=0.0,
                reasoning=f"Evaluation error: {str(e)}",
                input_tokens=0,
                output_tokens=0,
            )


def get_user_id(request: Request) -> str:
    """Get user ID based on IP address and user agent for rate limiting"""
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    
    # Include additional headers to better differentiate sessions
    # This helps distinguish between regular and incognito sessions
    sec_fetch_site = request.headers.get("sec-fetch-site", "")
    accept_language = request.headers.get("accept-language", "")
    
    # Simple hash to create consistent user ID
    import hashlib
    user_string = f"{client_ip}:{user_agent}:{sec_fetch_site}:{accept_language}"
    return hashlib.md5(user_string.encode()).hexdigest()[:16]


async def get_current_daily_evaluations(user_id: str) -> int:
    """Get current daily evaluation count for user"""
    from datetime import datetime, timezone
    
    if not analytics or not analytics.db:
        return 0
    
    try:
        user_ref = analytics.db.collection('user_metrics').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return 0  # New user
        
        user_data = user_doc.to_dict()
        last_activity = user_data.get('last_activity')
        daily_evaluations = user_data.get('daily_evaluations', 0)
        
        # Reset daily count if last activity was yesterday or earlier
        if last_activity:
            if isinstance(last_activity, str):
                last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
            
            now = datetime.now(timezone.utc)
            if last_activity.date() < now.date():
                # Reset daily count for new day
                return 0
        
        return daily_evaluations
        
    except Exception as e:
        # Return 0 on error to avoid blocking users
        return 0


async def check_daily_evaluation_limit(user_id: str) -> bool:
    """Check if user has exceeded daily evaluation limit (3 per day)"""
    current_count = await get_current_daily_evaluations(user_id)
    return current_count < 3


async def increment_daily_evaluation_count(user_id: str) -> None:
    """Increment user's daily evaluation count"""
    from datetime import datetime, timezone
    
    if not analytics or not analytics.db:
        return
    
    try:
        user_ref = analytics.db.collection('user_metrics').document(user_id)
        user_doc = user_ref.get()
        
        # Get current count using consistent logic
        current_count = await get_current_daily_evaluations(user_id)
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            user_data['daily_evaluations'] = current_count + 1
            user_data['last_activity'] = datetime.now(timezone.utc)
            user_ref.set(user_data)
        else:
            # New user
            user_data = {
                'first_seen': datetime.now(timezone.utc),
                'total_visits': 0,
                'total_evaluations': 0,
                'total_pdf_downloads': 0,
                'dataset_sizes': [],
                'daily_evaluations': 1,
                'last_activity': datetime.now(timezone.utc)
            }
            user_ref.set(user_data)
            
    except Exception as e:
        # Continue silently on error
        pass


def process_file(file_content, filename: str) -> List[Dict[str, str]]:
    """Process uploaded file and extract question, reference, answer columns"""
    rows = []

    if filename.endswith(".csv"):
        # Parse CSV
        lines = file_content.strip().split("\n")
        reader = csv.DictReader(lines)
        for row in reader:
            rows.append(
                {
                    "question": row.get("question", ""),
                    "reference": row.get("reference", ""),
                    "answer": row.get("answer", ""),
                }
            )

    elif filename.endswith(".jsonl"):
        # Parse JSONL
        for line in file_content.strip().split("\n"):
            if line.strip():
                try:
                    data = json.loads(line)
                    rows.append(
                        {
                            "question": data.get("question", ""),
                            "reference": data.get("reference", ""),
                            "answer": data.get("answer", ""),
                        }
                    )
                except json.JSONDecodeError:
                    continue

    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        # Parse Excel files using pandas
        try:
            import io
            # For Excel files, we need to handle binary content differently
            # We'll need to modify the calling code to handle this properly
            df = pd.read_excel(io.StringIO(file_content) if isinstance(file_content, str) else io.BytesIO(file_content))
            
            # Check if required columns exist
            required_columns = ["question", "reference", "answer"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Excel file missing columns: {', '.join(missing_columns)}")
                return []
            
            # Convert to list of dictionaries
            for _, row in df.iterrows():
                # Skip rows where all required fields are empty
                question = str(row.get("question", "")).strip()
                reference = str(row.get("reference", "")).strip() 
                answer = str(row.get("answer", "")).strip()
                
                if question and reference and answer:  # Only include rows with all fields
                    rows.append({
                        "question": question,
                        "reference": reference,
                        "answer": answer,
                    })
        except Exception as e:
            print(f"Error processing Excel file: {e}")
            return []

    return rows


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page with upload widget"""
    language = detect_language(request)
    return templates.TemplateResponse("landing.html", {
        "request": request, 
        "language": language
    })


@app.post("/evaluation")
async def upload_and_evaluate(
    request: Request,
    file: UploadFile = File(...),
):
    """Handle file upload and show loading, then run evaluation"""

    # Validate file type
    if not file.filename.endswith((".csv", ".jsonl", ".xlsx", ".xls")):
        return templates.TemplateResponse(
            "landing.html",
            {"request": request, "error": "Please upload a CSV, JSONL, or Excel file"},
        )

    try:
        # Read file content
        content = await file.read()
        
        # Process file - handle Excel files differently (binary) vs text files
        if file.filename.endswith((".xlsx", ".xls")):
            rows = process_file(content, file.filename)  # Pass binary content for Excel
        else:
            content_str = content.decode("utf-8")
            rows = process_file(content_str, file.filename)  # Pass text content for CSV/JSONL

        if not rows:
            return templates.TemplateResponse(
                "landing.html",
                {
                    "request": request,
                    "error": "Could not process file or file is empty",
                },
            )

        # Validate required columns
        missing_columns = []
        for row in rows[:1]:  # Check first row
            if not row.get("question"):
                missing_columns.append("question")
            if not row.get("reference"):
                missing_columns.append("reference")
            if not row.get("answer"):
                missing_columns.append("answer")

        if missing_columns:
            return templates.TemplateResponse(
                "landing.html",
                {
                    "request": request,
                    "error": f"Missing required columns: {', '.join(missing_columns)}",
                },
            )

        # At this point we know the actual number of questions
        total_questions = len(rows)
        
        # Get user ID for rate limiting
        user_id = get_user_id(request)
        
        # Check file size limit (50 questions max)
        # Note: For CSV files, headers are automatically excluded from the count
        if total_questions > 50:
            file_type = "rows" if file.filename.endswith(".jsonl") else "questions"
            return templates.TemplateResponse(
                "landing.html",
                {
                    "request": request,
                    "error": f"File contains {total_questions} {file_type}. Maximum allowed is 50 questions per file. Please join our waiting list for higher limits.",
                    "show_waitlist": True
                },
            )
        
        # Check daily evaluation limit (3 evaluations per day)
        if not await check_daily_evaluation_limit(user_id):
            return templates.TemplateResponse(
                "landing.html",
                {
                    "request": request,
                    "error": "You have reached the daily limit of 3 evaluations. Please join our waiting list for higher limits.",
                    "show_waitlist": True
                },
            )
        
        # Increment daily evaluation count
        await increment_daily_evaluation_count(user_id)
        
        # Run evaluation
        evaluator = SimpleEvaluator()
        results = []

        for row in rows:
            result = await evaluator.evaluate_answer(
                row["question"], row["reference"], row["answer"]
            )
            results.append(result)

        # Detect language for AI analysis
        language = detect_language(request)
        
        # Generate AI-powered summary and insights
        ai_analysis = await evaluator.generate_summary_insights(results, file.filename, language)

        # Calculate key insights
        total_questions = len(results)
        average_score = (
            sum(r.score for r in results) / total_questions
            if total_questions > 0
            else 0
        )
        passing_threshold = 0.7
        passing_count = sum(1 for r in results if r.score >= passing_threshold)
        pass_rate = passing_count / total_questions if total_questions > 0 else 0

        # Calculate token usage
        total_input_tokens = sum(r.input_tokens for r in results)
        total_output_tokens = sum(r.output_tokens for r in results)
        total_tokens = total_input_tokens + total_output_tokens

        # Categorize results
        excellent = sum(1 for r in results if r.score >= 0.9)
        good = sum(1 for r in results if 0.7 <= r.score < 0.9)
        needs_improvement = sum(1 for r in results if 0.5 <= r.score < 0.7)
        poor = sum(1 for r in results if r.score < 0.5)

        insights = {
            "total_questions": total_questions,
            "average_score": average_score,
            "pass_rate": pass_rate,
            "total_tokens": total_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "distribution": {
                "excellent": excellent,
                "good": good,
                "needs_improvement": needs_improvement,
                "poor": poor,
            },
            "ai_analysis": ai_analysis
        }

        # Track evaluation completion in analytics
        if analytics:
            try:
                await analytics.track_event("dataset_uploaded", user_id, {
                    "dataset_size": total_questions,
                    "filename": file.filename,
                    "average_score": average_score,
                    "pass_rate": pass_rate,
                    "total_tokens": total_tokens,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens
                })
            except Exception as e:
                # Don't fail the evaluation if analytics fails
                print(f"Analytics tracking failed: {e}")

        # Store results in cache with a unique ID for PDF generation
        result_id = str(uuid.uuid4())
        evaluation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        results_cache[result_id] = {
            "results": results,
            "insights": insights,
            "filename": file.filename,
            "evaluation_time": evaluation_time,
            "language": language
        }

        
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "results": results,
                "insights": insights,
                "filename": file.filename,
                "evaluation_time": evaluation_time,
                "result_id": result_id,
                "language": language,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "landing.html",
            {"request": request, "error": f"Error processing file: {str(e)}"},
        )


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_page(request: Request):
    """Evaluation page - redirects to landing for upload"""
    return RedirectResponse(url="/", status_code=302)


@app.get("/evaluations", response_class=HTMLResponse)
async def evaluations_page(request: Request):
    """Simple evaluations page - redirects to landing for now"""
    return RedirectResponse(url="/", status_code=302)


@app.get("/download-pdf/{result_id}")
async def download_pdf(result_id: str, request: Request, user_id: str = None):
    """Generate and download PDF report"""
    if result_id not in results_cache:
        raise HTTPException(status_code=404, detail="Results not found")
    
    cached_data = results_cache[result_id]
    
    try:
        # Use cached language or detect as fallback
        language = cached_data.get("language", detect_language(request))
        
        # Generate PDF
        pdf_generator = EvaluationPDFGenerator()
        pdf_buffer = pdf_generator.generate_pdf(
            results=cached_data["results"],
            insights=cached_data["insights"],
            filename=cached_data["filename"],
            evaluation_time=cached_data["evaluation_time"],
            language=language
        )
        
        # Track PDF download analytics
        if analytics:
            # Use provided user_id or generate a fallback
            analytics_user_id = user_id or f"anon_{request.client.host}_{datetime.now().strftime('%Y%m%d')}"
            await analytics.track_event("pdf_downloaded", analytics_user_id, {"result_id": result_id})
        
        # Create safe filename
        safe_filename = cached_data["filename"].replace(" ", "_").replace("/", "_")
        pdf_filename = f"evalnow_report_{safe_filename}_{result_id[:8]}.pdf"
        
        # Return PDF as download
        return StreamingResponse(
            io.BytesIO(pdf_buffer.read()),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={pdf_filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")


@app.get("/waitlist", response_class=HTMLResponse)
async def waitlist_page(request: Request):
    """Waiting list signup page"""
    return templates.TemplateResponse("waitlist.html", {"request": request})


@app.post("/waitlist")
async def join_waitlist(
    request: Request,
    email: str = Form(...),
    company: str = Form(default=""),
    use_case: str = Form(default=""),
):
    """Handle waiting list signup"""
    try:
        # Validate email format
        import re
        if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            return templates.TemplateResponse(
                "waitlist.html",
                {
                    "request": request,
                    "error": "Please enter a valid email address",
                    "email": email,
                    "company": company,
                    "use_case": use_case,
                }
            )
        
        # Check if already on waiting list
        if analytics and analytics.db:
            existing = analytics.db.collection('waiting_list').where('email', '==', email).limit(1).get()
            if len(list(existing)) > 0:
                return templates.TemplateResponse(
                    "waitlist.html",
                    {
                        "request": request,
                        "success": True,
                        "message": "You're already on our waiting list! We'll notify you when higher limits are available.",
                    }
                )
        
        # Add to waiting list
        await add_to_waitlist(email, company, use_case, request)
        
        return templates.TemplateResponse(
            "waitlist.html",
            {
                "request": request,
                "success": True,
                "message": "Successfully joined the waiting list! We'll notify you when higher limits are available.",
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "waitlist.html",
            {
                "request": request,
                "error": f"Error joining waiting list: {str(e)}",
                "email": email,
                "company": company,
                "use_case": use_case,
            }
        )


async def add_to_waitlist(email: str, company: str, use_case: str, request: Request):
    """Add user to waiting list"""
    if not analytics or not analytics.db:
        return
    
    from datetime import datetime, timezone
    
    try:
        user_id = get_user_id(request)
        
        waitlist_data = {
            'email': email,
            'company': company,
            'use_case': use_case,
            'user_id': user_id,
            'requested_at': datetime.now(timezone.utc),
            'status': 'pending',
            'priority': 0,  # Can be used for prioritization later
            'ip_address': request.client.host,
            'user_agent': request.headers.get("user-agent", "")
        }
        
        analytics.db.collection('waiting_list').add(waitlist_data)
        
    except Exception as e:
        # Log error but don't fail
        pass


@app.get("/api/waitlist")
async def get_waitlist(request: Request, credentials: HTTPBasicCredentials = Depends(verify_admin)):
    """Get waiting list entries - admin only"""
    try:
        if not analytics or not analytics.db:
            return JSONResponse(content={"error": "Analytics not enabled"})
        
        # Get all waiting list entries, ordered by requested_at
        entries = analytics.db.collection('waiting_list').order_by('requested_at').stream()
        
        waitlist = []
        for entry in entries:
            data = entry.to_dict()
            data['id'] = entry.id
            # Convert datetime fields to ISO string for JSON serialization
            if 'requested_at' in data:
                data['requested_at'] = data['requested_at'].isoformat()
            if 'approved_at' in data:
                data['approved_at'] = data['approved_at'].isoformat()
            waitlist.append(data)
        
        return JSONResponse(content={'waitlist': waitlist})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/waitlist/{entry_id}/approve")
async def approve_waitlist_entry(
    entry_id: str, 
    credentials: HTTPBasicCredentials = Depends(verify_admin)
):
    """Approve waiting list entry - admin only"""
    from datetime import datetime, timezone
    
    try:
        if not analytics or not analytics.db:
            return JSONResponse(content={"error": "Analytics not enabled"})
        
        # Update entry status to approved
        entry_ref = analytics.db.collection('waiting_list').document(entry_id)
        entry_ref.update({
            'status': 'approved',
            'approved_at': datetime.now(timezone.utc)
        })
        
        return JSONResponse(content={"status": "approved"})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.delete("/api/waitlist/{entry_id}")
async def remove_waitlist_entry(
    entry_id: str, 
    credentials: HTTPBasicCredentials = Depends(verify_admin)
):
    """Remove waiting list entry - admin only"""
    try:
        if not analytics or not analytics.db:
            return JSONResponse(content={"error": "Analytics not enabled"})
        
        # Delete entry
        analytics.db.collection('waiting_list').document(entry_id).delete()
        
        return JSONResponse(content={"status": "deleted"})
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/analytics/track")
async def track_analytics(request: Request):
    """Track analytics events from client-side"""
    try:
        data = await request.json()
        if analytics:
            await analytics.track_event(
                event_type=data.get("event_type"),
                user_id=data.get("user_id"),
                data={k: v for k, v in data.items() if k not in ["event_type", "user_id"]}
            )
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.get("/api/analytics/kpis")
async def get_kpis(request: Request, credentials: HTTPBasicCredentials = Depends(verify_admin)):
    """Get KPI dashboard data - admin only"""
    try:
        if analytics:
            kpis = await analytics.get_kpi_summary()
        else:
            kpis = {"error": "Analytics not available"}
        return JSONResponse(content=kpis)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/api/feedback")
async def submit_feedback(request: Request):
    """Submit user feedback"""
    try:
        data = await request.json()
        
        # Add server-side timestamp
        feedback_entry = {
            **data,
            "server_timestamp": datetime.utcnow().isoformat(),
            "feedback_id": str(uuid.uuid4())
        }
        
        # Store in analytics if available
        if analytics:
            await analytics.track_event(
                event_type="feedback_submitted",
                user_id=data.get("result_id", "anonymous"),
                data=feedback_entry
            )
            
            # Also store in a dedicated feedback collection
            try:
                from google.cloud import firestore
                # Use the same custom database as analytics
                database_name = os.getenv("FIRESTORE_DATABASE", "evalnow-db")
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                db = firestore.Client(project=project_id, database=database_name)
                doc_ref = db.collection("feedback").document(feedback_entry["feedback_id"])
                doc_ref.set(feedback_entry)
            except Exception as e:
                print(f"Error saving feedback to Firestore: {e}")
        
        return {"success": True, "feedback_id": feedback_entry["feedback_id"]}
        
    except Exception as e:
        print(f"Feedback submission error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


@app.get("/api/feedback")
async def get_feedback(request: Request, credentials: HTTPBasicCredentials = Depends(verify_admin)):
    """Get all feedback - admin only"""
    try:
        feedback_list = []
        
        if analytics:
            try:
                from google.cloud import firestore
                # Use the same custom database as analytics
                database_name = os.getenv("FIRESTORE_DATABASE", "evalnow-db")
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                db = firestore.Client(project=project_id, database=database_name)
                
                # Get feedback from the past 30 days
                from datetime import timedelta
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                
                feedback_ref = db.collection("feedback")
                query = feedback_ref.where("server_timestamp", ">=", cutoff_date.isoformat()).order_by("server_timestamp", direction=firestore.Query.DESCENDING)
                
                for doc in query.stream():
                    feedback_data = doc.to_dict()
                    feedback_list.append(feedback_data)
                    
            except Exception as e:
                print(f"Error fetching feedback: {e}")
                return {"feedback": [], "error": f"Database error: {str(e)}"}
        
        return {"feedback": feedback_list, "total": len(feedback_list)}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_dashboard(request: Request, credentials: HTTPBasicCredentials = Depends(verify_admin)):
    """Analytics dashboard - admin only"""
    return templates.TemplateResponse("analytics.html", {"request": request})



@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page"""
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/privacy", response_class=HTMLResponse)
async def privacy_page(request: Request):
    """Privacy policy page"""
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    return templates.TemplateResponse("privacy.html", {"request": request, "current_date": current_date})


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "evalnow"}
