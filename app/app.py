import os
import json
import tempfile
import csv
import pandas as pd
import io
import uuid
from typing import List, Dict, Any
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

    async def generate_summary_insights(self, results: List[EvaluationResult], filename: str) -> Dict[str, str]:
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
        
        # Create comprehensive prompt for analysis
        analysis_prompt = f"""You are an expert AI evaluation analyst. Please analyze the following evaluation results and provide a comprehensive summary with actionable insights.

EVALUATION OVERVIEW:
- File: {filename}
- Total Questions: {total_questions}
- Average Score: {average_score:.1%}
- Pass Rate (≥70%): {pass_rate:.1%}

DETAILED EVALUATION DATA:
{''.join(evaluation_data)}

Please provide a JSON response with the following structure:
{{
    "summary": "A 2-3 sentence executive summary of the overall AI performance",
    "key_insights": [
        "List 3-5 specific insights about patterns, strengths, and weaknesses you observe",
        "Focus on actionable observations from the data"
    ],
    "recommendations": [
        "List 3-5 specific recommendations for improving the AI system",
        "Base recommendations on the actual evaluation results"
    ]
}}

Focus on:
1. Common patterns in correct vs incorrect responses
2. Types of questions where the AI excels or struggles
3. Quality of reasoning in AI judgments
4. Specific areas for improvement
5. Actionable recommendations based on the data

Provide concrete, specific insights rather than generic advice."""

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

        prompt = f"""You are an expert evaluator. Please evaluate the given answer to the question against the reference answer.

Question: {question}
Reference Answer: {reference}
Given Answer: {answer}

Evaluate on these criteria:
1. Accuracy: Is the answer factually correct?
2. Completeness: Does it fully answer the question?
3. Relevance: Does it directly address what was asked?

You must provide a BINARY evaluation - either the answer is acceptable (1) or not acceptable (0).

An answer gets a score of 1 (PASS) if:
- It is factually correct AND addresses the question adequately
- Minor wording differences are acceptable if the meaning is correct
- The core information matches the reference answer

An answer gets a score of 0 (FAIL) if:
- It contains factual errors
- It doesn't answer the question asked
- It's completely irrelevant or nonsensical
- It's missing critical information

Format your response as:
SCORE: [either 0 or 1]
REASONING: [your explanation - be specific about why it passed or failed]"""

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
    # Simple hash to create consistent user ID
    import hashlib
    user_string = f"{client_ip}:{user_agent}"
    return hashlib.md5(user_string.encode()).hexdigest()[:16]


async def check_daily_evaluation_limit(user_id: str) -> bool:
    """Check if user has exceeded daily evaluation limit (3 per day)"""
    from datetime import datetime, timezone, timedelta
    
    if not analytics or not analytics.db:
        return True  # Allow if analytics disabled
    
    try:
        user_ref = analytics.db.collection('user_metrics').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return True  # New user, allow
        
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
                daily_evaluations = 0
        else:
            daily_evaluations = 0
        
        return daily_evaluations < 3
        
    except Exception as e:
        # Allow on error to avoid blocking users
        return True


async def increment_daily_evaluation_count(user_id: str) -> None:
    """Increment user's daily evaluation count"""
    from datetime import datetime, timezone
    
    if not analytics or not analytics.db:
        return
    
    try:
        user_ref = analytics.db.collection('user_metrics').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            last_activity = user_data.get('last_activity')
            daily_evaluations = user_data.get('daily_evaluations', 0)
            
            # Reset daily count if last activity was yesterday or earlier
            if last_activity:
                if isinstance(last_activity, str):
                    last_activity = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                
                now = datetime.now(timezone.utc)
                if last_activity.date() < now.date():
                    daily_evaluations = 0
            
            user_data['daily_evaluations'] = daily_evaluations + 1
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


def process_file(file_content: str, filename: str) -> List[Dict[str, str]]:
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
        # For Excel files, we'll need to handle this differently
        # For now, return empty to trigger error
        pass

    return rows


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page with upload widget"""
    return templates.TemplateResponse("landing.html", {"request": request})


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
        content_str = content.decode("utf-8")

        # Process file
        rows = process_file(content_str, file.filename)

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
        if total_questions > 50:
            return templates.TemplateResponse(
                "landing.html",
                {
                    "request": request,
                    "error": f"File contains {total_questions} questions. Maximum allowed is 50 questions per file. Please join our waiting list for higher limits.",
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

        # Generate AI-powered summary and insights
        ai_analysis = await evaluator.generate_summary_insights(results, file.filename)

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
            "evaluation_time": evaluation_time
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
        # Generate PDF
        pdf_generator = EvaluationPDFGenerator()
        pdf_buffer = pdf_generator.generate_pdf(
            results=cached_data["results"],
            insights=cached_data["insights"],
            filename=cached_data["filename"],
            evaluation_time=cached_data["evaluation_time"]
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
