import os
import json
import tempfile
import csv
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="Simple Eval", description="Simple AI evaluation tool")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Configure Gemini (AI Vertex default)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


class EvaluationResult:
    def __init__(
        self, question: str, reference: str, answer: str, score: float, reasoning: str
    ):
        self.question = question
        self.reference = reference
        self.answer = answer
        self.score = score
        self.reasoning = reasoning


class SimpleEvaluator:
    def __init__(self):
        self.model = genai.GenerativeModel(GEMINI_MODEL) if GEMINI_API_KEY else None

    async def generate_summary_insights(self, results: List[EvaluationResult], filename: str) -> Dict[str, str]:
        """Generate intelligent summary and insights from all evaluation results"""
        if not self.model:
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
            response = self.model.generate_content(analysis_prompt)
            analysis_text = response.text
            
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
        if not self.model:
            # Fallback simple scoring
            return EvaluationResult(
                question=question,
                reference=reference,
                answer=answer,
                score=0.5,
                reasoning="No AI model configured. Using default score.",
            )

        prompt = f"""You are an expert evaluator. Please evaluate the given answer to the question against the reference answer.

Question: {question}
Reference Answer: {reference}
Given Answer: {answer}

Evaluate on these criteria:
1. Accuracy: Is the answer factually correct?
2. Completeness: Does it fully answer the question?
3. Relevance: Does it directly address what was asked?

Please provide:
1. A score from 0.0 to 1.0 (where 1.0 is perfect)
2. Brief reasoning for your score

Format your response as:
SCORE: [number between 0.0 and 1.0]
REASONING: [your explanation]"""

        try:
            response = self.model.generate_content(prompt)
            text = response.text

            # Parse score and reasoning
            score = 0.5
            reasoning = "Could not parse evaluation"

            lines = text.split("\n")
            for line in lines:
                if line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                    except:
                        pass
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            return EvaluationResult(
                question=question,
                reference=reference,
                answer=answer,
                score=score,
                reasoning=reasoning,
            )

        except Exception as e:
            return EvaluationResult(
                question=question,
                reference=reference,
                answer=answer,
                score=0.0,
                reasoning=f"Evaluation error: {str(e)}",
            )


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

        # Categorize results
        excellent = sum(1 for r in results if r.score >= 0.9)
        good = sum(1 for r in results if 0.7 <= r.score < 0.9)
        needs_improvement = sum(1 for r in results if 0.5 <= r.score < 0.7)
        poor = sum(1 for r in results if r.score < 0.5)

        insights = {
            "total_questions": total_questions,
            "average_score": average_score,
            "pass_rate": pass_rate,
            "distribution": {
                "excellent": excellent,
                "good": good,
                "needs_improvement": needs_improvement,
                "poor": poor,
            },
            "ai_analysis": ai_analysis
        }

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "results": results,
                "insights": insights,
                "filename": file.filename,
                "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/evaluations", response_class=HTMLResponse)
async def evaluations_page(request: Request):
    """Simple evaluations page - redirects to landing for now"""
    return templates.TemplateResponse("landing.html", {"request": request})


@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "simple-eval"}
