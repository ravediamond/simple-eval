import os
import json
from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from app.database import init_db, get_db
from app.models import Dataset, DatasetVersion, Agent, AgentVersion, Run, TestCase, MetricResult
from app.dataset_utils import DatasetProcessor
from app.run_utils import AnswersProcessor, RunProcessor
from app.export_utils import RunExporter, ResultsFilter
from app.connectors import ConnectorFactory, ConnectorConfig

load_dotenv()

app = FastAPI(title="Simple Eval", description="Lightweight chatbot evaluation platform")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "simple-eval"}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Main dashboard page"""
    # Get recent runs for dashboard
    recent_runs = db.query(Run).order_by(Run.created_at.desc()).limit(5).all()
    
    # Get summary stats
    total_agents = db.query(Agent).count()
    total_datasets = db.query(Dataset).count()
    total_runs = db.query(Run).count()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "recent_runs": recent_runs,
        "total_agents": total_agents,
        "total_datasets": total_datasets,
        "total_runs": total_runs
    })

@app.get("/agents", response_class=HTMLResponse)
async def agents(request: Request, db: Session = Depends(get_db)):
    """Agents page"""
    agents = db.query(Agent).all()
    return templates.TemplateResponse("agents.html", {
        "request": request,
        "agents": agents
    })

@app.get("/datasets", response_class=HTMLResponse)
async def datasets(request: Request, db: Session = Depends(get_db)):
    """Datasets page"""
    datasets = db.query(Dataset).all()
    return templates.TemplateResponse("datasets.html", {
        "request": request, 
        "datasets": datasets
    })

@app.get("/datasets/upload", response_class=HTMLResponse)
async def upload_dataset_form(request: Request):
    """Upload dataset form"""
    return templates.TemplateResponse("upload_dataset.html", {"request": request})

@app.post("/datasets/upload")
async def upload_dataset(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Handle dataset upload"""
    # Read file content
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
    except Exception as e:
        return templates.TemplateResponse("upload_dataset.html", {
            "request": request,
            "error": f"Error reading file: {str(e)}"
        })
    
    # Process file
    rows, errors = DatasetProcessor.process_file(content_str, file.filename)
    
    # Check for duplicate IDs
    if rows:
        duplicate_errors = DatasetProcessor.check_duplicate_ids(rows)
        errors.extend(duplicate_errors)
    
    # If there are errors, show preview with errors
    if errors:
        return templates.TemplateResponse("upload_dataset.html", {
            "request": request,
            "errors": errors,
            "preview_rows": rows[:10] if rows else [],
            "total_rows": len(rows),
            "name": name,
            "description": description
        })
    
    # Create dataset and version
    try:
        # Create dataset
        dataset = Dataset(name=name, description=description)
        db.add(dataset)
        db.flush()  # Get the ID
        
        # Save normalized data
        file_path = DatasetProcessor.save_normalized_data(rows, dataset.id, 1)
        
        # Create dataset version
        content_hash = DatasetVersion.generate_content_hash(rows)
        version = DatasetVersion(
            dataset_id=dataset.id,
            version_number=1,
            notes="Initial version",
            row_count=len(rows),
            content_hash=content_hash,
            file_path=file_path
        )
        db.add(version)
        db.commit()
        
        return RedirectResponse(url="/datasets", status_code=302)
    
    except Exception as e:
        db.rollback()
        return templates.TemplateResponse("upload_dataset.html", {
            "request": request,
            "error": f"Error saving dataset: {str(e)}"
        })

@app.get("/datasets/{dataset_id}/versions/{version_id}/preview")
async def preview_dataset(dataset_id: int, version_id: int, db: Session = Depends(get_db)):
    """Get preview of dataset version"""
    version = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id,
        DatasetVersion.id == version_id
    ).first()
    
    if not version:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    
    # Read first 10 rows from file
    try:
        rows = []
        with open(version.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                rows.append(json.loads(line))
        
        return JSONResponse({
            "rows": rows,
            "total_rows": version.row_count,
            "content_hash": version.content_hash
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

@app.get("/agents/new", response_class=HTMLResponse)
async def new_agent_form(request: Request):
    """New agent form"""
    return templates.TemplateResponse("new_agent.html", {"request": request})

@app.post("/agents")
async def create_agent(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    tags: str = Form(""),
    model_provider: str = Form(...),
    model_name: str = Form(...),
    temperature: float = Form(0.7),
    max_tokens: int = Form(1000),
    llm_as_judge_enabled: bool = Form(True),
    faithfulness_enabled: bool = Form(True),
    judge_model_provider: str = Form("openai"),
    judge_model_name: str = Form("gpt-4"),
    judge_temperature: float = Form(0.0),
    judge_prompt: str = Form(""),
    # Connector fields
    connector_enabled: bool = Form(False),
    connector_type: str = Form("openai"),
    connector_endpoint: str = Form(""),
    connector_api_key: str = Form(""),
    connector_model: str = Form(""),
    connector_timeout: int = Form(30),
    connector_rate_limit: int = Form(60),
    # Evaluation settings
    store_verbose_artifacts: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Create a new agent"""
    try:
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Create agent
        agent = Agent(
            name=name,
            description=description,
            tags=tag_list
        )
        db.add(agent)
        db.flush()  # Get the ID
        
        # Create initial agent version
        model_config = {
            "provider": model_provider,
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        judge_model_config = {
            "provider": judge_model_provider,
            "model": judge_model_name,
            "temperature": judge_temperature
        }
        
        default_thresholds = {
            "llm_as_judge": 0.8,
            "faithfulness": 0.8
        }
        
        default_judge_prompt = judge_prompt or "Please evaluate the following response for quality and accuracy. Rate from 0 to 1 where 1 is excellent and 0 is poor."
        
        # Configure connector if enabled
        connector_config_dict = None
        if connector_enabled and connector_endpoint:
            connector_config_dict = {
                "connector_type": connector_type,
                "endpoint_url": connector_endpoint,
                "api_key": connector_api_key if connector_api_key else None,
                "model_name": connector_model if connector_model else None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": connector_timeout,
                "rate_limit_per_minute": connector_rate_limit
            }
        
        version = AgentVersion(
            agent_id=agent.id,
            version_number=1,
            notes="Initial version",
            model_config=model_config,
            llm_as_judge_enabled=llm_as_judge_enabled,
            faithfulness_enabled=faithfulness_enabled,
            default_thresholds=default_thresholds,
            judge_model_config=judge_model_config,
            judge_prompt=default_judge_prompt,
            connector_enabled=connector_enabled,
            connector_config=connector_config_dict,
            store_verbose_artifacts=store_verbose_artifacts
        )
        db.add(version)
        db.commit()
        
        return RedirectResponse(url="/agents", status_code=302)
    
    except Exception as e:
        db.rollback()
        return templates.TemplateResponse("new_agent.html", {
            "request": request,
            "error": f"Error creating agent: {str(e)}"
        })

@app.get("/agents/{agent_id}", response_class=HTMLResponse)
async def agent_detail(request: Request, agent_id: int, db: Session = Depends(get_db)):
    """Agent detail page"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return templates.TemplateResponse("agent_detail.html", {
        "request": request,
        "agent": agent
    })

@app.post("/api/test-connector")
async def test_connector(
    request: Request,
    connector_type: str = Form(...),
    endpoint_url: str = Form(...),
    api_key: str = Form(""),
    model_name: str = Form(""),
    temperature: float = Form(0.7),
    max_tokens: int = Form(100),
    timeout_seconds: int = Form(10)
):
    """Test a connector configuration"""
    try:
        config = ConnectorConfig(
            connector_type=connector_type,
            endpoint_url=endpoint_url,
            api_key=api_key if api_key else None,
            model_name=model_name if model_name else None,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=1  # Fast test
        )
        
        connector = ConnectorFactory.create_connector(config)
        
        async with connector:
            # Test with a simple question
            response = await connector.evaluate("What is 2+2? Answer briefly.")
            
            if response.success:
                return JSONResponse({
                    "success": True,
                    "message": "Connection successful!",
                    "test_answer": response.answer,
                    "response_time_ms": response.response_time_ms
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": f"Connection failed: {response.error_message}"
                })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Test failed: {str(e)}"
        })

@app.get("/runs", response_class=HTMLResponse)
async def runs(request: Request, db: Session = Depends(get_db)):
    """Runs page"""
    runs = db.query(Run).order_by(Run.created_at.desc()).all()
    return templates.TemplateResponse("runs.html", {
        "request": request,
        "runs": runs
    })

@app.get("/runs/new", response_class=HTMLResponse)
async def new_run_form(request: Request, db: Session = Depends(get_db)):
    """New run wizard - step 1"""
    agents = db.query(Agent).all()
    datasets = db.query(Dataset).all()
    return templates.TemplateResponse("new_run.html", {
        "request": request,
        "agents": agents,
        "datasets": datasets
    })

@app.post("/runs/new/upload")
async def upload_run_answers(
    request: Request,
    name: str = Form(...),
    agent_version_id: int = Form(...),
    dataset_version_id: int = Form(...),
    evaluation_source: str = Form("upload"),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """Handle run creation with upload or connector evaluation"""
    
    # Handle file reading for upload mode
    answers = []
    if evaluation_source == "upload":
        if not file:
            agents = db.query(Agent).all()
            datasets = db.query(Dataset).all()
            return templates.TemplateResponse("new_run.html", {
                "request": request,
                "error": "Answers file is required for upload mode",
                "agents": agents,
                "datasets": datasets
            })
        
        try:
            content = await file.read()
            content_str = content.decode('utf-8')
        except Exception as e:
            agents = db.query(Agent).all()
            datasets = db.query(Dataset).all()
            return templates.TemplateResponse("new_run.html", {
                "request": request,
                "error": f"Error reading file: {str(e)}",
                "agents": agents,
                "datasets": datasets
            })
    
    # Get agent and dataset versions
    agent_version = db.query(AgentVersion).filter(AgentVersion.id == agent_version_id).first()
    dataset_version = db.query(DatasetVersion).filter(DatasetVersion.id == dataset_version_id).first()
    
    if not agent_version or not dataset_version:
        agents = db.query(Agent).all()
        datasets = db.query(Dataset).all()
        return templates.TemplateResponse("new_run.html", {
            "request": request,
            "error": "Invalid agent or dataset version selected",
            "agents": agents,
            "datasets": datasets
        })
    
    # Validate connector configuration for connector mode
    if evaluation_source == "connector":
        if not agent_version.connector_enabled or not agent_version.connector_config:
            agents = db.query(Agent).all()
            datasets = db.query(Dataset).all()
            return templates.TemplateResponse("new_run.html", {
                "request": request,
                "error": "Selected agent does not have connector configured",
                "agents": agents,
                "datasets": datasets
            })
    
    # Process answers file for upload mode
    errors = []
    if evaluation_source == "upload":
        answers, errors = AnswersProcessor.process_answers_file(content_str, file.filename)
        
        if errors:
            agents = db.query(Agent).all()
            datasets = db.query(Dataset).all()
            return templates.TemplateResponse("new_run.html", {
                "request": request,
                "errors": errors,
                "agents": agents,
                "datasets": datasets,
                "name": name,
                "selected_agent_version": agent_version_id,
                "selected_dataset_version": dataset_version_id
            })
    
    # Load dataset rows for validation
    try:
        dataset_rows = []
        with open(dataset_version.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset_rows.append(json.loads(line))
    except Exception as e:
        return templates.TemplateResponse("new_run.html", {
            "request": request,
            "error": f"Error reading dataset: {str(e)}"
        })
    
    # Validate 100% coverage for upload mode
    if evaluation_source == "upload":
        coverage_errors = AnswersProcessor.validate_coverage(answers, dataset_rows)
        if coverage_errors:
            agents = db.query(Agent).all()
            datasets = db.query(Dataset).all()
            return templates.TemplateResponse("new_run.html", {
                "request": request,
                "errors": coverage_errors,
                "agents": agents,
                "datasets": datasets,
                "name": name,
                "selected_agent_version": agent_version_id,
                "selected_dataset_version": dataset_version_id
            })
    
    # Create run
    try:
        run = Run(
            name=name,
            agent_version_id=agent_version_id,
            dataset_version_id=dataset_version_id,
            status="pending",
            evaluation_source=evaluation_source
        )
        db.add(run)
        db.commit()
        
        # Start processing in background
        import asyncio
        from threading import Thread
        
        def start_run_processing():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create new DB session for background processing
            from app.database import SessionLocal
            bg_db = SessionLocal()
            try:
                run_processor = RunProcessor(bg_db)
                if evaluation_source == "upload":
                    loop.run_until_complete(run_processor.start_run(run, answers, dataset_rows))
                else:  # connector
                    loop.run_until_complete(run_processor.start_run(run, dataset_rows=dataset_rows))
            finally:
                bg_db.close()
                loop.close()
        
        thread = Thread(target=start_run_processing)
        thread.start()
        
        return RedirectResponse(url=f"/runs/{run.id}", status_code=302)
    
    except Exception as e:
        db.rollback()
        agents = db.query(Agent).all()
        datasets = db.query(Dataset).all()
        return templates.TemplateResponse("new_run.html", {
            "request": request,
            "error": f"Error creating run: {str(e)}",
            "agents": agents,
            "datasets": datasets
        })

@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(
    request: Request, 
    run_id: int, 
    db: Session = Depends(get_db),
    status: str = None,
    search: str = None,
    min_score: float = None,
    max_score: float = None,
    sort_by: str = "case_id",
    sort_desc: bool = False
):
    """Run detail page with results and filtering"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Get all test cases with metric results
    all_test_cases = db.query(TestCase).filter(TestCase.run_id == run_id).order_by(TestCase.case_id).all()
    
    # Apply filters
    filters = {
        'status': status,
        'search': search,
        'min_score': min_score,
        'max_score': max_score,
        'sort_by': sort_by,
        'sort_desc': sort_desc
    }
    
    filtered_test_cases = ResultsFilter.filter_test_cases(all_test_cases, filters)
    
    # Calculate summary stats
    total_cases = len(all_test_cases)
    passed_cases = sum(1 for tc in all_test_cases if tc.passed)
    failed_cases = total_cases - passed_cases
    
    return templates.TemplateResponse("run_detail.html", {
        "request": request,
        "run": run,
        "test_cases": filtered_test_cases,
        "all_test_cases": all_test_cases,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "current_filters": filters
    })

@app.get("/api/runs/{run_id}/status")
async def get_run_status(run_id: int, db: Session = Depends(get_db)):
    """Get current run status (for progress polling)"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return {
        "status": run.status,
        "total_test_cases": run.total_test_cases,
        "completed_test_cases": run.completed_test_cases,
        "progress": (run.completed_test_cases / run.total_test_cases * 100) if run.total_test_cases > 0 else 0
    }

@app.get("/runs/{run_id}/export/csv")
async def export_run_csv(run_id: int, db: Session = Depends(get_db)):
    """Export run results as CSV"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.status != 'completed':
        raise HTTPException(status_code=400, detail="Run must be completed to export")
    
    try:
        exporter = RunExporter()
        csv_path = exporter.export_csv(run, db)
        
        # Update run with export path
        run.csv_export_path = csv_path
        db.commit()
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=csv_path,
            filename=f"{run.name}_results.csv",
            media_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/runs/{run_id}/export/json")
async def export_run_json(run_id: int, db: Session = Depends(get_db)):
    """Export run results as JSON"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.status != 'completed':
        raise HTTPException(status_code=400, detail="Run must be completed to export")
    
    try:
        exporter = RunExporter()
        json_path = exporter.export_json(run, db)
        
        # Update run with export path
        run.json_export_path = json_path
        db.commit()
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=json_path,
            filename=f"{run.name}_results.json",
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/runs/{run_id}/export/html")
async def export_run_html(run_id: int, db: Session = Depends(get_db)):
    """Generate and download HTML report"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.status != 'completed':
        raise HTTPException(status_code=400, detail="Run must be completed to export")
    
    try:
        exporter = RunExporter()
        html_path = exporter.generate_html_report(run, db)
        
        # Update run with export path
        run.html_report_path = html_path
        db.commit()
        
        from fastapi.responses import FileResponse
        return FileResponse(
            path=html_path,
            filename=f"{run.name}_report.html",
            media_type="text/html"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@app.get("/api/runs/{run_id}/case/{case_id}/details")
async def get_case_details(run_id: int, case_id: str, db: Session = Depends(get_db)):
    """Get detailed information for a specific test case"""
    test_case = db.query(TestCase).filter(
        TestCase.run_id == run_id,
        TestCase.case_id == case_id
    ).first()
    
    if not test_case:
        raise HTTPException(status_code=404, detail="Test case not found")
    
    # Get metric results with reasoning
    metric_details = []
    for metric in test_case.metric_results:
        metric_details.append({
            'name': metric.metric_name,
            'score': metric.score,
            'passed': metric.passed,
            'threshold': metric.threshold,
            'reasoning': metric.reasoning or 'No reasoning provided',
            'execution_time_ms': metric.execution_time_ms
        })
    
    return {
        'case_id': test_case.case_id,
        'question': test_case.question,
        'actual_answer': test_case.actual_answer,
        'expected_answer': test_case.expected_answer,
        'context': test_case.context,
        'overall_score': test_case.overall_score,
        'passed': test_case.passed,
        'metrics': metric_details
    }

@app.post("/api/runs/{run_id}/rescore")
async def rescore_run(
    run_id: int, 
    request: Request,
    db: Session = Depends(get_db)
):
    """Phase 8: Re-score a run with new threshold overrides (no re-querying)"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    if run.status != 'completed':
        raise HTTPException(status_code=400, detail="Can only re-score completed runs")
    
    try:
        # Parse request body
        body = await request.json()
        threshold_overrides = body.get('threshold_overrides', {})
        
        # Validate thresholds
        for metric_name, threshold in threshold_overrides.items():
            if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                raise HTTPException(status_code=400, detail=f"Invalid threshold for {metric_name}: must be between 0 and 1")
        
        # Apply threshold overrides to existing metric results
        test_cases = db.query(TestCase).filter(TestCase.run_id == run_id).all()
        
        for test_case in test_cases:
            metric_scores = []
            for metric_result in test_case.metric_results:
                # Get effective threshold (override or default)
                effective_threshold = threshold_overrides.get(
                    metric_result.metric_name, 
                    run.agent_version.default_thresholds.get(metric_result.metric_name, 0.8)
                )
                
                # Update metric result with new threshold and pass status
                metric_result.threshold = effective_threshold
                metric_result.passed = metric_result.score >= effective_threshold
                metric_scores.append(metric_result.score)
            
            # Recalculate overall test case score and pass status
            if metric_scores:
                test_case.overall_score = sum(metric_scores) / len(metric_scores)
                test_case.passed = all(mr.passed for mr in test_case.metric_results)
            
        # Recalculate run aggregates
        _recalculate_run_aggregates(run, db)
        
        # Store threshold overrides
        run.threshold_overrides = threshold_overrides
        db.commit()
        
        return {"success": True, "message": "Run re-scored successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Re-scoring failed: {str(e)}")

def _recalculate_run_aggregates(run: Run, db: Session):
    """Helper function to recalculate run aggregate metrics"""
    test_cases = db.query(TestCase).filter(TestCase.run_id == run.id).all()
    
    if not test_cases:
        return
    
    # Overall score (average of all test case scores)
    scores = [tc.overall_score for tc in test_cases if tc.overall_score is not None]
    if scores:
        run.overall_score = sum(scores) / len(scores)
    
    # Pass rate
    passed_cases = sum(1 for tc in test_cases if tc.passed)
    run.pass_rate = passed_cases / len(test_cases)
    
    # Aggregate results by metric
    aggregate_results = {}
    
    # Get all metric results
    all_metrics = db.query(MetricResult).join(TestCase).filter(TestCase.run_id == run.id).all()
    
    # Group by metric name
    metrics_by_name = {}
    for metric in all_metrics:
        if metric.metric_name not in metrics_by_name:
            metrics_by_name[metric.metric_name] = []
        metrics_by_name[metric.metric_name].append(metric)
    
    # Calculate aggregates for each metric
    for metric_name, metric_results in metrics_by_name.items():
        scores = [m.score for m in metric_results]
        passed_count = sum(1 for m in metric_results if m.passed)
        
        aggregate_results[metric_name] = {
            'score': sum(scores) / len(scores) if scores else 0,
            'pass_rate': passed_count / len(metric_results) if metric_results else 0,
            'total_cases': len(metric_results)
        }
    
    run.aggregate_results = aggregate_results