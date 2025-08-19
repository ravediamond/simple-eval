import os
import json
from datetime import datetime
from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from app.database import init_db, get_db
from app.models import Dataset, DatasetVersion, Agent, AgentVersion, Run, TestCase, MetricResult, ReferenceDataset, LLMConfiguration, JudgeConfiguration
from app.dataset_utils import DatasetProcessor
from app.run_utils import AnswersProcessor, RunProcessor
from app.export_utils import RunExporter, ResultsFilter
from app.connectors import ConnectorFactory, ConnectorConfig

load_dotenv()

app = FastAPI(title="EvalNow", description="Lightweight chatbot evaluation platform")

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
    return {"status": "ok", "service": "evalnow"}

@app.get("/api/chatbots")
async def get_chatbots_list(db: Session = Depends(get_db)):
    """Get chatbots list for sidebar"""
    chatbots = db.query(Agent).filter(Agent.is_active == True).all()
    return [
        {
            "id": bot.id,
            "name": bot.name,
            "description": bot.description or "",
            "run_count": sum(len(v.runs) for v in bot.versions),
            "dataset_count": sum(len(v.reference_datasets) for v in bot.versions),
            "latest_version": bot.versions[0].version_number if bot.versions else 1
        }
        for bot in chatbots
    ]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Main entry point - redirect to chatbots (like MLflow)"""
    return RedirectResponse(url="/chatbots", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Legacy dashboard page - redirect to chatbots"""
    return RedirectResponse(url="/chatbots", status_code=302)

@app.get("/chatbots", response_class=HTMLResponse)
async def chatbots_page(request: Request, db: Session = Depends(get_db)):
    """Chatbots page - main entry point (like MLflow)"""
    agents = db.query(Agent).filter(Agent.is_active == True).all()
    return templates.TemplateResponse("experiments.html", {
        "request": request,
        "experiments": agents  # Template still uses experiments variable
    })

@app.get("/experiments", response_class=HTMLResponse)
async def experiments_redirect(request: Request):
    """Legacy redirect to chatbots"""
    return RedirectResponse(url="/chatbots", status_code=302)


@app.get("/agents", response_class=HTMLResponse)
async def agents(request: Request, db: Session = Depends(get_db)):
    """Legacy redirect to chatbots"""
    agents = db.query(Agent).all()
    return templates.TemplateResponse("chatbots.html", {
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

@app.get("/configs", response_class=HTMLResponse)
async def configs_page(request: Request, db: Session = Depends(get_db)):
    """AI Setup page"""
    configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).order_by(LLMConfiguration.created_at.desc()).all()
    available_types = ConnectorFactory.get_available_types()
    return templates.TemplateResponse("configs.html", {
        "request": request,
        "configs": configs,
        "available_types": available_types
    })

@app.get("/judges", response_class=HTMLResponse)
async def judges_page(request: Request, db: Session = Depends(get_db)):
    """Judge Configurations page"""
    judges = db.query(JudgeConfiguration).filter(JudgeConfiguration.is_active == True).order_by(JudgeConfiguration.created_at.desc()).all()
    llm_configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
    return templates.TemplateResponse("judges.html", {
        "request": request,
        "judges": judges,
        "llm_configs": llm_configs
    })

@app.get("/chatbots/{chatbot_id}/datasets/upload", response_class=HTMLResponse)
async def upload_dataset_for_chatbot(request: Request, chatbot_id: int, db: Session = Depends(get_db)):
    """Upload reference dataset form for specific chatbot"""
    chatbot = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Get current version
    current_version = chatbot.versions[0] if chatbot.versions else None
    if not current_version:
        raise HTTPException(status_code=400, detail="Chatbot has no versions. Create a version first.")
    
    return templates.TemplateResponse("upload_dataset_focused.html", {
        "request": request,
        "chatbot": chatbot,
        "agent": chatbot,  # For backward compatibility with template
        "current_version": current_version
    })

@app.get("/experiments/{experiment_id}/datasets/upload", response_class=HTMLResponse)
async def upload_dataset_for_experiment_legacy(request: Request, experiment_id: int, db: Session = Depends(get_db)):
    """Legacy redirect to chatbot dataset upload"""
    return RedirectResponse(url=f"/chatbots/{experiment_id}/datasets/upload", status_code=302)

@app.get("/datasets/upload", response_class=HTMLResponse)
async def upload_dataset_form(request: Request):
    """Legacy upload dataset form - redirect to chatbots"""
    return RedirectResponse(url="/chatbots", status_code=302)

@app.post("/chatbots/{chatbot_id}/datasets/upload")
async def upload_reference_dataset_for_chatbot(
    request: Request,
    chatbot_id: int,
    name: str = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload reference dataset for specific chatbot"""
    # Get chatbot and current version
    chatbot = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    current_version = chatbot.versions[0] if chatbot.versions else None
    if not current_version:
        return templates.TemplateResponse("upload_dataset_focused.html", {
            "request": request,
            "chatbot": chatbot,
            "agent": chatbot,
            "error": "Chatbot has no versions. Create a version first."
        })
    
    # Read and process file (same logic as before)
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
    except Exception as e:
        return templates.TemplateResponse("upload_dataset_focused.html", {
            "request": request,
            "chatbot": chatbot,
            "agent": chatbot,
            "current_version": current_version,
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
        return templates.TemplateResponse("upload_dataset_focused.html", {
            "request": request,
            "chatbot": chatbot,
            "agent": chatbot,
            "current_version": current_version,
            "errors": errors,
            "preview_rows": rows[:10] if rows else [],
            "total_rows": len(rows),
            "name": name,
            "description": description
        })
    
    # Create reference dataset
    try:
        # Save normalized data
        file_path = DatasetProcessor.save_normalized_data(rows, f"chatbot_{chatbot_id}_v{current_version.version_number}", 1)
        
        # Create reference dataset
        content_hash = ReferenceDataset.generate_content_hash(rows)
        reference_dataset = ReferenceDataset(
            agent_version_id=current_version.id,
            name=name,
            description=description,
            row_count=len(rows),
            content_hash=content_hash,
            file_path=file_path
        )
        db.add(reference_dataset)
        db.commit()
        
        return RedirectResponse(url=f"/chatbots/{chatbot_id}", status_code=302)
    
    except Exception as e:
        db.rollback()
        return templates.TemplateResponse("upload_dataset_focused.html", {
            "request": request,
            "chatbot": chatbot,
            "agent": chatbot,
            "current_version": current_version,
            "error": f"Error saving reference dataset: {str(e)}"
        })


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
        
        # Check if this was uploaded from a chatbot context
        referer = request.headers.get('referer', '')
        if '/chatbots/' in referer and '/datasets/upload' in referer:
            # Extract agent_id from referer
            import re
            match = re.search(r'/chatbots/(\d+)/', referer)
            if match:
                agent_id = match.group(1)
                return RedirectResponse(url=f"/chatbots/{agent_id}", status_code=302)
        
        return RedirectResponse(url="/datasets", status_code=302)
    
    except Exception as e:
        db.rollback()
        # Check if this was from a chatbot context
        referer = request.headers.get('referer', '')
        if '/chatbots/' in referer and '/datasets/upload' in referer:
            import re
            match = re.search(r'/chatbots/(\d+)/', referer)
            if match:
                agent_id = match.group(1)
                agent = db.query(Agent).filter(Agent.id == agent_id).first()
                if agent:
                    return templates.TemplateResponse("upload_dataset_focused.html", {
                        "request": request,
                        "agent": agent,
                        "error": f"Error saving dataset: {str(e)}"
                    })
        
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

@app.get("/chatbots/new", response_class=HTMLResponse)
async def new_chatbot_form(request: Request, db: Session = Depends(get_db)):
    """New chatbot form"""
    llm_configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
    judge_configs = db.query(JudgeConfiguration).filter(JudgeConfiguration.is_active == True).all()
    return templates.TemplateResponse("new_agent.html", {
        "request": request,
        "llm_configs": llm_configs,
        "judge_configs": judge_configs
    })

@app.get("/agents/new", response_class=HTMLResponse)
async def new_agent_form_legacy(request: Request):
    """Legacy redirect to chatbots/new"""
    return RedirectResponse(url="/chatbots/new", status_code=302)

@app.post("/chatbots")
async def create_chatbot(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    llm_config_id: int = Form(...),
    judge_config_id: str = Form(""),
    judge_prompt: str = Form(""),
    # Evaluation settings
    store_verbose_artifacts: bool = Form(False),
    db: Session = Depends(get_db)
):
    """Create a new chatbot"""
    try:
        # Validate LLM configuration exists
        llm_config = db.query(LLMConfiguration).filter(LLMConfiguration.id == llm_config_id).first()
        if not llm_config:
            llm_configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
            judge_configs = db.query(JudgeConfiguration).filter(JudgeConfiguration.is_active == True).all()
            return templates.TemplateResponse("new_agent.html", {
                "request": request,
                "llm_configs": llm_configs,
                "judge_configs": judge_configs,
                "error": "Selected LLM configuration not found"
            })
        
        # Validate judge configuration if provided
        judge_config = None
        if judge_config_id and judge_config_id.strip():
            try:
                judge_config_id_int = int(judge_config_id)
                judge_config = db.query(JudgeConfiguration).filter(JudgeConfiguration.id == judge_config_id_int).first()
                if not judge_config:
                    llm_configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
                    judge_configs = db.query(JudgeConfiguration).filter(JudgeConfiguration.is_active == True).all()
                    return templates.TemplateResponse("new_agent.html", {
                        "request": request,
                        "llm_configs": llm_configs,
                        "judge_configs": judge_configs,
                        "error": "Selected judge configuration not found"
                    })
            except ValueError:
                llm_configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
                judge_configs = db.query(JudgeConfiguration).filter(JudgeConfiguration.is_active == True).all()
                return templates.TemplateResponse("new_agent.html", {
                    "request": request,
                    "llm_configs": llm_configs,
                    "judge_configs": judge_configs,
                    "error": "Invalid judge configuration ID"
                })
        else:
            # Use default judge if no specific judge config selected
            judge_config = db.query(JudgeConfiguration).filter(
                JudgeConfiguration.is_default_judge == True,
                JudgeConfiguration.is_active == True
            ).first()
        
        # Create agent
        agent = Agent(
            name=name,
            description=description,
            tags=[]
        )
        db.add(agent)
        db.flush()  # Get the ID
        
        # Set up default thresholds - only LLM-as-judge now
        default_thresholds = {
            "llm_as_judge": judge_config.llm_as_judge_threshold if judge_config else 0.8
        }
        
        # Use judge prompt from form, or from judge config, or default
        effective_judge_prompt = judge_prompt
        if not effective_judge_prompt and judge_config and judge_config.judge_prompt:
            effective_judge_prompt = judge_config.judge_prompt
        if not effective_judge_prompt:
            effective_judge_prompt = """You are an expert evaluator assessing the quality and correctness of AI responses.

Please evaluate the given response to the question on the following criteria:
1. Accuracy: Is the response factually correct?
2. Completeness: Does it fully answer the question?
3. Clarity: Is it well-written and understandable?
4. Relevance: Does it directly address what was asked?

Provide a score from 0 to 1 (where 1 is excellent and 0 is poor) and explain your reasoning."""
        
        # Connector functionality removed - CSV upload only
        
        version = AgentVersion(
            agent_id=agent.id,
            version_number=1,
            notes="Initial version",
            llm_config_id=llm_config.id,
            judge_config_id=judge_config.id if judge_config else None,
            model_config={},  # Empty dict for new chatbots using LLM configurations
            judge_model_config={},  # Empty dict for new chatbots using LLM configurations
            default_thresholds=default_thresholds,
            judge_prompt=effective_judge_prompt,
            store_verbose_artifacts=store_verbose_artifacts
        )
        db.add(version)
        db.commit()
        
        return RedirectResponse(url="/chatbots", status_code=302)
    
    except Exception as e:
        db.rollback()
        llm_configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
        judge_configs = db.query(JudgeConfiguration).filter(JudgeConfiguration.is_active == True).all()
        return templates.TemplateResponse("new_agent.html", {
            "request": request,
            "llm_configs": llm_configs,
            "judge_configs": judge_configs,
            "error": f"Error creating chatbot: {str(e)}"
        })

@app.post("/agents")
async def create_agent_legacy(
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
    endpoint_url: str = Form(""),
    api_key: str = Form(""),
    judge_endpoint_url: str = Form(""),
    judge_api_key: str = Form(""),
    db: Session = Depends(get_db)
):
    """Legacy create agent - redirect to chatbots"""
    # Forward all form data to the chatbots endpoint
    form_data = {
        "name": name,
        "description": description,
        "tags": tags,
        "model_provider": model_provider,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "llm_as_judge_enabled": llm_as_judge_enabled,
        "faithfulness_enabled": faithfulness_enabled,
        "judge_model_provider": judge_model_provider,
        "judge_model_name": judge_model_name,
        "judge_temperature": judge_temperature,
        "endpoint_url": endpoint_url,
        "api_key": api_key,
        "judge_endpoint_url": judge_endpoint_url,
        "judge_api_key": judge_api_key
    }
    # Create form request to forward to chatbots endpoint
    from fastapi import Request as FastAPIRequest
    from starlette.datastructures import FormData
    new_form = FormData([(k, v) for k, v in form_data.items()])
    
    # Call the chatbots endpoint directly
    return await create_chatbot(request, name, description, tags, model_provider, model_name, temperature, max_tokens, llm_as_judge_enabled, faithfulness_enabled, judge_model_provider, judge_model_name, judge_temperature, endpoint_url, api_key, judge_endpoint_url, judge_api_key, db)

@app.get("/chatbots/{chatbot_id}", response_class=HTMLResponse)
async def chatbot_overview(request: Request, chatbot_id: int, version: int = None, db: Session = Depends(get_db)):
    """Chatbot overview - High-level summary view"""
    chatbot = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Get current version - either specified version or latest
    if version:
        current_version = db.query(AgentVersion).filter(
            AgentVersion.id == version, 
            AgentVersion.agent_id == chatbot_id
        ).first()
        if not current_version:
            # Fall back to latest if specified version not found
            current_version = chatbot.versions[0] if chatbot.versions else None
    else:
        current_version = chatbot.versions[0] if chatbot.versions else None
    
    reference_datasets = []
    if current_version:
        reference_datasets = current_version.reference_datasets
    
    # Get runs for this chatbot version (or all runs if no specific version)
    if version and current_version:
        runs = db.query(Run).filter(Run.agent_version_id == current_version.id).order_by(Run.created_at.desc()).all()
    else:
        runs = db.query(Run).join(AgentVersion).filter(AgentVersion.agent_id == chatbot_id).order_by(Run.created_at.desc()).all()
    
    return templates.TemplateResponse("business_dashboard.html", {
        "request": request,
        "experiment": chatbot,  # Template still uses experiment variable
        "current_version": current_version,
        "reference_datasets": reference_datasets,
        "runs": runs
    })

@app.get("/chatbots/{chatbot_id}/detailed", response_class=HTMLResponse)
async def chatbot_detailed(request: Request, chatbot_id: int, db: Session = Depends(get_db)):
    """Chatbot detailed view - In-depth technical information"""
    chatbot = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    
    # Get current version (latest) and its reference datasets
    current_version = chatbot.versions[0] if chatbot.versions else None
    reference_datasets = []
    if current_version:
        reference_datasets = current_version.reference_datasets
    
    # Get runs for this chatbot
    runs = db.query(Run).join(AgentVersion).filter(AgentVersion.agent_id == chatbot_id).order_by(Run.created_at.desc()).all()
    
    return templates.TemplateResponse("evaluation_dashboard.html", {
        "request": request,
        "experiment": chatbot,  # Template still uses experiment variable
        "current_version": current_version,
        "reference_datasets": reference_datasets,
        "runs": runs
    })

@app.get("/experiments/{experiment_id}", response_class=HTMLResponse)
async def experiment_redirect(request: Request, experiment_id: int, db: Session = Depends(get_db)):
    """Legacy redirect to chatbot dashboard"""
    return RedirectResponse(url=f"/chatbots/{experiment_id}", status_code=302)

@app.get("/chatbots/{chatbot_id}/technical", response_class=HTMLResponse)
async def chatbot_technical_redirect(request: Request, chatbot_id: int, db: Session = Depends(get_db)):
    """Legacy redirect from technical to detailed view"""
    return RedirectResponse(url=f"/chatbots/{chatbot_id}/detailed", status_code=302)

@app.get("/agents/{agent_id}", response_class=HTMLResponse)
async def agent_detail(request: Request, agent_id: int, db: Session = Depends(get_db)):
    """Legacy agent detail - redirect to chatbot detail"""
    return RedirectResponse(url=f"/chatbots/{agent_id}", status_code=302)

# Connector test endpoint removed - CSV upload only

@app.get("/runs", response_class=HTMLResponse)
async def runs(request: Request, db: Session = Depends(get_db)):
    """Runs page"""
    runs = db.query(Run).order_by(Run.created_at.desc()).all()
    return templates.TemplateResponse("runs.html", {
        "request": request,
        "runs": runs
    })

@app.get("/chatbots/{agent_id}/datasets/{reference_dataset_id}/run", response_class=HTMLResponse)
async def new_run_for_chatbot_dataset(request: Request, agent_id: int, reference_dataset_id: int, db: Session = Depends(get_db)):
    """New run form for specific chatbot and reference dataset"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    reference_dataset = db.query(ReferenceDataset).filter(ReferenceDataset.id == reference_dataset_id).first()
    
    if not agent or not reference_dataset:
        raise HTTPException(status_code=404, detail="Chatbot or reference dataset not found")
    
    # Verify the reference dataset belongs to this chatbot
    if reference_dataset.agent_version.agent_id != agent_id:
        raise HTTPException(status_code=404, detail="Reference dataset does not belong to this chatbot")
    
    return templates.TemplateResponse("new_run_focused.html", {
        "request": request,
        "agent": agent,
        "reference_dataset": reference_dataset,
        "agent_version": reference_dataset.agent_version
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
    reference_dataset_id: int = Form(...),
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
    
    # Get agent version and reference dataset
    agent_version = db.query(AgentVersion).filter(AgentVersion.id == agent_version_id).first()
    reference_dataset = db.query(ReferenceDataset).filter(ReferenceDataset.id == reference_dataset_id).first()
    
    if not agent_version or not reference_dataset:
        agents = db.query(Agent).all()
        datasets = db.query(Dataset).all()
        return templates.TemplateResponse("new_run.html", {
            "request": request,
            "error": "Invalid agent or dataset version selected",
            "agents": agents,
            "datasets": datasets
        })
    
    # Connector mode removed - CSV upload only
    
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
                "selected_reference_dataset": reference_dataset_id
            })
    
    # Load dataset rows for validation
    try:
        dataset_rows = []
        with open(reference_dataset.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset_rows.append(json.loads(line))
    except Exception as e:
        return templates.TemplateResponse("new_run.html", {
            "request": request,
            "error": f"Error reading dataset: {str(e)}"
        })
    
    # Validate 100% coverage (CSV upload only mode)
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
                "selected_reference_dataset": reference_dataset_id
            })
    
    # Create run
    try:
        run = Run(
            name=name,
            agent_version_id=agent_version_id,
            reference_dataset_id=reference_dataset_id,
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
                # Get fresh run object in background session
                bg_run = bg_db.query(Run).filter(Run.id == run.id).first()
                if not bg_run:
                    return
                
                run_processor = RunProcessor(bg_db)
                if evaluation_source == "upload":
                    # CSV upload mode - use provided answers
                    loop.run_until_complete(run_processor.start_run(bg_run, answers, dataset_rows))
                else:
                    # Connector mode - run live evaluation
                    loop.run_until_complete(run_processor.start_run(bg_run, None, dataset_rows))
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

# AI Setup Management APIs

@app.post("/api/configs")
async def create_llm_config(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    connector_type: str = Form(...),
    endpoint_url: str = Form(...),
    api_key: str = Form(""),
    model_name: str = Form(...),
    temperature: float = Form(0.7),
    max_tokens: int = Form(1000),
    timeout_seconds: int = Form(30),
    rate_limit_per_minute: int = Form(60),
    additional_params: str = Form("{}"),
    db: Session = Depends(get_db)
):
    """Create new LLM configuration"""
    try:
        # Parse additional parameters
        import json
        try:
            additional_params_dict = json.loads(additional_params) if additional_params.strip() else {}
        except json.JSONDecodeError:
            return JSONResponse({
                "success": False,
                "message": "Invalid JSON in additional parameters"
            }, status_code=400)
        
        # Check if name already exists
        existing = db.query(LLMConfiguration).filter(LLMConfiguration.name == name).first()
        if existing:
            return JSONResponse({
                "success": False,
                "message": "Configuration name already exists"
            }, status_code=400)
        
        
        # Create configuration
        config = LLMConfiguration(
            name=name,
            description=description,
            connector_type=connector_type,
            endpoint_url=endpoint_url,
            api_key=api_key if api_key else None,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            rate_limit_per_minute=rate_limit_per_minute,
            additional_params=additional_params_dict
        )
        
        db.add(config)
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": "Configuration created successfully",
            "config_id": config.id
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error creating configuration: {str(e)}"
        }, status_code=500)

@app.post("/api/configs/{config_id}/test")
async def test_llm_config(config_id: int, db: Session = Depends(get_db)):
    """Test an LLM configuration"""
    config = db.query(LLMConfiguration).filter(LLMConfiguration.id == config_id).first()
    if not config:
        return JSONResponse({
            "success": False,
            "message": "Configuration not found"
        }, status_code=404)
    
    try:
        connector_config = config.to_connector_config()
        connector = ConnectorFactory.create_connector(connector_config)
        
        async with connector:
            response = await connector.evaluate("What is 2+2? Answer briefly.")
            
            if response.success:
                return JSONResponse({
                    "success": True,
                    "message": "Configuration test successful!",
                    "test_answer": response.answer,
                    "response_time_ms": response.response_time_ms,
                    "provider": response.metadata.get('provider') if response.metadata else config.display_provider
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": f"Test failed: {response.error_message}"
                })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Test failed: {str(e)}"
        })

@app.put("/api/configs/{config_id}")
async def update_llm_config(
    config_id: int,
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    connector_type: str = Form(...),
    endpoint_url: str = Form(...),
    api_key: str = Form(""),
    model_name: str = Form(...),
    temperature: float = Form(0.7),
    max_tokens: int = Form(1000),
    timeout_seconds: int = Form(30),
    rate_limit_per_minute: int = Form(60),
    additional_params: str = Form("{}"),
    db: Session = Depends(get_db)
):
    """Update LLM configuration"""
    config = db.query(LLMConfiguration).filter(LLMConfiguration.id == config_id).first()
    if not config:
        return JSONResponse({
            "success": False,
            "message": "Configuration not found"
        }, status_code=404)
    
    try:
        # Parse additional parameters
        import json
        try:
            additional_params_dict = json.loads(additional_params) if additional_params.strip() else {}
        except json.JSONDecodeError:
            return JSONResponse({
                "success": False,
                "message": "Invalid JSON in additional parameters"
            }, status_code=400)
        
        # Check if name already exists (excluding current config)
        existing = db.query(LLMConfiguration).filter(
            LLMConfiguration.name == name,
            LLMConfiguration.id != config_id
        ).first()
        if existing:
            return JSONResponse({
                "success": False,
                "message": "Configuration name already exists"
            }, status_code=400)
        
        
        # Update configuration
        config.name = name
        config.description = description
        config.connector_type = connector_type
        config.endpoint_url = endpoint_url
        config.api_key = api_key if api_key else None
        config.model_name = model_name
        config.temperature = temperature
        config.max_tokens = max_tokens
        config.timeout_seconds = timeout_seconds
        config.rate_limit_per_minute = rate_limit_per_minute
        config.additional_params = additional_params_dict
        config.updated_at = datetime.utcnow()
        
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": "Configuration updated successfully"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error updating configuration: {str(e)}"
        }, status_code=500)

@app.delete("/api/configs/{config_id}")
async def delete_llm_config(config_id: int, db: Session = Depends(get_db)):
    """Delete LLM configuration (soft delete)"""
    config = db.query(LLMConfiguration).filter(LLMConfiguration.id == config_id).first()
    if not config:
        return JSONResponse({
            "success": False,
            "message": "Configuration not found"
        }, status_code=404)
    
    try:
        # Soft delete
        config.is_active = False
        config.updated_at = datetime.utcnow()
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": "Configuration deleted successfully"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error deleting configuration: {str(e)}"
        }, status_code=500)

@app.get("/api/configs/{config_id}")
async def get_llm_config(config_id: int, db: Session = Depends(get_db)):
    """Get LLM configuration details"""
    config = db.query(LLMConfiguration).filter(LLMConfiguration.id == config_id).first()
    if not config:
        return JSONResponse({
            "success": False,
            "message": "Configuration not found"
        }, status_code=404)
    
    return JSONResponse({
        "success": True,
        "config": {
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "connector_type": config.connector_type,
            "endpoint_url": config.endpoint_url,
            "api_key": config.api_key or "",
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "timeout_seconds": config.timeout_seconds,
            "rate_limit_per_minute": config.rate_limit_per_minute,
            "additional_params": json.dumps(config.additional_params or {}, indent=2),
            "display_provider": config.display_provider,
            "created_at": config.created_at.isoformat(),
            "updated_at": config.updated_at.isoformat()
        }
    })

@app.get("/api/configs")
async def get_llm_configs(db: Session = Depends(get_db)):
    """Get all active LLM configurations"""
    configs = db.query(LLMConfiguration).filter(LLMConfiguration.is_active == True).all()
    return [
        {
            "id": config.id,
            "name": config.name,
            "description": config.description,
            "display_provider": config.display_provider,
            "model_name": config.model_name
        }
        for config in configs
    ]

# Judge Configuration Management APIs

@app.post("/api/judges")
async def create_judge_config(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    base_llm_config_id: int = Form(...),
    judge_profile: str = Form("simple"),
    judge_prompt: str = Form(...),
    llm_as_judge_threshold: float = Form(0.8),
    is_default_judge: bool = Form(False),
    judge_temperature: str = Form(""),
    judge_max_tokens: str = Form(""),
    db: Session = Depends(get_db)
):
    """Create new Judge configuration"""
    try:
        # Check if name already exists
        existing = db.query(JudgeConfiguration).filter(JudgeConfiguration.name == name).first()
        if existing:
            return JSONResponse({
                "success": False,
                "message": "Judge configuration name already exists"
            }, status_code=400)
        
        # Validate base LLM config exists
        base_config = db.query(LLMConfiguration).filter(LLMConfiguration.id == base_llm_config_id).first()
        if not base_config:
            return JSONResponse({
                "success": False,
                "message": "Base LLM configuration not found"
            }, status_code=400)
        
        # Parse optional overrides
        parsed_temperature = None
        parsed_max_tokens = None
        
        if judge_temperature.strip():
            try:
                parsed_temperature = float(judge_temperature)
            except ValueError:
                return JSONResponse({
                    "success": False,
                    "message": "Invalid temperature value"
                }, status_code=400)
        
        if judge_max_tokens.strip():
            try:
                parsed_max_tokens = int(judge_max_tokens)
            except ValueError:
                return JSONResponse({
                    "success": False,
                    "message": "Invalid max tokens value"
                }, status_code=400)
        
        # If this is set as default judge, unset others
        if is_default_judge:
            db.query(JudgeConfiguration).update({JudgeConfiguration.is_default_judge: False})
        
        # Create configuration
        judge = JudgeConfiguration(
            name=name,
            description=description,
            base_llm_config_id=base_llm_config_id,
            judge_profile=judge_profile,
            judge_prompt=judge_prompt,
            llm_as_judge_threshold=llm_as_judge_threshold,
            is_default_judge=is_default_judge,
            judge_temperature=parsed_temperature,
            judge_max_tokens=parsed_max_tokens
        )
        
        db.add(judge)
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": "Judge configuration created successfully",
            "judge_id": judge.id
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error creating judge configuration: {str(e)}"
        }, status_code=500)

@app.post("/api/judges/{judge_id}/test")
async def test_judge_config(judge_id: int, db: Session = Depends(get_db)):
    """Test a Judge configuration"""
    judge = db.query(JudgeConfiguration).filter(JudgeConfiguration.id == judge_id).first()
    if not judge:
        return JSONResponse({
            "success": False,
            "message": "Judge configuration not found"
        }, status_code=404)
    
    try:
        connector_config = judge.to_connector_config()
        connector = ConnectorFactory.create_connector(connector_config)
        
        async with connector:
            # Use the judge prompt with a sample question and answer
            test_prompt = f"""{judge.judge_prompt}

Question: What is 2+2?
Answer: 4

Please provide your evaluation:"""
            
            response = await connector.evaluate(test_prompt)
            
            if response.success:
                return JSONResponse({
                    "success": True,
                    "message": "Judge configuration test successful!",
                    "test_answer": response.answer,
                    "response_time_ms": response.response_time_ms,
                    "provider": response.metadata.get('provider') if response.metadata else judge.base_llm_config.display_provider
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": f"Test failed: {response.error_message}"
                })
    
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Test failed: {str(e)}"
        })

@app.put("/api/judges/{judge_id}")
async def update_judge_config(
    judge_id: int,
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    base_llm_config_id: int = Form(...),
    judge_profile: str = Form("simple"),
    judge_prompt: str = Form(...),
    llm_as_judge_threshold: float = Form(0.8),
    is_default_judge: bool = Form(False),
    judge_temperature: str = Form(""),
    judge_max_tokens: str = Form(""),
    db: Session = Depends(get_db)
):
    """Update Judge configuration"""
    judge = db.query(JudgeConfiguration).filter(JudgeConfiguration.id == judge_id).first()
    if not judge:
        return JSONResponse({
            "success": False,
            "message": "Judge configuration not found"
        }, status_code=404)
    
    try:
        # Check if name already exists (excluding current config)
        existing = db.query(JudgeConfiguration).filter(
            JudgeConfiguration.name == name,
            JudgeConfiguration.id != judge_id
        ).first()
        if existing:
            return JSONResponse({
                "success": False,
                "message": "Judge configuration name already exists"
            }, status_code=400)
        
        # Validate base LLM config exists
        base_config = db.query(LLMConfiguration).filter(LLMConfiguration.id == base_llm_config_id).first()
        if not base_config:
            return JSONResponse({
                "success": False,
                "message": "Base LLM configuration not found"
            }, status_code=400)
        
        # Parse optional overrides
        parsed_temperature = None
        parsed_max_tokens = None
        
        if judge_temperature.strip():
            try:
                parsed_temperature = float(judge_temperature)
            except ValueError:
                return JSONResponse({
                    "success": False,
                    "message": "Invalid temperature value"
                }, status_code=400)
        
        if judge_max_tokens.strip():
            try:
                parsed_max_tokens = int(judge_max_tokens)
            except ValueError:
                return JSONResponse({
                    "success": False,
                    "message": "Invalid max tokens value"
                }, status_code=400)
        
        # If this is set as default judge, unset others
        if is_default_judge and not judge.is_default_judge:
            db.query(JudgeConfiguration).update({JudgeConfiguration.is_default_judge: False})
        
        # Update configuration
        judge.name = name
        judge.description = description
        judge.base_llm_config_id = base_llm_config_id
        judge.judge_profile = judge_profile
        judge.judge_prompt = judge_prompt
        judge.llm_as_judge_threshold = llm_as_judge_threshold
        judge.is_default_judge = is_default_judge
        judge.judge_temperature = parsed_temperature
        judge.judge_max_tokens = parsed_max_tokens
        judge.updated_at = datetime.utcnow()
        
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": "Judge configuration updated successfully"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error updating judge configuration: {str(e)}"
        }, status_code=500)

@app.delete("/api/judges/{judge_id}")
async def delete_judge_config(judge_id: int, db: Session = Depends(get_db)):
    """Delete Judge configuration (soft delete)"""
    judge = db.query(JudgeConfiguration).filter(JudgeConfiguration.id == judge_id).first()
    if not judge:
        return JSONResponse({
            "success": False,
            "message": "Judge configuration not found"
        }, status_code=404)
    
    try:
        # Soft delete
        judge.is_active = False
        judge.updated_at = datetime.utcnow()
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": "Judge configuration deleted successfully"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error deleting judge configuration: {str(e)}"
        }, status_code=500)

@app.get("/api/judges/{judge_id}")
async def get_judge_config(judge_id: int, db: Session = Depends(get_db)):
    """Get Judge configuration details"""
    judge = db.query(JudgeConfiguration).filter(JudgeConfiguration.id == judge_id).first()
    if not judge:
        return JSONResponse({
            "success": False,
            "message": "Judge configuration not found"
        }, status_code=404)
    
    return JSONResponse({
        "success": True,
        "judge": {
            "id": judge.id,
            "name": judge.name,
            "description": judge.description,
            "base_llm_config_id": judge.base_llm_config_id,
            "judge_profile": judge.judge_profile,
            "judge_prompt": judge.judge_prompt,
            "llm_as_judge_threshold": judge.llm_as_judge_threshold,
            "is_default_judge": judge.is_default_judge,
            "judge_temperature": judge.judge_temperature,
            "judge_max_tokens": judge.judge_max_tokens,
            "base_llm_config": {
                "id": judge.base_llm_config.id,
                "name": judge.base_llm_config.name,
                "display_provider": judge.base_llm_config.display_provider,
                "model_name": judge.base_llm_config.model_name
            },
            "created_at": judge.created_at.isoformat(),
            "updated_at": judge.updated_at.isoformat()
        }
    })

@app.get("/api/judges")
async def get_judge_configs(db: Session = Depends(get_db)):
    """Get all active Judge configurations"""
    judges = db.query(JudgeConfiguration).filter(JudgeConfiguration.is_active == True).all()
    return [
        {
            "id": judge.id,
            "name": judge.name,
            "description": judge.description,
            "is_default_judge": judge.is_default_judge,
            "base_llm_config": {
                "id": judge.base_llm_config.id,
                "name": judge.base_llm_config.name,
                "display_provider": judge.base_llm_config.display_provider,
                "model_name": judge.base_llm_config.model_name
            }
        }
        for judge in judges
    ]

# Additional APIs for complete testing flow

@app.post("/api/chatbots/{chatbot_id}/datasets")
async def upload_dataset_api(
    chatbot_id: int,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db)
):
    """API endpoint to upload dataset for a chatbot"""
    from app.dataset_utils import DatasetProcessor
    
    # Get chatbot
    agent = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
    if not agent:
        return JSONResponse({
            "success": False,
            "message": "Chatbot not found"
        }, status_code=404)
    
    # Get latest version
    latest_version = agent.versions[0] if agent.versions else None
    if not latest_version:
        return JSONResponse({
            "success": False,
            "message": "Chatbot has no versions"
        }, status_code=404)
    
    try:
        # Read file content
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Process dataset
        processor = DatasetProcessor()
        rows, errors = processor.process_file(content_str, file.filename)
        
        if errors:
            return JSONResponse({
                "success": False,
                "message": f"Dataset processing errors: {', '.join(errors)}"
            }, status_code=400)
        
        # Save dataset
        file_path = processor.save_normalized_data(rows, f"chatbot_{chatbot_id}_v{latest_version.version_number}", 1)
        
        # Create reference dataset
        content_hash = ReferenceDataset.generate_content_hash(rows)
        reference_dataset = ReferenceDataset(
            agent_version_id=latest_version.id,
            name=name,
            description=description,
            row_count=len(rows),
            content_hash=content_hash,
            file_path=file_path
        )
        db.add(reference_dataset)
        db.commit()
        
        dataset_id = reference_dataset.id
        
        return JSONResponse({
            "success": True,
            "message": "Dataset uploaded successfully",
            "dataset_id": dataset_id,
            "row_count": len(rows)
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error uploading dataset: {str(e)}"
        }, status_code=500)

@app.post("/api/runs")
async def create_run_api(
    chatbot_id: int = Form(...),
    dataset_id: int = Form(...),
    answers_file: UploadFile = File(...),
    run_name: str = Form(...),
    db: Session = Depends(get_db)
):
    """API endpoint to create and execute a run"""
    from app.run_utils import AnswersProcessor, RunProcessor
    
    try:
        # Validate chatbot and dataset
        agent = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
        if not agent:
            return JSONResponse({
                "success": False,
                "message": "Chatbot not found"
            }, status_code=404)
        
        dataset = db.query(ReferenceDataset).filter(ReferenceDataset.id == dataset_id).first()
        if not dataset:
            return JSONResponse({
                "success": False,
                "message": "Dataset not found"
            }, status_code=404)
        
        # Process answers file
        content = await answers_file.read()
        content_str = content.decode('utf-8')
        
        processor = AnswersProcessor()
        answers, errors = processor.process_answers_file(content_str, answers_file.filename)
        
        if errors:
            return JSONResponse({
                "success": False,
                "message": f"Answers processing errors: {', '.join(errors)}"
            }, status_code=400)
        
        # Create run
        latest_version = agent.versions[0] if agent.versions else None
        if not latest_version:
            return JSONResponse({
                "success": False,
                "message": "Chatbot has no versions"
            }, status_code=404)
        
        run = Run(
            name=run_name,
            agent_version_id=latest_version.id,
            reference_dataset_id=dataset_id,
            status="pending",
            evaluation_source="upload",
            total_test_cases=len(answers)
        )
        db.add(run)
        db.flush()
        
        # Load dataset rows
        import json
        with open(dataset.file_path, 'r') as f:
            dataset_rows = [json.loads(line) for line in f]
        
        # Process answers and start evaluation
        run_processor = RunProcessor(db)
        await run_processor.start_run(run, answers=answers, dataset_rows=dataset_rows)
        
        return JSONResponse({
            "success": True,
            "message": "Run created and evaluation started",
            "run_id": run.id,
            "status": run.status,
            "total_test_cases": run.total_test_cases
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error creating run: {str(e)}"
        }, status_code=500)

@app.get("/api/runs/{run_id}")
async def get_run_api(run_id: int, db: Session = Depends(get_db)):
    """API endpoint to get run details and results"""
    run = db.query(Run).filter(Run.id == run_id).first()
    if not run:
        return JSONResponse({
            "success": False,
            "message": "Run not found"
        }, status_code=404)
    
    # Get test cases with results
    test_cases = []
    for test_case in run.test_cases:
        metrics = {}
        for metric in test_case.metric_results:
            metrics[metric.metric_name] = {
                "score": metric.score,
                "passed": metric.passed,
                "threshold": metric.threshold,
                "reasoning": metric.reasoning
            }
        
        test_cases.append({
            "id": test_case.id,
            "case_id": test_case.case_id,
            "question": test_case.question,
            "context": test_case.context,
            "expected_answer": test_case.expected_answer,
            "actual_answer": test_case.actual_answer,
            "overall_score": test_case.overall_score,
            "passed": test_case.passed,
            "metrics": metrics
        })
    
    return JSONResponse({
        "success": True,
        "run": {
            "id": run.id,
            "name": run.name,
            "status": run.status,
            "overall_score": run.overall_score,
            "pass_rate": run.pass_rate,
            "total_test_cases": run.total_test_cases,
            "completed_test_cases": run.completed_test_cases,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "aggregate_results": run.aggregate_results,
            "test_cases": test_cases
        }
    })

@app.get("/api/chatbots/{chatbot_id}")
async def get_chatbot_api(chatbot_id: int, db: Session = Depends(get_db)):
    """API endpoint to get chatbot details"""
    agent = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
    if not agent:
        return JSONResponse({
            "success": False,
            "message": "Chatbot not found"
        }, status_code=404)
    
    latest_version = agent.versions[0] if agent.versions else None
    
    return JSONResponse({
        "success": True,
        "chatbot": {
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "created_at": agent.created_at.isoformat(),
            "latest_version": {
                "id": latest_version.id,
                "version_number": latest_version.version_number,
                "model_config": latest_version.model_config,
                "llm_config_id": latest_version.llm_config_id,
                "judge_config_id": latest_version.judge_config_id,
                "judge_prompt": latest_version.judge_prompt,
                "default_thresholds": latest_version.default_thresholds
            } if latest_version else None,
            "datasets": [
                {
                    "id": dataset.id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "row_count": dataset.row_count,
                    "created_at": dataset.created_at.isoformat()
                }
                for dataset in (latest_version.reference_datasets if latest_version else [])
            ],
            "runs": [
                {
                    "id": run.id,
                    "name": run.name,
                    "status": run.status,
                    "overall_score": run.overall_score,
                    "pass_rate": run.pass_rate,
                    "created_at": run.created_at.isoformat()
                }
                for run in (latest_version.runs if latest_version else [])
            ]
        }
    })

@app.get("/api/datasets/{dataset_id}/preview")
async def preview_dataset_api(dataset_id: int, db: Session = Depends(get_db)):
    """API endpoint to preview dataset contents"""
    reference_dataset = db.query(ReferenceDataset).filter(ReferenceDataset.id == dataset_id).first()
    if not reference_dataset:
        return JSONResponse({
            "success": False,
            "message": "Dataset not found"
        }, status_code=404)
    
    # Load and parse the dataset content
    try:
        import json
        rows = []
        if reference_dataset.file_path and reference_dataset.file_path.endswith('.jsonl'):
            with open(reference_dataset.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        row = json.loads(line)
                        rows.append({
                            "question": row.get("question", ""),
                            "reference": row.get("reference", "")
                        })
        
        # Limit to first 10 rows for preview
        preview_rows = rows[:10]
        
        return JSONResponse({
            "success": True,
            "total_rows": len(rows),
            "rows": preview_rows,
            "created_at": reference_dataset.created_at.strftime('%Y-%m-%d %H:%M')
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Error reading dataset: {str(e)}"
        }, status_code=500)

@app.delete("/api/chatbots/{chatbot_id}")
async def delete_chatbot(chatbot_id: int, db: Session = Depends(get_db)):
    """Delete chatbot (soft delete)"""
    agent = db.query(Agent).filter(Agent.id == chatbot_id, Agent.is_active == True).first()
    if not agent:
        return JSONResponse({
            "success": False,
            "message": "Chatbot not found"
        }, status_code=404)
    
    try:
        # Soft delete the agent
        agent.is_active = False
        agent.updated_at = datetime.utcnow()
        db.commit()
        
        return JSONResponse({
            "success": True,
            "message": f"Chatbot '{agent.name}' deleted successfully"
        })
        
    except Exception as e:
        db.rollback()
        return JSONResponse({
            "success": False,
            "message": f"Error deleting chatbot: {str(e)}"
        }, status_code=500)