import os
import json
from fastapi import FastAPI, Request, Depends, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from app.database import init_db, get_db
from app.models import Dataset, DatasetVersion
from app.dataset_utils import DatasetProcessor

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
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/agents", response_class=HTMLResponse)
async def agents(request: Request):
    """Agents page"""
    return templates.TemplateResponse("agents.html", {"request": request})

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

@app.get("/runs", response_class=HTMLResponse)
async def runs(request: Request):
    """Runs page"""
    return templates.TemplateResponse("runs.html", {"request": request})