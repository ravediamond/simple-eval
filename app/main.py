import os
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from app.database import init_db

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
async def datasets(request: Request):
    """Datasets page"""
    return templates.TemplateResponse("datasets.html", {"request": request})

@app.get("/runs", response_class=HTMLResponse)
async def runs(request: Request):
    """Runs page"""
    return templates.TemplateResponse("runs.html", {"request": request})