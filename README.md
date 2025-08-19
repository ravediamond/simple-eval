# EvalNow

A lightweight local web app to evaluate chatbot "agents" against verified datasets.

## 🚀 Quick Start

### Launch the Application

```bash
# Install dependencies
poetry install

# Start the server
python -m uvicorn app.main:app --reload --port 8000

# Or using Poetry
poetry run uvicorn app.main:app --reload --port 8000
```

**Application URL**: http://localhost:8000

### First Steps
1. **Upload Evaluation Questions** → http://localhost:8000/datasets/upload
2. **Create an Agent** → http://localhost:8000/agents/new  
3. **Create a Test Attempt** → http://localhost:8000/runs/new

### Available Pages
- **Dashboard**: http://localhost:8000/ - Overview and recent test attempts
- **Agents**: http://localhost:8000/agents - Agent management
- **Evaluation Questions**: http://localhost:8000/datasets - Question set management
- **Test Attempts**: http://localhost:8000/runs - Evaluation attempts

## What We're Building (MVP)

A local-first evaluation platform that lets you:

- **Create Agents** - Define what to test with configurable metrics, thresholds, and judge models
- **Two Evaluation Modes**:
  1. **Connector** - App queries the agent live and scores responses
  2. **Uploaded Answers** - Upload pre-generated answers for scoring only
- **Upload Evaluation Questions** - Questions + verified reference answers with optional contexts
- **View Results** - Per-case scores, pass/fail status, explanations, and aggregate metrics
- **Version Control** - Immutable snapshots of Agents and Datasets for reproducible evaluations

**Tech Stack**: LLM-based evaluation engine, FastAPI (backend), server-rendered UI (Jinja2 + Bootstrap), SQLite (storage)

## Why This Architecture

- **Local-first & Simple**: Runs on laptop; single binary/container; minimal ops
- **Reproducible Truth**: Immutable versions ensure stable evaluation snapshots
- **Flexible Input**: Evaluate via connector or upload answers - no vendor lock-in
- **Start Small**: Two core metrics with room to expand

## Core Concepts

### Agent
Represents what you're testing.

**Fields:**
- Name, description, tags
- Model config (provider, model name, params for Connector mode)
- Metric suite toggles: `llm_as_judge`, `faithfulness`
- Default thresholds per metric (e.g., 0.70)
- Judge model config + customizable judge prompt template

**Versioning:** AgentVersion is immutable, freezing all config with notes explaining changes.

### Dataset
What counts as "correct."

**DatasetVersion** contains normalized rows:
- `id`, `question`, `reference`
- Optional: `contexts` (array), `tag`

Creating/changing questions/references → new DatasetVersion.

### Run
One evaluation execution.

- References specific AgentVersion + DatasetVersion
- Evaluation source: `connector` or `uploaded_answers`
- Stores per-case results, aggregates, and timing

## Evaluation Modes

### A) Connector (Live)
App sends each question to the agent via connector (OpenAI-style, Ollama, custom HTTP), records answers, then scores them.

### B) Uploaded Answers (Offline)
Upload an Answers file with `id`, `answer` for the chosen DatasetVersion. No connector calls - scoring only. Useful for batch outputs, cost control, or systems we can't call directly.

## 🎯 Implemented Features

### ✅ Complete Implementation Status
- **Phase 1**: Dataset upload and management
- **Phase 2**: Agent configuration with model settings  
- **Phase 3**: Run creation with uploaded answers
- **Phase 4**: Results visualization and export
- **Phase 5**: Real evaluation with LLM-as-Judge
- **Phase 6**: Live evaluation via connectors

### 🧠 Evaluation Metrics

#### LLM-as-Judge
- Compares agent answer to reference using customizable rubric
- Customizable judge prompt (stored on AgentVersion)
- Threshold: pass if score ≥ threshold
- Outputs: numeric score [0-1], detailed reasoning
- Supports multiple score formats (0-1, 0-10, percentages, fractions)

#### Faithfulness
- Checks if agent's answer stays grounded in provided contexts
- Only runs when contexts exist; auto-skips with clear note otherwise
- Uses LLM to evaluate context adherence
- Configurable thresholds and detailed explanations

### 🔧 Connector System
- **OpenAI-Compatible**: Standard chat completions API
- **Generic HTTP**: Custom API endpoints with flexible response parsing
- **Rate Limiting**: Configurable requests per minute
- **Error Handling**: Retry logic with exponential backoff
- **Connection Testing**: Built-in endpoint validation

## Data Contracts

### Dataset Files
**Single file (preferred):**
- CSV (header) or JSONL
- Required: `question` (string), `reference` (string)
- Optional: `id`, `tag`, `contexts` (array; JSON-encoded in CSV)

**Two-file dataset:**
- Questions file: `id`, `question` (+ optional `tag`, `contexts`)
- Answers file: `id`, `reference`
- Inner join on `id`; unknown/duplicate ids → error

### Answers File (Uploaded Answers mode)
- CSV or JSONL with: `id` (must match DatasetVersion), `answer`
- Optional: `model_name`, `timestamp`, `meta`
- Requires 100% id coverage

## User Interface

### Key Pages
- **Dashboard**: Recent runs (status, pass rate, duration)
- **Agents**: Create/version agents; configure metrics, thresholds, judge prompts
- **Datasets**: Upload with preview; create new DatasetVersions
- **New Run**: Pick AgentVersion + DatasetVersion; choose evaluation source
- **Results**: Summary stats, detailed table with filters, export options

### Results View
- Summary: avg per metric, overall pass%
- Table: question | answer | reference | scores | pass/fail | explanation
- Filters (e.g., failures first)
- Export JSON/CSV
- Clear header showing AgentVersion, DatasetVersion, evaluation source

## 💾 Technical Details

### Architecture
- **Backend**: FastAPI with SQLAlchemy ORM
- **Database**: SQLite for local storage
- **Templates**: Jinja2 with Bootstrap 5 UI
- **Evaluation**: Custom LLM-based evaluation engine
- **Dependencies**: Poetry for package management

### Configuration
Create a `.env` file in the project root:
```
DATABASE_URL=sqlite:///./data/evalnow.db
DEBUG=true
HOST=127.0.0.1
PORT=8000
OPENAI_API_KEY=your_openai_key_here  # Optional, for judge models
```

### Thresholds & Prompts
- **Thresholds**: Configurable per metric (default: 0.8)
- **Judge Prompt**: Fully customizable evaluation prompts
- **Verbose Artifacts**: Optional storage of judge prompts and raw responses
- **Score Normalization**: Automatic handling of different score formats

### Live Evaluation Workflow
1. **Agent Setup**: Configure connector (OpenAI/HTTP) with API details
2. **Run Creation**: Choose "Live Evaluation (Connector)" mode  
3. **Processing**: 
   - Connector evaluates each dataset question automatically
   - Responses captured as "actual answers"
   - Evaluation engine scores via LLM-as-Judge & Faithfulness
   - Results stored with full scoring breakdown
4. **Results**: Rich results view with scores, reasoning, and artifacts

### Export Features
- **CSV Export**: Structured data for analysis
- **JSON Export**: Complete data with metadata
- **HTML Reports**: Self-contained reports for sharing
- **Filtering**: Results by pass/fail status, score ranges, text search

### Error Handling
- **Graceful Degradation**: Partial results visible if evaluation fails
- **Rate Limiting**: Built-in protection with configurable limits
- **Retry Logic**: Exponential backoff for API failures
- **Validation**: Comprehensive input validation with clear error messages

## Out of Scope (v1)
- Multi-user auth, roles, SSO
- Complex dataset schemas beyond question/reference/contexts
- Human labeling UI, crowd workflows
- Rich visual dashboards beyond tables and summaries
- Side-by-side agent comparisons

## 📋 Development Setup

### Prerequisites
- Python 3.10+
- Poetry package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd simple-eval

# Install dependencies
poetry install

# Set up environment variables (optional)
cp .env.example .env  # Edit with your configuration

# Run database migrations (required)
python migrate_db.py

# Start development server
python -m uvicorn app.main:app --reload --port 8000
```

### Project Structure
```
simple-eval/
├── app/
│   ├── main.py              # FastAPI application
│   ├── models.py            # SQLAlchemy models
│   ├── database.py          # Database configuration
│   ├── connectors.py        # Live evaluation connectors
│   ├── evaluation_engine.py # LLM-based evaluation
│   ├── run_utils.py         # Run processing logic
│   ├── dataset_utils.py     # Dataset processing
│   └── export_utils.py      # Export functionality
├── templates/               # Jinja2 HTML templates
├── static/                  # CSS, JS, images
├── data/                    # SQLite database and files
└── pyproject.toml          # Poetry dependencies
```

## ✅ Acceptance Criteria (All Completed)
- ✅ Create Agent v1 with LLM-as-Judge, custom thresholds and judge prompt
- ✅ Import Dataset v1 (question + reference + optional contexts)
- ✅ Run Uploaded Answers mode with 100% coverage → scores + explanations
- ✅ Run Connector mode → live evaluation with clear source indication
- ✅ Results export (CSV/JSON/HTML)
- ✅ Versioning: prompt/threshold changes → Agent v2; question changes → Dataset v2
- ✅ Runs always reference immutable AgentVersion + DatasetVersion in UI
- ✅ Real LLM-based evaluation with detailed reasoning
- ✅ Verbose artifacts for debugging and transparency
- ✅ Connector system with rate limiting and error handling

## 🚀 Usage Examples

### Creating an Agent
1. Navigate to http://localhost:8000/agents/new
2. Configure basic info (name, description, tags)
3. Set model configuration (provider, model, temperature)
4. Enable metrics (LLM-as-Judge, Faithfulness)
5. Configure judge model and custom prompts
6. Optionally enable live evaluation connector

### Uploading a Dataset
1. Go to http://localhost:8000/datasets/upload
2. Upload CSV/JSONL with required fields: `question`, `reference`
3. Optional fields: `id`, `context`, `tag`
4. Review preview and validation
5. Confirm upload to create dataset version

### Running Evaluations

#### Upload Mode (Pre-computed Answers)
1. Create run at http://localhost:8000/runs/new
2. Select "Upload Answers File" mode
3. Choose agent version and dataset version
4. Upload answers file with `id` and `answer` fields
5. Submit to start evaluation

#### Connector Mode (Live Evaluation)
1. Ensure agent has connector configured
2. Select "Live Evaluation (Connector)" mode
3. Choose agent version and dataset version
4. Submit - no file upload needed
5. Agent will evaluate each question live via API

### Viewing Results
- Real-time progress tracking during evaluation
- Summary cards with overall scores and pass rates
- Detailed per-case results with scores and reasoning
- Filtering by pass/fail status, score ranges, text search
- Export options: CSV, JSON, HTML reports
- Expandable verbose artifacts showing judge prompts and responses

## 🔧 API Endpoints

- `GET /` - Dashboard
- `GET /agents` - List agents
- `POST /agents` - Create agent
- `GET /datasets` - List datasets  
- `POST /datasets/upload` - Upload dataset
- `GET /runs` - List runs
- `POST /runs/new/upload` - Create run
- `GET /runs/{id}` - View run results
- `POST /api/test-connector` - Test connector configuration
- `GET /api/runs/{id}/status` - Get run status (for polling)

## 🔧 Troubleshooting

### Database Schema Errors
If you see errors like `no such column: runs.evaluation_source`, run the migration script:

```bash
python migrate_db.py
```

This will update your database schema to match the current models.

### Port Already in Use
If you get `Address already in use` errors:

```bash
# Kill any processes using port 8000
lsof -ti:8000 | xargs kill -9

# Then restart the server
python -m uvicorn app.main:app --reload --port 8000
```

### Missing Dependencies
If you encounter import errors:

```bash
# Reinstall dependencies
poetry install

# Or install specific missing packages
poetry add <package-name>
```

## 🔮 Future Enhancements

### Planned Features
- **DeepEval Integration**: Full integration when compatibility issues are resolved
- **Advanced Metrics**: BLEU, ROUGE, semantic similarity
- **Batch Operations**: Multi-agent comparisons
- **Custom Metrics**: Plugin system for custom evaluation logic
- **Visualization**: Charts and graphs for trend analysis
- **Docker Support**: Containerized deployment
- **Cloud Deployment**: AWS/GCP deployment guides

### Potential Improvements
- **Schema Mapping**: Handle different dataset formats
- **Partial Coverage**: Allow incomplete answer sets
- **Human Labeling**: Manual scoring interface
- **A/B Testing**: Side-by-side agent comparisons
- **Performance Optimization**: Caching and parallel processing
- **Multi-user Support**: Authentication and user management
