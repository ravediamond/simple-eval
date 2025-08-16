# Simple Eval

A lightweight local web app to evaluate chatbot "agents" against verified datasets.

## What We're Building (MVP)

A local-first evaluation platform that lets you:

- **Create Agents** - Define what to test with configurable metrics, thresholds, and judge models
- **Two Evaluation Modes**:
  1. **Connector** - App queries the agent live and scores responses
  2. **Uploaded Answers** - Upload pre-generated answers for scoring only
- **Upload Datasets** - Questions + verified reference answers with optional contexts
- **View Results** - Per-case scores, pass/fail status, explanations, and aggregate metrics
- **Version Control** - Immutable snapshots of Agents and Datasets for reproducible evaluations

**Tech Stack**: DeepEval (metrics), FastAPI/Flask (backend), server-rendered UI (Jinja + HTMX), SQLite (storage)

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

## Metrics (MVP)

### LLM-as-Judge (via DeepEval)
- Compares agent answer to reference using customizable rubric
- Customizable judge prompt (stored on AgentVersion)
- Threshold: pass if score ≥ threshold
- Outputs: numeric score [0-1], short explanation

### Faithfulness (via DeepEval)
- Checks if agent's answer stays grounded in provided contexts
- Only runs when contexts exist; auto-skips with clear note otherwise

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

## Technical Details

### Thresholds & Prompts
- **Thresholds**: Defaults on Agent; each Run snapshots thresholds used
- **Judge Prompt**: Editable template with variables `{question}`, `{reference}`, `{answer}`, `{contexts}`
- Stored on AgentVersion; "Reset to default" option
- Bias warning when tested model equals judge model

### Validation (Light for v1)
- Required non-empty: `question`, `reference`, `answer`
- UTF-8, configurable length guards, whitespace trimming
- Duplicate/unknown ids → error
- Faithfulness silently skipped if no contexts

### Non-Functional Requirements
- **Local-first**: Single process, SQLite storage, files under `./data`
- **No telemetry**: Keys/config via local env file
- **Determinism**: Record model params; capture context (not guaranteeing reproducibility)
- **Resilience**: Partial results remain visible if run stops
- **Logs**: Simple per-run log; last N lines in UI

## Out of Scope (v1)
- Multi-user auth, roles, SSO
- Complex dataset schemas beyond question/reference/contexts
- Human labeling UI, crowd workflows
- Rich visual dashboards beyond tables and summaries
- Side-by-side agent comparisons

## Acceptance Criteria (v1)
- ✅ Create Agent v1 with LLM-as-Judge, custom thresholds and judge prompt
- ✅ Import Dataset v1 (question + reference)
- ✅ Run Uploaded Answers mode with 100% coverage → scores + explanations
- ✅ Run Connector mode → scores with clear source indication
- ✅ Results export (CSV/JSON)
- ✅ Versioning: prompt/threshold changes → Agent v2; question changes → Dataset v2
- ✅ Runs always reference immutable AgentVersion + DatasetVersion in UI

## Future Iterations
After v1: schema mapping, partial coverage, richer dashboards, RAG contexts by default, agent comparisons, containerization, cloud deployment options.
