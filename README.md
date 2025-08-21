# Simple Eval

A simple AI evaluation tool for quick prototyping and validation. Upload your data, get insights - no complex setup required.

## 🚀 Quick Start

### Setup
```bash
# Install dependencies
poetry install

# Add your Gemini API key to .env
GEMINI_API_KEY=your_gemini_api_key_here

# Start the server
python main.py
```

**Application URL**: http://localhost:8000

## What This Tool Does

Simple Eval helps you quickly evaluate AI responses by:

1. **Upload your data** - CSV, JSONL, or Excel files with question, reference, and answer columns
2. **Get instant evaluation** - AI-powered scoring using Google's Gemini
3. **See key insights** - Performance breakdown, score distribution, and detailed results
4. **No data storage** - Everything is processed in memory and discarded immediately

## Key Features

### ✅ Privacy First
- **No data storage** - Files processed in memory only
- **No user tracking** - Simple, anonymous evaluation
- **Local processing** - Runs entirely on your machine

### ✅ Simple Upload
- Support for CSV, JSONL, and Excel files
- Required columns: `question`, `reference`, `answer`
- Drag & drop or click to upload
- Instant validation and error checking

### ✅ AI-Powered Evaluation
- Uses Google Gemini for intelligent scoring
- Evaluates accuracy, completeness, and relevance
- Provides detailed reasoning for each score
- Scores from 0-100% with clear thresholds

### ✅ Rich Insights
- **Performance Overview**: Total questions, average score, pass rate
- **Score Distribution**: Excellent (90%+), Good (70-89%), Needs Work (50-69%), Poor (<50%)
- **Detailed Results**: Question-by-question breakdown with reasoning
- **Interactive Interface**: Modern, responsive design

## File Format

Your file should have three columns:

| question | reference | answer |
|----------|-----------|---------|
| What is 2+2? | 4 | 2+2 equals 4 |
| Capital of France? | Paris | The capital city of France is Paris |

## Configuration

Edit `.env` file:
```bash
# App Configuration
DEBUG=true
HOST=127.0.0.1
PORT=8000

# AI Configuration - Add your Gemini API key here
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash-exp
```

## Test Dataset

Sample datasets included in `data/`:
- `test_dataset.csv` - 10 sample questions in CSV format
- `test_dataset.jsonl` - Same data in JSONL format

## Tech Stack

- **Backend**: FastAPI + Python
- **AI**: Google Gemini (via AI Vertex)
- **Frontend**: Bootstrap 5 + vanilla JavaScript
- **No Database**: In-memory processing only
- **Dependencies**: Minimal set focused on core functionality

## Why Simple Eval?

This tool was built for rapid prototyping and user validation:

- **Quick Setup**: Single command to start
- **No Complexity**: No databases, user management, or complex configuration
- **Focus on Value**: Evaluate AI quality instantly
- **Privacy Focused**: No data collection or storage
- **Prototype Ready**: Perfect for testing user demand

## Development

### Project Structure
```
simple-eval/
├── app/
│   └── simple_main.py      # Main FastAPI application
├── templates/
│   ├── landing.html        # Landing page with upload
│   └── results.html        # Results with insights
├── static/                 # CSS, JS, images
├── data/                   # Test datasets
├── main.py                 # Application entry point
└── pyproject.toml         # Dependencies
```

### Running Tests
Test the application with sample data:
1. Start the server: `python main.py`
2. Go to http://localhost:8000
3. Upload `data/test_dataset.csv`
4. Review the evaluation results

## API Reference

- `GET /` - Landing page with upload widget
- `POST /upload` - Process uploaded file and run evaluation
- `GET /evaluations` - Simple evaluations page (redirects to landing)
- `GET /healthz` - Health check endpoint

## Limitations

This is a simplified prototype with intentional limitations:
- No data persistence
- Single AI model (Gemini)
- Basic evaluation metrics
- No user accounts or sessions
- No advanced analytics

## Future Considerations

If user validation is successful, consider adding:
- Multiple AI model support
- Advanced evaluation metrics
- Data persistence options
- User accounts and history
- Batch processing capabilities
- API integrations