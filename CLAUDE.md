# EvalNow - AI Chatbot Evaluation Platform

## 🎯 Project Goals

EvalNow is designed to make AI chatbot evaluation simple and accessible for non-technical users. The platform addresses the core problem: **"How do I know if my AI chatbot is actually working?"**

### Target Users
- **Product Managers**: Need clear metrics to assess chatbot performance
- **Team Leads**: Require shareable reports for stakeholder meetings  
- **Founders**: Want measurable progress to show investors/clients

### Value Proposition
> "Know if your AI chatbot works — in minutes."

Transform complex AI evaluation from a technical burden into a simple business tool that provides:
- Clear pass/fail scores
- Plain-English feedback  
- Shareable PDF reports
- Actionable improvement recommendations

## 🏗️ Technical Architecture

### Core Technology Stack
- **Backend**: FastAPI (Python 3.10+)
- **Frontend**: Bootstrap 5 + Vanilla JavaScript
- **Database**: Google Cloud Firestore
- **AI Evaluation**: Google Gemini (LLM-as-a-Judge)
- **File Processing**: Pandas + openpyxl
- **PDF Generation**: ReportLab
- **Dependency Management**: Poetry
- **Deployment**: Google Cloud Run

### Project Structure
```
simple-eval/
├── app/
│   ├── app.py              # Main FastAPI application
│   ├── analytics.py        # Analytics and tracking
│   └── pdf_generator.py    # PDF report generation
├── templates/              # Jinja2 HTML templates
│   ├── landing.html        # Main landing page
│   ├── results.html        # Evaluation results
│   ├── analytics.html      # Admin analytics dashboard
│   └── waitlist.html       # Waiting list signup
├── static/
│   ├── js/analytics.js     # Client-side analytics
│   └── public/             # Static assets
├── pyproject.toml          # Poetry dependencies
├── setup_firestore.sh     # Firestore initialization
└── CLAUDE.md              # This documentation
```

## 🔧 Development Setup

### Prerequisites
- Python 3.10+
- Poetry (dependency management)
- Google Cloud SDK (for deployment)
- Google Cloud Project with Firestore enabled

### Local Development
```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your Google Cloud credentials

# Run development server
poetry run uvicorn app.app:app --reload

# Set up Firestore (one-time)
./setup_firestore.sh
```

### Key Environment Variables
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
FIRESTORE_DATABASE=evalnow-db
GEMINI_API_KEY=your-gemini-api-key
ADMIN_USERNAME=your-admin-username
ADMIN_PASSWORD=your-admin-password
```

## 📊 Core Features

### 1. File Upload & Processing
- **Supported Formats**: CSV, JSONL, Excel (.xlsx, .xls)
- **Required Columns**: `question`, `reference`, `answer`
- **Limits**: 50 questions/file, 3 evaluations/day (beta)

### 2. AI-Powered Evaluation ("AI as Judge")

**Simple Explanation**: Think of it like having a smart, unbiased critic review your chatbot's answers.

**How It Works**:
1. **The Judge**: We use Google's Gemini AI as an expert evaluator
2. **The Process**: For each chatbot answer, the AI judge compares it against the correct reference answer
3. **The Decision**: The judge asks: "Does this answer actually help the user?" and gives a simple Pass/Fail
4. **The Reasoning**: Just like a human reviewer, the AI explains why it passed or failed each answer

**What Makes It Reliable**:
- **Consistent**: Unlike humans, the AI judge doesn't have bad days or biases
- **Fast**: Can review hundreds of answers in minutes, not hours
- **Detailed**: Explains every decision so you understand the "why"
- **Smart**: Recognizes when answers are correct even if worded differently than the reference

**Global Analysis & Insights**:
After reviewing all individual answers, the AI steps back and looks at the bigger picture:
- **Patterns**: "Your chatbot struggles with technical questions but excels at general knowledge"
- **Trends**: "Answers are accurate but often too wordy for customer service"
- **Recommendations**: "Focus training on concise responses" or "Add more examples for complex topics"
- **Business Impact**: "87% pass rate means customers are getting helpful answers"

This gives you actionable insights, not just numbers - so you know exactly what to improve and why it matters for your business.

**Real Example**:
- **Question**: "How do I reset my password?"
- **Reference Answer**: "Click 'Forgot Password' on the login page"
- **Chatbot's Answer**: "To reset your password, please navigate to the login screen and select the 'Forgot Password' link"
- **AI Judge Decision**: ✅ **PASS** - "Answer is correct and helpful, just uses slightly more formal language"
- **Your Takeaway**: Chatbot is working well, maybe train it to be more conversational

**What This Means for Product Owners**:
- No need to manually read through hundreds of chat logs
- Get confidence that your chatbot is actually helping customers
- Identify specific areas that need improvement before customers complain
- Have concrete data to show stakeholders: "Our chatbot has a 94% success rate"

### 3. Analytics & Insights
- **User Metrics**: Visitors, evaluators, retention rates
- **Usage Analytics**: Evaluations, dataset sizes, token usage
- **Token Tracking**: Input/output tokens, cost estimation
- **PDF Metrics**: Download rates, conversion tracking

### 4. Reporting
- **Real-time Results**: Immediate web-based results
- **PDF Export**: Professional reports with scores and insights
- **AI Analysis**: Global insights and recommendations
- **Performance Breakdown**: Score distribution and patterns

## 🚀 Deployment (Google Cloud Run)

### Deployment Architecture
- **Platform**: Google Cloud Run (serverless containers)
- **Database**: Firestore (NoSQL, serverless)
- **CDN**: Cloud Run handles static files
- **Scaling**: Automatic based on traffic
- **Region**: Configurable (default: europe-west1)

### Deployment Process
```bash
# Build and deploy to Cloud Run
gcloud run deploy evalnow \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated

# Set environment variables
gcloud run services update evalnow \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=your-project" \
  --region europe-west1
```

### Production Configuration
- **Container**: Python 3.10 with Poetry
- **Port**: 8080 (Cloud Run standard)
- **Memory**: 1GB (configurable)
- **CPU**: 1 vCPU (configurable)
- **Concurrency**: 100 requests/instance
- **Timeout**: 300 seconds

## 📈 Analytics Implementation

### Data Collection
- **User Tracking**: Anonymous user IDs, session-based
- **Event Tracking**: Page visits, evaluations, PDF downloads
- **Performance Metrics**: Token usage, response times
- **Business Metrics**: Conversion rates, retention

### Firestore Collections
```
analytics_events/        # Individual user actions
user_metrics/           # Aggregated user data
waiting_list/           # Beta signup requests
```

### Admin Dashboard
- **Access**: `/analytics` (admin auth required)
- **Metrics**: KPIs, usage charts, waiting list management
- **Real-time**: Auto-refreshing dashboard
- **Export**: Analytics API for external tools

## 🎨 User Experience Design

### Design Philosophy
- **Business language, not tech speak**: "Is your chatbot working?" instead of "LLM evaluation metrics"
- **Clarity over complexity**: Simple, focused interface that anyone can understand
- **Visual storytelling**: Clear progression from problem → solution → results
- **Mobile responsive**: Works on all device sizes
- **Outcome-focused**: Emphasizes what you learn, not how the technology works

### Landing Page Flow
1. **Problem**: "Your chatbot talks a lot. But is it actually helping?"
2. **Solution**: 3-step evaluation process
3. **Social Proof**: Personas and use cases
4. **Call to Action**: "Upload & Evaluate Free"

### Results Experience  
- **Immediate feedback**: Results page with clear pass/fail
- **Visual elements**: Charts, score badges, color coding
- **Actionable insights**: AI-generated recommendations
- **Shareable output**: Professional PDF reports

## 🔒 Security & Privacy

### Data Handling
- **No persistent storage**: Files processed and discarded immediately
- **Temporary processing**: Results cached only during session
- **Analytics**: Anonymous user tracking only
- **API keys**: Secure environment variable management

### Admin Security
- **Basic Auth**: Username/password protection for admin endpoints
- **Firestore Rules**: Server-side only access to analytics
- **Input validation**: File type and size restrictions
- **Rate limiting**: Daily evaluation limits per user

## 🚦 Current Status & Roadmap

### Beta Release (Current)
- ✅ Core evaluation functionality
- ✅ PDF report generation
- ✅ Analytics dashboard
- ✅ Waiting list management
- ✅ Cloud Run deployment

### Future Enhancements
- **Enterprise Features**: Higher limits, custom models
- **Advanced Analytics**: Trend analysis, A/B testing
- **API Access**: Programmatic evaluation endpoints
- **Integration**: Slack, Teams, webhook notifications
- **Multi-language**: Support for non-English evaluations

## 🧪 Testing & Quality

### Test Commands
```bash
# Run evaluation with test file
poetry run python -c "
import asyncio
from app.app import upload_and_evaluate
# Test evaluation logic
"

# Test analytics
curl -u admin:password http://localhost:8000/api/analytics/kpis

# Test file processing
python -c "from app.app import process_file; print(process_file('test.csv'))"
```

### Quality Assurance
- **Input validation**: File format, column requirements
- **Error handling**: Graceful failure with user feedback
- **Performance**: Async processing, efficient token usage
- **Monitoring**: Analytics tracking, error logging

## 📞 Support & Maintenance

### Key Commands
- **Start server**: `poetry run uvicorn app.app:app --reload`
- **Deploy**: `gcloud run deploy evalnow --source .`
- **Setup Firestore**: `./setup_firestore.sh`
- **View logs**: `gcloud run logs tail evalnow`

### Monitoring
- **Health check**: `/` endpoint returns landing page
- **Analytics**: `/api/analytics/kpis` for system metrics
- **Errors**: Cloud Run logs for debugging
- **Performance**: Firestore query monitoring

This project represents a shift from "AI evaluation tool" to "business confidence platform" - making AI assessment accessible to everyone who deploys chatbots.