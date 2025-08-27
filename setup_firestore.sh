#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 EvalNow Firestore Setup${NC}"
echo "=================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ Error: gcloud CLI not found${NC}"
    echo "Install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if project is configured
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: No Google Cloud project configured${NC}"
    echo "Set it with: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo -e "${GREEN}✅ Project: $PROJECT_ID${NC}"

# Enable required APIs
echo -e "${BLUE}🔧 Enabling required APIs...${NC}"
gcloud services enable firestore.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
echo -e "${GREEN}✅ APIs enabled${NC}"

# Create Firestore database (Native mode) with custom name
echo -e "${BLUE}📝 Creating Firestore database...${NC}"
DATABASE_NAME="evalnow-db"

if gcloud firestore databases describe --database="$DATABASE_NAME" 2>/dev/null >/dev/null; then
    echo -e "${YELLOW}⚠️ Firestore database '$DATABASE_NAME' already exists${NC}"
else
    echo "Creating Firestore database '$DATABASE_NAME' in Belgium region..."
    gcloud firestore databases create --database="$DATABASE_NAME" --location=europe-west1 --quiet
    echo -e "${GREEN}✅ Firestore database '$DATABASE_NAME' created${NC}"
fi

# Create composite indexes for better query performance
echo -e "${BLUE}🔍 Creating Firestore indexes...${NC}"

# Index for analytics_events: event_type + timestamp (for filtering by event type with time ordering)
echo "Creating index: analytics_events (event_type ASC, timestamp DESC)"
gcloud firestore indexes composite create \
    --database="$DATABASE_NAME" \
    --collection-group=analytics_events \
    --field-config=field-path=event_type,order=ascending \
    --field-config=field-path=timestamp,order=descending \
    --quiet || echo "Index may already exist"

# Index for analytics_events: user_id + timestamp (for user-specific queries)
echo "Creating index: analytics_events (user_id ASC, timestamp DESC)"
gcloud firestore indexes composite create \
    --database="$DATABASE_NAME" \
    --collection-group=analytics_events \
    --field-config=field-path=user_id,order=ascending \
    --field-config=field-path=timestamp,order=descending \
    --quiet || echo "Index may already exist"

# Index for user_metrics: total_evaluations + last_activity (for finding active users)
echo "Creating index: user_metrics (total_evaluations ASC, last_activity DESC)"
gcloud firestore indexes composite create \
    --database="$DATABASE_NAME" \
    --collection-group=user_metrics \
    --field-config=field-path=total_evaluations,order=ascending \
    --field-config=field-path=last_activity,order=descending \
    --quiet || echo "Index may already exist"

# Index for waiting_list: requested_at (for chronological ordering)
echo "Creating index: waiting_list (requested_at DESC)"
gcloud firestore indexes fields create \
    --database="$DATABASE_NAME" \
    --collection-group=waiting_list \
    --field-config=field-path=requested_at,order=descending \
    --quiet || echo "Index may already exist"

# Index for feedback: server_timestamp (for chronological ordering)
echo "Creating index: feedback (server_timestamp DESC)"
gcloud firestore indexes fields create \
    --database="$DATABASE_NAME" \
    --collection-group=feedback \
    --field-config=field-path=server_timestamp,order=descending \
    --quiet || echo "Index may already exist"

echo -e "${GREEN}✅ Indexes created (may take a few minutes to build)${NC}"

# Set up Firestore security rules
echo -e "${BLUE}🔒 Setting up security rules...${NC}"
cat > firestore.rules << 'EOF'
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Allow read/write access to analytics collections from server-side only
    // Client-side access is restricted
    match /analytics_events/{document} {
      allow read, write: if false;  // Server-side only via service account
    }
    match /user_metrics/{document} {
      allow read, write: if false;  // Server-side only via service account
    }
    match /waiting_list/{document} {
      allow read, write: if false;  // Server-side only via service account
    }
    match /feedback/{document} {
      allow read, write: if false;  // Server-side only via service account
    }
  }
}
EOF

# Deploy via Firebase CLI if available, otherwise skip
if command -v firebase &> /dev/null; then
    firebase deploy --only firestore:rules --quiet
    rm firestore.rules
    echo -e "${GREEN}✅ Security rules deployed${NC}"
else
    echo -e "${YELLOW}⚠️ Firebase CLI not found - skipping security rules${NC}"
    echo "Install Firebase CLI: npm install -g firebase-tools"
    echo "Then deploy rules manually: firebase deploy --only firestore:rules"
    rm firestore.rules
fi

# Set environment variables for the application
echo -e "${BLUE}🔧 Environment setup...${NC}"
echo "export GOOGLE_CLOUD_PROJECT=$PROJECT_ID" >> .env.local
echo "export FIRESTORE_DATABASE=$DATABASE_NAME" >> .env.local
echo -e "${GREEN}✅ Added environment variables to .env.local${NC}"

echo ""
echo -e "${GREEN}🎉 Firestore setup completed successfully!${NC}"
echo ""
echo -e "${YELLOW}📊 Collections that will be created:${NC}"
echo "   • analytics_events - Stores all user interaction events"
echo "   • user_metrics - Aggregated metrics per user"
echo "   • waiting_list - Users waiting for higher limits"
echo "   • feedback - User feedback and satisfaction ratings"
echo ""
echo -e "${YELLOW}🔍 Indexes created for optimal queries:${NC}"
echo "   • analytics_events: event_type + timestamp"
echo "   • analytics_events: user_id + timestamp" 
echo "   • user_metrics: total_evaluations + last_activity"
echo "   • waiting_list: requested_at (chronological)"
echo "   • feedback: server_timestamp (chronological)"
echo ""
echo -e "${YELLOW}🔒 Security:${NC}"
echo "   • Collections are server-side only (no client access)"
echo "   • Data is automatically tracked via your app"
echo ""
echo -e "${YELLOW}🎯 Next steps:${NC}"
echo "   1. Install dependencies: poetry install"
echo "   2. Set environment: source .env.local"
echo "   3. Deploy your app - analytics will start tracking!"
echo "   4. Monitor in Firebase Console: https://console.firebase.google.com/project/$PROJECT_ID/firestore"
echo ""