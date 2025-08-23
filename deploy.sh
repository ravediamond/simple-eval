#!/bin/bash

# Simple Eval - Cloud Run Deployment Script
# This script builds and pushes the Docker image to Google Container Registry

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo -e "${BLUE}📄 Loading environment variables from .env file...${NC}"
    export $(grep -v '^#' .env | xargs)
    echo -e "${GREEN}✅ Environment variables loaded${NC}"
else
    echo -e "${YELLOW}⚠️  No .env file found, using default values${NC}"
fi

# Set defaults if not provided in .env
PROJECT_ID=${PROJECT_ID:-"evalnow-prod"}
SERVICE_NAME=${SERVICE_NAME:-"evalnow"}
REGION=${REGION:-"europe-west1"}
IMAGE_TAG="europe-west1-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/$SERVICE_NAME:latest"

echo -e "${BLUE}🚀 Starting deployment for $SERVICE_NAME...${NC}"
echo -e "${YELLOW}Project: $PROJECT_ID${NC}"
echo -e "${YELLOW}Region: $REGION${NC}"

# Build the Docker image
echo -e "${BLUE}📦 Building Docker image...${NC}"
docker build -t $IMAGE_TAG .

# Configure Docker to use gcloud as a credential helper
echo -e "${BLUE}🔧 Configuring Docker authentication...${NC}"
gcloud auth configure-docker europe-west1-docker.pkg.dev

# Push the image to Google Container Registry
echo -e "${BLUE}⬆️  Pushing image to Container Registry...${NC}"
docker push $IMAGE_TAG

echo -e "${GREEN}✅ Image pushed successfully!${NC}"

# Prepare environment variables for Cloud Run
echo -e "${BLUE}🔧 Preparing environment variables...${NC}"
ENV_VARS=""

# Add environment variables if they exist
if [ ! -z "$GOOGLE_CLOUD_PROJECT" ]; then
    ENV_VARS="$ENV_VARS,GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT"
    echo -e "${GREEN}  • GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT${NC}"
fi

if [ ! -z "$FIRESTORE_DATABASE" ]; then
    ENV_VARS="$ENV_VARS,FIRESTORE_DATABASE=$FIRESTORE_DATABASE"
    echo -e "${GREEN}  • FIRESTORE_DATABASE=$FIRESTORE_DATABASE${NC}"
fi

if [ ! -z "$GEMINI_API_KEY" ]; then
    ENV_VARS="$ENV_VARS,GEMINI_API_KEY=$GEMINI_API_KEY"
    echo -e "${GREEN}  • GEMINI_API_KEY=***${NC}"
fi

if [ ! -z "$GEMINI_JUDGE_MODEL" ]; then
    ENV_VARS="$ENV_VARS,GEMINI_JUDGE_MODEL=$GEMINI_JUDGE_MODEL"
    echo -e "${GREEN}  • GEMINI_JUDGE_MODEL=$GEMINI_JUDGE_MODEL${NC}"
fi

if [ ! -z "$GEMINI_ANALYSIS_MODEL" ]; then
    ENV_VARS="$ENV_VARS,GEMINI_ANALYSIS_MODEL=$GEMINI_ANALYSIS_MODEL"
    echo -e "${GREEN}  • GEMINI_ANALYSIS_MODEL=$GEMINI_ANALYSIS_MODEL${NC}"
fi

if [ ! -z "$ADMIN_USERNAME" ]; then
    ENV_VARS="$ENV_VARS,ADMIN_USERNAME=$ADMIN_USERNAME"
    echo -e "${GREEN}  • ADMIN_USERNAME=$ADMIN_USERNAME${NC}"
fi

if [ ! -z "$ADMIN_PASSWORD" ]; then
    ENV_VARS="$ENV_VARS,ADMIN_PASSWORD=$ADMIN_PASSWORD"
    echo -e "${GREEN}  • ADMIN_PASSWORD=***${NC}"
fi

if [ ! -z "$DEBUG" ]; then
    ENV_VARS="$ENV_VARS,DEBUG=$DEBUG"
    echo -e "${GREEN}  • DEBUG=$DEBUG${NC}"
fi

# Remove leading comma
ENV_VARS=${ENV_VARS#,}

# Deploy to Cloud Run
echo -e "${BLUE}🚀 Deploying to Cloud Run...${NC}"
if [ ! -z "$ENV_VARS" ]; then
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_TAG \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8080 \
        --memory 1Gi \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 10 \
        --set-env-vars "$ENV_VARS"
else
    echo -e "${YELLOW}⚠️  No environment variables to set${NC}"
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_TAG \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --port 8080 \
        --memory 1Gi \
        --cpu 1 \
        --min-instances 0 \
        --max-instances 10
fi

echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${BLUE}🌍 Your app is available at:${NC}"
gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"

echo ""
echo -e "${YELLOW}📝 Environment variables configured:${NC}"
if [ ! -z "$ENV_VARS" ]; then
    echo -e "${GREEN}  Environment variables have been set on Cloud Run${NC}"
else
    echo -e "${YELLOW}  No environment variables were configured${NC}"
    echo -e "${YELLOW}  Create a .env file to set environment variables for deployment${NC}"
fi