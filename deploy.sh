#!/bin/bash

# Simple Eval - Cloud Run Deployment Script
# This script builds and pushes the Docker image to Google Container Registry

set -e

PROJECT_ID="evalnow-prod"
SERVICE_NAME="evalnow"
REGION="europe-west9"
IMAGE_TAG="europe-west9-docker.pkg.dev/$PROJECT_ID/$SERVICE_NAME/$SERVICE_NAME:latest"

echo "🚀 Starting deployment for $SERVICE_NAME..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_TAG .

# Configure Docker to use gcloud as a credential helper
echo "🔧 Configuring Docker authentication..."
gcloud auth configure-docker europe-west9-docker.pkg.dev

# Push the image to Google Container Registry
echo "⬆️  Pushing image to Container Registry..."
docker push $IMAGE_TAG

echo "✅ Image pushed successfully!"

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
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

echo ""
echo "✅ Deployment complete!"
echo "🌍 Your app is available at:"
gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"