#!/bin/bash

# Simple Eval - Cloud Run Deployment Script
# This script builds and pushes the Docker image to Google Container Registry

set -e

PROJECT_ID="mcph-dev"
SERVICE_NAME="simple-eval"
REGION="us-central1"
IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Starting deployment for $SERVICE_NAME..."

# Build the Docker image
echo "📦 Building Docker image..."
docker build -t $IMAGE_TAG .

# Configure Docker to use gcloud as a credential helper
echo "🔧 Configuring Docker authentication..."
gcloud auth configure-docker

# Push the image to Google Container Registry
echo "⬆️  Pushing image to Container Registry..."
docker push $IMAGE_TAG

echo "✅ Image pushed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Go to Google Cloud Console"
echo "2. Navigate to Cloud Run"
echo "3. Click 'Create Service'"
echo "4. Use this container image: $IMAGE_TAG"
echo "5. Set port to: 8080"
echo "6. Allow unauthenticated invocations"
echo "7. Set minimum instances to 0 (for cost optimization)"
echo ""
echo "🌍 Your app will be available at: https://$SERVICE_NAME-[hash]-$REGION.a.run.app"