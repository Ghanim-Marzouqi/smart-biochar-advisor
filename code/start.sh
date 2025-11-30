#!/bin/bash

# Smart Biochar Advisor - Docker Startup Script

echo "🌱 Smart Biochar Advisor - Docker Setup"
echo "========================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"

# Create necessary directories
echo "📁 Creating data and log directories..."
mkdir -p data/samples data/models logs

# Build and start containers
echo "🔨 Building Docker image..."
docker-compose build

echo "🚀 Starting containers..."
docker-compose up -d

# Wait for app to be ready
echo "⏳ Waiting for app to start..."
sleep 5

# Check if container is running
if docker ps | grep -q smart-biochar-advisor; then
    echo ""
    echo "✅ Smart Biochar Advisor is running!"
    echo "🌐 Access the app at: http://localhost:8501"
    echo ""
    echo "Useful commands:"
    echo "  - View logs:    docker-compose logs -f"
    echo "  - Stop app:     docker-compose down"
    echo "  - Restart:      docker-compose restart"
    echo "  - Rebuild:      docker-compose up -d --build"
else
    echo "❌ Failed to start container. Check logs with: docker-compose logs"
    exit 1
fi
