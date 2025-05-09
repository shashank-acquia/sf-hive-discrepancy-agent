#!/bin/bash
set -e

IMAGE_NAME="sf-hive-discrepancy-agent"
CONTAINER_NAME="sf-hive-discrepancy-resolver"
PORT=5000

echo "====================================================="
echo "Running the SF-Hive Discrepancy Agent"
echo "====================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with the properties in the README."
    exit 1
fi

# Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

echo "Starting container with name: $CONTAINER_NAME..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $PORT:$PORT \
  --env-file .env \
  $IMAGE_NAME

echo "====================================================="
echo "Container started successfully!"
echo "Access the application at: http://localhost:$PORT"
echo "====================================================="