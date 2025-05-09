#!/bin/bash
set -e

IMAGE_NAME="sf-hive-discrepancy-agent"

echo "====================================================="
echo "Building the SF-Hive Discrepancy Agent Docker image"
echo "====================================================="

echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

echo "====================================================="
echo "Docker image built successfully: $IMAGE_NAME"
echo "====================================================="