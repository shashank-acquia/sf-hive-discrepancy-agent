#!/bin/bash

# ===================================================================================
#
#  DOCKER BUILD SCRIPT for CDP AI TOOLS
#
#  Description:
#  This script builds the Docker image for the application. It tags the image
#  with a specified name and version.
#
#  Usage:
#  ./build_docker.sh [version]
#  Example: ./build_docker.sh 1.0.0
#
# ===================================================================================

# --- Configuration ---
IMAGE_NAME="cdp-ai-tools"
# Use the first argument as the tag/version, or default to "latest"
IMAGE_TAG=${1:-"latest"}

# --- Colors for Logging ---
COLOR_BLUE='\033[0;34m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_RESET='\033[0m'

# --- Logging Functions ---
log_info() {
    echo -e "${COLOR_BLUE}[INFO] $1${COLOR_RESET}"
}
log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS] $1${COLOR_RESET}"
}
log_error() {
    echo -e "${COLOR_RED}[ERROR] $1${COLOR_RESET}"
    exit 1
}

# --- Main Script Logic ---
main() {
    echo -e "${COLOR_BLUE}===========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE} CDP AI TOOLS - DOCKER IMAGE BUILDER         ${COLOR_RESET}"
    echo -e "${COLOR_BLUE}===========================================${COLOR_RESET}"

    # Step 1: Check for Docker
    log_info "Step 1: Checking if Docker is running..."
    if ! docker info > /dev/null 2>&1; then
        log_error "Docker does not seem to be running. Please start Docker and try again."
    fi
    log_success "Docker is running."
    echo

    # Step 2: Build the Docker image
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
    log_info "Step 2: Starting Docker build for image: ${FULL_IMAGE_NAME}"
    echo

    # The 'DOCKER_BUILDKIT=1' enables a more efficient and modern builder.
    DOCKER_BUILDKIT=1 docker build -t ${FULL_IMAGE_NAME} .

    if [ $? -ne 0 ]; then
        log_error "Docker build failed. Please check the output above for errors."
    fi
    echo

    # --- Final Instructions ---
    log_success "Docker image '${FULL_IMAGE_NAME}' built successfully!"
    echo -e "You can now run the container using the '${COLOR_GREEN}./run_docker.sh${COLOR_RESET}' script."
    echo -e "To push this image to a registry, use: ${COLOR_GREEN}docker push ${FULL_IMAGE_NAME}${COLOR_RESET}"
    echo -e "${COLOR_BLUE}===========================================${COLOR_RESET}"
}

# --- Execute Main Function ---
main