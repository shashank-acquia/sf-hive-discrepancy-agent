#!/bin/bash

# ===================================================================================
#
#  DOCKER RUN SCRIPT for CDP AI TOOLS (Improved Version)
#
#  Description:
#  This script runs the CDP AI Tools Docker container. It maps the necessary ports
#  and loads environment variables from the .env file.
#  Includes a more robust cleanup step.
#
# ===================================================================================

# --- Configuration ---
CONTAINER_NAME="cdp-ai-tools-app"
IMAGE_NAME="cdp-ai-tools"
IMAGE_TAG=${1:-"latest"}
HOST_PORT="8081"
CONTAINER_PORT="8081"
ENV_FILE=".env"

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
log_warn() {
    echo -e "${COLOR_YELLOW}[WARNING] $1${COLOR_RESET}"
}
log_error() {
    echo -e "${COLOR_RED}[ERROR] $1${COLOR_RESET}"
    exit 1
}

# --- Main Script Logic ---
main() {
    echo -e "${COLOR_BLUE}===========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE} CDP AI TOOLS - DOCKER CONTAINER RUNNER      ${COLOR_RESET}"
    echo -e "${COLOR_BLUE}===========================================${COLOR_RESET}"

    # Step 1: Check for .env file
    log_info "Step 1: Checking for '${ENV_FILE}'..."
    if [ ! -f "$ENV_FILE" ]; then
        log_error "Configuration file '${ENV_FILE}' not found. Please create it before running the container."
    fi
    log_success "'${ENV_FILE}' found."
    echo

    # Step 2: Stop and remove any existing container with the same name (Robust version)
    log_info "Step 2: Checking for and removing existing containers named '${CONTAINER_NAME}'..."
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        log_warn "Found existing container. Forcefully removing it..."
        docker rm -f ${CONTAINER_NAME}
    fi
    log_success "Cleanup complete. No conflicting containers found."
    echo

    # Step 3: Run the new container
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
    log_info "Step 3: Starting container '${CONTAINER_NAME}' from image '${FULL_IMAGE_NAME}'..."
    echo -e "  - Mapping host port ${COLOR_YELLOW}${HOST_PORT}${COLOR_RESET} to container port ${COLOR_YELLOW}${CONTAINER_PORT}${COLOR_RESET}"
    echo -e "  - Loading environment variables from ${COLOR_YELLOW}${ENV_FILE}${COLOR_RESET}"
    echo

    docker run -d \
      --name ${CONTAINER_NAME} \
      -p ${HOST_PORT}:${CONTAINER_PORT} \
      --env-file ${ENV_FILE} \
      ${FULL_IMAGE_NAME}

    if [ $? -ne 0 ]; then
        log_error "Failed to start the Docker container. Check `docker logs ${CONTAINER_NAME}` for details."
    fi

    # --- Final Instructions ---
    log_success "Container '${CONTAINER_NAME}' started successfully!"
    echo -e "Application should be accessible at: ${COLOR_GREEN}http://localhost:${HOST_PORT}${COLOR_RESET}"
    echo -e "To view live logs, run: ${COLOR_GREEN}docker logs -f ${CONTAINER_NAME}${COLOR_RESET}"
    echo -e "To stop the container, run: ${COLOR_GREEN}docker stop ${CONTAINER_NAME}${COLOR_RESET}"
    echo -e "${COLOR_BLUE}===========================================${COLOR_RESET}"
}

# --- Execute Main Function ---
main