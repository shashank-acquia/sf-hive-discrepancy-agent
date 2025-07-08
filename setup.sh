#!/bin/bash
set -e

echo "====================================================="
echo "SF-Hive Discrepancy Agent - Environment Setup"
echo "====================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH."
    echo "Please install Conda first: https://docs.conda.io/projects/conda/en/latest/user-guide/install/"
    exit 1
fi

# Check if .env file exists, if not create a template
if [ ! -f .env ]; then
    echo "Creating .env template file..."
    cat > .env << EOL
SNOWFLAKE_USER=your-snowflake-username
SNOWFLAKE_ACCOUNT=your-snowflake-account
SNOWFLAKE_WAREHOUSE=your-snowflake-warehouse
SNOWFLAKE_DATABASE=your-snowflake-database
SNOWFLAKE_SCHEMA=your-snowflake-schema
HIVE_SCRIPT_DIR=/path/to/hive/scripts
SNOWFLAKE_SCRIPT_DIR=/path/to/snowflake/scripts
OPENAI_API_KEY=your-openai-api-key
EOL
    echo "Created .env template file. Please update it with your actual values."
fi

# Create conda environment
echo "Creating conda environment: snowflake_ai with Python 3.10..."
conda create -n snowflake_ai python=3.10 -y

# Activate the environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snowflake_ai

# Install dependencies
echo "Installing Python dependencies..."
pip install snowflake-connector-python pandas python-dotenv langchain openai
pip install -U langchain-ollama
pip install flask langchain_community docx2txt faiss-cpu

# Install IPython kernel
echo "Installing IPython kernel..."
conda install ipykernel -y
python -m ipykernel install --user --name=snowflake_ai

echo "====================================================="
echo "Setup complete! You can now run ./build_docker.sh to build the Docker image."
echo "====================================================="
echo "Don't forget to edit the .env file with your configuration before running the agent."