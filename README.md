# sf-hive-discrepancy-agent

## Overview
This project is a Flask-based application designed to process and resolve discrepancies in Snowflake Hive data. It uses OpenAI's API for generating suggestions and resolving discrepancies.

## Prerequisites

1. Install [Docker](https://www.docker.com/) on your system.
2. Ensure you have a valid OpenAI API key.
3. Create a `.env` file in the project root with the following content:
   ```env
   SNOWFLAKE_USER=your-snowflake-username
   SNOWFLAKE_ACCOUNT=your-snowflake-account
   SNOWFLAKE_WAREHOUSE=your-snowflake-warehouse
   SNOWFLAKE_DATABASE=your-snowflake-database
   SNOWFLAKE_SCHEMA=your-snowflake-schema
   HIVE_SCRIPT_DIR=/path/to/hive/scripts    # Path to directory containing Hive SQL scripts
   SNOWFLAKE_SCRIPT_DIR=/path/to/snowflake/scripts    # Path to directory containing Snowflake SQL scripts
   OPENAI_API_KEY=your-openai-api-key    # Your OpenAI API key for LLM integration
   ```

## Installation

If you are not using Docker, you can set up the environment manually:

1. Create a Python virtual environment:
   ```bash
   conda create -n snowflake_ai python=3.10 -y
   conda activate snowflake_ai
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install the IPython kernel:
   ```bash
   conda install ipykernel
   python -m ipykernel install --user --name=snowflake_ai
   ```

## Using Docker

### Build the Docker Image
Run the following command to build the Docker image:
```bash
./build.sh
```

### Run the Docker Container
Run the following command to start the container:
```bash
./run.sh
```

The application will be accessible at: [http://localhost:5000](http://localhost:5000)

## Project Structure

- `app.py`: Main Flask application.
- `templates/`: Contains HTML templates for the web interface.
- `tools/`: Utility scripts for processing discrepancies.
- `agents/`: Agents for generating suggestions and resolving discrepancies.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Configuration for containerizing the application.
- `build.sh`: Script to build the Docker image.
- `run.sh`: Script to run the Docker container.

## Troubleshooting

### Common Issues

1. **Missing or incomplete `.env` file**:
   Ensure the `.env` file exists in the project root and contains all required configuration variables (Snowflake credentials, script directories, and OpenAI API key).

2. **Port already in use**:
   If port 5000 is already in use, stop the conflicting process or modify the port in `run.sh`.

3. **Dependency issues**:
   If you encounter dependency issues, ensure you are using Python 3.10 and the required packages are installed.

## License
This project is licensed under the MIT License.