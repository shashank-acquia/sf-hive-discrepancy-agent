# SF-Hive Discrepancy Agent

A tool for detecting and resolving discrepancies between Snowflake and Hive data sources.

## Overview

The SF-Hive Discrepancy Agent automatically monitors and reports differences between data stored in Snowflake and Hive environments, helping ensure data consistency across platforms.

## Prerequisites

- Docker
- Snowflake account credentials
- Python 3.8+
- OpenAI API Key

## Setup and Installation

### Quick Setup

1. Make the setup script executable and run it:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. Build the Docker image:
   ```bash
   ./build_docker.sh
   ```

### Manual Setup

If you prefer manual setup, you'll need to:
1. Configure your environment variables
2. Install required dependencies
3. Build the Docker image using `build_docker.sh`

## Usage

```bash
./run_docker.sh
```

## Configuration

Configure the agent by setting the following environment variables:

- `SNOWFLAKE_USER` - Snowflake username
- `SNOWFLAKE_ACCOUNT` - Your Snowflake account identifier
- `SNOWFLAKE_WAREHOUSE` - Snowflake warehouse to use
- `SNOWFLAKE_DATABASE` - Snowflake database name
- `SNOWFLAKE_SCHEMA` - Snowflake schema name
- `HIVE_SCRIPT_DIR` - Path to directory containing Hive SQL scripts
- `SNOWFLAKE_SCRIPT_DIR` - Path to directory containing Snowflake SQL scripts
- `OPENAI_API_KEY` - Your OpenAI API key for LLM integration
- `METADATA_DIR` - (Optional) Path to directory containing metadata CSV files for script expansion

## Script Expansion Feature

The Script Expansion feature allows the agent to process SQL scripts with templated patterns like:

```sql
${columns:customer::%1$s string:%1$s boolean:%1$s bigint:%1$s double:%1$s decimal}
```

These patterns are expanded to include the full list of column names and types based on metadata.

### How Script Expansion Works

1. The agent reads SQL scripts containing templated patterns
2. It uses metadata from CSV files (schema_table.csv and schema_column.csv) to expand these patterns
3. The expanded scripts are then fed to the AI model for analysis

### Using Script Expansion

To use script expansion in your workflow:

```python
from tools.script_expansion_tool import ScriptExpansionTool

# Initialize the expander with metadata directory
expander = ScriptExpansionTool("/path/to/metadata")

# Expand a specific script
expanded_script = expander.expand_script_file("/path/to/script.sql")

# Or use the integrated agent with script expansion
python examples/script_expansion_example.py
```

### Example Patterns

- `${columns:customer::%1$s}` - List all columns from customer table
- `${columns:customer:~Email,FirstName:%1$s}` - List all columns except Email and FirstName
- `${columns:customer::%1$s string:%1$s boolean:%1$s bigint:%1$s double:%1$s decimal}` - List columns with type-specific formatting

## Scripts

- `setup.sh` - Prepares the environment, installs dependencies, and validates configurations
- `build_docker.sh` - Builds the Docker image for the agent

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.