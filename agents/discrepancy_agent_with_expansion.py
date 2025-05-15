import requests
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from tools.data_matrix_tool import matrix_tables_tool
from tools.script_expansion_tool import ScriptExpansionTool, expand_scripts_for_agent
from typing import List, Dict
from langchain_community.chat_models import ChatOpenAI
import ast
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable SSL verification for requests
session = requests.Session()
session.verify = False
requests.sessions.Session = lambda: session

# Load environment variables
load_dotenv()

def get_related_sql_scripts(table_name: str) -> List[str]:
    """
    Find SQL script files related to the given table name.
    
    Args:
        table_name: The table name to search for
        
    Returns:
        List of paths to related SQL scripts
    """
    # Get directories from environment variables or use defaults
    hive_script_dir = os.getenv("HIVE_SCRIPT_DIR")
    snowflake_script_dir = os.getenv("SNOWFLAKE_SCRIPT_DIR")
    
    related_scripts = []
    
    # Helper function to search for scripts containing the table name
    def find_scripts_with_table(directory, extension):
        if not directory or not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            return []
            
        scripts = []
        # Make table name case insensitive
        table_pattern = table_name.lower()
        
        for filename in os.listdir(directory):
            if filename.lower().endswith(extension):
                file_path = os.path.join(directory, filename)
                # Quick check if the table name might be in the file
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    if table_pattern in content:
                        scripts.append(file_path)
        
        return scripts
    
    # Search for Hive scripts (.hql)
    if hive_script_dir:
        hive_scripts = find_scripts_with_table(hive_script_dir, ".hql")
        related_scripts.extend(hive_scripts)
    
    # Search for Snowflake scripts (.sql)
    if snowflake_script_dir:
        sf_scripts = find_scripts_with_table(snowflake_script_dir, ".sql")
        related_scripts.extend(sf_scripts)
    
    return related_scripts

def lookup_with_expanded_scripts(table_name: str, list_ids: List[str]) -> Dict:
    """
    Enhanced version of lookup that includes expanded SQL scripts.
    
    Args:
        table_name: The table name to search for discrepancies
        list_ids: List of record IDs to look for discrepancies
        
    Returns:
        Dictionary containing discrepancies and expanded SQL scripts
    """
    # First, find related SQL scripts
    related_scripts = get_related_sql_scripts(table_name)
    logger.info(f"Found {len(related_scripts)} related SQL scripts for table {table_name}")
    
    # Expand the scripts
    expanded_scripts = {}
    if related_scripts:
        # Get metadata directory from environment variable or use default
        metadata_dir = os.getenv("METADATA_DIR")
        expander = ScriptExpansionTool(metadata_dir)
        
        try:
            expanded_scripts = expand_scripts_for_agent(related_scripts, metadata_dir)
            logger.info(f"Successfully expanded {len(expanded_scripts)} SQL scripts")
        except Exception as e:
            logger.error(f"Error expanding SQL scripts: {e}")
    
    # Get discrepancies using the original lookup function
    discrepancies = lookup(table_name, list_ids)
    
    # Return both discrepancies and expanded scripts
    return {
        "discrepancies": discrepancies,
        "expanded_scripts": expanded_scripts
    }

def lookup(table_name: str, list_ids: List[str]) -> str:
    """
    Original lookup function for finding discrepancies.
    
    Args:
        table_name: The table name to search for discrepancies
        list_ids: List of record IDs to look for discrepancies
        
    Returns:
        Discrepancies as a string
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Prompt template for context
    prompt_template = PromptTemplate(
        template="""Given the input, return a list of discrepancy JSONs like:
        [{"columnName": "firstName", "hive": "abcd", "snowflake": "def", "id": "123"}]""",
        input_variables=[]
    )

    # Tools for the agent
    tools_for_agent = [
        Tool(
            name="ExtractDiscrepancy",
            func=matrix_tables_tool,
            description="Takes input as a dictionary with keys: 'table_name' and 'list_ids' (list of IDs).",
            return_direct=True
        )
    ]

    # Create the agent
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # Input data for the agent
    input_data = {
        "table_name": table_name,
        "list_ids": list_ids
    }

    # Execute the agent
    result = agent_executor.invoke(input={"input": input_data})

    return result["output"]

if __name__ == "__main__":
    # Example usage
    result = lookup_with_expanded_scripts(
        table_name="CUSTOMER",
        list_ids=["0035Y00003t5hjkQAA", "0035Y00003sacKmQAI"]
    )
    
    # Print summary of the results
    print("\n=== DISCREPANCY RESULTS ===")
    print(f"Found discrepancies: {result['discrepancies']}")
    print("\n=== EXPANDED SCRIPTS ===")
    for script_path, content in result['expanded_scripts'].items():
        print(f"Script: {os.path.basename(script_path)}")
        # Print first few lines as preview
        content_preview = "\n".join(content.split("\n")[:5]) + "\n..."
        print(content_preview)
        print("-" * 50)