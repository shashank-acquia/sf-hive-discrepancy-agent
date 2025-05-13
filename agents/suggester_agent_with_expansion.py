from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import codecs
from tools.discrepancy_suggester_tool import discrepancy_suggester
from langchain_community.chat_models import ChatOpenAI
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def generate_discrepancy_suggestions_with_scripts(table_name: str, discrepancies: str, expanded_scripts: dict = None) -> str:
    """
    Enhanced version of generate_discrepancy_suggestions that includes expanded SQL scripts.
    
    Args:
        table_name: The table name for discrepancies
        discrepancies: JSON string of discrepancies
        expanded_scripts: Dictionary of expanded SQL scripts {path: content}
        
    Returns:
        Suggestions for fixing the discrepancies
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # Custom discrepancy suggester function that includes expanded scripts
    def discrepancy_suggester_with_scripts(input_data: dict) -> str:
        # Add expanded scripts to the input data
        if expanded_scripts:
            if isinstance(input_data, str):
                try:
                    # Parse input data if it's a string
                    if input_data.strip().startswith("input_data ="):
                        input_data = input_data.strip().replace("input_data =", "", 1).strip()
                    input_data = json.loads(input_data) if isinstance(input_data, str) else input_data
                except Exception as e:
                    return f"❌ Failed to parse input string: {e}"
                    
            # Add expanded scripts to input data
            input_data["expanded_scripts"] = expanded_scripts
            
        # Call the original discrepancy suggester
        return discrepancy_suggester(input_data)
    
    tools = [
        Tool(
            name="SuggestFixesForDiscrepancy",
            func=discrepancy_suggester_with_scripts,
            description="Given (table_name, discrepancies, expanded_scripts), suggests changes to Snowflake SQL to match Hive",
            return_direct=True
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    # Prepare input data with or without expanded scripts
    input_data = {
        "table_name": table_name,
        "discrepancies": discrepancies
    }
    
    if expanded_scripts:
        # Only include the first 1000 characters of each script to avoid token limits
        condensed_scripts = {}
        for path, content in expanded_scripts.items():
            preview = content[:1000] + ("..." if len(content) > 1000 else "")
            condensed_scripts[path] = preview
            
        input_data["expanded_scripts"] = condensed_scripts
        logger.info(f"Added {len(condensed_scripts)} expanded scripts to the input data")
    
    result = executor.invoke({"input": json.dumps(input_data)})
    logger.info("Generated discrepancy suggestions")
    
    return result["output"]

def generate_discrepancy_suggestions(table_name: str, discrepancies: dict) -> str:
    """
    Original function for generating discrepancy suggestions.
    
    Args:
        table_name: The table name for discrepancies
        discrepancies: JSON string of discrepancies
        
    Returns:
        Suggestions for fixing the discrepancies
    """
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    tools = [
        Tool(
            name="SuggestFixesForDiscrepancy",
            func=discrepancy_suggester,
            description="Given (table_name, discrepancies), suggests changes to Snowflake SQL to match Hive",
            return_direct=True
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    input_data_str = json.dumps({
        "table_name": table_name,
        "discrepancies": discrepancies
    })
    
    result = executor.invoke({"input": input_data_str})
    
    return result["output"]

if __name__ == "__main__":
    # Example usage with mock expanded scripts
    mock_expanded_scripts = {
        "/path/to/snowflake/script.sql": """
        -- This is an expanded Snowflake script
        CREATE OR REPLACE TABLE customer (
            ID string,
            TenantID bigint,
            FIRSTNAME string,
            LASTNAME string,
            EMAIL string,
            AGE bigint,
            GENDER string,
            PHONEVALIDATIONRESULTCODES string,
            DELETEFLAG boolean,
            PREFERENCES string,
            ROWCREATED timestamp,
            ROWMODIFIED timestamp
        );
        
        -- Default time values that might be causing the discrepancy
        INSERT INTO customer (C_ESPOPTINDATEFA, C_EMAILOPTOUTDATEFA, C_SMSOPTOUTDATEFA)
        VALUES ('1900-01-01 00:00:01.000000000', '1900-01-01 00:00:01.000000000', '1900-01-01 00:00:01.000000000');
        """
    }
    
    result = generate_discrepancy_suggestions_with_scripts(
        table_name="CUSTOMER",
        discrepancies='[{"columnName":"C_ESPOPTINDATEFA","hive":["1900-01-01 00:01:00.000000000"],"snowflake":["1900-01-01 00:00:01.000000000"],"id":["0035Y00003saN8GQAU","0035Y00003sxU9GQAU","0035Y00003szoziQAA","0035Y00005xQPhUQAW","003Ux000002xKrEIAU","003Ux000003dhwEIAQ","003Ux000005GrIPIA0","003Ux000007ZjFxIAK","003Ux00000QQiJMIA1"]}]',
        expanded_scripts=mock_expanded_scripts
    )
    
    print("\n=== SUGGESTIONS ===")
    print(result)