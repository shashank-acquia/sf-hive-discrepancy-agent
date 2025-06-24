from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from tools.fetch_tables_tool import fetch_tables_tool
from typing import List, Tuple, Union
import pandas as pd

# Load environment variables from a .env file
load_dotenv()

def extract_tablename() -> Union[Tuple[pd.DataFrame, str], None]:
    """
    Extracts table names with discrepancies using a language model agent.

    Returns:
        A tuple containing the DataFrame and a string of table names with discrepancies,
        or None if an error occurs.
    """
    # Initialize the language model (LLM) with specified parameters
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Define the tool for the agent to use
    tools_for_agent = [
        Tool(
            name="ExtractTables",
            func=fetch_tables_tool,
            description="Get the list of tables that have discrepancies",
            return_direct=True
        )
    ]

    # Pull the default react prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    # Create the agent with the specified LLM, tools, and prompt
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    # Create an executor for the agent
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # Invoke the agent with a specific input
    try:
        result = agent_executor.invoke({"input": "Which tables have discrepancies?"})
        
        # Check if the result is a tuple and contains a DataFrame and string
        if isinstance(result['output'], tuple) and len(result['output']) == 2:
            df = result['output'][0]
            table_names_str = result['output'][1]
            if isinstance(df, pd.DataFrame) and isinstance(table_names_str, str):
                return df, table_names_str
        return None
    except Exception as e:
        print(f"Error invoking agent: {e}")
        return None

if __name__ == "__main__":
    # Fetch and print the extracted table names and DataFrame
    output = extract_tablename()
    if output:
        df, table_names_str = output
        print("DataFrame:")
        print(df)
        print("\nTable Names:")
        print(table_names_str)
    else:
        print("No discrepancies found or an error occurred.")