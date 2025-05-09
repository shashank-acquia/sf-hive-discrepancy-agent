import requests
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from tools.data_matrix_tool import matrix_tables_tool
from typing import List
from langchain_community.chat_models import ChatOpenAI
import ast


session = requests.Session()
session.verify = False
requests.sessions.Session = lambda: session


load_dotenv()

def lookup(table_name: str, list_ids: List[str]) -> str:
    # llm = OllamaLLM(model="llama3.1:8b")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # This prompt is for LLM's context, not the tool itself
    prompt_template = PromptTemplate(
        template="""Given the input, return a list of discrepancy JSONs like:
        [{{"columnName": "firstName", "hive": "abcd", "snowflake": "def", "id": "123"}}]""",
        input_variables=[]
     )

    tools_for_agent = [
        Tool(
            name="ExtractDiscrepancy",
            func=matrix_tables_tool,
            description="Takes input as a dictionary with keys: 'table_name' and 'list_ids' (list of IDs).",
            return_direct=True
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # Pass the tuple as a string so the tool can parse it
   
    input_data = {
    "table_name": table_name,
    "list_ids": list_ids
  }

    result = agent_executor.invoke(input={"input": input_data})

    return result["output"]


if __name__ == "__main__":
    print(lookup(
        table_name="CUSTOMER",
        list_ids=["0035Y00003t5hjkQAA", "0035Y00003sacKmQAI"]
    ))
