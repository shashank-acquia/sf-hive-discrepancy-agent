from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import Tool
from langchain_community.chat_models import ChatOpenAI
from tools.fetch_tables_tool import fetch_tables_tool
from typing import List

load_dotenv()

def extract_tablename() -> List[str]:
    # Use OpenAI or Ollama as needed
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    # llm = OllamaLLM(model="llama3.1:8b")  # Optional alternative

    tools_for_agent = [
        Tool(
            name="ExtractTables",
            func=fetch_tables_tool,
            description="Get the list of tables that have discrepancies",
            return_direct=True
        )
    ]

    # Use default react prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

    # Call the agent with an empty input (unless you need to provide specific inputs)
    result = agent_executor.invoke({"input": "Which tables have discrepancies?"})

    # result will be a dict with "output" key if return_direct=True
    return result["output"] if isinstance(result, dict) else result

if __name__ == "__main__":
    print(extract_tablename())
