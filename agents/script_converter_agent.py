from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import codecs
# from tools.rag_tool import rag_tool
from  tools.script_converter_tool import script_converter
from langchain_community.chat_models import ChatOpenAI
import json

load_dotenv()

def script_converter_suggestions(script:str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    tools = [
        Tool(
            name="ScriptConverter",
            func=script_converter,
            description="Given hive script,convert to snowflake",
            return_direct=True
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True )
    input_data_str = json.dumps({
        "script": script
    })
    result = executor.invoke({"input": input_data_str})
    print("result in generate_discrepancy_suggestions")
    print(result)
    return result["output"]


if __name__ == "__main__":
    script_converter_suggestions(
        script="select * from customer"
    )