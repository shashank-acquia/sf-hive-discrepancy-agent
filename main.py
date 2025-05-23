from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from langchain.prompts.prompt import PromptTemplate
from agents.extract_agent import lookup
from agents.discrepancy_agent import lookup as getDis
from agents.suggester_agent import generate_discrepancy_suggestions

load_dotenv()


import json

def dw_validation(name: str):
    ids = lookup(name=name)
    discrepancy_json = getDis(table_name=name, list_ids=ids)
    print("here")
    print(discrepancy_json)

    if discrepancy_json == 'No discrepancies found.':
        print("⚠️ No discrepancies found for the given table. No suggestions will be generated.")
        return [], []

    print("🧠 Generating discrepancy suggestions with LLM...\n")
    if isinstance(discrepancy_json, str):
        try:
            discrepancy_json = json.loads(discrepancy_json)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to decode discrepancy JSON: {e}")
            return [], []
    print("generate_discrepancy_suggestions")
    result = generate_discrepancy_suggestions(table_name=name, discrepancies=discrepancy_json)
    print("result is :")
    print(result)

    if not result:
        print("⚠️ No suggestions were generated. There might be no discrepancies or the LLM could not provide suggestions.")
        return [], discrepancy_json

    # 👇 Convert stringified list to real Python list if needed
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError as e:
            print("❌ Failed to decode LLM result as JSON:", e)
            return [], discrepancy_json

    print("✅ Suggestions generated:\n")
    print(result)

    return result if isinstance(result, list) else [], discrepancy_json

if __name__ == "__main__":
    load_dotenv()

    print("DW validation Enter")
    dw_validation(name='CUSTOMER')