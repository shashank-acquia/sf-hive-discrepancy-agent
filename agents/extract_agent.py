from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
from typing import List
from tools.data_discrepancy_tool import discrepancy_tables_tool
from langchain_community.chat_models import ChatOpenAI

load_dotenv()


def lookup(name: str) -> List[str]:
#    llm = OllamaLLM(model="llama3.1:8b")
   llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
   template = """given the table name {name_of_table} I want you to get  me a list of ID that have discrepancies.
                              Your answer should contain only a list of ID"""

   prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_table"]
    )
   tools_for_agent = [
       Tool(
        name="ExtractDiscrepancy",
        func=discrepancy_tables_tool,
        description="Pass the table name directly to extract discrepancy IDs, e.g. TRANSACTION",
        return_direct=True
    )
    ]

   react_prompt = hub.pull("hwchase17/react")
   agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)
   agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)

   result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_table=name)}
    )

   list_ids = result["output"]
   return list_ids


if __name__ == "__main__":
    print(lookup(name="TRANSACTION")) 