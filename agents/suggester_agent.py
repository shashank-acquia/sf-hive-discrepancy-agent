from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import codecs
# from tools.rag_tool import rag_tool
from  tools.discrepancy_suggester_tool import discrepancy_suggester
from langchain_community.chat_models import ChatOpenAI
import json

load_dotenv()

def generate_discrepancy_suggestions(table_name: str, discrepancies: dict) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    print(discrepancies)
    tools = [
        Tool(
            name="SuggestFixesForDiscrepancy",
            func=discrepancy_suggester,
            description="Given (table_name, and array of discrepancies json), suggests changes to Snowflake SQL to match Hive",
            return_direct=True
        )
    ]

    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True,handle_parsing_errors=True )
    input_data_str = json.dumps({
        "table_name": table_name,
        "discrepancies": discrepancies
    })
    result = executor.invoke({"input": input_data_str})
    print("result in generate_discrepancy_suggestions")
    print(result)
    return result["output"]


if __name__ == "__main__":
    generate_discrepancy_suggestions(
        table_name="CUSTOMER",
        discrepancies='[{"columnName":"C_ESPOPTINDATEFA","hive":["1900-01-01 00:01:00.000000000"],"snowflake":["1900-01-01 00:00:01.000000000"],"id":["0035Y00003saN8GQAU","0035Y00003sxU9GQAU","0035Y00003szoziQAA","0035Y00005xQPhUQAW","003Ux000002xKrEIAU","003Ux000003dhwEIAQ","003Ux000005GrIPIA0","003Ux000007ZjFxIAK","003Ux00000QQiJMIA1"]},{"columnName":"C_EMAILOPTOUTDATEJC","hive":["1900-01-01 00:01:00.000000000"],"snowflake":["1900-01-01 00:00:01.000000000"],"id":["0035Y00003saN8GQAU","0035Y00003szoziQAA","003Ux000003dhwEIAQ","003Ux000005GrIPIA0","003Ux000007ZjFxIAK"]},{"columnName":"C_SMSOPTOUTDATEJC","hive":["1900-01-01 00:01:00.000000000"],"snowflake":["1900-01-01 00:00:01.000000000"],"id":["0035Y00003saN8GQAU","0035Y00003sxU9GQAU","0035Y00003szoziQAA","0035Y00003ty7iNQAQ","0035Y00005xQPhUQAW","003Ux000002xKrEIAU","003Ux000003dhwEIAQ","003Ux000005GrIPIA0","003Ux000007ZjFxIAK","003Ux00000QQiJMIA1"]},{"columnName":"C_SMSOPTOUTDATEFA","hive":["1900-01-01 00:01:00.000000000"],"snowflake":["1900-01-01 00:00:01.000000000"],"id":["0035Y00003saN8GQAU","0035Y00003sxU9GQAU","0035Y00003szoziQAA","0035Y00003ty7iNQAQ","0035Y00005xQPhUQAW","003Ux000002xKrEIAU","003Ux000003dhwEIAQ","003Ux000005GrIPIA0","003Ux000007ZjFxIAK","003Ux00000QQiJMIA1"]},{"columnName":"C_DSCLVREVENUE1YEARJC","hive":["717.2786"],"snowflake":["717.278634433"],"id":["003Ux000007ZjFxIAK"]},{"columnName":"C_DSCLVPROFIT1YEARJC","hive":["458.3408"],"snowflake":["458.340828832"],"id":["003Ux000007ZjFxIAK"]},{"columnName":"C_ESPOPTINDATEJC","hive":["1900-01-01 00:01:00.000000000"],"snowflake":["1900-01-01 00:00:01.000000000"],"id":["0035Y00003sxU9GQAU","0035Y00003ty7iNQAQ","0035Y00005xQPhUQAW","003Ux000002xKrEIAU","003Ux00000QQiJMIA1"]},{"columnName":"C_EMAILOPTOUTDATEFA","hive":["1900-01-01 00:01:00.000000000"],"snowflake":["1900-01-01 00:00:01.000000000"],"id":["0035Y00003ty7iNQAQ"]},{"columnName":"C_DSCLVREVENUE1YEARFA","hive":["18.6129","351.4206","55.3056"],"snowflake":["18.612891172","351.420568676","55.305560176"],"id":["0035Y00005xQPhUQAW","003Ux000002xKrEIAU","003Ux00000QQiJMIA1"]},{"columnName":"C_DSCLVPROFIT1YEARFA","hive":["-6.0049","264.4981","9.893"],"snowflake":["-6.004948973","264.498054757","9.892951795"],"id":["0035Y00005xQPhUQAW","003Ux000002xKrEIAU","003Ux00000QQiJMIA1"]}]'
    )