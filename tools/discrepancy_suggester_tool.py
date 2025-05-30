import os
import json
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from typing import List, Dict
import ast
import re
from langchain_community.chat_models import ChatOpenAI
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.script_expansion_tool import ScriptExpansionTool
from tools.rag_tool import RAGTool
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

HIVE_SCRIPT_DIR = os.getenv("HIVE_SCRIPT_DIR")
SNOWFLAKE_SCRIPT_DIR = os.getenv("SNOWFLAKE_SCRIPT_DIR")
table_name = os.getenv("TARGET_TABLE", "CUSTOMER")
metadata_dir = os.getenv("METADATA_DIR", "resources/prod-gcp")
    
if not os.path.isdir(metadata_dir):
    logger.warning(f"Metadata directory not found: {metadata_dir}")
    logger.warning("Script expansion may not work correctly without metadata.")

# rag_tool = RAGTool()

llm = ChatOpenAI(temperature=0.7, model_name="gpt-4.1",cache=False)

def extract_suffix(file_name: str, prefix: str) -> str:
    if file_name.startswith(prefix):
        base = file_name[len(prefix):]
        return os.path.splitext(base)[0]
    return None

def match_file_pairs(hive_files: dict, snowflake_files: dict) -> List[tuple]:
    matched_pairs = []
    
    hive_suffix_map = {
        extract_suffix(fname, "nw_"): fname
        for fname in hive_files if extract_suffix(fname, "nw_")
    }

    snowflake_suffix_map = {
        extract_suffix(fname, "sf_dw_"): fname
        for fname in snowflake_files if extract_suffix(fname, "sf_dw_")
    }

    for suffix, hive_file in hive_suffix_map.items():
        if suffix in snowflake_suffix_map:
            matched_pairs.append((hive_file, snowflake_suffix_map[suffix]))

    return matched_pairs

def read_files_by_extension(folder_path: str, extension: str, table_name: str) -> dict:
    # Determine the subdirectory to append based on the file extension
    if extension.lower() == ".sql":
        subdirectory = f"sf_dw_{table_name.lower()}"
    elif extension.lower() == ".hql":
        subdirectory = f"nw_{table_name.lower()}"
    else:
        subdirectory = ""

    # Append the subdirectory to the folder path
    full_path = os.path.join(folder_path, subdirectory)

    file_map = {}
    # Ensure the directory exists before attempting to list files
    if os.path.exists(full_path) and os.path.isdir(full_path):
        for file_name in os.listdir(full_path):
            if file_name.lower().endswith(extension):
                with open(os.path.join(full_path, file_name), "r") as f:
                    file_map[file_name] = f.read()

    return file_map


def extract_relevant_sql(sql_text: str, column: str, context_lines: int = 50) -> str:
    lines = sql_text.splitlines()
    pattern = re.compile(r'\b' + re.escape(column) + r'\b', re.IGNORECASE)
    for idx, line in enumerate(lines):
        if pattern.search(line):
            start = max(idx - context_lines, 0)
            end = min(idx + context_lines + 1, len(lines))
            return "\n".join(lines[start:end])
    return ""

def escape_curly_braces(text: str) -> str:
    # Escape all {placeholder} that are NOT already escaped (i.e., not already {{...}})
    return re.sub(r'(?<!{){([^{}]+)}(?!})', r'{{\1}}', text)

def ask_llm_to_fix(column: str, hive_snippet: str, snowflake_snippet: str, hive_val: str, sf_val: str, hive_file: str, sf_file: str) -> str:


    prompt = f"""
    You are an expert data engineer skilled in both Hive and Snowflake SQL. Your task is to diagnose data mismatches and generate high-quality, production-ready SQL fixes.
     Instructions:
        - Always generate a clean, multi-line fix in valid Snowflake SQL syntax.
        - Do NOT skip explanation or return a one-liner.
        - Color you explanation or suggestion in green color
        - Return a fix that is more than 100 characters.
        - If unsure, reply with: "I do not know."
        - Follow the output format strictly.
    There is a discrepancy in the column `{column}`: 
    Hive output = `{hive_val}`, Snowflake output = `{sf_val}`.

    Here is the relevant Hive SQL snippet from `{hive_file}`:
    ```
    {hive_snippet}
    ```

    Here is the relevant Snowflake SQL snippet from `{sf_file}`:
    ```
    {snowflake_snippet}
    ```
    """

    prompt_template = PromptTemplate(
    input_variables=[
     column, hive_snippet, snowflake_snippet,
    hive_val, sf_val, hive_file, sf_file
   ],
    template=escape_curly_braces(prompt),
   )
    chain = prompt_template | llm
    logger.info("🚀 Invoking LLM...")
    return chain.invoke({"prompt": prompt})

def clean_llm_output(output) -> str:
    # output is an AIMessage, so extract its content
    if hasattr(output, "content"):
        output = output.content

    if "```" in output:
        return output.split("```")[1].strip()
    return output.strip()


def find_column_in_sql(column: str, hive_sql: str, sf_sql: str) -> bool:
    pattern = re.compile(r'\b' + re.escape(column) + r'\b', re.IGNORECASE)
    return pattern.search(hive_sql) is not None or pattern.search(sf_sql) is not None

def enhance_suggestion_with_rag(column: str, hive_snippet: str, snowflake_snippet: str, 
                               hive_val: str, sf_val: str, basic_suggestion: str) -> str:
    if not rag_tool or not rag_tool.chain:
        logger.warning("RAG tool not available, returning basic suggestion")
        return basic_suggestion
    
    try:
        logger.info(f"Enhancing suggestion for column {column} with RAG")
        enhanced_suggestion = rag_tool.enhance_sql_suggestion(
            column=column,
            hive_sql=hive_snippet,
            snowflake_sql=snowflake_snippet,
            hive_val=hive_val,
            sf_val=sf_val,
            suggestion=basic_suggestion
        )
        
        if enhanced_suggestion:
            return enhanced_suggestion
        return basic_suggestion
    except Exception as e:
        logger.error(f"Error enhancing suggestion with RAG: {e}")
        return basic_suggestion

def discrepancy_suggester(input_data: dict) -> str:
    logger.info("⚙️ Calling discrepancy_suggester")
    logger.info(f"🧪 Received input: {type(input_data)} -> {input_data}")

    if isinstance(input_data, str):
        try:
            if input_data.strip().startswith("input_data ="):
                input_data = input_data.strip().replace("input_data =", "", 1).strip()
            input_data = ast.literal_eval(input_data)
        except Exception as e:
            return f"❌ Failed to parse input string: {e}"

    try:
        table_name = input_data["table_name"]
        discrepancy_json = input_data["discrepancies"]

        if isinstance(discrepancy_json, str):
            discrepancies = json.loads(discrepancy_json)
        else:
            discrepancies = discrepancy_json
    except Exception as e:
        return f"❌ Failed to extract input: {e}"

    hive_files = read_files_by_extension(HIVE_SCRIPT_DIR, ".hql",table_name)
    snowflake_files = read_files_by_extension(SNOWFLAKE_SCRIPT_DIR, ".sql",table_name)
    matched_pairs = match_file_pairs(hive_files, snowflake_files)

    logger.info(f"📄 Matched file pairs: {matched_pairs}")
    metadata_dir = os.getenv("METADATA_DIR")
    expander = ScriptExpansionTool(metadata_dir)
    results = {}
    expanded_sql_map = {}  
    
    for entry in discrepancies:
        column = entry["columnName"]
        hive_val = entry["hive"][0] if entry["hive"] else ""
        snowflake_val = entry["snowflake"][0] if entry["snowflake"] else ""
        column_found = False

        for hive_file, sf_file in matched_pairs:
            hive_sql = hive_files[hive_file]
            sf_sql = snowflake_files[sf_file]
            expanded_hive_sql = expander.expand_script(hive_sql)
            expanded_sf_sql = expander.expand_script(sf_sql)
            expanded_sql_map[hive_file] = expanded_hive_sql
            expanded_sql_map[sf_file] = expanded_sf_sql

            if find_column_in_sql(column, expanded_hive_sql, expanded_sf_sql):
                column_found = True

                hive_snippet = extract_relevant_sql(expanded_hive_sql, column)
                snowflake_snippet = extract_relevant_sql(expanded_sf_sql, column)

                if not hive_snippet and not snowflake_snippet:
                    logger.warning(f"⚠️ No relevant SQL snippet found for column {column}")
                    continue

                # Get basic suggestion
                basic_suggestion = clean_llm_output(ask_llm_to_fix(
                    column, hive_snippet, snowflake_snippet, hive_val, snowflake_val, hive_file, sf_file
                ))
                
                # Enhance with RAG if available
                # enhanced_suggestion = enhance_suggestion_with_rag(
                #     column, hive_snippet, snowflake_snippet, hive_val, snowflake_val, basic_suggestion
                # )

                logger.info(f"✅ Generated suggestion for column: {column}, files: ({hive_file}, {sf_file})")
                if column not in results:
                  results[column] = []
                results[column].append({
                    "column": column,
                    "hive_val": hive_val,
                    "snowflake_val": snowflake_val,
                    "hive_file": hive_file,
                    "snowflake_file": sf_file,
                    "suggestion": basic_suggestion
                })

        if not column_found:
            logger.warning(f"⚠️ Column '{column}' not found in any SQL files.")

    if not results:
        logger.warning("⚠️ No suggestions were generated.")

    return json.dumps({
    "results": results,
    "expanded_sql_map": expanded_sql_map
}, indent=2)