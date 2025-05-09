
# import os
# import json
# from dotenv import load_dotenv
# from langchain.prompts.prompt import PromptTemplate
# from langchain_ollama.llms import OllamaLLM
# from typing import List, Dict
# import ast
# import re
# from langchain_community.chat_models import ChatOpenAI

# load_dotenv()

# HIVE_SCRIPT_DIR = os.getenv("HIVE_SCRIPT_DIR")
# SNOWFLAKE_SCRIPT_DIR = os.getenv("SNOWFLAKE_SCRIPT_DIR")

# llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# def extract_suffix(file_name: str, prefix: str) -> str:
#     """
#     Removes the prefix and extension to extract a suffix used to match files.
#     Example: 'nw_customer_PRE_445.hql' with prefix 'nw_' → 'customer_PRE_445'
#     """
#     if file_name.startswith(prefix):
#         base = file_name[len(prefix):]
#         return os.path.splitext(base)[0]
#     return None

# def match_file_pairs(hive_files: dict, snowflake_files: dict) -> List[tuple]:
#     matched_pairs = []
    
#     hive_suffix_map = {
#         extract_suffix(fname, "nw_"): fname
#         for fname in hive_files if extract_suffix(fname, "nw_")
#     }

#     snowflake_suffix_map = {
#         extract_suffix(fname, "sf_dw_"): fname
#         for fname in snowflake_files if extract_suffix(fname, "sf_dw_")
#     }

#     # Match by common suffix
#     for suffix, hive_file in hive_suffix_map.items():
#         if suffix in snowflake_suffix_map:
#             matched_pairs.append((hive_file, snowflake_suffix_map[suffix]))

#     return matched_pairs


# def read_files_by_extension(folder_path: str, extension: str) -> dict:
#     file_map = {}
#     for file_name in os.listdir(folder_path):
#         if file_name.lower().endswith(extension):
#             with open(os.path.join(folder_path, file_name), "r") as f:
#                 file_map[file_name] = f.read()
#     return file_map

# def ask_llm_to_fix(column: str, hive_sql: str, snowflake_sql: str, hive_val: str, sf_val: str, hive_file: str, sf_file: str) -> str:
#     prompt = f"""
#         You are a Snowflake SQL expert. There is a discrepancy in the column `{column}`: Hive output `{hive_val}` and Snowflake  `{sf_val}`.

    
#         Here is the Snowflake SQL logic from file `{sf_file}`:
#         ```
#         {snowflake_sql}
#         ```

#         1>Suggest a modification to the Snowflake SQL to make it match Hive output.
#         2>Be specific to this file.
#         3>Return a clean SQL fix for the discrepancy 
#         4>Along with summary why this fix is required this is very important
#         5>If unsure, say "I do not know".
#         6>y
#         """
#     prompt_template = PromptTemplate(
#         input_variables=["prompt"],
#         template="You are a data engineering assistant.\nUser: {prompt}\nAssistant"
#     )
#     chain = prompt_template | llm
#     print("invoking llm")
#     return chain.invoke({"prompt": prompt})

# def clean_llm_output(output: str) -> str:
#     # Strip out markdown code blocks
#     if "```" in output:
#         return output.split("```")[1].strip()
#     return output.strip()

# def find_column_in_sql(column, hive_sql, sf_sql):
#     # Create a regex pattern to match the column name as a whole word
#     pattern = re.compile(r'\b' + re.escape(column) + r'\b', re.IGNORECASE)
    
#     # Check if the column name is present in either SQL script
#     return pattern.search(hive_sql) is not None or pattern.search(sf_sql) is not None


# def discrepancy_suggester(input_data: dict) -> str:
    
#     print("calling discrepancy_suggester")
#     print(f"🧪 Received input: {type(input_data)} -> {input_data}")
#     print(input_data)

#     # If input_data is a string, clean and parse it
#     if isinstance(input_data, str):
#         try:
#             if input_data.strip().startswith("input_data ="):
#                 input_data = input_data.strip().replace("input_data =", "", 1).strip()
#             input_data = ast.literal_eval(input_data)
#         except Exception as e:
#             return f"❌ Failed to parse input string: {e}"

#     try:
#         table_name = input_data["table_name"]
#         discrepancy_json = input_data["discrepancies"]

#         if isinstance(discrepancy_json, str):
#             discrepancies = json.loads(discrepancy_json)
#         else:
#             discrepancies = discrepancy_json
#     except Exception as e:
#         return f"❌ Failed to extract input: {e}"

#     # continue with your logic here
    
    
#     hive_files = read_files_by_extension(HIVE_SCRIPT_DIR, ".hql")
#     snowflake_files = read_files_by_extension(SNOWFLAKE_SCRIPT_DIR, ".sql")
#     matched_pairs = match_file_pairs(hive_files, snowflake_files)

#     print(f"📄 Matched file pairs: {matched_pairs}")
#     print(discrepancies)

#     results = []
#     for entry in discrepancies:
#         column = entry["columnName"]
#         hive_val = entry["hive"][0] if entry["hive"] else ""
#         snowflake_val = entry["snowflake"][0] if entry["snowflake"] else ""
#         column_found = False

#         for hive_file, sf_file in matched_pairs:
#             hive_sql = hive_files[hive_file]
#             sf_sql = snowflake_files[sf_file]
#             print(f"finding for column : {column}")
#             if find_column_in_sql(column, hive_sql, sf_sql):
#                 column_found = True
#                 suggestion = clean_llm_output(ask_llm_to_fix(
#                     column, hive_sql, sf_sql, hive_val, snowflake_val, hive_file, sf_file
#                 ))
#                 print(f"✅ Suggestion for column: {column}, file pair: ({hive_file}, {sf_file})")
#                 results.append({
#                     "column": column,
#                     "hive_file": hive_file,
#                     "snowflake_file": sf_file,
#                     "suggestion": suggestion
#                 })

#         if not column_found:
#             print(f"⚠️ Column '{column}' not found in any of the provided SQL files.")

#     if not results:
#         print("⚠️ No suggestions were generated.")

#     return json.dumps(results, indent=2)


import os
import json
from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from typing import List, Dict
import ast
import re
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

HIVE_SCRIPT_DIR = os.getenv("HIVE_SCRIPT_DIR")
SNOWFLAKE_SCRIPT_DIR = os.getenv("SNOWFLAKE_SCRIPT_DIR")

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

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

def read_files_by_extension(folder_path: str, extension: str) -> dict:
    file_map = {}
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(extension):
            with open(os.path.join(folder_path, file_name), "r") as f:
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

def ask_llm_to_fix(column: str, hive_snippet: str, snowflake_snippet: str, hive_val: str, sf_val: str, hive_file: str, sf_file: str) -> str:
    prompt = f"""
    You are a Snowflake SQL expert. There is a discrepancy in the column `{column}`: 
    Hive output = `{hive_val}`, Snowflake output = `{sf_val}`.

    Here is the relevant Hive SQL snippet from `{hive_file}`:
    ```
    {hive_snippet}
    ```

    Here is the relevant Snowflake SQL snippet from `{sf_file}`:
    ```
    {snowflake_snippet}
    ```

    1. Suggest a modification to the Snowflake SQL to make it match Hive output.
    2. Be specific to this file and column.
    3. Return a clean , valid snowflake script for fix, which i can run directly on snowflake
    4. Provide a summary why this fix is required and what is done in this fix.
    5. If unsure, say "I do not know".
    """

    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="You are a data engineering assistant.\nUser: {prompt}\nAssistant:"
    )

    chain = prompt_template | llm
    print("🚀 Invoking LLM...")
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

def discrepancy_suggester(input_data: dict) -> str:
    print("⚙️ Calling discrepancy_suggester")
    print(f"🧪 Received input: {type(input_data)} -> {input_data}")

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

    hive_files = read_files_by_extension(HIVE_SCRIPT_DIR, ".hql")
    snowflake_files = read_files_by_extension(SNOWFLAKE_SCRIPT_DIR, ".sql")
    matched_pairs = match_file_pairs(hive_files, snowflake_files)

    print(f"📄 Matched file pairs: {matched_pairs}")

    results = []
    for entry in discrepancies:
        column = entry["columnName"]
        hive_val = entry["hive"][0] if entry["hive"] else ""
        snowflake_val = entry["snowflake"][0] if entry["snowflake"] else ""
        column_found = False

        for hive_file, sf_file in matched_pairs:
            hive_sql = hive_files[hive_file]
            sf_sql = snowflake_files[sf_file]

            if find_column_in_sql(column, hive_sql, sf_sql):
                column_found = True

                hive_snippet = extract_relevant_sql(hive_sql, column)
                snowflake_snippet = extract_relevant_sql(sf_sql, column)

                if not hive_snippet and not snowflake_snippet:
                    print(f"⚠️ No relevant SQL snippet found for column {column}")
                    continue

                suggestion = clean_llm_output(ask_llm_to_fix(
                    column, hive_snippet, snowflake_snippet, hive_val, snowflake_val, hive_file, sf_file
                ))

                print(f"✅ Suggestion for column: {column}, files: ({hive_file}, {sf_file})")
                results.append({
                    "column": column,
                    "hive_val":hive_val,
                    "snowflake_val":snowflake_val,
                    "hive_file": hive_file,
                    "snowflake_file": sf_file,
                    "suggestion": suggestion
                })

        if not column_found:
            print(f"⚠️ Column '{column}' not found in any SQL files.")

    if not results:
        print("⚠️ No suggestions were generated.")

    return json.dumps(results, indent=2)
