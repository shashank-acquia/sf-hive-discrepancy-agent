import ast
from typing import List
import pandas as pd
from io import StringIO
from collections import defaultdict
import json
from tools.snowflake_tool import fetch_mismatch_table

def matrix_tables_tool(input_data: dict) -> List[dict]:
    print(f"🧪 Received input: {type(input_data)} -> {input_data}")
    discrepancies = []
    if isinstance(input_data, str):
        try:
            input_data = ast.literal_eval(input_data)
        except Exception as e:
            print(f"❌ Failed to parse input: {e}")
            return []

    # If input is a tuple like (table_name, list_ids)
    if isinstance(input_data, tuple):
        try:
            table_name, list_ids = input_data
            if isinstance(list_ids, str):
                list_ids = ast.literal_eval(list_ids)
        except Exception as e:
            print(f"❌ Failed to unpack tuple: {e}")
            return []

    # If input is a dict
    elif isinstance(input_data, dict):
        try:
            table_name = input_data["table_name"]
            list_ids = input_data["list_ids"]
            if isinstance(list_ids, str):
                list_ids = ast.literal_eval(list_ids)
        except Exception as e:
            print(f"❌ Failed to extract from dict: {e}")
            return []
    else:
        print("❌ Unsupported input format.")
        return []
    
    list_ids = list_ids[:3]
    print(f"🧪 list_ids type: {type(list_ids)}")
    
    if not list_ids:
        print("⚠️ No IDs provided in list_ids")
        return "No discrepancies found."
    
    print(f"🧪 First item: {list_ids[0]}")

    for id_val in list_ids:
        try:
            comparison_data = fetch_mismatch_table(table_name, id_val)
            

            for row in comparison_data:
               if len(row) == 3:
                discrepancies.append({
                    "columnName": row[0],
                    "hive": row[1],
                    "snowflake": row[2],
                    "id": id_val
                })
                
        except Exception as e:
            return f"❌ Error fetching data for ID {id_val}: {e}"
    print(discrepancies)
    merged = defaultdict(lambda: {'hive': set(), 'snowflake': set(), 'id': set()})

    for item in discrepancies:
            key = item['columnName']
            merged[key]['hive'].add(item['hive'])
            merged[key]['snowflake'].add(item['snowflake'])
            merged[key]['id'].add(item['id'])

# Convert sets to sorted lists and structure output
    merged_list = [
    {
        'columnName': col,
        'hive': sorted(list(data['hive'])),
        'snowflake': sorted(list(data['snowflake'])),
        'id': sorted(list(data['id']))
    }
    for col, data in merged.items()
  ]
# Optional: pretty print
    print(json.dumps(merged_list, indent=2))
    return json.dumps(merged_list, indent=2) if merged_list else "No discrepancies found."
