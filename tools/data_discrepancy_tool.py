
import pandas as pd
from io import StringIO
from langchain.tools import Tool
from tools.snowflake_tool import fetch_metrics_table
import re

def discrepancy_tables_tool(tableName: str) -> str:
    

    try:
        csv = fetch_metrics_table()
    except Exception as e:
        return f"❌ Error fetching tables: {e}"

    df1 = pd.read_csv(StringIO(csv))

    print(df1.columns);
    print("......");
   

    if 'TABLE_NAME' not in df1.columns or 'DATA_DISCREPANCY_PK_VALUES' not in df1.columns:
        return "❌ Required columns ('TABLE_NAME', 'DATA_DISCREPANCY_PK_VALUES') not found in metrics table."
    
  
    
    table_name = f"{tableName}"

    print("✅ Available table names in metrics:")
    print(df1["TABLE_NAME"].unique())
    print(f"🔍 Checking for table_name='{table_name}' (after cleanup: {table_name.strip().upper()})")

    matched_row = df1[df1["TABLE_NAME"].str.upper().str.strip() == table_name.strip().upper()]
    if matched_row.empty:
        return f"⚠️ Table {table_name} not found in metrics."

    # Get PK discrepancy values
    pk_values = matched_row["DATA_DISCREPANCY_PK_VALUES"].values[0]

    if pd.isna(pk_values) or not str(pk_values).strip():
        return f"✅ No discrepancy IDs found for '{table_name}'."
    
    return f"{pk_values}"


# LangChain Tool definition
