import pandas as pd
from io import StringIO
from langchain.tools import Tool
from tools.snowflake_tool import fetch_table
import re

def compare_tables_tool(query: str) -> str:
    """
    Expects query like: "Compare CUSTOMER and ACCOUNT by 'id'"
    Parses table names and compares them.
    """
    match = re.search(r"Compare ['\"]?(\w+)['\"]?\s+and\s+['\"]?(\w+)['\"]?", query, re.IGNORECASE)
    if not match:
        return "❌ Couldn't parse table names. Use format: Compare 'TABLE1' and 'TABLE2'."

    table1, table2 = match.groups()

    try:
        csv1 = fetch_table(table1)
        csv2 = fetch_table(table2)
    except Exception as e:
        return f"❌ Error fetching tables: {e}"

    df1 = pd.read_csv(StringIO(csv1))
    df2 = pd.read_csv(StringIO(csv2))

    print(df1.columns);
    print("......");
    print(df2.columns);

    if 'ID' not in df1.columns or 'ID' not in df2.columns:
        return "❌ One or both tables do not contain an 'id' column."

    merged = pd.merge(df1, df2, on="ID", how="outer", indicator=True)
    missing1 = merged[merged['_merge'] == 'right_only']
    missing2 = merged[merged['_merge'] == 'left_only']

    result = []
    if not missing1.empty:
        result.append(f"🟡 IDs in {table2} but missing in {table1}:\n{missing1['ID'].tolist()}")
    if not missing2.empty:
        result.append(f"🟡 IDs in {table1} but missing in {table2}:\n{missing2['ID'].tolist()}")

    return "\n\n".join(result) if result else "✅ Both tables match on 'id'."
