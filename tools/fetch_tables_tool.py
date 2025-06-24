def fetch_tables_tool(_input: str = "") -> tuple:
    import pandas as pd
    from io import StringIO
    from tools.snowflake_tool import fetch_metrics_table

    try:
        csv = fetch_metrics_table()
    except Exception as e:
        return f"❌ Error fetching tables: {e}"

    df1 = pd.read_csv(StringIO(csv))

    if 'TABLE_NAME' not in df1.columns or 'DATA_DISCREPANCY_PK_VALUES' not in df1.columns:
        return "❌ Required columns ('TABLE_NAME', 'DATA_DISCREPANCY_PK_VALUES') not found in metrics table."
    
    table_names = df1["TABLE_NAME"].unique().tolist()

    # Return as a clean comma-separated string (LLMs handle this format better)
    return df1,", ".join(table_names)