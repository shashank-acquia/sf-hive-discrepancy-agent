import os
import pandas as pd
from snowflake.connector import connect
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

def get_snowflake_connection():
    return connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        authenticator="externalbrowser"
    )


def fetch_table(table_name: str) -> str:
    """
    Fetches all rows from a given table in Snowflake and returns it as a CSV string.
    """
    conn = get_snowflake_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10", conn)
        return df.to_csv(index=False)
    finally:
        conn.close()


def fetch_metrics_table() -> str:
    """
    Fetches rows from dw.dw_data_metrics in Snowflake and returns them as a CSV string.
    Only returns rows where record counts are > 0 and there are discrepancies in PK arrays.
    """
    conn = get_snowflake_connection()
    try:
        query = """
            SELECT *
            FROM dw.dw_data_metrics
            WHERE 
                (total_record_count_hive > 0 OR total_record_count_sf > 0)
                AND (
                    ARRAY_SIZE(hive_only_pk_values) > 0 OR 
                    ARRAY_SIZE(sf_only_pk_values) > 0 OR 
                    ARRAY_SIZE(data_discrepancy_pk_values) > 0
                )
            ORDER BY 
                hive_only_pk_values, 
                sf_only_pk_values, 
                data_discrepancy_pk_values, 
                table_name
        """
        df = pd.read_sql(query, conn)
        return df.to_csv(index=False)
    finally:
        conn.close()



# if __name__ == "__main__":
#     table_name = input("Enter table name to fetch: ")
#     csv_output = fetch_table(table_name)
#     print("\n✅ Table loaded successfully. Preview:")
#     print(csv_output[:500])  # print first 500 characters of CSV


def fetch_mismatch_table(table_name: str, id_val: str) -> list:
    """
    Executes the Snowflake procedure and returns the result as CSV string.
    """
    print(f"🔍 fetching query for id and entity '{table_name}' and {id_val} )")
    conn = get_snowflake_connection()
    try:
        cursor = conn.cursor()
        try:
            # Execute SET statements separately
            cursor.execute(f"SET entity_name = '{table_name}'")
            cursor.execute(f"SET record_id = '{id_val}'")

            # Execute the CALL statement
            cursor.execute("""
                CALL sandbox.COMPARE_MISMATCH_IGNORE_EXCLUDED_COL1(
                    'DW_HIVE_INC',
                    'DW',
                    $entity_name,
                    'DELTA_STAGE_' || $entity_name,
                    'ID',
                    $record_id
                )
            """)

            # Get result set
            result = cursor.fetchall()
            if not result or not result[0]:
                return []

            raw_str = result[0][0]  # extract string from first row/column
            eva_out= eval(raw_str) 
            return eva_out;

        finally:
            cursor.close()

    finally:
        conn.close()
