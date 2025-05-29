import os
import pandas as pd
from snowflake.connector import connect
from dotenv import load_dotenv
from langchain.tools import tool
import threading
import time
import logging

load_dotenv()

class DWValidationConfig:
    def __init__(self):
        self.mode = os.getenv("MODE", "TESTING")
        if self.mode == "TESTING":
            self.snowflake_dw_schema = "SANDBOX"
            self.snowflake_hive_schema = "SANDBOX"
        else:
            self.snowflake_dw_schema = "DW"
            self.snowflake_hive_schema = "DW_HIVE_INC"


    def getMetricsQuery(self):
        return f"""
            SELECT *
            FROM {self.snowflake_dw_schema}.dw_data_metrics
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
    
    def getMismatchQuery(self, table_name: str, id_val: str):
        table_name_suffix = "_RK" if self.mode == "TESTING" else ""
        return f"""
            CALL sandbox.COMPARE_MISMATCH_IGNORE_EXCLUDED_COL1(
                '{self.snowflake_hive_schema}',
                '{self.snowflake_dw_schema}',
                '{table_name}{table_name_suffix}',
                'DELTA_STAGE_{table_name}{table_name_suffix}',
                'ID',
                '{id_val}'
            )
        """

dw_validation_config = DWValidationConfig()

def get_snowflake_connection():
    params = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "authenticator": "externalbrowser"
    }
    return connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        authenticator="externalbrowser"
    )

class ConnectionManager:
    def __init__(self):
        self.connection = None
        self.max_retries = 3
        self.validation_interval = 60  # 5 minutes in seconds
        self._stop_validation = False
        self._validation_thread = None
        self._start_validation_thread()
        

    def _connection_validator(self):
        """Background thread that periodically validates the connection."""
        logging.info("Connection validation thread started")
        while not self._stop_validation and self.max_retries > 0:
            try:
                self.validate_connection()
                logging.debug("Connection validated successfully")
                self.max_retries = 3  # Reset retries on success
            except Exception as e:
                self.max_retries -= 1
                logging.error(f"Connection validation failed: {str(e)}")
            time.sleep(self.validation_interval)
        logging.info("Connection validation thread stopped")

    def _start_validation_thread(self):
        """Starts the connection validation thread."""
        if self._validation_thread is None or not self._validation_thread.is_alive():
            self._stop_validation = False
            self._validation_thread = threading.Thread(
                target=self._connection_validator,
                daemon=True
            )
            self._validation_thread.start()

    def stop_validation_thread(self):
        """Stops the connection validation thread."""
        self._stop_validation = True
        if self._validation_thread and self._validation_thread.is_alive():
            self._validation_thread.join(timeout=1.0)

    def connect(self):
        if self.connection:
            return self.connection
        self.connection = get_snowflake_connection()
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def validate_connection(self):
        """
        Invalidates and closes the current Snowflake connection.
        This forces a new connection to be created on the next query.
        """
        if self.connection:
            try:
                conn = self.connection
                conn.execute_string("SELECT 1")  # Test the connection
            except:
                self.connection = None  # Invalidate the connection
                self.connect()
                conn.execute_string("SELECT 1")
                pass  # Ignore errors during close

connection_manager = ConnectionManager()

def fetch_table(table_name: str) -> str:
    """
    Fetches all rows from a given table in Snowflake and returns it as a CSV string.
    """
    conn = connection_manager.connect()
    df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 10", conn)
    return df.to_csv(index=False)


def fetch_metrics_table() -> str:
    """
    Fetches rows from dw.dw_data_metrics in Snowflake and returns them as a CSV string.
    Only returns rows where record counts are > 0 and there are discrepancies in PK arrays.
    """
    conn = connection_manager.connect()

    query = dw_validation_config.getMetricsQuery()
    print(f"🔍 fetching query for metrics table: {query}")
    df = pd.read_sql(query, conn)
    return df.to_csv(index=False)



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
    conn = connection_manager.connect()
    cursor = conn.cursor()
    try:

        # Execute the CALL statement
        cursor.execute(
            dw_validation_config.getMismatchQuery(table_name, id_val)
        )

        # Get result set
        result = cursor.fetchall()
        if not result or not result[0]:
            return []

        raw_str = result[0][0]  # extract string from first row/column
        eva_out= eval(raw_str) 
        return eva_out

    finally:
        cursor.close()

