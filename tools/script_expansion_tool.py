import re
import os
from typing import Dict, List, Tuple, Optional
import json

class ScriptExpansionTool:
    """
    Tool for expanding script templates with shorthand patterns like:
    ${columns:customer::%1$s string:%1$s boolean:%1$s bigint:%1$s double:%1$s decimal}
    
    This tool will replace these patterns with their fully expanded form by using
    metadata about tables and columns.
    """
    
    def __init__(self, metadata_dir: Optional[str] = None):
        """
        Initialize the script expansion tool.
        
        Args:
            metadata_dir: Directory containing metadata CSV files for tables and columns
                         (schema_table.csv, schema_column.csv). If None, will try to use
                         environment variables or default locations.
        """
        self.metadata_dir = metadata_dir
        if not self.metadata_dir:
            # Try to get from environment variables or use default location
            self.metadata_dir = os.getenv("METADATA_DIR", os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "resources/prod-gcp"
            ))
        
        # Cache for table and column metadata
        self._table_metadata = None
        self._column_metadata = None
        
    def _load_metadata(self):
        """Load table and column metadata from CSV files."""
        if self._table_metadata is not None and self._column_metadata is not None:
            return
            
        # Load table metadata
        table_csv_path = os.path.join(self.metadata_dir, "schema_table.csv")
        if os.path.exists(table_csv_path):
            import csv
            self._table_metadata = {}
            with open(table_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    table_name = row.get('name', '').lower()
                    table_id=row.get('table_id', '').lower()
                    if table_name:
                        self._table_metadata[table_name] = table_id
        
        # Load column metadata
                def get_data_type(type_id):
            # Define the mapping of type IDs to data types
                    type_map = {
                        '1': "LONG",
                        '0': "INTEGER",
                        '2': "STRING",
                        '3': "DOUBLE",
                        '4': "BOOLEAN",
                        '5': "DECIMAL"
                    }
                    return type_map.get(type_id, "Unknown Type")
        column_csv_path = os.path.join(self.metadata_dir, "schema_column.csv")
        if os.path.exists(column_csv_path):
            import csv
            self._column_metadata = {}
            with open(column_csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    table_id = row.get('tableId', '').lower()
                    column_name = row.get('name', '')
                    column_type = get_data_type(row.get('type', ''))
                    
                    if table_id and column_name:
                        if table_id not in self._column_metadata:
                            self._column_metadata[table_id] = []
                        
                        self._column_metadata[table_id].append({
                            'name': column_name,
                            'type': column_type,
                            'row': row
                        })
    
    def get_columns_for_table(self, table_name: str) -> List[Dict]:
            self._load_metadata()
            original_table_name = table_name.lower()
            columns = []

            # Attempt direct lookup
            table_id = self._table_metadata.get(original_table_name)
            if table_id and table_id in self._column_metadata:
                columns = self._column_metadata[table_id]
            else:
                # Try removing known prefixes
                fallback_prefixes = ['udm_sf_', 'udm_s_', 'delta_stage_', 'delta_udm_']
                for prefix in fallback_prefixes:
                    if original_table_name.startswith(prefix):
                        simplified_name = original_table_name[len(prefix):]
                        table_id = self._table_metadata.get(simplified_name)
                        if table_id and table_id in self._column_metadata:
                            columns = self._column_metadata[table_id]
                            break  # Found valid match, break out of loop

            if not columns:
                raise IOError(f"Schema definition for table '{table_name}' not found")

            return columns
            
    def expand_column_pattern(self, match: re.Match) -> str:
        try:
            full_match = match.group(0)         # e.g., ${columns:customer::c.%1$s AS %1$s}
            pattern_content = match.group(1)    # e.g., columns:customer::c.%1$s AS %1$s

            parts = pattern_content.split(":")
            if len(parts) < 3 or parts[0].lower() != "columns":
                return full_match

            table_name = parts[1].strip().lower()
            exclusions = []
            format_str_parts = parts[2:]  # everything after table name

            # Handle exclusion if it starts with ~
            if parts[2].startswith("~"):
                exclusion_part = parts[2][1:]
                exclusions = [e.strip().upper() for e in exclusion_part.split(",") if e.strip()]
                format_str_parts = parts[3:]

            # Join remaining parts into format string
            format_spec_str = ":".join(format_str_parts).rstrip("}")

            if not format_spec_str:
                default_format = "%1$s"  # fallback
            else:
                format_types = format_spec_str.split(":")
                default_format = format_types[0]
                format_map = {}

                for entry in format_types[1:]:
                    tf_parts = entry.strip().split(" ")
                    if len(tf_parts) == 2:
                        format_map[tf_parts[0].lower()] = tf_parts[1]

            # Get and filter columns
            columns = self.get_columns_for_table(table_name)
            if not columns:
                return full_match

            if exclusions:
                columns = [col for col in columns if col['name'].upper() not in exclusions]

            result = []
            for col in columns:
                col_name = col['name'].upper()
                col_type = (col.get('type') or 'string').lower()
                fmt = format_map.get(col_type, default_format)
                result.append(fmt.replace("%1$s", col_name))

            return ",".join(result)

        except Exception as e:
            print(f"Error expanding pattern {match.group(0)}: {e}")
            return match.group(0)
            
    def expand_script(self, script_content: str) -> str:
        """
        Expand all ${columns:...} patterns in a script.
        
        Args:
            script_content: The SQL script with template patterns
            
        Returns:
            The expanded SQL script with patterns replaced by actual column lists
        """
        # Pattern to match ${columns:table:format} or ${columns:table:~col1,col2:format}
        pattern = r'\$\{(columns:[^}]+)\}'
        
        # Replace all matches using the expand_column_pattern function
        expanded_script = re.sub(pattern, self.expand_column_pattern, script_content)
        
        return expanded_script
        
    def expand_script_file(self, script_path: str) -> str:
        """
        Read a script file and expand all patterns.
        
        Args:
            script_path: Path to the script file
            
        Returns:
            The expanded script content
        """
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script file not found: {script_path}")
            
        with open(script_path, 'r') as f:
            content = f.read()
            
        return self.expand_script(content)
    
    def expand_script_files_in_dir(self, directory: str, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Expand all scripts in a directory.
        
        Args:
            directory: Directory containing SQL scripts
            output_dir: Optional directory to write expanded scripts to
            
        Returns:
            Dictionary mapping script names to expanded content
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")
            
        results = {}
        for file_name in os.listdir(directory):
            if file_name.endswith('.sql') or file_name.endswith('.hql'):
                script_path = os.path.join(directory, file_name)
                try:
                    expanded_content = self.expand_script_file(script_path)
                    results[file_name] = expanded_content
                    
                    # Write expanded script to output directory if specified
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        output_path = os.path.join(output_dir, file_name)
                        with open(output_path, 'w') as f:
                            f.write(expanded_content)
                            
                except Exception as e:
                    print(f"Error expanding script {file_name}: {e}")
        
        return results
        
    def create_example_metadata(self, output_dir: str):
        """
        Create example metadata files for testing.
        
        Args:
            output_dir: Directory to write example metadata files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Example table metadata
        table_data = [
            "TABLE_NAME,TABLE_TYPE,REMARKS",
            "customer,TABLE,Customer information"
        ]
        
        # Example column metadata
        column_data = [
            "TABLE_NAME,COLUMN_NAME,DATA_TYPE,TYPE_NAME,COLUMN_SIZE,BUFFER_LENGTH,DECIMAL_DIGITS,NUM_PREC_RADIX,NULLABLE,REMARKS,COLUMN_DEF,SQL_DATA_TYPE,SQL_DATETIME_SUB,CHAR_OCTET_LENGTH,ORDINAL_POSITION,IS_NULLABLE,SCOPE_CATALOG,SCOPE_SCHEMA,SCOPE_TABLE,SOURCE_DATA_TYPE,IS_AUTOINCREMENT,IS_GENERATEDCOLUMN,JMX_ID",
            "customer,ID,12,string,255,0,0,10,1,,null,0,0,255,1,YES,null,null,null,0,NO,NO,",
            "customer,TenantID,-5,bigint,19,0,0,10,1,,null,0,0,0,2,YES,null,null,null,0,NO,NO,",
            "customer,FIRSTNAME,12,string,255,0,0,10,1,,null,0,0,255,3,YES,null,null,null,0,NO,NO,",
            "customer,LASTNAME,12,string,255,0,0,10,1,,null,0,0,255,4,YES,null,null,null,0,NO,NO,",
            "customer,EMAIL,12,string,255,0,0,10,1,,null,0,0,255,5,YES,null,null,null,0,NO,NO,",
            "customer,AGE,-5,bigint,19,0,0,10,1,,null,0,0,0,6,YES,null,null,null,0,NO,NO,",
            "customer,GENDER,12,string,10,0,0,10,1,,null,0,0,10,7,YES,null,null,null,0,NO,NO,",
            "customer,PHONEVALIDATIONRESULTCODES,12,string,255,0,0,10,1,,null,0,0,255,8,YES,null,null,null,0,NO,NO,",
            "customer,DELETEFLAG,16,boolean,1,0,0,10,1,,null,0,0,0,9,YES,null,null,null,0,NO,NO,",
            "customer,PREFERENCES,12,string,1000,0,0,10,1,,null,0,0,1000,10,YES,null,null,null,0,NO,NO,",
            "customer,ROWCREATED,93,timestamp,23,0,3,10,1,,null,0,0,0,11,YES,null,null,null,0,NO,NO,",
            "customer,ROWMODIFIED,93,timestamp,23,0,3,10,1,,null,0,0,0,12,YES,null,null,null,0,NO,NO,"
        ]
        
        # Write files
        with open(os.path.join(output_dir, "schema_table.csv"), "w") as f:
            f.write('\n'.join(table_data))
            
        with open(os.path.join(output_dir, "schema_column.csv"), "w") as f:
            f.write('\n'.join(column_data))
        
        print(f"Created example metadata files in {output_dir}")

# Example usage
def expand_scripts_for_agent(input_scripts: List[str], metadata_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Helper function to expand scripts for the AI agent.
    
    Args:
        input_scripts: List of paths to scripts that need expansion
        metadata_dir: Optional path to metadata directory
        
    Returns:
        Dictionary of script paths to expanded content
    """
    expander = ScriptExpansionTool(metadata_dir)
    results = {}
    
    for script_path in input_scripts:
        try:
            expanded_content = expander.expand_script_file(script_path)
            results[script_path] = expanded_content
        except Exception as e:
            print(f"Error expanding script {script_path}: {e}")
            results[script_path] = f"ERROR: {str(e)}"
    
    return results


if __name__ == "__main__":
    # Example: Create test metadata
    import tempfile
    test_dir = tempfile.mkdtemp()
    expander = ScriptExpansionTool()
    expander.create_example_metadata(test_dir)
    
    # Example script with patterns
    test_script = """
    CREATE OR REPLACE TABLE customer (
        ${columns:customer::%1$s string:%1$s boolean:%1$s bigint:%1$s double:%1$s decimal}
    );
    
    INSERT INTO customer (${columns:customer::%1$s})
    VALUES (...);
    
    SELECT ${columns:customer:~Email,EmailStatus,FirstName,LastName:%1$s}
    FROM customer;
    """
    
    # Set the metadata directory for this test
    expander.metadata_dir = test_dir
    
    # Expand the script
    expanded = expander.expand_script(test_script)
    print("Original script:")
    print(test_script)
    print("\nExpanded script:")
    print(expanded)