from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from main import dw_validation
import os
import re

load_dotenv()

app = Flask(__name__)

HIVE_SCRIPT_DIR = os.getenv("HIVE_SCRIPT_DIR", "scripts/DW_MAPPER_DEFAULT")
SNOWFLAKE_SCRIPT_DIR = os.getenv("SNOWFLAKE_SCRIPT_DIR", "scripts/SF_DW_MAPPER_DEFAULT")

def get_entity_names():
    """Extract entity names from script files in both directories."""
    # Keep track of matched scripts
    entities = []
    
    # Maps of filenames by entity name (preserving case as in filename)
    hive_map = {}
    sf_map = {}
    
    # Extract entity name from Hive files (format: nw_NAME_*.hql)
    if os.path.exists(HIVE_SCRIPT_DIR):
        for filename in os.listdir(HIVE_SCRIPT_DIR):
            if filename.endswith(".hql") and filename.startswith("nw_"):
                # Extract NAME part from nw_NAME_POST_600.hql format
                match = re.match(r'nw_([^_]+)_.*\.hql', filename)
                if match:
                    entity_name = match.group(1)
                    hive_map[entity_name.upper()] = filename
    
    # Extract entity name from Snowflake files (format: sf_dw_NAME_*.sql)
    if os.path.exists(SNOWFLAKE_SCRIPT_DIR):
        for filename in os.listdir(SNOWFLAKE_SCRIPT_DIR):
            if filename.endswith(".sql") and filename.startswith("sf_dw_"):
                # Extract NAME part from sf_dw_NAME_POST_600.sql format
                match = re.match(r'sf_dw_([^_]+)_.*\.sql', filename)
                if match:
                    entity_name = match.group(1)
                    sf_map[entity_name.upper()] = filename

    # Find entities that have both Hive and Snowflake scripts
    for entity in hive_map.keys():
        if entity in sf_map:
            # Store entity_name in original case from filename
            hive_filename = hive_map[entity]
            orig_case_entity = re.match(r'nw_([^_]+)_.*\.hql', hive_filename).group(1)
            
            # Add as tuple: (display_name, value_to_submit)
            entities.append((orig_case_entity.upper(), orig_case_entity.upper()))
            
    return sorted(entities)

@app.route("/")
def index():
    entities = get_entity_names()
    return render_template("index.html", entities=entities)


@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    
    suggestions, discrepancy_json = dw_validation(name=name)

    return jsonify({
        "suggestions": suggestions,
        "discrepancies": discrepancy_json
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)