from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import json
from main import dw_validation,getColumnList,getConvertedScript
from agents.slack_search_agent import SlackSearchAgent
import logging
import os

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

search_agent = SlackSearchAgent()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    
    suggestions, discrepancy_json , expanded_scr_map = dw_validation(name=name)

    return jsonify({
        "suggestions": suggestions,
        "discrepancies": discrepancy_json,
        "expanded_scr_map":expanded_scr_map
    })

@app.route("/convert", methods=["POST"])
def convert():
    script = request.form["script"]
    
    suggestions = getConvertedScript(script=script)
    data = json.loads(suggestions)

    return jsonify({
        "converted_script": data['results'],
    })
@app.route("/metadata", methods=["GET"])
def metadata():
    col_list,df_html = getColumnList()

    return jsonify({
        "col_list": col_list,
        "df_html": df_html
    })


@app.route("/slack/search", methods=["POST"])
def slack_search():
    """API endpoint for searching across Jira, Slack, and Confluence"""
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400
        
        query = data['query']
        channel = data.get('channel', 'general')
        
        logger.info(f"API search request: {query}")
        
        # Perform search across all platforms
        search_results = search_agent._search_all_platforms(query)
        
        # Generate LLM response
        formatted_response = search_agent._generate_llm_response(query, search_results)
        
        return jsonify({
            'success': True,
            'query': query,
            'search_results': search_results,
            'formatted_response': formatted_response
        })
        
    except Exception as e:
        logger.error(f"Error in slack search API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



if __name__ == "__main__":
    print("🚀 Starting Unified SF-Hive Discrepancy & Slack Search API...")
    print("Available services:")
    print("- SF-Hive Discrepancy Agent (existing functionality)")
    print("- Slack Search Integration")
    print("  - POST /slack/search - Search across platforms")
    print("  - POST /slack/send - Send message to Slack")
    print("  - POST /slack/search-and-send - Search and send to Slack")
    print("  - POST /jira/search - Search Jira issues")
    print("  - POST /confluence/search - Search Confluence pages")
    
    app.run(
        host="0.0.0.0", 
        debug=os.getenv("FLASK_ENV") == "development",
        port=int(os.getenv("PORT", 8081))
    )
