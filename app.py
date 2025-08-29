from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import json
from main import dw_validation,getColumnList,getConvertedScript
from agents.slack_search_agent import SlackSearchAgent
import logging
import os
import threading
import re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

search_agent = SlackSearchAgent()

# Initialize Slack bot
slack_app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Bot mention keywords that trigger the search
TRIGGER_KEYWORDS = [
    'help', 'issue', 'problem', 'error', 'bug', 'question', 
    'how to', 'search', 'find', 'similar', 'related', 'authentication',
    'login', 'access', 'permission', 'connection', 'timeout'
]

def perform_search(query, channel=None):
    """Perform search using the integrated agent"""
    try:
        if not search_agent:
            return {
                'success': False,
                'error': 'Search agent not available'
            }
        
        # # Use semantic search if available for better results
        # if hasattr(search_agent, 'semantic_search_cross_platform') and search_agent.vector_service:
        #     results = search_agent.semantic_search_cross_platform(query)
        #     logger.info("Using semantic search for enhanced results")
        # else:
        #     results = search_agent._search_all_platforms(query)
        #     logger.info("Using traditional search")
        results = search_agent._search_all_platforms(query)
        # Generate AI response
        formatted_response = search_agent._generate_llm_response(query, results)
        
        return {
            'success': True,
            'formatted_response': formatted_response,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error performing search: {e}")
        return {
            'success': False,
            'error': f'Search failed: {str(e)}'
        }

@slack_app.event("app_mention")
def handle_app_mention(event, say, logger):
    """Handle when the bot is mentioned"""
    try:
        user = event['user']
        channel = event['channel']
        text = event['text']
        thread_ts = event.get('ts')  # Use the message timestamp as thread_ts for replies
        
        logger.info(f"🎯 Bot mentioned by {user} in {channel}: {text}")
        
        # Remove the bot mention from the text
        clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        
        if not clean_text:
            say("Hi! 👋 I can help you search for related issues in Jira, Slack, and Confluence. Just mention me with your question or issue description.", thread_ts=thread_ts)
            return
        
        # Show typing indicator
        say("🔍 Searching for relevant information...", thread_ts=thread_ts)
        
        # Perform the search
        result = perform_search(clean_text, channel)
        
        if result.get('success'):
            # Send the formatted response
            formatted_response = result.get('formatted_response', 'No response generated')
            say(formatted_response, thread_ts=thread_ts)
        else:
            say(f"❌ Sorry, I encountered an error while processing your request: {result.get('error', 'Unknown error')}", thread_ts=thread_ts)
            
    except Exception as e:
        logger.error(f"Error handling app mention: {e}")
        say("❌ Sorry, I encountered an unexpected error. Please try again later.", thread_ts=event.get('ts'))

@slack_app.message()
def handle_message(message, say, logger):
    """Handle direct messages or messages in channels where the bot is present"""
    try:
        # Only respond to direct messages or if the message contains trigger keywords
        channel_type = message.get('channel_type', '')
        text = message.get('text', '').lower()
        user = message['user']
        channel = message['channel']
        thread_ts = message.get('ts')
        
        # Skip messages from bots
        if message.get('bot_id'):
            return
        
        # Only respond to direct messages or messages with trigger keywords
        is_dm = channel_type == 'im'
        has_trigger = any(keyword in text for keyword in TRIGGER_KEYWORDS)
        
        if not (is_dm or has_trigger):
            return
        
        logger.info(f"💬 Processing message from {user} in {channel}: {text[:100]}...")
        
        # Show typing indicator
        say("🔍 Searching for relevant information...", thread_ts=thread_ts)
        
        # Perform the search
        result = perform_search(message['text'], channel)
        
        if result.get('success'):
            # Send the formatted response
            formatted_response = result.get('formatted_response', 'No response generated')
            say(formatted_response, thread_ts=thread_ts)
        else:
            say(f"❌ Sorry, I encountered an error while processing your request: {result.get('error', 'Unknown error')}", thread_ts=thread_ts)
            
    except Exception as e:
        logger.error(f"Error handling message: {e}")

@slack_app.command("/search-help")
def handle_search_command(ack, respond, command, logger):
    """Handle the /search-help slash command"""
    try:
        ack()
        
        query = command['text'].strip()
        channel = command['channel_id']
        user = command['user_id']
        
        if not query:
            respond("Please provide a search query. Example: `/search-help login issue`")
            return
        
        logger.info(f"🔍 Search command from {user}: {query}")
        
        # Perform the search
        result = perform_search(query, channel)
        
        if result.get('success'):
            formatted_response = result.get('formatted_response', 'No response generated')
            respond(f"Here are the search results:\n\n{formatted_response}")
        else:
            respond(f"❌ Sorry, I encountered an error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error handling search command: {e}")
        respond("❌ Sorry, I encountered an unexpected error. Please try again later.")

def start_bot():
    """Start the Slack bot in a separate thread"""
    try:
        # Use Socket Mode for development (requires SLACK_APP_TOKEN)
        socket_token = os.getenv("SLACK_APP_TOKEN")
        if socket_token:
            handler = SocketModeHandler(slack_app, socket_token)
            logger.info("🚀 Starting Slack bot in Socket Mode...")
            handler.start()
        else:
            # Use HTTP mode (requires proper webhook setup)
            logger.info("🚀 Starting Slack bot in HTTP Mode...")
            slack_app.start(port=int(os.getenv("SLACK_PORT", 3000)))
            
    except Exception as e:
        logger.error(f"❌ Error starting bot: {e}")
        raise

def start_bot_thread():
    """Start the Slack bot in a separate thread"""
    try:
        bot_thread = threading.Thread(target=start_bot, daemon=True)
        bot_thread.start()
        logger.info("✅ Slack bot thread started")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start bot thread: {e}")
        return False

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

        #      # Use semantic search if available for better results
        # if hasattr(search_agent, 'semantic_search_cross_platform') and search_agent.vector_service:
        #     search_results = search_agent.semantic_search_cross_platform(query)
        #     logger.info("Using semantic search for enhanced results")
        # else:
        #     search_results = search_agent._search_all_platforms(query)
        #     logger.info("Using traditional search")
        
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

@app.route("/bot/start", methods=["POST"])
def start_bot_endpoint():
    """API endpoint to start the Slack bot"""
    try:
        if not os.getenv("SLACK_BOT_TOKEN") or not os.getenv("SLACK_APP_TOKEN"):
            return jsonify({
                'success': False,
                'error': 'Missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN environment variables'
            }), 400
        
        bot_started = start_bot_thread()
        
        if bot_started:
            return jsonify({
                'success': True,
                'message': 'Slack bot started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start Slack bot'
            }), 500
            
    except Exception as e:
        logger.error(f"Error starting bot via API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/bot/status", methods=["GET"])
def bot_status():
    """API endpoint to check bot status"""
    try:
        has_tokens = bool(os.getenv("SLACK_BOT_TOKEN") and os.getenv("SLACK_APP_TOKEN"))
        
        return jsonify({
            'success': True,
            'bot_configured': has_tokens,
            'slack_bot_token': bool(os.getenv("SLACK_BOT_TOKEN")),
            'slack_app_token': bool(os.getenv("SLACK_APP_TOKEN")),
            'message': 'Bot is configured and ready' if has_tokens else 'Bot requires SLACK_BOT_TOKEN and SLACK_APP_TOKEN'
        })
        
    except Exception as e:
        logger.error(f"Error checking bot status: {e}")
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
    print("- Slack Bot Integration")
    print("  - Responds to @mentions")
    print("  - Processes direct messages")
    print("  - Handles /search-help slash commands")
    print("  - Searches Jira, Confluence, and Slack")
    
    # Start Slack bot if tokens are available
    if os.getenv("SLACK_BOT_TOKEN") and os.getenv("SLACK_APP_TOKEN"):
        print("\n🤖 Starting Slack Bot...")
        bot_started = start_bot_thread()
        if bot_started:
            print("✅ Slack bot started successfully in background thread")
        else:
            print("❌ Failed to start Slack bot")
    else:
        print("\n⚠️  Slack bot not started - missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN")
        print("   Set these environment variables to enable bot functionality")
    
    print(f"\n🌐 Flask API server starting on port {os.getenv('PORT', 8081)}...")
    
    app.run(
        host="0.0.0.0", 
        debug=os.getenv("FLASK_ENV") == "development",
        port=int(os.getenv("PORT", 8081))
    )
