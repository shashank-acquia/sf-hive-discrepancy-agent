#!/usr/bin/env python3
"""
Standalone Slack Search Bot - No external API server needed
Integrates directly with Jira, Confluence, and Slack search
"""

import os
import logging
import re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# Import our agents and tools directly
from agents.slack_search_agent import SlackSearchAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Slack app
app = App(token=os.getenv("SLACK_BOT_TOKEN"))

# Initialize the search agent
try:
    search_agent = SlackSearchAgent()
    logger.info("✅ Search agent initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize search agent: {e}")
    search_agent = None

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
        
        # Use the search agent to find relevant content
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

@app.event("app_mention")
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

@app.message()
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

@app.command("/search-help")
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
    """Start the Slack bot"""
    try:
        # Use Socket Mode for development (requires SLACK_APP_TOKEN)
        socket_token = os.getenv("SLACK_APP_TOKEN")
        if socket_token:
            handler = SocketModeHandler(app, socket_token)
            logger.info("🚀 Starting Slack bot in Socket Mode...")
            handler.start()
        else:
            # Use HTTP mode (requires proper webhook setup)
            logger.info("🚀 Starting Slack bot in HTTP Mode...")
            app.start(port=int(os.getenv("PORT", 3000)))
            
    except Exception as e:
        logger.error(f"❌ Error starting bot: {e}")
        raise

if __name__ == "__main__":
    print("🤖 Starting Standalone Slack Search Bot...")
    print("=" * 50)
    print("The bot will:")
    print("- ✅ Respond to @mentions")
    print("- ✅ Process direct messages")
    print("- ✅ React to messages with trigger keywords")
    print("- ✅ Handle /search-help slash commands")
    print("- ✅ Search Jira, Confluence, and Slack directly")
    print("- ✅ Provide AI-powered responses")
    print()
    print("Required environment variables:")
    print("- SLACK_BOT_TOKEN")
    print("- SLACK_APP_TOKEN (for Socket Mode)")
    print("- OPENAI_API_KEY")
    print("- JIRA_SERVER, JIRA_USERNAME, JIRA_API_TOKEN")
    print("- CONFLUENCE_SERVER, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN")
    print("- SLACK_SEARCH_CHANNELS (comma-separated)")
    print()
    
    start_bot()
