import os
import logging
import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

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

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8081")

# Bot mention keywords that trigger the search
TRIGGER_KEYWORDS = [
    'help', 'issue', 'problem', 'error', 'bug', 'question', 
    'how to', 'search', 'find', 'similar', 'related'
]

def call_search_api(query, channel):
    """Call the unified API to perform search"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/slack/search",
            json={"query": query, "channel": channel},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API call failed with status {response.status_code}: {response.text}")
            return {
                'success': False,
                'error': f'API call failed with status {response.status_code}'
            }
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling search API: {e}")
        return {
            'success': False,
            'error': f'Failed to connect to search API: {str(e)}'
        }

def send_message_via_api(channel, message, thread_ts=None):
    """Send message via the unified API"""
    try:
        payload = {
            "channel": channel,
            "message": message
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
            
        response = requests.post(
            f"{API_BASE_URL}/slack/send",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Send message API call failed: {response.text}")
            return {'success': False, 'error': 'Failed to send message'}
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message via API: {e}")
        return {'success': False, 'error': str(e)}

@app.event("app_mention")
def handle_app_mention(event, say, logger):
    """Handle when the bot is mentioned"""
    try:
        user = event['user']
        channel = event['channel']
        text = event['text']
        thread_ts = event.get('ts')  # Use the message timestamp as thread_ts for replies
        
        logger.info(f"Bot mentioned by {user} in {channel}: {text}")
        
        # Remove the bot mention from the text
        # The text usually starts with <@BOTID> so we need to clean it
        import re
        clean_text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        
        if not clean_text:
            say("Hi! I can help you search for related issues in Jira, Slack, and Confluence. Just mention me with your question or issue description.", thread_ts=thread_ts)
            return
        
        # Call the search API
        result = call_search_api(clean_text, channel)
        
        if result.get('success'):
            # Send the formatted response
            formatted_response = result.get('formatted_response', 'No response generated')
            say(formatted_response, thread_ts=thread_ts)
        else:
            say(f"Sorry, I encountered an error while processing your request: {result.get('error', 'Unknown error')}", thread_ts=thread_ts)
            
    except Exception as e:
        logger.error(f"Error handling app mention: {e}")
        say("Sorry, I encountered an unexpected error. Please try again later.", thread_ts=event.get('ts'))

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
        
        logger.info(f"Processing message from {user} in {channel}: {text[:100]}...")
        
        # Call the search API
        result = call_search_api(message['text'], channel)
        
        if result.get('success'):
            # Send the formatted response
            formatted_response = result.get('formatted_response', 'No response generated')
            say(formatted_response, thread_ts=thread_ts)
        else:
            say(f"Sorry, I encountered an error while processing your request: {result.get('error', 'Unknown error')}", thread_ts=thread_ts)
            
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
        
        logger.info(f"Search command from {user}: {query}")
        
        # Call the search-and-send API endpoint
        try:
            response = requests.post(
                f"{API_BASE_URL}/slack/search-and-send",
                json={"query": query, "channel": channel},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    respond("Search completed! I've posted the results in the channel.")
                else:
                    respond(f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}")
            else:
                respond(f"Sorry, the search service is currently unavailable (HTTP {response.status_code})")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling search-and-send API: {e}")
            respond("Sorry, I couldn't connect to the search service. Please try again later.")
            
    except Exception as e:
        logger.error(f"Error handling search command: {e}")
        respond("Sorry, I encountered an unexpected error. Please try again later.")

@app.event("message")
def handle_message_events(event, logger):
    """Handle message events (for thread replies, etc.)"""
    try:
        # Skip if it's a bot message or doesn't have text
        if event.get('bot_id') or not event.get('text'):
            return
        
        # Check if this is a thread reply to one of our messages
        thread_ts = event.get('thread_ts')
        if thread_ts:
            # You could implement logic here to continue conversations in threads
            pass
            
    except Exception as e:
        logger.error(f"Error handling message event: {e}")

def start_bot():
    """Start the Slack bot"""
    try:
        # Use Socket Mode for development (requires SLACK_APP_TOKEN)
        socket_token = os.getenv("SLACK_APP_TOKEN")
        if socket_token:
            handler = SocketModeHandler(app, socket_token)
            logger.info("Starting Slack bot in Socket Mode...")
            handler.start()
        else:
            # Use HTTP mode (requires proper webhook setup)
            logger.info("Starting Slack bot in HTTP Mode...")
            app.start(port=int(os.getenv("PORT", 3000)))
            
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise

if __name__ == "__main__":
    print("🤖 Starting Slack Search Bot...")
    print("The bot will:")
    print("- Respond to @mentions")
    print("- Process direct messages")
    print("- React to messages with trigger keywords")
    print("- Handle /search-help slash commands")
    print("\nMake sure your environment variables are set:")
    print("- SLACK_BOT_TOKEN")
    print("- SLACK_APP_TOKEN (for Socket Mode)")
    print("- OPENAI_API_KEY")
    print("- JIRA_SERVER, JIRA_USERNAME, JIRA_API_TOKEN")
    print("- CONFLUENCE_SERVER, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN")
    print("- SLACK_SEARCH_CHANNELS (comma-separated)")
    print()
    
    start_bot()
