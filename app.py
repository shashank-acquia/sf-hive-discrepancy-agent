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

from tools.utils import download_nltk_data

# Import MCP integration
try:
    from mcp_integration.flask_integration import register_mcp_routes, enhance_existing_search_results
    from mcp_integration.mcp_config import mcp_config
    MCP_AVAILABLE = True
    # Validate MCP configuration on startup
    mcp_status = mcp_config.validate_configuration()
    enabled_servers = mcp_config.get_enabled_servers()
    MCP_SERVERS_READY = len(enabled_servers) > 0
    logging.info(f"MCP integration loaded - {len(enabled_servers)} servers ready: {list(enabled_servers.keys())}")
except ImportError as e:
    logging.warning(f"MCP integration not available: {e}")
    MCP_AVAILABLE = False
    MCP_SERVERS_READY = False

load_dotenv()
download_nltk_data()

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
    """Perform search using enhanced SlackSearchAgent with JIRA thread detection"""
    try:
        logger.info(f"🔍 Starting enhanced search for query: '{query}'")
        
        # Use the enhanced SlackSearchAgent directly
        search_results = search_agent._search_all_platforms(query)
        
        # Format results for UI compatibility
        formatted_results = {
            'success': True,
            'query': query,
            'total_results': (
                len(search_results.get('slack_messages', [])) + 
                len(search_results.get('jira_issues', [])) + 
                len(search_results.get('confluence_pages', []))
            ),
            'platforms_searched': ['slack', 'jira', 'confluence'],
            'results': [],
            'formatted_response': '',
            'search_results': search_results,
            'mcp_enhanced': MCP_AVAILABLE and MCP_SERVERS_READY,
            'mcp_available': MCP_AVAILABLE,
            'mcp_servers_ready': MCP_SERVERS_READY
        }
        
        # Convert to unified results format
        for msg in search_results.get('slack_messages', []):
            formatted_results['results'].append({
                'platform': 'slack',
                'title': f"Message in #{msg.get('channel', 'unknown')}",
                'content': msg.get('text', ''),
                'url': msg.get('permalink', ''),
                'metadata': {
                    'channel': msg.get('channel'),
                    'user': msg.get('user'),
                    'timestamp': msg.get('timestamp'),
                    'thread_summary': msg.get('thread_summary', ''),
                    'reply_count': msg.get('reply_count', 0),
                    'score': msg.get('score', 0)
                }
            })
        
        for issue in search_results.get('jira_issues', []):
            formatted_results['results'].append({
                'platform': 'jira',
                'key': issue.get('key'),
                'title': issue.get('summary', ''),
                'content': issue.get('description', ''),
                'url': issue.get('url', ''),
                'metadata': {
                    'status': issue.get('status'),
                    'priority': issue.get('priority'),
                    'assignee': issue.get('assignee'),
                    'project': issue.get('project'),
                    'issue_type': issue.get('issue_type'),
                    'relevance_score': issue.get('relevance_score', 0),
                    'search_strategy': issue.get('search_strategy', 'direct_query')
                }
            })
        
        for page in search_results.get('confluence_pages', []):
            formatted_results['results'].append({
                'platform': 'confluence',
                'title': page.get('title', ''),
                'content': page.get('excerpt', ''),
                'url': page.get('url', ''),
                'metadata': {
                    'space': page.get('space'),
                    'last_modified': page.get('lastModified'),
                    'author': page.get('author'),
                    'type': page.get('type')
                }
            })
        
        # Generate formatted response using the enhanced agent
        try:
            formatted_response = search_agent._generate_llm_response(query, search_results)
            formatted_results['formatted_response'] = formatted_response
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
            formatted_results['formatted_response'] = search_agent._generate_fallback_response(search_results)
        
        logger.info(f"✅ Enhanced search completed: {formatted_results['total_results']} results found")
        logger.info(f"📊 Results breakdown - Slack: {len(search_results.get('slack_messages', []))}, JIRA: {len(search_results.get('jira_issues', []))}, Confluence: {len(search_results.get('confluence_pages', []))}")
        
        # Log JIRA results with relevance scores for debugging
        jira_results = search_results.get('jira_issues', [])
        if jira_results:
            logger.info("🎫 JIRA Results with relevance scores:")
            for i, issue in enumerate(jira_results[:5], 1):
                logger.info(f"  {i}. {issue.get('key', 'Unknown')}: {issue.get('summary', 'No summary')}")
                logger.info(f"     Relevance: {issue.get('relevance_score', 0)}, Strategy: {issue.get('search_strategy', 'unknown')}")
                logger.info(f"     Status: {issue.get('status', 'Unknown')}, URL: {issue.get('url', 'No URL')}")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"❌ Error performing enhanced search: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Enhanced search failed: {str(e)}'
        }

def perform_search_mcp_fallback(query, channel=None):
    """MCP enhanced search function with same functionality as perform_search"""
    try:
        logger.info(f"🔍 Starting MCP enhanced search for query: '{query}'")
        
        # First, try MCP enhanced search if available
        if MCP_AVAILABLE and MCP_SERVERS_READY:
            try:
                # Import MCP agent directly
                from mcp_integration.mcp_enhanced_search_agent import MCPEnhancedSearchAgent
                import asyncio
                
                # Create MCP agent
                mcp_agent = MCPEnhancedSearchAgent()
                
                # Run MCP search with timeout
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    logger.info(f"🚀 Searching platforms via MCP: ['jira', 'slack', 'confluence', 'google_docs']")
                    
                    # Add timeout to prevent hanging (30 seconds)
                    mcp_results = loop.run_until_complete(
                        asyncio.wait_for(
                            mcp_agent.enhanced_search(query, ['jira', 'slack', 'confluence', 'google_docs']),
                            timeout=30.0
                        )
                    )
                    
                    # Convert MCP results to the same format as perform_search
                    search_results = {
                        'slack_messages': [],
                        'jira_issues': [],
                        'confluence_pages': []
                    }
                    
                    # Parse MCP results and organize by platform (same as perform_search)
                    results = mcp_results.get('results', []) if isinstance(mcp_results, dict) else []
                    for item in results:
                        platform = item.get('platform', '').lower()
                        metadata = item.get('metadata', {})
                        
                        if platform == 'slack':
                            search_results['slack_messages'].append({
                                'channel': metadata.get('channel', 'unknown'),
                                'user': metadata.get('user', 'Unknown'),
                                'timestamp': metadata.get('timestamp', metadata.get('ts')),
                                'text': item.get('content', item.get('text', 'No content')),
                                'permalink': item.get('url', item.get('permalink')),
                                'thread_summary': metadata.get('thread_summary', ''),
                                'reply_count': metadata.get('reply_count', 0),
                                'score': metadata.get('score', 0)
                            })
                        elif platform == 'jira':
                            search_results['jira_issues'].append({
                                'key': item.get('key', item.get('id', 'Unknown')),
                                'summary': item.get('title', item.get('summary', 'No summary')),
                                'status': metadata.get('status', 'Unknown'),
                                'priority': metadata.get('priority', 'Unknown'),
                                'assignee': metadata.get('assignee', 'Unassigned'),
                                'description': item.get('content', item.get('description')),
                                'url': item.get('url'),
                                'project': metadata.get('project', 'Unknown'),
                                'issue_type': metadata.get('issue_type', 'Unknown'),
                                'relevance_score': metadata.get('relevance_score', 0),
                                'search_strategy': 'mcp_enhanced'
                            })
                        elif platform == 'confluence':
                            search_results['confluence_pages'].append({
                                'title': item.get('title', 'Untitled'),
                                'space': metadata.get('space', 'Unknown'),
                                'lastModified': metadata.get('last_modified', metadata.get('updated')),
                                'excerpt': item.get('content', item.get('excerpt')),
                                'url': item.get('url'),
                                'type': metadata.get('type', 'page'),
                                'author': metadata.get('author', 'Unknown')
                            })
                    
                    # Generate formatted response using the enhanced agent (same as perform_search)
                    try:
                        formatted_response = search_agent._generate_llm_response(query, search_results)
                    except Exception as e:
                        logger.warning(f"LLM response generation failed: {e}")
                        formatted_response = search_agent._generate_fallback_response(search_results)
                    
                    # Format results for UI compatibility (same structure as perform_search)
                    formatted_results = {
                        'success': True,
                        'query': query,
                        'total_results': (
                            len(search_results.get('slack_messages', [])) + 
                            len(search_results.get('jira_issues', [])) + 
                            len(search_results.get('confluence_pages', []))
                        ),
                        'platforms_searched': ['slack', 'jira', 'confluence'],
                        'results': [],
                        'formatted_response': formatted_response,
                        'search_results': search_results,
                        'mcp_enhanced': True,
                        'mcp_available': MCP_AVAILABLE,
                        'mcp_servers_ready': MCP_SERVERS_READY
                    }
                    
                    # Convert to unified results format (same as perform_search)
                    for msg in search_results.get('slack_messages', []):
                        formatted_results['results'].append({
                            'platform': 'slack',
                            'title': f"Message in #{msg.get('channel', 'unknown')}",
                            'content': msg.get('text', ''),
                            'url': msg.get('permalink', ''),
                            'metadata': {
                                'channel': msg.get('channel'),
                                'user': msg.get('user'),
                                'timestamp': msg.get('timestamp'),
                                'thread_summary': msg.get('thread_summary', ''),
                                'reply_count': msg.get('reply_count', 0),
                                'score': msg.get('score', 0)
                            }
                        })
                    
                    for issue in search_results.get('jira_issues', []):
                        formatted_results['results'].append({
                            'platform': 'jira',
                            'key': issue.get('key'),
                            'title': issue.get('summary', ''),
                            'content': issue.get('description', ''),
                            'url': issue.get('url', ''),
                            'metadata': {
                                'status': issue.get('status'),
                                'priority': issue.get('priority'),
                                'assignee': issue.get('assignee'),
                                'project': issue.get('project'),
                                'issue_type': issue.get('issue_type'),
                                'relevance_score': issue.get('relevance_score', 0),
                                'search_strategy': issue.get('search_strategy', 'mcp_enhanced')
                            }
                        })
                    
                    for page in search_results.get('confluence_pages', []):
                        formatted_results['results'].append({
                            'platform': 'confluence',
                            'title': page.get('title', ''),
                            'content': page.get('excerpt', ''),
                            'url': page.get('url', ''),
                            'metadata': {
                                'space': page.get('space'),
                                'last_modified': page.get('lastModified'),
                                'author': page.get('author'),
                                'type': page.get('type')
                            }
                        })
                    
                    logger.info(f"✅ MCP enhanced search completed: {formatted_results['total_results']} results found")
                    logger.info(f"📊 Results breakdown - Slack: {len(search_results.get('slack_messages', []))}, JIRA: {len(search_results.get('jira_issues', []))}, Confluence: {len(search_results.get('confluence_pages', []))}")
                    
                    return formatted_results
                    
                finally:
                    loop.close()
                    
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ MCP search timed out for query: '{query}', falling back to direct API search")
                # Fall back to direct API search
                pass
            except Exception as e:
                logger.warning(f"⚠️ MCP search failed: {e}, falling back to direct API search")
                # Fall back to direct API search
                pass
        
        # Fallback to direct API search (same logic as perform_search)
        logger.info(f"🔍 Using direct API search as fallback for query: '{query}'")
        
        # Use the enhanced SlackSearchAgent directly (same as perform_search)
        search_results = search_agent._search_all_platforms(query)
        
        # Format results for UI compatibility (same as perform_search)
        formatted_results = {
            'success': True,
            'query': query,
            'total_results': (
                len(search_results.get('slack_messages', [])) + 
                len(search_results.get('jira_issues', [])) + 
                len(search_results.get('confluence_pages', []))
            ),
            'platforms_searched': ['slack', 'jira', 'confluence'],
            'results': [],
            'formatted_response': '',
            'search_results': search_results,
            'mcp_enhanced': False,  # Fallback mode
            'mcp_available': MCP_AVAILABLE,
            'mcp_servers_ready': MCP_SERVERS_READY
        }
        
        # Convert to unified results format (same as perform_search)
        for msg in search_results.get('slack_messages', []):
            formatted_results['results'].append({
                'platform': 'slack',
                'title': f"Message in #{msg.get('channel', 'unknown')}",
                'content': msg.get('text', ''),
                'url': msg.get('permalink', ''),
                'metadata': {
                    'channel': msg.get('channel'),
                    'user': msg.get('user'),
                    'timestamp': msg.get('timestamp'),
                    'thread_summary': msg.get('thread_summary', ''),
                    'reply_count': msg.get('reply_count', 0),
                    'score': msg.get('score', 0)
                }
            })
        
        for issue in search_results.get('jira_issues', []):
            formatted_results['results'].append({
                'platform': 'jira',
                'key': issue.get('key'),
                'title': issue.get('summary', ''),
                'content': issue.get('description', ''),
                'url': issue.get('url', ''),
                'metadata': {
                    'status': issue.get('status'),
                    'priority': issue.get('priority'),
                    'assignee': issue.get('assignee'),
                    'project': issue.get('project'),
                    'issue_type': issue.get('issue_type'),
                    'relevance_score': issue.get('relevance_score', 0),
                    'search_strategy': issue.get('search_strategy', 'direct_query')
                }
            })
        
        for page in search_results.get('confluence_pages', []):
            formatted_results['results'].append({
                'platform': 'confluence',
                'title': page.get('title', ''),
                'content': page.get('excerpt', ''),
                'url': page.get('url', ''),
                'metadata': {
                    'space': page.get('space'),
                    'last_modified': page.get('lastModified'),
                    'author': page.get('author'),
                    'type': page.get('type')
                }
            })
        
        # Generate formatted response using the enhanced agent (same as perform_search)
        try:
            formatted_response = search_agent._generate_llm_response(query, search_results)
            formatted_results['formatted_response'] = formatted_response
        except Exception as e:
            logger.warning(f"LLM response generation failed: {e}")
            formatted_results['formatted_response'] = search_agent._generate_fallback_response(search_results)
        
        logger.info(f"✅ Fallback search completed: {formatted_results['total_results']} results found")
        logger.info(f"📊 Results breakdown - Slack: {len(search_results.get('slack_messages', []))}, JIRA: {len(search_results.get('jira_issues', []))}, Confluence: {len(search_results.get('confluence_pages', []))}")
        
        # Log JIRA results with relevance scores for debugging (same as perform_search)
        jira_results = search_results.get('jira_issues', [])
        if jira_results:
            logger.info("🎫 JIRA Results with relevance scores:")
            for i, issue in enumerate(jira_results[:5], 1):
                logger.info(f"  {i}. {issue.get('key', 'Unknown')}: {issue.get('summary', 'No summary')}")
                logger.info(f"     Relevance: {issue.get('relevance_score', 0)}, Strategy: {issue.get('search_strategy', 'unknown')}")
                logger.info(f"     Status: {issue.get('status', 'Unknown')}, URL: {issue.get('url', 'No URL')}")
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"❌ Error performing MCP enhanced search: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Enhanced search failed: {str(e)}'
        }

@slack_app.event("app_mention")
def handle_app_mention(event, say, client, logger):
    """Handle when the bot is mentioned"""
    try:
        user = event['user']
        channel = event['channel']
        text = event['text']
        thread_ts = event.get('thread_ts') or event.get('ts')
        
        logger.info(f"🎯 Bot mentioned by {user} in {channel}: {text}")

        # Fetch Thread Context
        full_query_text = text
        if event.get('thread_ts'):
            try:
                logger.info(f"Mention is in a thread. Fetching context from thread {thread_ts}...")
                replies = client.conversations_replies(channel=channel, ts=event.get('thread_ts'), limit=10)
                # Combine messages, oldest to newest, to form a coherent story
                thread_messages = [msg['text'] for msg in replies['messages']]
                full_query_text = "\n".join(thread_messages)
                logger.info(f"Full context query:\n{full_query_text}")
            except Exception as e:
                logger.error(f"Failed to fetch thread context: {e}")
                full_query_text = text
        
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
    """API endpoint for searching across Jira, Slack, and Confluence with conditional MCP/API search"""
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing query parameter'
            }), 400
        
        query = data['query']
        channel = data.get('channel', 'general')
        
        # Check if MCP search is enabled via environment variable
        is_mcp_enabled = os.getenv('IS_MCP_SEARCH_ENABLED', 'true').lower() == 'true'
        
        logger.info(f"API search request: {query} (MCP enabled: {is_mcp_enabled})")

        # Conditionally use MCP or API search based on environment variable
        if is_mcp_enabled and MCP_AVAILABLE and MCP_SERVERS_READY:
            logger.info("🚀 Using MCP enhanced search")
            result = perform_search_mcp_fallback(query, channel)
        else:
            logger.info("🔍 Using direct API search")
            result = perform_search(query, channel)
        
        if result.get('success'):
            # Generate solution recommendations using MCP enhanced search agent with direct API fallback
            solution_analysis = None
            try:
                from mcp_integration.mcp_enhanced_search_agent import MCPEnhancedSearchAgent
                from direct_api_fallback import direct_api_fallback
                import asyncio
                
                mcp_agent = MCPEnhancedSearchAgent()
                
                # Run solution analysis with reduced timeout and fallback
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    logger.info(f"🧠 Analyzing technical issues and generating solutions for query: '{query}'")
                    raw_solution_analysis = loop.run_until_complete(
                        asyncio.wait_for(
                            mcp_agent.analyze_technical_issue_and_generate_solution(result.get('results', []), query),
                            timeout=5.0  # Reduced to 5s to prevent UI timeouts
                        )
                    )
                    
                    # Transform the solution analysis to match UI expectations with source links
                    solution_analysis = {
                        'solutions': [],
                        'error_analysis': raw_solution_analysis.get('error_analysis', {}),
                        'detailed_solutions': raw_solution_analysis.get('detailed_solutions', []),
                        'metadata': raw_solution_analysis.get('metadata', {}),
                        'source_links': []  # Add source links array
                    }
                    
                    # Extract error patterns and convert to UI-friendly format
                    error_patterns = raw_solution_analysis.get('error_analysis', {}).get('error_patterns', [])
                    solution_recommendations = raw_solution_analysis.get('solution_recommendations', {})
                    next_steps = raw_solution_analysis.get('next_steps', [])
                    
                    # Extract source links from solution recommendations
                    source_links = solution_recommendations.get('source_links', [])
                    solution_analysis['source_links'] = source_links
                    
                    # Create primary solution from recommendations with source link and detailed solutions
                    if solution_recommendations:
                        primary_solution_source = solution_recommendations.get('primary_solution_source', '')
                        detailed_solutions_data = raw_solution_analysis.get('detailed_solutions', [])
                        
                        # Extract Jira comments from detailed solutions with enhanced data
                        jira_comments = []
                        slack_thread_solutions = []
                        confluence_solutions = []
                        
                        logger.info(f"🔍 Processing {len(detailed_solutions_data)} detailed solution sources for UI display")
                        
                        for detailed_sol in detailed_solutions_data:
                            source = detailed_sol.get('source', '')
                            logger.info(f"📊 Processing {source} solution data: {list(detailed_sol.keys())}")
                            
                            if source == 'jira':
                                # Extract Jira comments and solutions with enhanced metadata
                                comments = detailed_sol.get('comments', [])
                                solutions = detailed_sol.get('solutions', [])
                                ticket_key = detailed_sol.get('ticket', 'Unknown')
                                ticket_url = detailed_sol.get('url', '')
                                
                                logger.info(f"🎫 Processing Jira ticket {ticket_key}: {len(comments)} comments, {len(solutions)} solutions")
                                
                                # Add actual Jira comments with enhanced metadata
                                for comment in comments[:10]:  # Increased to 10 comments for better coverage
                                    jira_comment = {
                                        'author': comment.get('author', 'Unknown'),
                                        'content': comment.get('body', comment.get('content', 'No content')),
                                        'created': comment.get('created', ''),
                                        'type': 'jira_comment',
                                        'ticket_key': ticket_key,
                                        'ticket_url': ticket_url,
                                        'source_platform': 'jira'
                                    }
                                    jira_comments.append(jira_comment)
                                    logger.info(f"  📝 Added Jira comment from {jira_comment['author']}")
                                
                                # Add solution-specific comments with enhanced metadata
                                for solution in solutions[:5]:  # Increased to 5 solutions
                                    solution_comment = {
                                        'author': solution.get('author', 'AI Analysis'),
                                        'content': solution.get('content', 'No content'),
                                        'created': solution.get('created', ''),
                                        'keywords_matched': solution.get('keywords_matched', []),
                                        'confidence': solution.get('confidence', 0),
                                        'type': solution.get('type', 'jira_solution'),
                                        'ticket_key': ticket_key,
                                        'ticket_url': ticket_url,
                                        'source_platform': 'jira'
                                    }
                                    jira_comments.append(solution_comment)
                                    logger.info(f"  💡 Added Jira solution from {solution_comment['author']} (confidence: {solution_comment['confidence']:.2f})")
                            
                            elif source == 'slack':
                                # Extract Slack thread solutions with enhanced metadata
                                solutions = detailed_sol.get('solutions', [])
                                channel = detailed_sol.get('channel', 'unknown')
                                thread_url = detailed_sol.get('url', '')
                                
                                logger.info(f"💬 Processing Slack channel #{channel}: {len(solutions)} solutions")
                                
                                for solution in solutions[:5]:  # Increased to 5 solutions
                                    slack_solution = {
                                        'author': solution.get('author', 'Unknown'),
                                        'content': solution.get('content', 'No content'),
                                        'timestamp': solution.get('timestamp', ''),
                                        'confidence': solution.get('confidence', 0),
                                        'type': solution.get('type', 'slack_thread_solution'),
                                        'channel': channel,
                                        'thread_url': thread_url,
                                        'source_platform': 'slack'
                                    }
                                    slack_thread_solutions.append(slack_solution)
                                    logger.info(f"  💬 Added Slack solution from {slack_solution['author']} in #{channel}")
                            
                            elif source == 'confluence':
                                # Extract Confluence solutions with enhanced metadata
                                solutions = detailed_sol.get('solutions', [])
                                page_title = detailed_sol.get('page_title', 'Unknown Page')
                                page_url = detailed_sol.get('url', '')
                                
                                logger.info(f"📄 Processing Confluence page '{page_title}': {len(solutions)} solutions")
                                
                                for solution in solutions[:5]:  # Increased to 5 solutions
                                    confluence_solution = {
                                        'content': solution.get('content', 'No content'),
                                        'pattern': solution.get('pattern', ''),
                                        'confidence': solution.get('confidence', 0),
                                        'type': solution.get('type', 'confluence_documentation_solution'),
                                        'page_title': page_title,
                                        'page_url': page_url,
                                        'source_platform': 'confluence'
                                    }
                                    confluence_solutions.append(confluence_solution)
                                    logger.info(f"  📄 Added Confluence solution from '{page_title}'")
                        
                        # Enhanced primary solution with comprehensive data
                        primary_solution = {
                            'error_pattern': f"Technical Issue Analysis for: {query}",
                            'technical_context': f"Found {len(error_patterns)} error patterns across platforms",
                            'root_cause': solution_recommendations.get('root_cause_analysis', 'Analysis in progress...'),
                            'solution': solution_recommendations.get('primary_solution', 'No specific solution available yet'),
                            'prevention_measures': solution_recommendations.get('prevention_measures', []),
                            'next_steps': next_steps,
                            'related_tickets': [ep.get('title', 'Unknown') for ep in error_patterns if ep.get('platform') == 'jira'],
                            'confidence_score': 0.8,  # Default confidence
                            'source_url': primary_solution_source,  # Add source URL to primary solution
                            'source_links': [link for link in source_links if link.get('type') == 'primary_solution'],  # Primary solution links
                            # Enhanced detailed solutions for UI display with comprehensive data
                            'jira_comments': jira_comments,
                            'slack_thread_solutions': slack_thread_solutions,
                            'confluence_solutions': confluence_solutions,
                            # Additional metadata for debugging and UI enhancement
                            'solution_metadata': {
                                'total_jira_comments': len(jira_comments),
                                'total_slack_solutions': len(slack_thread_solutions),
                                'total_confluence_solutions': len(confluence_solutions),
                                'detailed_sources_processed': len(detailed_solutions_data),
                                'query_analyzed': query
                            }
                        }
                        
                        logger.info(f"✅ Primary solution created with:")
                        logger.info(f"  🎫 Jira comments: {len(jira_comments)}")
                        logger.info(f"  💬 Slack solutions: {len(slack_thread_solutions)}")
                        logger.info(f"  📄 Confluence solutions: {len(confluence_solutions)}")
                        
                        solution_analysis['solutions'].append(primary_solution)
                        
                        # Add alternative solutions if available with their source links
                        alt_solutions = solution_recommendations.get('alternative_solutions', [])
                        alt_source_links = [link for link in source_links if link.get('type') == 'alternative_solution']
                        
                        for i, alt_sol in enumerate(alt_solutions[:2]):  # Limit to 2 alternatives
                            # Find corresponding source link for this alternative
                            alt_source_link = alt_source_links[i] if i < len(alt_source_links) else {}
                            
                            alt_solution = {
                                'error_pattern': f"Alternative Solution {i+1}",
                                'solution': alt_sol,
                                'confidence_score': 0.6 - (i * 0.1),  # Decreasing confidence
                                'source_url': alt_source_link.get('url', ''),  # Add source URL to alternative
                                'source_title': alt_source_link.get('title', ''),
                                'source_platform': alt_source_link.get('platform', '')
                            }
                            solution_analysis['solutions'].append(alt_solution)
                    
                    # If no structured solutions, create a basic one from error patterns with empty arrays
                    if not solution_analysis['solutions'] and error_patterns:
                        basic_solution = {
                            'error_pattern': f"Error Pattern Analysis for: {query}",
                            'technical_context': f"Detected {len(error_patterns)} error patterns",
                            'solution': f"Based on the analysis of {len(error_patterns)} error patterns, consider reviewing the related tickets and documentation for similar issues.",
                            'next_steps': next_steps or [
                                "Review error patterns in detail",
                                "Check related tickets for solutions",
                                "Consult team members who worked on similar issues"
                            ],
                            'confidence_score': 0.5,
                            # Ensure these arrays exist even for basic solutions
                            'jira_comments': [],
                            'slack_thread_solutions': [],
                            'confluence_solutions': [],
                            'solution_metadata': {
                                'total_jira_comments': 0,
                                'total_slack_solutions': 0,
                                'total_confluence_solutions': 0,
                                'detailed_sources_processed': 0,
                                'query_analyzed': query
                            }
                        }
                        solution_analysis['solutions'].append(basic_solution)
                        logger.info("📝 Created basic solution with empty solution arrays")
                    
                    logger.info(f"✅ Solution analysis completed: {len(solution_analysis.get('solutions', []))} solutions generated")
                finally:
                    loop.close()
                    
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ MCP solution analysis timed out for query: '{query}', using direct API fallback")
                try:
                    # Use direct API fallback when MCP times out
                    solution_analysis = direct_api_fallback.generate_solution_analysis_fallback(result.get('results', []), query)
                    logger.info(f"✅ Direct API fallback completed successfully")
                except Exception as fallback_error:
                    logger.error(f"❌ Direct API fallback also failed: {fallback_error}")
                    solution_analysis = {
                        'solutions': [{
                            'error_pattern': 'Analysis Timeout - Fallback Failed',
                            'solution': 'Both MCP and direct API analysis failed. Basic search results are available above.',
                            'confidence_score': 0.3,
                            'jira_comments': [],
                            'slack_thread_solutions': [],
                            'confluence_solutions': [],
                            'solution_metadata': {
                                'total_jira_comments': 0,
                                'total_slack_solutions': 0,
                                'total_confluence_solutions': 0,
                                'detailed_sources_processed': 0,
                                'query_analyzed': query,
                                'error': 'Timeout and fallback failed'
                            }
                        }],
                        'error': 'Solution analysis timed out and fallback failed'
                    }
            except Exception as e:
                logger.error(f"❌ MCP solution analysis failed: {e}, using direct API fallback")
                try:
                    # Use direct API fallback when MCP fails
                    solution_analysis = direct_api_fallback.generate_solution_analysis_fallback(result.get('results', []), query)
                    logger.info(f"✅ Direct API fallback completed successfully after MCP failure")
                except Exception as fallback_error:
                    logger.error(f"❌ Direct API fallback also failed: {fallback_error}")
                    solution_analysis = {
                        'solutions': [{
                            'error_pattern': 'Analysis Error - Fallback Failed',
                            'solution': f'Both MCP analysis and direct API fallback failed. Basic search results are available above. Error: {str(e)}',
                            'confidence_score': 0.2,
                            'jira_comments': [],
                            'slack_thread_solutions': [],
                            'confluence_solutions': [],
                            'solution_metadata': {
                                'total_jira_comments': 0,
                                'total_slack_solutions': 0,
                                'total_confluence_solutions': 0,
                                'detailed_sources_processed': 0,
                                'query_analyzed': query,
                                'error': f'MCP and fallback failed: {str(e)}'
                            }
                        }],
                        'error': f'Solution analysis failed: {str(e)}'
                    }
            
            # Structure the response to match what the UI expects
            search_results = {
                'slack_messages': [],
                'jira_issues': [],
                'confluence_pages': []
            }
            
            # Parse results from MCP and organize by platform
            results = result.get('results', [])
            if isinstance(results, list):
                for item in results:
                    platform = item.get('platform', '').lower()
                    metadata = item.get('metadata', {})
                    
                    if platform == 'slack':
                        search_results['slack_messages'].append({
                            'channel': metadata.get('channel', 'unknown'),
                            'user': metadata.get('user', 'Unknown'),
                            'timestamp': metadata.get('timestamp', metadata.get('ts')),
                            'text': item.get('content', item.get('text', 'No content')),
                            'permalink': item.get('url', item.get('permalink')),
                            'thread_summary': metadata.get('thread_summary', ''),
                            'reply_count': metadata.get('reply_count', 0),
                            'score': metadata.get('score', 0)
                        })
                    elif platform == 'jira':
                        search_results['jira_issues'].append({
                            'key': item.get('key', item.get('id', 'Unknown')),
                            'summary': item.get('title', item.get('summary', 'No summary')),
                            'status': metadata.get('status', 'Unknown'),
                            'priority': metadata.get('priority', 'Unknown'),
                            'assignee': metadata.get('assignee', 'Unassigned'),
                            'description': item.get('content', item.get('description')),
                            'url': item.get('url'),
                            'project': metadata.get('project', 'Unknown'),
                            'issue_type': metadata.get('issue_type', 'Unknown')
                        })
                    elif platform == 'confluence':
                        search_results['confluence_pages'].append({
                            'title': item.get('title', 'Untitled'),
                            'space': metadata.get('space', 'Unknown'),
                            'lastModified': metadata.get('last_modified', metadata.get('updated')),
                            'excerpt': item.get('content', item.get('excerpt')),
                            'url': item.get('url'),
                            'type': metadata.get('type', 'page'),
                            'author': metadata.get('author', 'Unknown')
                        })
                    elif platform == 'google_docs':
                        search_results['google_docs'] = search_results.get('google_docs', [])
                        search_results['google_docs'].append({
                            'title': item.get('title', 'Untitled Document'),
                            'type': metadata.get('type', 'Document'),
                            'lastModified': metadata.get('last_modified', metadata.get('updated')),
                            'owner': metadata.get('owner', metadata.get('author')),
                            'folder': metadata.get('folder', metadata.get('parent_folder')),
                            'excerpt': item.get('content', item.get('excerpt')),
                            'url': item.get('url'),
                            'metadata': {
                                'relevance_score': metadata.get('relevance_score', 0)
                            }
                        })
            
            # CRITICAL FIX: Add Jira tickets extracted from Slack threads to jira_issues array
            # This ensures cross-platform Jira tickets appear in Platform Intelligence Insights and Jira Issues sections
            if solution_analysis and solution_analysis.get('solutions'):
                logger.info("🔍 Processing solution analysis for cross-platform Jira tickets...")
                
                for solution in solution_analysis.get('solutions', []):
                    jira_comments = solution.get('jira_comments', [])
                    
                    # Extract unique Jira tickets from solution analysis
                    processed_tickets = set()
                    for comment in jira_comments:
                        ticket_key = comment.get('ticket_key')
                        ticket_url = comment.get('ticket_url')
                        found_via = comment.get('found_via', '')
                        
                        if ticket_key and ticket_key not in processed_tickets:
                            # Check if this ticket is already in the jira_issues array
                            existing_ticket = next((t for t in search_results['jira_issues'] if t.get('key') == ticket_key), None)
                            
                            if not existing_ticket:
                                # Add the cross-platform Jira ticket to the jira_issues array
                                cross_platform_ticket = {
                                    'key': ticket_key,
                                    'summary': f"Cross-platform ticket from Slack analysis: {comment.get('content', 'No summary')[:100]}...",
                                    'status': 'Referenced in Slack',
                                    'priority': 'Unknown',
                                    'assignee': 'Unknown',
                                    'description': comment.get('content', 'Jira ticket extracted from Slack thread analysis'),
                                    'url': ticket_url or f"https://acquia.atlassian.net/browse/{ticket_key}",
                                    'project': 'Cross-Platform',
                                    'issue_type': 'Referenced',
                                    'found_via': found_via,
                                    'cross_platform_source': 'slack_thread_analysis'
                                }
                                
                                search_results['jira_issues'].append(cross_platform_ticket)
                                processed_tickets.add(ticket_key)
                                
                                logger.info(f"✅ Added cross-platform Jira ticket {ticket_key} to jira_issues array (found via: {found_via})")
                            else:
                                logger.info(f"ℹ️ Jira ticket {ticket_key} already exists in jira_issues array")
                
                logger.info(f"🎫 Final jira_issues count: {len(search_results['jira_issues'])} (including cross-platform tickets)")
            
            # Get platform_insights from the correct location - try multiple sources
            platform_insights = result.get('platform_insights')
            if platform_insights is None:
                platform_insights = result.get('search_results', {}).get('platform_insights')
            if platform_insights is None:
                # Create default insights structure including Google Docs
                platform_insights = {
                    'jira': {'insights': ['No Jira data available'], 'count': 0, 'summary': 'No results'},
                    'slack': {'insights': ['No Slack data available'], 'count': 0, 'summary': 'No results'},
                    'confluence': {'insights': ['No Confluence data available'], 'count': 0, 'summary': 'No results'},
                    'google_docs': {'insights': ['No Google Docs data available'], 'count': 0, 'summary': 'No results'}
                }
            
            # Ensure Google Docs insights are included if we have Google Docs results
            google_docs_results = search_results.get('google_docs', [])
            if google_docs_results and 'google_docs' not in platform_insights:
                # Generate Google Docs insights if missing
                platform_insights['google_docs'] = {
                    'insights': [
                        f"Found {len(google_docs_results)} Google Docs documents",
                        f"• Document types: Various formats available",
                        f"• {len([d for d in google_docs_results if d.get('metadata', {}).get('relevance_score', 0) > 0.5])} highly relevant documents"
                    ],
                    'count': len(google_docs_results),
                    'summary': f"Found {len(google_docs_results)} Google Docs with relevant content"
                }
            elif google_docs_results and 'google_docs' in platform_insights:
                # Update count if insights exist but count is wrong
                platform_insights['google_docs']['count'] = len(google_docs_results)
            
            return jsonify({
                'success': True,
                'query': query,
                'search_results': search_results,
                'formatted_response': result.get('formatted_response', ''),
                'mcp_enhanced': result.get('mcp_enhanced', False),
                'additional_insights': result.get('additional_insights', []),
                'cross_platform_results': result.get('cross_platform_results', []),
                'semantic_score': result.get('semantic_score', 0),
                'processing_time': result.get('metadata', {}).get('processing_time', 'N/A'),
                'solution_analysis': solution_analysis or {
                    'solutions': [],
                    'error': 'No solution analysis available'
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Search failed')
            }), 500
        
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

@app.route("/search/config", methods=["GET"])
def search_config():
    """API endpoint to check current search configuration"""
    try:
        is_mcp_enabled = os.getenv('IS_MCP_SEARCH_ENABLED', 'true').lower() == 'true'
        
        return jsonify({
            'success': True,
            'search_configuration': {
                'is_mcp_search_enabled': is_mcp_enabled,
                'mcp_available': MCP_AVAILABLE,
                'mcp_servers_ready': MCP_SERVERS_READY,
                'current_search_method': 'MCP Enhanced Search' if (is_mcp_enabled and MCP_AVAILABLE and MCP_SERVERS_READY) else 'Direct API Search',
                'environment_variable': os.getenv('IS_MCP_SEARCH_ENABLED', 'true'),
                'enabled_servers': list(mcp_config.get_enabled_servers().keys()) if MCP_AVAILABLE else []
            },
            'message': f"Search method: {'MCP Enhanced' if (is_mcp_enabled and MCP_AVAILABLE and MCP_SERVERS_READY) else 'Direct API'}"
        })
        
    except Exception as e:
        logger.error(f"Error checking search config: {e}")
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
    
    # Register MCP routes if available
    if MCP_AVAILABLE:
        print("- MCP Enhanced Search Integration")
        print("  - GET /api/mcp/status - Check MCP server status")
        print("  - POST /api/mcp/search - Enhanced cross-platform search")
        print("  - GET /api/mcp/platforms - Available search platforms")
        print("  - POST /api/mcp/config/validate - Validate MCP configuration")
        register_mcp_routes(app)
        print("✅ MCP Enhanced Search routes registered")
    else:
        print("⚠️  MCP Enhanced Search not available - install dependencies to enable")
    
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
