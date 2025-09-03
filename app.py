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
    """Fallback MCP search function (original implementation)"""
    try:
        if not MCP_AVAILABLE:
            return {
                'success': False,
                'error': 'MCP integration not available'
            }
        
        # Import MCP agent directly
        from mcp_integration.mcp_enhanced_search_agent import MCPEnhancedSearchAgent
        import asyncio
        
        # Create MCP agent
        mcp_agent = MCPEnhancedSearchAgent()
        
        # Run MCP-only search with timeout
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                logger.info(f"🔍 Starting MCP fallback search for query: '{query}'")
                logger.info(f"🚀 Searching platforms: ['jira', 'slack', 'confluence']")
                
                # Add timeout to prevent hanging (30 seconds)
                mcp_results = loop.run_until_complete(
                    asyncio.wait_for(
                        mcp_agent.enhanced_search(query, ['jira', 'slack', 'confluence']),
                        timeout=30.0
                    )
                )
                
                # Log detailed MCP server responses (using both logger and print for visibility)
                log_separator = "=" * 80
                mcp_header = "🚀 MCP SERVER RESPONSES"
                
                print(log_separator)
                print(mcp_header)
                print(log_separator)
                logger.info(log_separator)
                logger.info(mcp_header)
                logger.info(log_separator)
                
                # Log full MCP results structure
                full_results_msg = f"📊 FULL MCP RESULTS: {json.dumps(mcp_results, indent=2, default=str)}"
                print(full_results_msg)
                logger.info(full_results_msg)
                
                # Log individual platform results
                results = mcp_results.get('results', []) if isinstance(mcp_results, dict) else []
                slack_results = [r for r in results if r.get('platform', '').lower() == 'slack']
                jira_results = [r for r in results if r.get('platform', '').lower() == 'jira']
                confluence_results = [r for r in results if r.get('platform', '').lower() == 'confluence']
                
                slack_header = f"💬 SLACK MCP SERVER RESPONSE ({len(slack_results)} results):"
                print(slack_header)
                logger.info(slack_header)
                for i, result in enumerate(slack_results, 1):
                    slack_result_msg = f"  Slack Result {i}: {json.dumps(result, indent=4, default=str)}"
                    print(slack_result_msg)
                    logger.info(slack_result_msg)
                
                jira_header = f"🎫 JIRA MCP SERVER RESPONSE ({len(jira_results)} results):"
                print(jira_header)
                logger.info(jira_header)
                for i, result in enumerate(jira_results, 1):
                    jira_result_msg = f"  Jira Result {i}: {json.dumps(result, indent=4, default=str)}"
                    print(jira_result_msg)
                    logger.info(jira_result_msg)
                
                confluence_header = f"📄 CONFLUENCE MCP SERVER RESPONSE ({len(confluence_results)} results):"
                print(confluence_header)
                logger.info(confluence_header)
                for i, result in enumerate(confluence_results, 1):
                    confluence_result_msg = f"  Confluence Result {i}: {json.dumps(result, indent=4, default=str)}"
                    print(confluence_result_msg)
                    logger.info(confluence_result_msg)
                
                # Log platform insights
                platform_insights = mcp_results.get('platform_insights', {})
                insights_msg = f"🧠 PLATFORM INSIGHTS: {json.dumps(platform_insights, indent=2, default=str)}"
                print(insights_msg)
                logger.info(insights_msg)
                
                # Log metadata
                metadata = mcp_results.get('metadata', {})
                metadata_msg = f"📈 METADATA: {json.dumps(metadata, indent=2, default=str)}"
                print(metadata_msg)
                logger.info(metadata_msg)
                
                print(log_separator)
                logger.info(log_separator)
                
            finally:
                loop.close()
            
            # Format MCP results for API response
            search_results = {
                'success': True,
                'mcp_enhanced': True,
                'query': mcp_results.get('query', query),
                'total_results': mcp_results.get('total_results', 0),
                'platforms_searched': mcp_results.get('platforms_searched', []),
                'results': mcp_results.get('results', []),
                'formatted_response': mcp_results.get('summary', ''),
                'additional_insights': mcp_results.get('summary', ''),
                'cross_platform_results': mcp_results.get('results', []),
                'semantic_score': mcp_results.get('metadata', {}).get('avg_relevance_score', 0),
                'platform_insights': mcp_results.get('platform_insights', {}),
                'metadata': mcp_results.get('metadata', {})
            }
            
            logger.info(f"✅ MCP-only search completed: {search_results['total_results']} results across {len(search_results['platforms_searched'])} platforms")
            return search_results
            
        except asyncio.TimeoutError:
            logger.error(f"❌ MCP search timed out after 30 seconds for query: '{query}'")
            return {
                'success': False,
                'error': 'Search timed out after 30 seconds. Please try a simpler query or try again later.'
            }
        except Exception as e:
            logger.error(f"❌ MCP search failed: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': f'MCP search failed: {str(e)}'
            }
        
    except Exception as e:
        logger.error(f"❌ Error performing MCP search: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
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
    """API endpoint for searching across Jira, Slack, and Confluence with MCP enhancement"""
    
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

        # Use the MCP-enhanced perform_search function
        result = perform_search(query, channel)
        
        if result.get('success'):
            # Generate solution recommendations using MCP enhanced search agent
            solution_analysis = None
            try:
                from mcp_integration.mcp_enhanced_search_agent import MCPEnhancedSearchAgent
                import asyncio
                
                mcp_agent = MCPEnhancedSearchAgent()
                
                # Run solution analysis with timeout
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    logger.info(f"🧠 Analyzing technical issues and generating solutions for query: '{query}'")
                    raw_solution_analysis = loop.run_until_complete(
                        asyncio.wait_for(
                            mcp_agent.analyze_technical_issue_and_generate_solution(result.get('results', []), query),
                            timeout=15.0
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
                    
                    # Create primary solution from recommendations with source link
                    if solution_recommendations:
                        primary_solution_source = solution_recommendations.get('primary_solution_source', '')
                        
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
                            'source_links': [link for link in source_links if link.get('type') == 'primary_solution']  # Primary solution links
                        }
                        
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
                    
                    # If no structured solutions, create a basic one from error patterns
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
                            'confidence_score': 0.5
                        }
                        solution_analysis['solutions'].append(basic_solution)
                    
                    logger.info(f"✅ Solution analysis completed: {len(solution_analysis.get('solutions', []))} solutions generated")
                finally:
                    loop.close()
                    
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Solution analysis timed out for query: '{query}'")
                solution_analysis = {
                    'solutions': [{
                        'error_pattern': 'Analysis Timeout',
                        'solution': 'Solution analysis timed out. Please try a more specific query or try again later.',
                        'confidence_score': 0.3
                    }],
                    'error': 'Solution analysis timed out'
                }
            except Exception as e:
                logger.error(f"❌ Solution analysis failed: {e}")
                solution_analysis = {
                    'solutions': [{
                        'error_pattern': 'Analysis Error',
                        'solution': f'Solution analysis encountered an error: {str(e)}. Please try again or contact support.',
                        'confidence_score': 0.2
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
            
            # Get platform_insights from the correct location in search_results
            platform_insights = result.get('search_results', {}).get('platform_insights')
            if platform_insights is None:
                platform_insights = {
                    'jira': {'insights': ['No Jira data available'], 'count': 0, 'summary': 'No results'},
                    'slack': {'insights': ['No Slack data available'], 'count': 0, 'summary': 'No results'},
                    'confluence': {'insights': ['No Confluence data available'], 'count': 0, 'summary': 'No results'}
                }
            
            return jsonify({
                'success': True,
                'query': query,
                'search_results': search_results,
                'formatted_response': result.get('formatted_response', ''),
                'mcp_enhanced': result.get('mcp_enhanced', False),
                'additional_insights': result.get('additional_insights', []),
                'cross_platform_results': result.get('cross_platform_results', []),
                'semantic_score': result.get('semantic_score', 0),
                'platform_insights': platform_insights,
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
