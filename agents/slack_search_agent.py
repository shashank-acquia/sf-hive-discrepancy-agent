import os
import logging
from typing import Dict, List, Optional
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        # Fallback for basic functionality without LLM
        ChatOpenAI = None
        HumanMessage = None
        SystemMessage = None

from tools.cdp_chat_tool.slack_tool import SlackTool
from tools.cdp_chat_tool.jira_tool import JiraTool
from tools.cdp_chat_tool.confluence_tool import ConfluenceTool

logger = logging.getLogger(__name__)

class SlackSearchAgent:
    def __init__(self):
        self.slack_tool = SlackTool()
        self.jira_tool = JiraTool()
        self.confluence_tool = ConfluenceTool()
        
        # Initialize LLM
        if ChatOpenAI is not None:
            try:
                self.llm = ChatOpenAI(
                    model_name=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
                    temperature=0.3,
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.llm = None
        else:
            logger.warning("ChatOpenAI not available, LLM responses will be disabled")
            self.llm = None
        
        # Configuration
        self.search_channels = os.getenv('SLACK_SEARCH_CHANNELS', '').split(',')
        self.confluence_spaces = os.getenv('CONFLUENCE_SPACES', '').split(',')
        self.jira_projects = os.getenv('JIRA_PROJECTS', '').split(',')
        
    def process_slack_message(self, message: str, channel: str, user: str, thread_ts: Optional[str] = None) -> Dict:
        """Process a Slack message and generate a comprehensive response"""
        try:
            logger.info(f"Processing message from {user} in {channel}: {message[:100]}...")
            
            # Step 1: Search across all platforms
            search_results = self._search_all_platforms(message)
            
            # Step 2: Generate LLM response
            formatted_response = self._generate_llm_response(message, search_results)
            
            # Step 3: Send response back to Slack
            response_result = self.slack_tool.send_message(
                channel=channel,
                text=formatted_response,
                thread_ts=thread_ts
            )
            
            return {
                'success': True,
                'message': 'Response sent successfully',
                'search_results': search_results,
                'response': formatted_response,
                'slack_response': response_result
            }
            
        except Exception as e:
            logger.error(f"Error processing Slack message: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _search_all_platforms(self, query: str) -> Dict:
        """Search across Jira, Slack, and Confluence"""
        results = {
            'jira_issues': [],
            'slack_messages': [],
            'confluence_pages': []
        }
        
        # Search Jira for similar issues
        try:
            jira_results = self.jira_tool.get_similar_issues(query, max_results=5)
            results['jira_issues'] = jira_results
            logger.info(f"Found {len(jira_results)} similar Jira issues")
        except Exception as e:
            logger.error(f"Error searching Jira: {e}")
        
        # Search Slack channels
        try:
            if self.search_channels and self.search_channels[0]:  # Check if channels are configured
                slack_results = self.slack_tool.search_in_channels(
                    query=query,
                    channels=[ch.strip() for ch in self.search_channels if ch.strip()],
                    limit=5
                )
                results['slack_messages'] = slack_results
                logger.info(f"Found {len(slack_results)} relevant Slack messages")
        except Exception as e:
            logger.error(f"Error searching Slack: {e}")
        
        # Search Confluence (optional - gracefully handle if not available)
        try:
            if os.getenv('CONFLUENCE_SERVER') and os.getenv('CONFLUENCE_API_TOKEN'):
                confluence_results = self.confluence_tool.search_similar_content(query, limit=5)
                results['confluence_pages'] = confluence_results
                logger.info(f"Found {len(confluence_results)} relevant Confluence pages")
            else:
                logger.info("Confluence not configured - skipping Confluence search")
        except Exception as e:
            logger.warning(f"Confluence search unavailable (user may not have access): {e}")
            # Continue without Confluence - this is not a critical error
        
        return results
    
    def _generate_llm_response(self, original_message: str, search_results: Dict) -> str:
        """Generate a formatted response using LLM"""
        try:
            # If LLM is not available, use fallback response
            if self.llm is None or HumanMessage is None or SystemMessage is None:
                return self._generate_fallback_response(search_results)
            
            # Prepare context for LLM
            context = self._prepare_context(search_results)
            
            system_prompt = """You are a helpful assistant that analyzes user queries and provides comprehensive responses based on search results from Jira, Slack, and Confluence.

Your task is to:
1. Analyze the user's message/issue
2. Review the search results from different platforms
3. Provide a well-structured response that includes:
   - A brief summary of the issue
   - Relevant findings from each platform
   - Direct links to helpful resources
   - Actionable recommendations

Format your response in a clear, professional manner suitable for Slack. Use bullet points and clear sections. Always include links when available."""

            user_prompt = f"""
Original Message: {original_message}

Search Results:
{context}

Please provide a comprehensive response that helps the user with their query. Include relevant links and actionable insights.
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(search_results)
    
    def _prepare_context(self, search_results: Dict) -> str:
        """Prepare search results context for LLM"""
        context_parts = []
        
        # Jira Issues
        if search_results['jira_issues']:
            context_parts.append("JIRA ISSUES:")
            for issue in search_results['jira_issues'][:5]:  # Top 3 results
                context_parts.append(f"- {issue['key']}: {issue['summary']}")
                context_parts.append(f"  Status: {issue['status']}, Priority: {issue['priority']}")
                context_parts.append(f"  URL: {issue['url']}")
                if issue.get('description'):
                    context_parts.append(f"  Description: {issue['description'][:200]}...")
                context_parts.append("")
        
        # Slack Messages
        if search_results['slack_messages']:
            context_parts.append("SLACK MESSAGES:")
            for msg in search_results['slack_messages'][:5]:  # Top 3 results
                context_parts.append(f"- Channel: #{msg['channel']}")
                context_parts.append(f"  Message: {msg['text'][:200]}...")
                if msg.get('thread_summary'):
                    context_parts.append(f"  Thread Summary: {msg['thread_summary']}")
                if msg.get('permalink'):
                    context_parts.append(f"  Link: {msg['permalink']}")
                context_parts.append("")
        
        # Confluence Pages
        if search_results['confluence_pages']:
            context_parts.append("CONFLUENCE PAGES:")
            for page in search_results['confluence_pages'][:5]:  # Top 3 results
                context_parts.append(f"- {page['title']}")
                context_parts.append(f"  Space: {page['space']}")
                if page.get('excerpt'):
                    context_parts.append(f"  Excerpt: {page['excerpt'][:200]}...")
                context_parts.append(f"  URL: {page['url']}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_response(self, search_results: Dict) -> str:
        """Generate a fallback response when LLM fails"""
        response_parts = ["I found some relevant information that might help:"]
        
        # Jira Issues
        if search_results['jira_issues']:
            response_parts.append("\n🎫 *Related Jira Issues:*")
            for issue in search_results['jira_issues'][:3]:
                response_parts.append(f"• <{issue['url']}|{issue['key']}>: {issue['summary']}")
                response_parts.append(f"  Status: {issue['status']} | Priority: {issue['priority']}")
        
        # Slack Messages
        if search_results['slack_messages']:
            response_parts.append("\n💬 *Related Slack Discussions:*")
            for msg in search_results['slack_messages'][:3]:
                if msg.get('permalink'):
                    response_parts.append(f"• <{msg['permalink']}|Discussion in #{msg['channel']}>")
                else:
                    response_parts.append(f"• Discussion in #{msg['channel']}")
                response_parts.append(f"  {msg['text'][:150]}...")
                
                # Add thread summary if available
                if msg.get('thread_summary'):
                    response_parts.append(f"  💡 {msg['thread_summary']}")
                elif msg.get('reply_count', 0) > 0:
                    response_parts.append(f"  💬 {msg['reply_count']} replies in thread")
        
        # Confluence Pages
        if search_results['confluence_pages']:
            response_parts.append("\n📄 *Related Documentation:*")
            for page in search_results['confluence_pages'][:3]:
                response_parts.append(f"• <{page['url']}|{page['title']}>")
                response_parts.append(f"  Space: {page['space']}")
        
        if not any([search_results['jira_issues'], search_results['slack_messages'], search_results['confluence_pages']]):
            response_parts.append("\nI couldn't find any directly related content, but you might want to:")
            response_parts.append("• Check with your team members")
            response_parts.append("• Create a new Jira issue if this is a bug or feature request")
            response_parts.append("• Search for more specific keywords")
        
        return "\n".join(response_parts)
    
    def search_and_respond(self, query: str, channel: str) -> Dict:
        """Direct search and response without processing a Slack message"""
        try:
            search_results = self._search_all_platforms(query)
            formatted_response = self._generate_llm_response(query, search_results)
            
            response_result = self.slack_tool.send_message(
                channel=channel,
                text=formatted_response
            )
            
            return {
                'success': True,
                'search_results': search_results,
                'response': formatted_response,
                'slack_response': response_result
            }
            
        except Exception as e:
            logger.error(f"Error in search and respond: {e}")
            return {
                'success': False,
                'error': str(e)
            }
