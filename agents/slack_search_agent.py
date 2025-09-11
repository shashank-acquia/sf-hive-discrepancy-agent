import json
import os
import re
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
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


from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tools.cdp_chat_tool.slack_tool import SlackTool
from tools.cdp_chat_tool.jira_tool import JiraTool
from tools.cdp_chat_tool.confluence_tool import ConfluenceTool

logger = logging.getLogger(__name__)


class AdvancedRelevanceCalculator:

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"SentenceTransformer model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model: {e}")
            self.model = None

    def calculate_score(self, ticket: Dict, query: str, keywords: List[str]) -> float:
        if not self.model:
            return 0.0

        ticket_text = f"{ticket.get('summary', '')}. {ticket.get('description', '')[:500]}"
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        ticket_embedding = self.model.encode(ticket_text, convert_to_tensor=True)
        semantic_score = util.pytorch_cos_sim(query_embedding, ticket_embedding).item()

        summary_lower = ticket.get('summary', '').lower()
        description_lower = (ticket.get('description', '') or '').lower()

        summary_matches = sum(1 for kw in keywords if kw.lower() in summary_lower)
        description_matches = sum(1 for kw in keywords if kw.lower() in description_lower)

        keyword_score = (2 * summary_matches + description_matches) / (
                    2 * len(keywords) + len(keywords)) if keywords else 0

        status_weight = {
            'closed': 1.0, 'resolved': 1.0, 'done': 1.0,
            'in progress': 0.6, 'in review': 0.6,
            'open': 0.3, 'to do': 0.3, 'backlog': 0.1
        }.get(ticket.get('status', '').lower(), 0.2)

        recency_score = 0.0
        try:
            updated_str = ticket.get('updated', '').split('.')[0]
            updated_date = datetime.fromisoformat(updated_str).replace(tzinfo=timezone.utc)
            days_since_update = (datetime.now(timezone.utc) - updated_date).days
            recency_score = max(0, 1 - (days_since_update / 730))
        except (ValueError, TypeError):
            recency_score = 0.3

        metadata_score = (status_weight + recency_score) / 2

        final_score = (
                (semantic_score * 0.5) +
                (keyword_score * 0.3) +
                (metadata_score * 0.2)
        )

        return round(final_score * 100, 2)

class SlackSearchAgent:
    def __init__(self):
        self.slack_tool = SlackTool()
        self.jira_tool = JiraTool()
        self.confluence_tool = ConfluenceTool()

        self.relevance_calculator = AdvancedRelevanceCalculator()
        try:
            self.nltk_stopwords = set(stopwords.words('english'))
        except Exception:
            self.nltk_stopwords = set()
        
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

    def _extract_keywords_with_llm(self, query: str) -> List[str]:
        """
        Uses an LLM to extract key technical terms. This version is robust against
        imperfect JSON responses from the LLM.
        """
        if not self.llm:
            logger.warning("LLM not available. Falling back to simple keyword extraction.")
            tokens = word_tokenize(query.lower())
            return [word for word in tokens if word.isalpha() and word not in self.nltk_stopwords]

        prompt = f"""
        Analyze the user query to identify critical technical keywords for a search.
        Focus on error messages, product names, technologies, and specific concepts.
        Exclude generic words like 'issue', 'problem', 'error'.
        Your response MUST be ONLY a single JSON array of strings and nothing else.

        Query: "{query}"
        """

        try:
            messages = [
                SystemMessage(
                    content="You are an API that returns JSON. You only respond with a JSON array of technical keywords."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            response_content = response.content

            json_match = re.search(r'\[.*\]', response_content, re.DOTALL)

            if not json_match:
                logger.error(f"LLM did not return a parsable JSON array. Response: '{response_content}'")
                raise ValueError("No JSON array found in LLM response")

            extracted_json = json.loads(json_match.group(0))
            logger.info(f"LLM extracted keywords: {extracted_json}")
            return [str(item) for item in extracted_json]

        except Exception as e:
            logger.error(f"LLM keyword extraction failed: {e}. Falling back to simple NLTK method.")
            tokens = word_tokenize(query.lower())
            return [word for word in tokens if word.isalpha() and word not in self.nltk_stopwords and len(word) > 2]
        
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
        """
        A streamlined search process using advanced keyword extraction and relevance ranking.
        """
        results = {
            'jira_issues': [],
            'slack_messages': [],
            'confluence_pages': []
        }

        keywords = self._extract_keywords_with_llm(query)
        search_query_for_jira = " ".join(keywords) if keywords else query

        # 1. Search Slack
        try:
            if self.search_channels and self.search_channels[0]:
                slack_results = self.slack_tool.search_in_channels(
                    query=query,
                    channels=[ch.strip() for ch in self.search_channels if ch.strip()],
                    limit=10
                )
                results['slack_messages'] = slack_results
                logger.info(f"Found {len(slack_results)} relevant Slack messages")
        except Exception as e:
            logger.error(f"Error searching Slack: {e}")

        # 2. Search and Rank Jira
        try:
            jira_candidates = self.jira_tool.search_issues(search_query_for_jira, max_results=25)
            scored_tickets = [
                {**t, 'relevance_score': self.relevance_calculator.calculate_score(t, query, keywords)}
                for t in jira_candidates
            ]
            sorted_tickets = sorted(scored_tickets, key=lambda x: x.get('relevance_score', 0), reverse=True)
            initial_top_tickets = sorted_tickets[:10]
            logger.info("Initial search completed.")
        except Exception as e:
            logger.error(f"Initial Jira search failed: {e}")
            initial_top_tickets = []

        final_jira_results = initial_top_tickets
        if initial_top_tickets and initial_top_tickets[0].get('relevance_score', 0) > 75:
            top_ticket = initial_top_tickets[0]
            logger.info(f"High confidence match found: {top_ticket['key']}. Refining search...")

            top_ticket_summary = top_ticket.get('summary', '')
            new_keywords = [word for word in word_tokenize(top_ticket_summary.lower())
                            if word.isalpha() and word not in self.nltk_stopwords and len(word) > 3]

            combined_keywords = list(set(keywords + new_keywords))
            refined_query = " ".join(combined_keywords)

            try:
                logger.info(f"Refined JQL query with terms: {refined_query}")
                refined_candidates = self.jira_tool.search_issues(refined_query, max_results=15)

                refined_scored = [
                    {**t, 'relevance_score': self.relevance_calculator.calculate_score(t, query, combined_keywords)}
                    for t in refined_candidates
                ]

                combined_tickets = {ticket['key']: ticket for ticket in initial_top_tickets}
                for ticket in refined_scored:
                    combined_tickets[ticket['key']] = ticket

                final_jira_results = sorted(combined_tickets.values(), key=lambda x: x.get('relevance_score', 0),
                                            reverse=True)[:10]
                logger.info("Search refined and results re-ranked.")

            except Exception as e:
                logger.error(f"Refined Jira search failed: {e}")

        results['jira_issues'] = final_jira_results

        # 3. Search Confluence
        try:
            if os.getenv('CONFLUENCE_SERVER'):
                confluence_results = self.confluence_tool.search_similar_content(query, limit=5)
                results['confluence_pages'] = confluence_results
                logger.info(f"Found {len(confluence_results)} relevant Confluence pages")
        except Exception as e:
            logger.warning(f"Confluence search failed: {e}")

        results['platform_insights'] = self._generate_platform_insights(results)
        return results
    
    def _generate_platform_insights(self, search_results: Dict) -> Dict:
        """Generate platform-specific insights based on search results"""
        insights = {}
        
        # JIRA Insights
        jira_issues = search_results.get('jira_issues', [])
        jira_insights = []
        jira_summary = "No JIRA results found"
        
        if jira_issues:
            # Count by status, priority, and project
            status_counts = {}
            priority_counts = {}
            project_counts = {}
            
            for issue in jira_issues:
                status = issue.get('status', 'Unknown')
                priority = issue.get('priority', 'Unknown')
                project = issue.get('project', issue.get('key', 'Unknown').split('-')[0] if issue.get('key') else 'Unknown')
                
                status_counts[status] = status_counts.get(status, 0) + 1
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
                project_counts[project] = project_counts.get(project, 0) + 1
            
            # Calculate detailed metrics
            total_issues = len(jira_issues)
            closed_issues = sum(count for status, count in status_counts.items() if status.lower() in ['closed', 'resolved', 'done'])
            open_issues = sum(count for status, count in status_counts.items() if status.lower() in ['open', 'in progress', 'to do', 'new'])
            high_priority_issues = sum(count for priority, count in priority_counts.items() if priority.lower() in ['high', 'critical', 'blocker'])
            
            jira_insights.append(f"Found {total_issues} relevant JIRA tickets")
            if closed_issues > 0:
                jira_insights.append(f"{closed_issues} tickets are resolved (likely contain solutions)")
            
            # Top status
            if status_counts:
                top_status = max(status_counts.items(), key=lambda x: x[1])
                jira_insights.append(f"Most common status: {top_status[0]} ({top_status[1]} tickets)")
            
            # Top priority
            if priority_counts:
                top_priority = max(priority_counts.items(), key=lambda x: x[1])
                jira_insights.append(f"Most common priority: {top_priority[0]} ({top_priority[1]} tickets)")
            
            # Top project
            if project_counts:
                top_project = max(project_counts.items(), key=lambda x: x[1])
                jira_insights.append(f"Most tickets from project: {top_project[0]} ({top_project[1]} tickets)")
            
            # Check for high relevance tickets
            high_relevance = [issue for issue in jira_issues if issue.get('relevance_score', 0) > 50]
            if high_relevance:
                jira_insights.append(f"{len(high_relevance)} tickets have high relevance scores")
            
            jira_summary = f"Found {total_issues} JIRA tickets, {closed_issues} resolved"
        
        insights['jira'] = {
            'insights': jira_insights,
            'count': len(jira_issues),
            'total_count': total_issues if jira_issues else 0,
            'open_count': open_issues if jira_issues else 0,
            'closed_count': closed_issues if jira_issues else 0,
            'high_priority_count': high_priority_issues if jira_issues else 0,
            'summary': jira_summary
        }
        
        # Slack Insights
        slack_messages = search_results.get('slack_messages', [])
        slack_insights = []
        slack_summary = "No Slack results found"
        
        if slack_messages:
            # Count by channel
            channel_counts = {}
            thread_count = 0
            
            for msg in slack_messages:
                channel = msg.get('channel', 'Unknown')
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
                
                if msg.get('reply_count', 0) > 0 or msg.get('thread_summary'):
                    thread_count += 1
            
            total_messages = len(slack_messages)
            slack_insights.append(f"Found {total_messages} relevant Slack messages")
            
            if thread_count > 0:
                slack_insights.append(f"{thread_count} messages have thread discussions")
            
            # Top channel
            if channel_counts:
                top_channel = max(channel_counts.items(), key=lambda x: x[1])
                slack_insights.append(f"Most messages from #{top_channel[0]} ({top_channel[1]} messages)")
            
            # Check for recent activity
            recent_messages = [msg for msg in slack_messages if msg.get('timestamp', 0) > 0]
            if recent_messages:
                slack_insights.append(f"Found recent discussions about this topic")
            
            slack_summary = f"Found {total_messages} Slack messages across {len(channel_counts)} channels"
        
        insights['slack'] = {
            'insights': slack_insights,
            'count': len(slack_messages),
            'summary': slack_summary
        }
        
        # Confluence Insights
        confluence_pages = search_results.get('confluence_pages', [])
        confluence_insights = []
        confluence_summary = "No Confluence results found"
        
        if confluence_pages:
            # Count by space
            space_counts = {}
            
            for page in confluence_pages:
                space = page.get('space', 'Unknown')
                space_counts[space] = space_counts.get(space, 0) + 1
            
            total_pages = len(confluence_pages)
            confluence_insights.append(f"Found {total_pages} relevant Confluence pages")
            
            # Top space
            if space_counts:
                top_space = max(space_counts.items(), key=lambda x: x[1])
                confluence_insights.append(f"Most pages from space: {top_space[0]} ({top_space[1]} pages)")
            
            confluence_summary = f"Found {total_pages} Confluence pages across {len(space_counts)} spaces"
        
        insights['confluence'] = {
            'insights': confluence_insights,
            'count': len(confluence_pages),
            'summary': confluence_summary
        }
        
        return insights
    
    def _enhanced_jira_search_from_slack(self, query: str, slack_messages: List[Dict]) -> List[Dict]:
        """Enhanced JIRA search based on Slack message content analysis"""
        enhanced_jira_results = []
        
        for msg in slack_messages:
            logger.info(f"Analyzing Slack message from #{msg.get('channel')} at {msg.get('ts')}")
            
            # Extract error details and search JIRA for similar errors
            error_details = self._extract_error_details(msg.get('text', ''))
            if error_details:
                logger.info(f"Extracted error details: {error_details}")
                
                # Search JIRA for tickets containing these error details
                for error_term in error_details:
                    try:
                        jira_tickets = self.jira_tool.get_similar_issues(error_term, max_results=3)
                        for ticket in jira_tickets:
                            ticket['search_strategy'] = f'error_term: {error_term}'
                            ticket['relevance_score'] = self._calculate_relevance(ticket, error_details)
                            enhanced_jira_results.append(ticket)
                    except Exception as e:
                        logger.warning(f"Error searching JIRA for term '{error_term}': {e}")
            
            # Extract job details and search for related tickets
            job_details = self._extract_job_details(msg.get('text', ''))
            if job_details:
                logger.info(f"Extracted job details: {job_details}")
                
                for job_term in job_details:
                    try:
                        jira_tickets = self.jira_tool.get_similar_issues(job_term, max_results=3)
                        for ticket in jira_tickets:
                            ticket['search_strategy'] = f'job_term: {job_term}'
                            ticket['relevance_score'] = self._calculate_relevance(ticket, job_details)
                            enhanced_jira_results.append(ticket)
                    except Exception as e:
                        logger.warning(f"Error searching JIRA for job term '{job_term}': {e}")
        
        return enhanced_jira_results
    
    def _extract_error_details(self, text: str) -> List[str]:
        """Extract specific error details from Slack message"""
        error_terms = []
        
        # Extract specific error messages
        error_patterns = [
            r'error_message\s*([^]]+)',  # error_message content
            r'SnowflakeExecutorException:\s*([^|]+)',  # Snowflake exceptions
            r'Timestamp\s+\'([^\']+)\'\s+is\s+not\s+recognized',  # Timestamp errors
            r'error-code\s+(\d+)',  # Error codes
            r'queryId\s+([a-f0-9-]+)',  # Query IDs
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            error_terms.extend(matches)
        
        # Also extract the main error components
        if 'SF_DW_MAPPER_DEFAULT' in text:
            error_terms.append('SF_DW_MAPPER_DEFAULT')
        
        if 'JCrew' in text:
            error_terms.append('JCrew')
        
        if re.search(r'\b1086\b', text):
            error_terms.append('1086')
        
        return [term.strip() for term in error_terms if term.strip()]
    
    def _extract_job_details(self, text: str) -> List[str]:
        """Extract job/workflow details from Slack message"""
        job_terms = []
        
        # Extract job-related information
        job_patterns = [
            r'Job ID\s*:\s*([^\n]+)',  # Job IDs
            r'SF_DW_MAPPER_DEFAULT',   # Job name
            r'JCrew\s*\((\d+)\)',      # Tenant info
            r'Status Detail:\s*([^\n]+)',  # Status details
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            job_terms.extend(matches)
        
        return [term.strip() for term in job_terms if term.strip()]
    
    def _calculate_relevance(self, ticket: Dict, search_terms: List[str]) -> float:
        """Calculate relevance score based on how many search terms appear in the ticket"""
        score = 0
        ticket_text = (ticket.get('summary', '') + ' ' + ticket.get('description', '')).lower()
        
        for term in search_terms:
            if term.lower() in ticket_text:
                score += 10
        
        # Bonus for closed tickets (likely resolved)
        if ticket.get('status', '').lower() in ['closed', 'resolved', 'done']:
            score += 5
        
        return score
    
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
            context_parts.append("JIRA ISSUES (sorted by relevance):")
            for issue in search_results['jira_issues'][:5]:
                score = issue.get('relevance_score', 0)
                context_parts.append(f"- {issue['key']}: {issue['summary']} (Relevance: {score:.1f}%)")
                context_parts.append(f"  Status: {issue['status']}, URL: {issue['url']}")

                if 'full_details' in issue and issue['full_details'].get('comments'):
                    context_parts.append("  - Relevant Comments:")
                    # Get the last 3 comments, as solutions are often at the end
                    for comment in issue['full_details']['comments'][-3:]:
                        author = comment.get('author', 'Unknown')
                        body = comment.get('body', '').strip()
                        # Sanitize and truncate comment for brevity
                        clean_body = re.sub(r'\{code:.*?\}', '', body, flags=re.DOTALL)
                        context_parts.append(f"    - From {author}: \"{clean_body[:400]}...\"")

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
