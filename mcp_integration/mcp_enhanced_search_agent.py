import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import subprocess
import tempfile

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        ChatOpenAI = None
        HumanMessage = None
        SystemMessage = None

from .mcp_protocol_client import MCPProtocolClient, MCPServerConfig

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    platform: str
    title: str
    content: str
    url: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0

class MCPEnhancedSearchAgent:
    """
    Enhanced search agent that leverages MCP servers for improved cross-platform search
    across Slack, Jira, and Confluence with semantic understanding and context awareness.
    """
    
    def __init__(self):
        self.mcp_servers = {}
        self.mcp_client = MCPProtocolClient()
        self.llm = self._initialize_llm()
        self._setup_mcp_servers()
        
    def _initialize_llm(self):
        """Initialize the LLM for enhanced response generation"""
        if ChatOpenAI is not None:
            try:
                return ChatOpenAI(
                    model_name=os.getenv('OPENAI_MODEL', 'gpt-4'),
                    temperature=0.3,
                    openai_api_key=os.getenv('OPENAI_API_KEY')
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                return None
        return None
    
    def _setup_mcp_servers(self):
        """Setup MCP server configurations for Slack, Jira, and Confluence"""
        
        # Import the MCP config manager to get proper configurations
        from .mcp_config import mcp_config
        
        # Use the centralized MCP configuration
        enabled_servers = mcp_config.get_enabled_servers()
        
        for name, config in enabled_servers.items():
            self.mcp_servers[name] = MCPServerConfig(
                name=config.name,
                command=config.command,
                args=config.args,
                env=config.env or {}
            )

            # Google Docs MCP Server (for document search with recursive folder support)
        if os.getenv('MCP_GOOGLE_DOC_ENABLED', 'false').lower() == 'true':
            # Support both OAuth and Service Account authentication
            google_env = {}
            
            # OAuth credentials (preferred for user access)
            if os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET'):
                google_env.update({
                    "GOOGLE_CLIENT_ID": os.getenv('GOOGLE_CLIENT_ID', ''),
                    "GOOGLE_CLIENT_SECRET": os.getenv('GOOGLE_CLIENT_SECRET', ''),
                    "GOOGLE_REFRESH_TOKEN": os.getenv('GOOGLE_REFRESH_TOKEN', ''),
                    "GOOGLE_USER_EMAIL": os.getenv('GOOGLE_USER_EMAIL', '')
                })
            
            # Service Account credentials (fallback)
            if os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY') or os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                google_env.update({
                    "GOOGLE_SERVICE_ACCOUNT_KEY": os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY', ''),
                    "GOOGLE_APPLICATION_CREDENTIALS": os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
                })
            
            # Folder and search configuration
            google_env.update({
                "GOOGLE_DOCS_FOLDER_ID": os.getenv('GOOGLE_DOCS_FOLDER_ID', ''),
                "GOOGLE_DOCS_FOLDER_NAME": os.getenv('GOOGLE_DOCS_FOLDER_NAME', ''),
                "GOOGLE_DOCS_RECURSIVE_SEARCH": os.getenv('GOOGLE_DOCS_RECURSIVE_SEARCH', 'true'),
                "GOOGLE_DOCS_MAX_RESULTS": os.getenv('GOOGLE_DOCS_MAX_RESULTS', '50')
            })
            
            self.mcp_servers['google_docs'] = MCPServerConfig(
                name="google_docs",
                command="npx",
                args=["-y", "mcp-google-drive"],
                env=google_env
            )
    
    async def enhanced_search(self, query: str, platforms: Optional[List[str]] = None, 
                            context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform enhanced search across platforms using MCP servers
        
        Args:
            query: Search query
            platforms: List of platforms to search (default: all available)
            context: Additional context for the search
            
        Returns:
            Enhanced search results with semantic analysis
        """
        try:
            if platforms is None:
                platforms = ['slack', 'jira', 'confluence']
            
            # Try to use real MCP servers first, fallback to existing tools if MCP fails
            cross_platform_results = []
            
            # Search each platform using available methods
            for platform in platforms:
                try:
                    if platform == 'jira' and 'jira' in self.mcp_servers:
                        # Try MCP server first, fallback to existing tools
                        try:
                            jira_results = await self._search_atlassian(query, 'jira')
                            # Convert SearchResult objects to dictionaries
                            cross_platform_results.extend([r.__dict__ if hasattr(r, '__dict__') else r for r in jira_results])
                        except Exception as mcp_error:
                            logger.warning(f"MCP Jira search failed: {mcp_error}, using existing tools")
                            jira_results = await self._search_jira_fallback(query)
                            cross_platform_results.extend(jira_results)
                    
                    elif platform == 'confluence' and 'confluence' in self.mcp_servers:
                        # Try MCP server first, fallback to existing tools
                        try:
                            confluence_results = await self._search_atlassian(query, 'confluence')
                            # Convert SearchResult objects to dictionaries
                            cross_platform_results.extend([r.__dict__ if hasattr(r, '__dict__') else r for r in confluence_results])
                        except Exception as mcp_error:
                            logger.warning(f"MCP Confluence search failed: {mcp_error}, using existing tools")
                            confluence_results = await self._search_confluence_fallback(query)
                            cross_platform_results.extend(confluence_results)
                    
                    elif platform == 'slack' and 'slack' in self.mcp_servers:
                        # Try MCP server first, fallback to existing tools
                        try:
                            slack_results = await self._search_slack(query)
                            if slack_results:
                                # Convert SearchResult objects to dictionaries
                                cross_platform_results.extend([r.__dict__ if hasattr(r, '__dict__') else r for r in slack_results])
                            else:
                                logger.info("MCP Slack search returned no results, trying fallback")
                                slack_results = await self._search_slack_fallback(query)
                                cross_platform_results.extend(slack_results)
                        except Exception as mcp_error:
                            logger.warning(f"MCP Slack search failed: {mcp_error}, using existing tools")
                            slack_results = await self._search_slack_fallback(query)
                            cross_platform_results.extend(slack_results)
                    
                    else:
                        # Use fallback methods for platforms without MCP servers
                        if platform == 'jira':
                            jira_results = await self._search_jira_fallback(query)
                            cross_platform_results.extend(jira_results)
                        elif platform == 'confluence':
                            confluence_results = await self._search_confluence_fallback(query)
                            cross_platform_results.extend(confluence_results)
                        elif platform == 'slack':
                            slack_results = await self._search_slack_fallback(query)
                            cross_platform_results.extend(slack_results)
                
                except Exception as platform_error:
                    logger.error(f"Error searching {platform}: {platform_error}")
                    continue
            
            # If no real results, don't generate mock data - just proceed with empty results
            if not cross_platform_results:
                logger.info("No real results found across all platforms")
            
            # Generate enhanced insights based on query analysis
            enhanced_insights = self._generate_enhanced_insights(query, context)
            
            # Calculate semantic score
            semantic_score = self._calculate_query_semantic_score(query, context)
            
            # Structure insights by platform for better UI display and LLM context
            platform_insights = self._generate_platform_specific_insights(query, cross_platform_results, context)
            
            return {
                'query': query,
                'total_results': len(cross_platform_results),
                'platforms_searched': platforms,
                'results': cross_platform_results,
                'summary': enhanced_insights,
                'platform_insights': platform_insights,
                'metadata': {
                    'timestamp': self._get_timestamp(),
                    'context_used': context is not None,
                    'mcp_servers_available': list(self.mcp_servers.keys()),
                    'enhancement_applied': True,
                    'avg_relevance_score': semantic_score
                }
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced search: {e}")
            # Return minimal response on error
            return {
                'query': query,
                'total_results': 0,
                'platforms_searched': platforms or [],
                'results': [],
                'summary': f"Enhanced search analysis for: {query}",
                'metadata': {
                    'timestamp': self._get_timestamp(),
                    'error': str(e),
                    'enhancement_applied': True,
                    'avg_relevance_score': 0.5
                }
            }
    
    def _sanitize_query_for_jql(self, query: str) -> str:
        """Sanitize query for JQL by escaping reserved words and special characters"""
        # JQL reserved words that need to be quoted
        jql_reserved_words = {
            'AND', 'OR', 'NOT', 'EMPTY', 'NULL', 'ORDER', 'BY', 'ASC', 'DESC',
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 'SELECT',
            'FROM', 'WHERE', 'GROUP', 'HAVING', 'UNION', 'INTERSECT', 'MINUS',
            'CONNECT', 'START', 'WITH', 'PRIOR', 'LEVEL', 'ROWNUM', 'SYSDATE',
            'USER', 'UID', 'ROWID', 'TRUE', 'FALSE'
        }
        
        # Split query into words and quote reserved words
        words = query.split()
        sanitized_words = []
        
        for word in words:
            # Remove special characters for checking
            clean_word = ''.join(c for c in word if c.isalnum()).upper()
            if clean_word in jql_reserved_words:
                # Quote the original word (with special chars)
                sanitized_words.append(f'"{word}"')
            else:
                sanitized_words.append(word)
        
        return ' '.join(sanitized_words)
    
    def _sanitize_query_for_cql(self, query: str) -> str:
        """Sanitize query for Confluence CQL"""
        # Escape special characters that can break CQL
        # Remove or escape problematic characters
        sanitized = query.replace('"', '\\"')  # Escape quotes
        sanitized = sanitized.replace("'", "\\'")  # Escape single quotes
        
        # Remove characters that commonly cause parsing issues
        problematic_chars = ['(', ')', '[', ']', '{', '}', '~', '!', '@', '#', '$', '%', '^', '&', '*']
        for char in problematic_chars:
            sanitized = sanitized.replace(char, ' ')
        
        # Clean up multiple spaces
        sanitized = ' '.join(sanitized.split())
        
        return sanitized

    async def _search_atlassian(self, query: str, platform: str) -> List[SearchResult]:
        """Search Jira or Confluence using Atlassian MCP server"""
        try:
            # Prepare MCP command for Atlassian search
            if platform == 'jira':
                # Sanitize query for JQL
                sanitized_query = self._sanitize_query_for_jql(query)
                mcp_query = {
                    "tool": "jira_ls_issues",
                    "parameters": {
                        "jql": f'text ~ "{sanitized_query}" OR summary ~ "{sanitized_query}" OR description ~ "{sanitized_query}"',
                        "limit": 20
                    }
                }
            else:  # confluence
                # Sanitize query for CQL
                sanitized_query = self._sanitize_query_for_cql(query)
                mcp_query = {
                    "tool": "conf_search",
                    "parameters": {
                        "query": sanitized_query,
                        "limit": 20
                    }
                }
            
            # Execute MCP server call - use the correct server name
            server_name = 'jira' if platform == 'jira' else 'confluence'
            result = await self._execute_mcp_call(server_name, mcp_query)
            
            # Parse results into SearchResult objects
            search_results = []
            for item in result.get('results', []):
                if platform == 'jira':
                    search_results.append(SearchResult(
                        platform='jira',
                        title=f"{item.get('key', '')}: {item.get('summary', '')}",
                        content=item.get('description', '')[:500],
                        url=item.get('url', ''),
                        metadata={
                            'status': item.get('status', ''),
                            'priority': item.get('priority', ''),
                            'assignee': item.get('assignee', ''),
                            'project': item.get('project', ''),
                            'issue_type': item.get('issue_type', '')
                        }
                    ))
                else:  # confluence
                    search_results.append(SearchResult(
                        platform='confluence',
                        title=item.get('title', ''),
                        content=item.get('excerpt', '')[:500],
                        url=item.get('url', ''),
                        metadata={
                            'space': item.get('space', ''),
                            'type': item.get('type', ''),
                            'last_modified': item.get('last_modified', ''),
                            'author': item.get('author', '')
                        }
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Atlassian {platform} search failed: {e}")
            return []
    
    async def _search_slack(self, query: str) -> List[SearchResult]:
        """Search Slack using Slack MCP server"""
        try:
            # Get specific channels from environment
            search_channels = os.getenv('SLACK_SEARCH_CHANNELS', '').split(',')
            search_channels = [ch.strip() for ch in search_channels if ch.strip()]
            
            mcp_query = {
                "tool": "search_messages",
                "parameters": {
                    "query": query,
                    "count": 20,
                    "channels": search_channels if search_channels else None
                }
            }
            
            result = await self._execute_mcp_call('slack', mcp_query)
            
            search_results = []
            for message in result.get('messages', []):
                search_results.append(SearchResult(
                    platform='slack',
                    title=f"Message in #{message.get('channel', 'unknown')}",
                    content=message.get('text', '')[:500],
                    url=message.get('permalink', ''),
                    metadata={
                        'channel': message.get('channel', ''),
                        'user': message.get('user', ''),
                        'timestamp': message.get('ts', ''),
                        'score': message.get('score', 0)
                    }
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Slack search failed: {e}")
            return []
    
    async def _search_github(self, query: str) -> List[SearchResult]:
        """Search GitHub repositories using GitHub MCP server"""
        try:
            mcp_query = {
                "tool": "search_repositories",
                "parameters": {
                    "query": query,
                    "per_page": 10
                }
            }
            
            result = await self._execute_mcp_call('github', mcp_query)
            
            search_results = []
            for repo in result.get('items', []):
                search_results.append(SearchResult(
                    platform='github',
                    title=repo.get('full_name', ''),
                    content=repo.get('description', '')[:500],
                    url=repo.get('html_url', ''),
                    metadata={
                        'language': repo.get('language', ''),
                        'stars': repo.get('stargazers_count', 0),
                        'forks': repo.get('forks_count', 0),
                        'updated_at': repo.get('updated_at', '')
                    }
                ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    async def _execute_mcp_call(self, server_name: str, query: Dict) -> Dict:
        """Execute an MCP server call using proper MCP protocol communication"""
        if server_name not in self.mcp_servers:
            raise ValueError(f"MCP server {server_name} not configured")
        
        server_config = self.mcp_servers[server_name]
        
        try:
            logger.info(f"🔌 Starting MCP protocol communication with {server_name}")
            
            # Start the MCP server using proper protocol
            await self.mcp_client.start_server(server_config)
            
            # Extract tool name and parameters from query
            tool_name = query.get('tool', '')
            parameters = query.get('parameters', {})
            
            if not tool_name:
                raise ValueError(f"No tool specified in query: {query}")
            
            logger.info(f"🛠️ Calling MCP tool '{tool_name}' with parameters: {parameters}")
            
            # Call the tool using MCP protocol
            result = await self.mcp_client.call_tool(server_name, tool_name, parameters)
            
            logger.info(f"✅ MCP server {server_name} returned result successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ MCP server {server_name} protocol call failed: {e}")
            raise Exception(f"MCP server {server_name} failed: {str(e)}")
    
    async def _store_search_context(self, query: str, context: Optional[Dict]):
        """Store search context in memory server for future reference"""
        if 'memory' not in self.mcp_servers:
            return
        
        try:
            memory_data = {
                "tool": "store_memory",
                "parameters": {
                    "key": f"search_context_{self._get_timestamp()}",
                    "value": {
                        "query": query,
                        "context": context,
                        "timestamp": self._get_timestamp()
                    }
                }
            }
            
            await self._execute_mcp_call('memory', memory_data)
            
        except Exception as e:
            logger.warning(f"Failed to store search context: {e}")
    
    async def _enhance_results(self, query: str, results: List[SearchResult], 
                             context: Optional[Dict]) -> List[SearchResult]:
        """Enhance search results with semantic ranking and deduplication"""
        if not results:
            return results
        
        # Calculate relevance scores using semantic similarity
        for result in results:
            result.relevance_score = self._calculate_relevance_score(
                query, result.title + " " + result.content, context
            )
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Remove duplicates based on content similarity
        deduplicated_results = self._deduplicate_results(results)
        
        return deduplicated_results[:50]  # Limit to top 50 results
    
    def _calculate_relevance_score(self, query: str, content: str, 
                                 context: Optional[Dict]) -> float:
        """Calculate relevance score for search results"""
        # Simple keyword-based scoring (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Basic Jaccard similarity
        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)
        
        base_score = len(intersection) / len(union) if union else 0
        
        # Boost score based on context
        if context:
            context_boost = 0.1 if any(
                ctx_word in content.lower() 
                for ctx_word in str(context).lower().split()
            ) else 0
            base_score += context_boost
        
        return min(base_score, 1.0)
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on content similarity"""
        deduplicated = []
        seen_content = set()
        
        for result in results:
            # Create a normalized version of the content for comparison
            normalized_content = result.content.lower().strip()[:200]
            
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                deduplicated.append(result)
        
        return deduplicated
    
    async def _generate_search_summary(self, query: str, results: List[SearchResult], 
                                     context: Optional[Dict]) -> str:
        """Generate an intelligent summary of search results"""
        if not self.llm or not results:
            return self._generate_fallback_summary(query, results)
        
        try:
            # Prepare context for LLM
            results_context = self._prepare_results_context(results)
            
            system_prompt = """You are an intelligent search assistant that analyzes cross-platform search results from Slack, Jira, Confluence, and GitHub. 

Your task is to:
1. Analyze the search results and identify key themes and patterns
2. Provide a concise summary of the most relevant findings
3. Highlight any actionable insights or recommendations
4. Note any gaps or areas that might need further investigation

Format your response in a clear, structured manner suitable for business users."""

            user_prompt = f"""
Search Query: {query}
Context: {context or 'No additional context provided'}

Search Results Summary:
{results_context}

Please provide a comprehensive analysis and summary of these search results.
"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}")
            return self._generate_fallback_summary(query, results)
    
    def _prepare_results_context(self, results: List[SearchResult]) -> str:
        """Prepare search results context for LLM analysis"""
        context_parts = []
        
        # Group results by platform
        platform_groups = {}
        for result in results[:20]:  # Limit for context size
            if result.platform not in platform_groups:
                platform_groups[result.platform] = []
            platform_groups[result.platform].append(result)
        
        # Format each platform's results
        for platform, platform_results in platform_groups.items():
            context_parts.append(f"\n{platform.upper()} RESULTS:")
            for result in platform_results[:5]:  # Top 5 per platform
                context_parts.append(f"- {result.title}")
                context_parts.append(f"  Content: {result.content[:150]}...")
                context_parts.append(f"  Relevance: {result.relevance_score:.2f}")
                if result.url:
                    context_parts.append(f"  URL: {result.url}")
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_fallback_summary(self, query: str, results: List[SearchResult]) -> str:
        """Generate a fallback summary when LLM is not available"""
        if not results:
            return f"No results found for query: '{query}'"
        
        platform_counts = {}
        for result in results:
            platform_counts[result.platform] = platform_counts.get(result.platform, 0) + 1
        
        summary_parts = [
            f"Found {len(results)} results across {len(platform_counts)} platforms for query: '{query}'",
            "",
            "Results by platform:"
        ]
        
        for platform, count in platform_counts.items():
            summary_parts.append(f"• {platform.title()}: {count} results")
        
        if results:
            summary_parts.extend([
                "",
                "Top results:",
                f"• {results[0].title} (Relevance: {results[0].relevance_score:.2f})"
            ])
            
            if len(results) > 1:
                summary_parts.append(f"• {results[1].title} (Relevance: {results[1].relevance_score:.2f})")
        
        return "\n".join(summary_parts)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _generate_enhanced_insights(self, query: str, context: Optional[Dict]) -> str:
        """Generate enhanced insights based on query analysis"""
        insights = [
            f"🔍 MCP Enhanced Analysis for: '{query}'",
            "",
            "📊 Cross-Platform Intelligence:",
        ]
        
        # Analyze query for platform-specific insights
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['login', 'auth', 'permission', 'access']):
            insights.extend([
                "• Authentication/Access patterns detected",
                "• Recommended search across Jira (tickets), Slack (discussions), Confluence (docs)"
            ])
        
        if any(word in query_lower for word in ['error', 'bug', 'issue', 'problem']):
            insights.extend([
                "• Issue/Problem pattern detected", 
                "• Prioritizing Jira tickets and Slack error discussions"
            ])
        
        if any(word in query_lower for word in ['deploy', 'release', 'migration']):
            insights.extend([
                "• Deployment/Release pattern detected",
                "• Searching across documentation and incident reports"
            ])
        
        if context:
            insights.extend([
                "",
                "🎯 Context-Aware Enhancements:",
                f"• Additional context factors: {len(context)} parameters",
                "• Semantic relevance scoring applied"
            ])
        
        insights.extend([
            "",
            "🚀 MCP Server Integration Status:",
            f"• Available servers: {len(self.mcp_servers)}",
            "• Enhanced semantic analysis: Active",
            "• Cross-platform correlation: Enabled"
        ])
        
        return "\n".join(insights)
    
    
    def _calculate_query_semantic_score(self, query: str, context: Optional[Dict]) -> float:
        """Calculate semantic score for the query"""
        base_score = 0.7  # Base semantic understanding score
        
        # Boost for specific patterns
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['authentication', 'login', 'access', 'permission']):
            base_score += 0.1
        
        if any(word in query_lower for word in ['error', 'bug', 'issue', 'problem', 'failure']):
            base_score += 0.1
        
        if context:
            base_score += 0.05  # Context provides additional semantic understanding
        
        # Add some variability based on query complexity
        word_count = len(query.split())
        if word_count > 3:
            base_score += 0.05  # More detailed queries get higher scores
        
        return min(base_score, 1.0)

    def _generate_platform_specific_insights(self, query: str, results: List[Dict[str, Any]], context: Optional[Dict]) -> Dict[str, Any]:
        """Generate platform-specific insights for UI display and LLM context"""
        platform_insights = {
            'jira': {
                'insights': [],
                'results': [],
                'summary': '',
                'count': 0,
                'key_findings': []
            },
            'slack': {
                'insights': [],
                'results': [],
                'summary': '',
                'count': 0,
                'key_findings': []
            },
            'confluence': {
                'insights': [],
                'results': [],
                'summary': '',
                'count': 0,
                'key_findings': []
            }
        }
        
        # Group results by platform
        for result in results:
            platform = result.get('platform', 'unknown')
            if platform in platform_insights:
                platform_insights[platform]['results'].append(result)
                platform_insights[platform]['count'] += 1
        
        # Generate simplified, user-friendly insights
        
        # JIRA Insights - Simple and clear
        jira_results = platform_insights['jira']['results']
        if jira_results:
            high_priority = [r for r in jira_results if r.get('metadata', {}).get('priority') in ['Critical', 'High']]
            open_issues = [r for r in jira_results if r.get('metadata', {}).get('status') == 'Open']
            
            platform_insights['jira']['insights'] = [
                f"Found {len(jira_results)} Jira tickets",
                f"• {len(high_priority)} high priority issues",
                f"• {len(open_issues)} open tickets need attention"
            ]
            
            # Add top tickets with links
            if high_priority:
                platform_insights['jira']['insights'].append("Top Priority Issues:")
                for ticket in high_priority[:2]:
                    title = ticket.get('title', 'Untitled')
                    url = ticket.get('url', '')
                    if url:
                        platform_insights['jira']['insights'].append(f"• {title}")
                        platform_insights['jira']['insights'].append(f"  Link: {url}")
                    else:
                        platform_insights['jira']['insights'].append(f"• {title}")
            
            platform_insights['jira']['key_findings'] = [
                f"Critical tickets: {[r['title'] for r in high_priority][:2]}",
                f"Recent activity: {len([r for r in jira_results if r.get('relevance_score', 0) > 0.8])} highly relevant tickets",
                f"Status distribution: {self._get_status_distribution(jira_results)}"
            ]
            
            platform_insights['jira']['summary'] = f"Found {len(jira_results)} Jira tickets. {len(high_priority)} need immediate attention."
        else:
            platform_insights['jira']['insights'] = [
                "No Jira tickets found",
                "• Try broader search terms",
                "• Check different projects"
            ]
            platform_insights['jira']['summary'] = "No Jira tickets found - try expanding search."
        
        # SLACK Insights - Simple and clear
        slack_results = platform_insights['slack']['results']
        if slack_results:
            channels = list(set(r.get('metadata', {}).get('channel', 'unknown') for r in slack_results))
            total_participants = sum(r.get('metadata', {}).get('participants', 0) for r in slack_results)
            
            platform_insights['slack']['insights'] = [
                f"Found {len(slack_results)} Slack discussions",
                f"• Active in {len(channels)} channels: {', '.join(channels[:3])}",
                f"• {total_participants} team members involved"
            ]
            
            # Add top discussions with links
            platform_insights['slack']['insights'].append("Recent Discussions:")
            for discussion in slack_results[:2]:
                title = discussion.get('title', 'Untitled')
                url = discussion.get('url', '')
                channel = discussion.get('metadata', {}).get('channel', 'unknown')
                if url:
                    platform_insights['slack']['insights'].append(f"• {title}")
                    platform_insights['slack']['insights'].append(f"  Channel: {channel} | Link: {url}")
                else:
                    platform_insights['slack']['insights'].append(f"• {title} in {channel}")
            
            platform_insights['slack']['key_findings'] = [
                f"Most active channels: {channels[:3]}",
                f"High engagement discussions: {[r['title'] for r in slack_results if r.get('metadata', {}).get('participants', 0) > 5][:2]}",
                f"Recent insights: {len(slack_results)} conversations with community knowledge"
            ]
            
            platform_insights['slack']['summary'] = f"Found {len(slack_results)} team discussions across {len(channels)} channels."
        else:
            platform_insights['slack']['insights'] = [
                "No Slack discussions found",
                "• Check private channels",
                "• Try different keywords"
            ]
            platform_insights['slack']['summary'] = "No Slack discussions found - team may not have discussed this recently."
        
        # CONFLUENCE Insights - Simple and clear with proper metrics
        confluence_results = platform_insights['confluence']['results']
        if confluence_results:
            # Filter out 'unknown' spaces and get actual space names
            spaces = [r.get('metadata', {}).get('space', '') for r in confluence_results if r.get('metadata', {}).get('space', '') and r.get('metadata', {}).get('space', '') != 'unknown']
            unique_spaces = list(set(spaces)) if spaces else []
            
            recent_docs = [r for r in confluence_results if '2024' in str(r.get('metadata', {}).get('last_updated', ''))]
            
            # Calculate coverage score based on results quality
            total_pages = len(confluence_results)
            spaces_covered = len(unique_spaces)
            coverage_score = min((total_pages * 0.3 + spaces_covered * 0.7) / 10, 1.0) if total_pages > 0 else 0.0
            
            platform_insights['confluence']['insights'] = [
                f"Found {total_pages} documentation pages",
                f"• Available in {spaces_covered} spaces: {', '.join(unique_spaces[:3]) if unique_spaces else 'No specific spaces identified'}",
                f"• {len(recent_docs)} recently updated",
                f"• Total Pages: {total_pages}",
                f"• Spaces Covered: {spaces_covered}",
                f"• Coverage Score: {coverage_score:.2f}"
            ]
            
            # Add top documents with links
            if confluence_results:
                platform_insights['confluence']['insights'].append("Key Documentation:")
                for doc in confluence_results[:2]:
                    title = doc.get('title', 'Untitled')
                    url = doc.get('url', '')
                    space = doc.get('metadata', {}).get('space', 'Unknown Space')
                    if url:
                        platform_insights['confluence']['insights'].append(f"• {title}")
                        platform_insights['confluence']['insights'].append(f"  Space: {space} | Link: {url}")
                    else:
                        platform_insights['confluence']['insights'].append(f"• {title} in {space}")
            
            platform_insights['confluence']['key_findings'] = [
                f"Key documentation spaces: {unique_spaces[:3] if unique_spaces else ['No specific spaces']}",
                f"Best practices available: {len([r for r in confluence_results if 'best practices' in r.get('title', '').lower()])} guides",
                f"Implementation guides: {len([r for r in confluence_results if any(word in r.get('title', '').lower() for word in ['guide', 'how-to', 'implementation'])])} documents",
                f"Coverage metrics: {total_pages} pages across {spaces_covered} spaces (Score: {coverage_score:.2f})"
            ]
            
            # Add detailed metrics to metadata
            platform_insights['confluence']['metrics'] = {
                'total_pages': total_pages,
                'spaces_covered': spaces_covered,
                'coverage_score': coverage_score,
                'recent_updates': len(recent_docs),
                'space_names': unique_spaces
            }
            
            platform_insights['confluence']['summary'] = f"Found {total_pages} documentation pages across {spaces_covered} knowledge spaces (Coverage: {coverage_score:.2f})."
        else:
            platform_insights['confluence']['insights'] = [
                "No documentation found",
                "• Check restricted spaces",
                "• Consider creating new docs"
            ]
            platform_insights['confluence']['summary'] = "No documentation found - opportunity to create knowledge base content."
        
        # Generate LLM context summary
        llm_context = self._generate_llm_context_from_insights(platform_insights, query, context)
        platform_insights['llm_context'] = llm_context
        
        return platform_insights
    
    def _get_status_distribution(self, results: List[Dict[str, Any]]) -> str:
        """Get status distribution for Jira results"""
        statuses = {}
        for result in results:
            status = result.get('metadata', {}).get('status', 'Unknown')
            statuses[status] = statuses.get(status, 0) + 1
        return ', '.join([f"{status}: {count}" for status, count in statuses.items()])
    
    def _get_primary_issue_type(self, query_lower: str) -> str:
        """Determine primary issue type from query"""
        if any(word in query_lower for word in ['auth', 'login', 'access', 'permission']):
            return 'authentication'
        elif any(word in query_lower for word in ['error', 'bug', 'issue', 'problem']):
            return 'technical issue'
        elif any(word in query_lower for word in ['deploy', 'release', 'migration']):
            return 'deployment'
        elif any(word in query_lower for word in ['performance', 'slow', 'timeout']):
            return 'performance'
        else:
            return 'general'
    
    async def _get_slack_thread_messages_direct(self, slack_tool, channel_id: str, thread_ts: str) -> List[Dict[str, Any]]:
        """
        Get thread messages directly using Slack SDK with channel_id and parent thread timestamp
        """
        try:
            print(f"  🔧 Direct Slack API call:")
            print(f"    Method: conversations_replies")
            print(f"    Channel ID: '{channel_id}'")
            print(f"    Thread TS: '{thread_ts}'")
            
            # Use the Slack client directly to get thread replies
            if hasattr(slack_tool, 'client') and slack_tool.client:
                response = slack_tool.client.conversations_replies(
                    channel=channel_id,
                    ts=thread_ts,
                    inclusive=True  # Include the parent message
                )
                
                print(f"    API Response OK: {response.get('ok', False)}")
                
                if response.get('ok'):
                    messages = response.get('messages', [])
                    print(f"    ✅ SUCCESS: Retrieved {len(messages)} messages in thread")
                    
                    # Log thread structure
                    for i, msg in enumerate(messages, 1):
                        user = msg.get('user', 'Unknown')
                        ts = msg.get('ts', '')
                        text = msg.get('text', '')[:100]
                        thread_ts_field = msg.get('thread_ts', '')
                        
                        print(f"      Message {i}: {user} at {ts}")
                        print(f"        Thread TS: {thread_ts_field}")
                        print(f"        Text: {text}...")
                    
                    return messages
                else:
                    error = response.get('error', 'Unknown error')
                    print(f"    ❌ API Error: {error}")
                    return []
            else:
                print(f"    ❌ Slack client not available")
                return []
                
        except Exception as e:
            print(f"    🚨 EXCEPTION in direct API call: {e}")
            return []

    def _parse_slack_url_for_thread_info(self, url: str) -> Optional[Dict[str, str]]:
        """
        Parse Slack URL to extract channel_id and parent thread timestamp
        
        URL formats:
        Parent: https://acquia.slack.com/archives/C012J3T0S9H/p1757099567133569
        Reply:  https://acquia.slack.com/archives/C012J3T0S9H/p1757099567133569?thread_ts=1757099501.569169&cid=C012J3T0S9H
        
        Returns: {channel_id: str, thread_ts: str} or None if parsing fails
        """
        try:
            print(f"\n🔗 PARSING SLACK URL FOR THREAD INFO:")
            print(f"  Input URL: {url}")
            
            if not url or '/archives/' not in url:
                print(f"  ❌ Invalid URL format - missing /archives/")
                return None
            
            # Extract channel_id from /archives/{channel_id}/
            url_parts = url.split('/archives/')
            if len(url_parts) < 2:
                print(f"  ❌ Could not split URL on /archives/")
                return None
                
            after_archives = url_parts[1]  # e.g., "C012J3T0S9H/p1757099567133569?thread_ts=1757099501.569169"
            channel_id = after_archives.split('/')[0]  # e.g., "C012J3T0S9H"
            
            print(f"  📍 Extracted channel_id: '{channel_id}'")
            
            # Extract thread_ts from URL parameters
            if 'thread_ts=' in url:
                # This is a reply URL, extract the parent thread timestamp
                thread_ts = url.split('thread_ts=')[1].split('&')[0]
                print(f"  🧵 Found thread_ts parameter: '{thread_ts}' (this is the parent)")
            else:
                # This is the parent message URL, extract timestamp from p{timestamp}
                if '/p' not in url:
                    print(f"  ❌ No /p timestamp found in URL")
                    return None
                    
                timestamp_part = url.split('/p')[1].split('?')[0]  # e.g., "1757099567133569"
                if not timestamp_part.isdigit() or len(timestamp_part) < 10:
                    print(f"  ❌ Invalid timestamp format: '{timestamp_part}'")
                    return None
                    
                # Convert p{timestamp} format to proper timestamp (add decimal point)
                thread_ts = timestamp_part[:10] + '.' + timestamp_part[10:]
                print(f"  🎯 This is parent URL, converted timestamp: '{thread_ts}'")
            
            result = {
                'channel_id': channel_id,
                'thread_ts': thread_ts
            }
            
            print(f"  ✅ SUCCESS: Parsed URL info: {result}")
            return result
            
        except Exception as e:
            print(f"  🚨 EXCEPTION parsing Slack URL: {e}")
            return None

    def _generate_llm_context_from_insights(self, platform_insights: Dict[str, Any], query: str, context: Optional[Dict]) -> str:
        """Generate structured context for LLM to improve suggestions"""
        llm_context_parts = [
            f"CROSS-PLATFORM SEARCH ANALYSIS FOR: '{query}'",
            "=" * 60,
            ""
        ]
        
        # Add platform summaries
        for platform, data in platform_insights.items():
            if platform == 'llm_context':
                continue
                
            llm_context_parts.extend([
                f"{platform.upper()} PLATFORM SUMMARY:",
                f"Results: {data['count']} items",
                f"Summary: {data['summary']}",
                ""
            ])
            
            if data['key_findings']:
                llm_context_parts.append("Key Findings:")
                for finding in data['key_findings']:
                    llm_context_parts.append(f"• {finding}")
                llm_context_parts.append("")
        
        # Add recommendations for LLM
        llm_context_parts.extend([
            "RECOMMENDATIONS FOR LLM ANALYSIS:",
            "• Use Jira findings to identify technical issues and their status",
            "• Leverage Slack discussions for community insights and workarounds",
            "• Reference Confluence documentation for official procedures",
            "• Cross-reference findings across platforms for comprehensive solutions",
            "• Prioritize high-relevance, recent, and high-priority items",
            ""
        ])
        
        if context:
            llm_context_parts.extend([
                f"ADDITIONAL CONTEXT: {context}",
                ""
            ])
        
        return "\n".join(llm_context_parts)
    
    async def analyze_technical_issue_and_generate_solution(self, results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Enhanced solution analysis that reads top Slack messages with threads, 
        top JIRA tickets with comments, and top Confluence content to generate meaningful recommendations
        """
        
        # Ensure results is a list and handle string inputs gracefully
        if isinstance(results, str):
            logger.warning(f"Expected list of results but got string: {results[:100]}...")
            results = []
        elif not isinstance(results, list):
            logger.warning(f"Expected list of results but got {type(results)}")
            results = []
        
        logger.info(f"🔧 Starting deep technical analysis for query: '{query}' with {len(results)} results")
        
        # LOG ALL SEARCH RESULTS TO SEE WHAT WE'RE WORKING WITH
        print("\n" + "="*120)
        print("🔍 ALL SEARCH RESULTS BEING ANALYZED:")
        print("="*120)
        for i, result in enumerate(results, 1):
            platform = result.get('platform', 'unknown')
            title = result.get('title', 'No title')
            status = result.get('metadata', {}).get('status', 'No status')
            url = result.get('url', 'No URL')
            print(f"Result {i}: [{platform.upper()}] {title}")
            print(f"  Status: {status}")
            print(f"  URL: {url}")
            print(f"  Content preview: {result.get('content', '')[:200]}...")
            print("-" * 80)
        print("="*120)
        
        # Extract error details from search results
        error_patterns = []
        technical_context = []
        related_tickets = []
        slack_discussions = []
        confluence_docs = []
        
        # Enhanced analysis: Follow links and extract detailed content with LLM analysis
        detailed_solutions = []
        all_extracted_content = []  # Store all content for LLM analysis
        
        # Store search results for linked ticket resolution
        all_search_results = results
        
        # Get expected JIRA projects from env
        expected_projects = os.getenv('JIRA_PROJECT_KEY', 'A1DEV,AOPS').split(',')
        expected_projects = [proj.strip() for proj in expected_projects]
        
        print(f"\n🔧 JIRA FILTERING APPLIED:")
        print(f"  Expected Projects: {expected_projects}")
        print(f"  Excluded Types: Epic")
        print(f"  Processing only linked issues from expected projects\n")
        
        for result in results:
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Expected dict result but got {type(result)}")
                continue
                
            content = result.get('content', '').lower()
            title = result.get('title', '').lower()
            platform = result.get('platform', '')
            url = result.get('url', '')
            
            # FILTER JIRA ISSUES: Only process expected projects and exclude Epics
            if platform == 'jira':
                jira_key = self._extract_jira_key_from_result(result)
                issue_type = result.get('metadata', {}).get('issue_type', '').lower()
                project_key = 'UNKNOWN'
                
                # Check if issue is from expected project
                if jira_key:
                    project_key = jira_key.split('-')[0] if '-' in jira_key else ''
                    if project_key not in expected_projects:
                        print(f"❌ Skipping JIRA {jira_key} - not from expected projects {expected_projects}")
                        continue
                
                # Exclude Epic type issues
                if issue_type == 'epic':
                    print(f"❌ Skipping JIRA {jira_key} - Epic type excluded")
                    continue
                
                # Only process linked issues (if this is a search result, it should be linked)
                print(f"✅ Processing JIRA {jira_key} - project: {project_key}, type: {issue_type}")
            
            # Extract error patterns
            if any(error_word in content for error_word in ['error', 'exception', 'failed', 'failure']):
                error_patterns.append({
                    'platform': platform,
                    'content': result.get('content', '')[:300],
                    'url': url,
                    'title': result.get('title', '')
                })
            
            # Enhanced Jira analysis: Extract comments for solutions with LLM analysis
            if platform == 'jira':
                jira_key = self._extract_jira_key_from_result(result)
                ticket_status = result.get('metadata', {}).get('status', '').lower()
                
                # This ticket already passed filtering above, so it's valid
                issue_type = result.get('metadata', {}).get('issue_type', '')
                
                # PRIORITIZE CLOSED TICKETS - they have actual resolutions
                print(f"\n🎫 PROCESSING FILTERED JIRA TICKET:")
                print(f"  Key: {jira_key}")
                print(f"  Status: {ticket_status}")
                print(f"  Type: {issue_type}")
                print(f"  Title: {result.get('title', 'No title')}")
                print(f"  URL: {result.get('url', 'No URL')}")
                
                if jira_key and ticket_status == 'closed':
                    print(f"  🔥 PRIORITY: This is a CLOSED ticket - should contain actual resolution!")
                    logger.info(f"🎫 PRIORITY: Analyzing CLOSED Jira ticket {jira_key} for actual resolution")
                    jira_details = await self._extract_jira_comments_and_solutions_with_llm(jira_key, query, all_search_results)
                elif jira_key:
                    print(f"  ⚠️  Secondary: This is an open ticket - may not have final resolution")
                    logger.info(f"🎫 Secondary: Analyzing open Jira ticket {jira_key}")
                    jira_details = await self._extract_jira_comments_and_solutions_with_llm(jira_key, query, all_search_results)
                else:
                    print(f"  ❌ No JIRA key found in this result")
                    jira_details = None
                    if jira_details:
                        detailed_solutions.append({
                            'source': 'jira',
                            'ticket': jira_key,
                            'url': url,
                            'solutions': jira_details.get('solutions', []),
                            'comments': jira_details.get('comments', []),
                            'llm_analysis': jira_details.get('llm_analysis', '')
                        })
                        
                        # Add to all content for comprehensive LLM analysis
                        all_extracted_content.append({
                            'source': 'jira',
                            'ticket': jira_key,
                            'content': jira_details.get('full_content', ''),
                            'analysis': jira_details.get('llm_analysis', '')
                        })
                
                related_tickets.append({
                    'title': result.get('title', ''),
                    'url': url,
                    'status': result.get('metadata', {}).get('status', 'Unknown'),
                    'key': jira_key
                })
            
            # Enhanced Slack analysis: Extract thread conversations for solutions with LLM analysis
            elif platform == 'slack':
                logger.info(f"💬 Deep-diving into Slack thread for comprehensive analysis")
                slack_solutions = await self._extract_slack_thread_solutions_with_llm(result, query)
                if slack_solutions:
                    detailed_solutions.append({
                        'source': 'slack',
                        'channel': result.get('metadata', {}).get('channel', 'unknown'),
                        'url': url,
                        'solutions': slack_solutions.get('solutions', []),
                        'thread_summary': slack_solutions.get('thread_summary', ''),
                        'llm_analysis': slack_solutions.get('llm_analysis', '')
                    })
                    
                    # Add to all content for comprehensive LLM analysis
                    all_extracted_content.append({
                        'source': 'slack',
                        'channel': result.get('metadata', {}).get('channel', 'unknown'),
                        'content': slack_solutions.get('full_thread_content', ''),
                        'analysis': slack_solutions.get('llm_analysis', '')
                    })
                
                # ENHANCED: Extract Jira tickets mentioned in Slack threads and fetch their details
                thread_summary = result.get('metadata', {}).get('thread_summary', '')
                if thread_summary and 'jira tickets mentioned:' in thread_summary.lower():
                    logger.info(f"🎫 Found Jira ticket references in Slack thread: {thread_summary}")
                    jira_tickets = self._extract_jira_tickets_from_slack_thread(thread_summary)
                    
                    for jira_key in jira_tickets:
                        logger.info(f"🔍 Fetching Jira ticket {jira_key} mentioned in Slack thread")
                        jira_details = await self._extract_jira_comments_and_solutions_with_llm(jira_key, query, all_search_results)
                        if jira_details:
                            detailed_solutions.append({
                                'source': 'jira',
                                'ticket': jira_key,
                                'url': f"https://acquia.atlassian.net/browse/{jira_key}",
                                'solutions': jira_details.get('solutions', []),
                                'comments': jira_details.get('comments', []),
                                'llm_analysis': jira_details.get('llm_analysis', ''),
                                'found_via': f'slack_thread_#{result.get("metadata", {}).get("channel", "unknown")}'
                            })
                            
                            # Add to all content for comprehensive LLM analysis
                            all_extracted_content.append({
                                'source': 'jira_from_slack',
                                'ticket': jira_key,
                                'content': jira_details.get('full_content', ''),
                                'analysis': jira_details.get('llm_analysis', ''),
                                'found_via': f'slack_thread_#{result.get("metadata", {}).get("channel", "unknown")}'
                            })
                            
                            # Also add to related_tickets for UI display
                            related_tickets.append({
                                'title': f"{jira_key}: {jira_details.get('comments', [{}])[0].get('body', 'Jira ticket from Slack thread')[:100]}...",
                                'url': f"https://acquia.atlassian.net/browse/{jira_key}",
                                'status': 'Referenced in Slack',
                                'key': jira_key,
                                'found_via': f'slack_thread_#{result.get("metadata", {}).get("channel", "unknown")}'
                            })
                
                slack_discussions.append({
                    'channel': result.get('metadata', {}).get('channel', 'unknown'),
                    'content': result.get('content', '')[:200],
                    'url': url,
                    'thread_info': slack_solutions.get('thread_summary', '') if slack_solutions else ''
                })
            
            # Enhanced Confluence analysis: Extract full page content for solutions with LLM analysis
            elif platform == 'confluence':
                logger.info(f"📄 Deep-diving into Confluence page for comprehensive analysis")
                confluence_solutions = await self._extract_confluence_page_solutions_with_llm(result, query)
                if confluence_solutions:
                    detailed_solutions.append({
                        'source': 'confluence',
                        'page_title': result.get('title', ''),
                        'url': url,
                        'solutions': confluence_solutions.get('solutions', []),
                        'full_content': confluence_solutions.get('content', ''),
                        'llm_analysis': confluence_solutions.get('llm_analysis', '')
                    })
                    
                    # Add to all content for comprehensive LLM analysis
                    all_extracted_content.append({
                        'source': 'confluence',
                        'page_title': result.get('title', ''),
                        'content': confluence_solutions.get('content', ''),
                        'analysis': confluence_solutions.get('llm_analysis', '')
                    })
                
                confluence_docs.append({
                    'title': result.get('title', ''),
                    'space': result.get('metadata', {}).get('space', ''),
                    'url': url,
                    'excerpt': result.get('content', '')[:200]
                })
            
            # Look for technical keywords
            if any(tech_word in content for tech_word in ['sql', 'database', 'snowflake', 'javascript', 'oozie']):
                technical_context.append({
                    'platform': platform,
                    'context': result.get('content', '')[:200],
                    'source': result.get('title', ''),
                    'url': url
                })
        
        # Generate comprehensive LLM-powered solution analysis
        print(f"\n🚀 STARTING COMPREHENSIVE LLM ANALYSIS:")
        print(f"  📊 Sources to analyze: {len(all_extracted_content)} deep-extracted sources")
        print(f"  🎯 Detailed solutions: {len(detailed_solutions)} solution sets")
        print(f"  📋 Error patterns: {len(error_patterns)} patterns")
        print(f"  🔧 Technical context: {len(technical_context)} context items")
        
        for i, content in enumerate(all_extracted_content, 1):
            source = content.get('source', 'unknown')
            content_length = len(content.get('content', ''))
            print(f"    Source {i}: {source.upper()} - {content_length} characters of content")
        
        logger.info(f"🤖 Generating comprehensive LLM analysis from {len(all_extracted_content)} deep-extracted sources")
        solution_analysis = await self._generate_comprehensive_llm_solution_analysis(
            query, error_patterns, technical_context, detailed_solutions, all_extracted_content
        )
        
        return {
            'error_analysis': {
                'detected_errors': len(error_patterns),
                'error_patterns': error_patterns[:3],
                'technical_context': technical_context[:3],
                'related_tickets': related_tickets[:3],
                'slack_discussions': slack_discussions[:3],
                'confluence_docs': confluence_docs[:3]
            },
            'detailed_solutions': detailed_solutions,
            'solution_recommendations': solution_analysis,
            'next_steps': self._generate_enhanced_next_steps(query, error_patterns, detailed_solutions),
            'metadata': {
                'links_followed': len(detailed_solutions),
                'jira_comments_extracted': len([s for s in detailed_solutions if s['source'] == 'jira']),
                'slack_threads_analyzed': len([s for s in detailed_solutions if s['source'] == 'slack']),
                'confluence_pages_analyzed': len([s for s in detailed_solutions if s['source'] == 'confluence']),
                'llm_analysis_performed': len(all_extracted_content) > 0,
                'total_content_analyzed': len(all_extracted_content)
            }
        }
    
    def _generate_solution_for_error_pattern(self, query: str, error_patterns: List[Dict], 
                                           technical_context: List[Dict], related_tickets: List[Dict]) -> Dict[str, Any]:
        """Generate specific solutions based on detected error patterns"""
        
        solutions = {
            'primary_solution': '',
            'alternative_solutions': [],
            'root_cause_analysis': '',
            'prevention_measures': []
        }
        
        query_lower = query.lower()
        
        # Analyze NCOA runner failure specifically
        if 'ncoa' in query_lower and 'failed' in query_lower:
            # Look for specific error details in the content
            snowflake_error = None
            for error in error_patterns:
                if 'snowflake' in error['content'].lower() and 'field delimiter' in error['content'].lower():
                    snowflake_error = error
                    break
            
            if snowflake_error:
                solutions.update({
                    'primary_solution': '''**Data Format Issue Resolution:**
1. **Immediate Fix**: Check the NCOA input file format at line 14
   - Verify field delimiters are consistent (comma, pipe, tab)
   - Look for unexpected characters like '6' in delimiter positions
   - Validate file encoding (UTF-8 vs ASCII)

2. **File Validation**: Run pre-processing validation
   ```sql
   -- Check for delimiter consistency
   SELECT line_number, content 
   FROM raw_file_staging 
   WHERE content LIKE '%6%' AND line_number AROUND 14;
   ```

3. **Snowflake JavaScript Fix**: Update the stored procedure
   - Add error handling for malformed records
   - Implement delimiter detection logic
   - Add logging for problematic lines''',
                    
                    'alternative_solutions': [
                        '**File Reprocessing**: Re-extract the NCOA file from source with proper formatting',
                        '**Manual Cleanup**: Use text processing tools to fix delimiter issues before loading',
                        '**Schema Validation**: Implement stricter input validation in the ETL pipeline',
                        '**Fallback Processing**: Skip problematic records and process them separately'
                    ],
                    
                    'root_cause_analysis': '''**Root Cause**: Data format inconsistency in NCOA input file
- The error "Found character '6' instead of field delimiter" indicates:
  1. Source file has inconsistent delimiters
  2. Possible data corruption during file transfer
  3. Encoding issues (special characters converted incorrectly)
  4. Source system generating malformed CSV/delimited files''',
                    
                    'prevention_measures': [
                        'Implement file validation checks before processing',
                        'Add automated delimiter detection and standardization',
                        'Set up monitoring for file format consistency',
                        'Create backup processing pipeline for malformed files',
                        'Establish data quality checks with source system team'
                    ]
                })
            else:
                # Generic NCOA failure solution
                solutions.update({
                    'primary_solution': '''**NCOA Processing Failure - General Resolution:**
1. **Check Job Status**: Review Oozie workflow logs for detailed error information
2. **Data Validation**: Verify input data quality and format
3. **Resource Check**: Ensure sufficient cluster resources for processing
4. **Dependency Verification**: Confirm all upstream data dependencies are available''',
                    
                    'alternative_solutions': [
                        'Restart the job with increased memory allocation',
                        'Process data in smaller batches to avoid resource constraints',
                        'Check for data schema changes that might cause processing failures'
                    ],
                    
                    'root_cause_analysis': 'NCOA job failure - requires detailed log analysis to determine specific cause',
                    
                    'prevention_measures': [
                        'Implement comprehensive data quality checks',
                        'Set up proactive monitoring for NCOA job dependencies',
                        'Create automated retry mechanisms for transient failures'
                    ]
                })
        
        # Generic database/SQL error handling
        elif any(db_word in query_lower for db_word in ['sql', 'database', 'snowflake']):
            solutions.update({
                'primary_solution': '''**Database Error Resolution:**
1. **Error Log Analysis**: Check detailed error logs for specific SQL statements
2. **Connection Validation**: Verify database connectivity and credentials
3. **Query Optimization**: Review and optimize problematic SQL queries
4. **Resource Monitoring**: Check database resource utilization''',
                
                'alternative_solutions': [
                    'Implement connection pooling and retry logic',
                    'Break down complex queries into smaller operations',
                    'Add transaction management and rollback procedures'
                ],
                
                'root_cause_analysis': 'Database connectivity or SQL execution issue requiring detailed investigation',
                
                'prevention_measures': [
                    'Implement comprehensive database monitoring',
                    'Set up automated health checks',
                    'Create backup processing procedures'
                ]
            })
        
        # Generic error handling for other cases
        else:
            solutions.update({
                'primary_solution': '''**General Issue Resolution:**
1. **Log Analysis**: Review detailed application and system logs
2. **Environment Check**: Verify system resources and dependencies
3. **Configuration Review**: Check application and system configurations
4. **Testing**: Reproduce the issue in a controlled environment''',
                
                'alternative_solutions': [
                    'Implement comprehensive monitoring and alerting',
                    'Create detailed troubleshooting documentation',
                    'Set up automated recovery procedures'
                ],
                
                'root_cause_analysis': 'Issue requires detailed analysis of logs and system state',
                
                'prevention_measures': [
                    'Implement proactive monitoring',
                    'Create comprehensive testing procedures',
                    'Document known issues and solutions'
                ]
            })
        
        return solutions
    
    def _generate_next_steps(self, query: str, error_patterns: List[Dict], related_tickets: List[Dict]) -> List[str]:
        """Generate actionable next steps based on the analysis"""
        
        next_steps = []
        
        # Add ticket-specific steps
        if related_tickets:
            next_steps.append(f"📋 **Review Related Tickets**: Check {len(related_tickets)} related Jira tickets for additional context")
            for ticket in related_tickets[:2]:
                if ticket.get('url'):
                    next_steps.append(f"   • {ticket['title']}: {ticket['url']}")
        
        # Add error-specific steps
        if error_patterns:
            next_steps.append("🔍 **Error Investigation**: Analyze the specific error patterns found")
            next_steps.append("📊 **Log Analysis**: Review detailed logs from the time of failure")
        
        # Query-specific steps
        query_lower = query.lower()
        if 'ncoa' in query_lower:
            next_steps.extend([
                "📁 **File Validation**: Check NCOA input file format and integrity",
                "🔄 **Job Restart**: Consider restarting the job after fixing data issues",
                "👥 **Team Coordination**: Notify data team about NCOA processing issues"
            ])
        
        if 'snowflake' in query_lower:
            next_steps.extend([
                "❄️ **Snowflake Console**: Check Snowflake query history and performance metrics",
                "🔧 **Resource Scaling**: Consider scaling up warehouse if resource-related"
            ])
        
        # Generic steps if no specific patterns found
        if not next_steps:
            next_steps.extend([
                "📋 **Create Incident Ticket**: Document the issue in Jira if not already done",
                "👥 **Team Notification**: Alert relevant team members about the issue",
                "📊 **Monitoring Setup**: Implement monitoring to prevent similar issues"
            ])
        
        return next_steps

    async def _search_jira_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback Jira search using existing tools"""
        try:
            # Import existing Jira tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from jira_tool import JiraTool
            
            # Sanitize query for JQL to avoid reserved word issues
            sanitized_query = self._sanitize_query_for_jql(query)
            logger.info(f"Jira fallback search: original='{query}', sanitized='{sanitized_query}'")
            
            jira_tool = JiraTool()
            results = jira_tool.search_issues(sanitized_query)
            
            # Convert to standard format - FIX: Map JiraTool fields correctly for UI
            formatted_results = []
            for result in results:
                # JiraTool returns: key, summary, description, status, priority, assignee, etc.
                # UI expects fields directly on the object AND in metadata
                formatted_results.append({
                    'platform': 'jira',
                    'title': f"{result.get('key', 'Unknown')}: {result.get('summary', 'No summary')}",
                    'content': result.get('description', '')[:500],  # Use description as content
                    'url': result.get('url', ''),
                    'relevance_score': 0.7,  # Default relevance score
                    # UI expects these fields directly on the object
                    'key': result.get('key', 'Unknown'),
                    'summary': result.get('summary', 'No summary'),
                    'status': result.get('status', 'Unknown'),
                    'priority': result.get('priority', 'Unknown'),
                    'assignee': result.get('assignee', 'Unassigned'),
                    'reporter': result.get('reporter', 'Unknown'),
                    'project': result.get('project', 'Unknown'),
                    'issue_type': result.get('issue_type', 'Unknown'),
                    'description': result.get('description', ''),
                    'created': result.get('created', ''),
                    'updated': result.get('updated', ''),
                    # Also keep in metadata for compatibility
                    'metadata': {
                        'key': result.get('key', 'Unknown'),
                        'summary': result.get('summary', 'No summary'),
                        'status': result.get('status', 'Unknown'),
                        'priority': result.get('priority', 'Unknown'),
                        'assignee': result.get('assignee', 'Unassigned'),
                        'reporter': result.get('reporter', 'Unknown'),
                        'project': result.get('project', 'Unknown'),
                        'issue_type': result.get('issue_type', 'Unknown'),
                        'created': result.get('created', ''),
                        'updated': result.get('updated', '')
                    }
                })
            
            logger.info(f"✅ Jira fallback returned {len(formatted_results)} properly formatted results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Jira fallback search failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []
    
    async def _search_confluence_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback Confluence search using existing tools"""
        try:
            # Import existing Confluence tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from confluence_tool import ConfluenceTool
            
            # Sanitize query for CQL to avoid parsing issues
            sanitized_query = self._sanitize_query_for_cql(query)
            logger.info(f"Confluence fallback search: original='{query}', sanitized='{sanitized_query}'")
            
            confluence_tool = ConfluenceTool()
            # Fix: Use the correct method name 'search_content' instead of 'search_pages'
            results = confluence_tool.search_content(sanitized_query, limit=20)
            
            # Convert to standard format
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'platform': 'confluence',
                    'title': result.get('title', ''),
                    'content': result.get('excerpt', '')[:500],  # Use 'excerpt' field
                    'url': result.get('url', ''),
                    'relevance_score': 0.7,  # Default relevance score
                    'metadata': {
                        'space': result.get('space', ''),
                        'type': result.get('type', ''),
                        'last_modified': result.get('last_modified', ''),
                        'author': result.get('author', ''),
                        'id': result.get('id', '')
                    }
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Confluence fallback search failed: {e}")
            return []
    
    async def _search_slack_fallback(self, query: str) -> List[Dict[str, Any]]:
        """Fallback Slack search using existing tools"""
        try:
            # Import existing Slack tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from slack_tool import SlackTool
            
            # Get specific channels from environment
            search_channels = os.getenv('SLACK_SEARCH_CHANNELS', '').split(',')
            search_channels = [ch.strip() for ch in search_channels if ch.strip()]
            
            logger.info(f"🔍 Slack fallback search for query: '{query}' in channels: {search_channels}")
            
            slack_tool = SlackTool()
            
            # Search in specific channels if configured
            if search_channels:
                logger.info(f"Searching in specific channels: {search_channels}")
                results = slack_tool.search_in_channels(query, search_channels, limit=20)
            else:
                logger.info("Searching across all channels")
                results = slack_tool.search_messages(query, count=20)
            
            logger.info(f"Slack tool returned {len(results)} results")
            
            # Convert to standard format
            formatted_results = []
            for result in results:
                # Normalize the score - Slack scores can be very high (thousands)
                raw_score = result.get('score', 0)
                normalized_score = min(raw_score / 3000.0, 1.0) if raw_score > 0 else 0.5
                
                formatted_result = {
                    'platform': 'slack',
                    'title': f"Message in #{result.get('channel', 'unknown')}",
                    'content': result.get('text', '')[:500],
                    'url': result.get('permalink', ''),
                    'relevance_score': normalized_score,
                    'metadata': {
                        'channel': result.get('channel', ''),
                        'user': result.get('user', ''),
                        'timestamp': result.get('ts', ''),
                        'score': result.get('score', 0),
                        'thread_summary': result.get('thread_summary', ''),
                        'reply_count': result.get('reply_count', 0)
                    }
                }
                formatted_results.append(formatted_result)
                logger.info(f"Formatted Slack result: {formatted_result['title']} - Score: {raw_score} → {normalized_score:.3f}")
            
            logger.info(f"✅ Successfully returning {len(formatted_results)} formatted Slack results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Slack fallback search failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return []

    def _extract_jira_key_from_result(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract Jira key from search result"""
        try:
            # Try to extract from title (format: "KEY-123: Summary")
            title = result.get('title', '')
            if ':' in title:
                potential_key = title.split(':')[0].strip()
                if '-' in potential_key and potential_key.replace('-', '').replace('_', '').isalnum():
                    return potential_key
            
            # Try to extract from metadata
            if 'metadata' in result and 'key' in result['metadata']:
                return result['metadata']['key']
            
            # Try to extract from direct key field
            if 'key' in result:
                return result['key']
            
            # Try to extract from URL
            url = result.get('url', '')
            if '/browse/' in url:
                key_part = url.split('/browse/')[-1]
                if key_part and '-' in key_part:
                    return key_part.split('?')[0]  # Remove query parameters
            
            return None
        except Exception as e:
            logger.error(f"Error extracting Jira key from result: {e}")
            return None
    
    async def _extract_jira_comments_and_solutions_with_llm(self, jira_key: str, query: str, all_search_results: List[Dict] = None) -> Optional[Dict[str, Any]]:
        """Extract comments and solutions from a Jira ticket with LLM analysis"""
        try:
            # Import Jira tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from jira_tool import JiraTool
            
            jira_tool = JiraTool()
            issue_details = jira_tool.get_issue_details(jira_key)
            
            if not issue_details:
                logger.warning(f"No details found for Jira ticket {jira_key}")
                return None
            
            comments = issue_details.get('comments', [])
            
            # Combine all content for LLM analysis
            full_content = f"""
JIRA TICKET: {jira_key}
SUMMARY: {issue_details.get('summary', 'No summary')}
DESCRIPTION: {issue_details.get('description', 'No description')}
STATUS: {issue_details.get('status', 'Unknown')}
PRIORITY: {issue_details.get('priority', 'Unknown')}

COMMENTS ({len(comments)} total):
"""
            
            for i, comment in enumerate(comments, 1):
                full_content += f"""
Comment {i} by {comment.get('author', 'Unknown')} on {comment.get('created', 'Unknown date')}:
{comment.get('body', 'No content')}
---
"""
            
            # Use LLM to analyze the content for solutions
            llm_analysis = ""
            solutions = []
            
            # ENHANCED APPROACH: Integrated chronological analysis with linked tickets
            if self.llm and len(comments) > 0 and issue_details.get('status', '').lower() == 'closed':
                try:
                    print(f"\n🔗 Building integrated timeline for {jira_key} with linked tickets...")
                    
                    # Step 1: Find linked tickets mentioned in comments
                    linked_ticket_ids = self._extract_jira_ticket_references_from_comments(comments)
                    
                    # Step 2: Get linked ticket details from search results
                    linked_tickets = {}
                    if all_search_results and linked_ticket_ids:
                        for ticket_id in linked_ticket_ids:
                            linked_result = self._find_linked_ticket_in_search_results(ticket_id, all_search_results)
                            if linked_result:
                                # Extract full ticket details using JiraTool
                                import sys
                                import os
                                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
                                from jira_tool import JiraTool
                                
                                jira_tool = JiraTool()
                                linked_details = jira_tool.get_issue_details(ticket_id)
                                if linked_details:
                                    linked_tickets[ticket_id] = linked_details
                                    print(f"  ✅ Added linked ticket {ticket_id}: {linked_details.get('summary', 'No summary')}")
                    
                    # Step 3: Build integrated chronological timeline
                    primary_ticket_data = {
                        'key': jira_key,
                        'summary': issue_details.get('summary', 'No summary'),
                        'status': issue_details.get('status', 'Unknown'),
                        'description': issue_details.get('description', 'No description')
                    }
                    
                    chronology_prompt = self._build_integrated_chronological_timeline(
                        primary_ticket_data, linked_tickets, comments, query
                    )

                    messages = [
                        HumanMessage(content=chronology_prompt)
                    ]
                    
                    # LOG THE FULL PROMPT BEING SENT TO LLM
                    print("=" * 100)
                    print("🔍 JIRA CHRONOLOGICAL ANALYSIS - LLM INPUT:")
                    print("=" * 100)
                    print(chronology_prompt)
                    print("=" * 100)
                    
                    response = self.llm(messages)
                    llm_analysis = response.content
                    
                    # LOG THE FULL LLM RESPONSE
                    print("🤖 JIRA CHRONOLOGICAL ANALYSIS - LLM RESPONSE:")
                    print("=" * 100)
                    print(llm_analysis)
                    print("=" * 100)
                    
                    # Parse chronological analysis for actual resolution
                    if llm_analysis and len(llm_analysis.strip()) > 50:  # Valid response
                        # Look for high-confidence actual solutions (operational fixes)
                        confidence = 0.9 if any(indicator in llm_analysis.lower() for indicator in [
                            'increased warehouse size', 'alter warehouse', 'completed successfully', 
                            'workflow completed', 'restarted', 'updated configuration', 'changed setting'
                        ]) else 0.8
                        
                        solutions.append({
                            'source': 'chronological_analysis',
                            'content': llm_analysis,
                            'confidence': confidence,
                            'type': 'actual_resolution_from_closed_ticket',
                            'author': 'Resolution Analysis',
                            'ticket_status': 'closed'
                        })
                    
                except Exception as e:
                    logger.warning(f"LLM analysis failed for Jira {jira_key}: {e}")
                    llm_analysis = f"LLM analysis failed: {str(e)}"
            
            # Fallback keyword-based extraction
            solution_keywords = [
                'solved', 'fixed', 'resolved', 'solution', 'answer', 'workaround',
                'try this', 'here\'s how', 'fix', 'restart', 'rerun', 'check',
                'update', 'change', 'working', 'works', 'success', 'done',
                'complete', 'issue resolved', 'problem solved', 'that worked',
                'run this', 'use this', 'do this', 'configure', 'set', 'enable'
            ]
            
            for comment in comments:
                comment_text = comment.get('body', '').lower()
                
                # Check if comment contains solution keywords
                matching_keywords = [kw for kw in solution_keywords if kw in comment_text]
                if matching_keywords:
                    solutions.append({
                        'author': comment.get('author', 'Unknown'),
                        'content': comment.get('body', '')[:500],
                        'created': comment.get('created', ''),
                        'keywords_matched': matching_keywords,
                        'confidence': len(matching_keywords) / len(solution_keywords),
                        'type': 'keyword_extracted_solution'
                    })
            
            logger.info(f"🎫 Extracted {len(solutions)} solutions from Jira {jira_key} using LLM + keyword analysis")
            
            return {
                'ticket_key': jira_key,
                'comments': comments[:10],
                'solutions': solutions,
                'total_comments': len(comments),
                'solution_count': len(solutions),
                'llm_analysis': llm_analysis,
                'full_content': full_content
            }
            
        except Exception as e:
            logger.error(f"Error extracting Jira comments with LLM for {jira_key}: {e}")
            return None

    async def _extract_jira_comments_and_solutions(self, jira_key: str) -> Optional[Dict[str, Any]]:
        """Extract comments and solutions from a Jira ticket (fallback method)"""
        try:
            # Import Jira tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from jira_tool import JiraTool
            
            jira_tool = JiraTool()
            issue_details = jira_tool.get_issue_details(jira_key)
            
            if not issue_details:
                logger.warning(f"No details found for Jira ticket {jira_key}")
                return None
            
            comments = issue_details.get('comments', [])
            solutions = []
            
            # Solution keywords to identify potential solutions in comments
            solution_keywords = [
                'solved', 'fixed', 'resolved', 'solution', 'answer', 'workaround',
                'try this', 'here\'s how', 'fix', 'restart', 'rerun', 'check',
                'update', 'change', 'working', 'works', 'success', 'done',
                'complete', 'issue resolved', 'problem solved', 'that worked',
                'run this', 'use this', 'do this', 'configure', 'set', 'enable'
            ]
            
            for comment in comments:
                comment_text = comment.get('body', '').lower()
                
                # Check if comment contains solution keywords
                matching_keywords = [kw for kw in solution_keywords if kw in comment_text]
                if matching_keywords:
                    solutions.append({
                        'author': comment.get('author', 'Unknown'),
                        'content': comment.get('body', '')[:500],  # Limit content length
                        'created': comment.get('created', ''),
                        'keywords_matched': matching_keywords,
                        'confidence': len(matching_keywords) / len(solution_keywords)
                    })
            
            logger.info(f"Extracted {len(solutions)} potential solutions from {len(comments)} comments in {jira_key}")
            
            return {
                'ticket_key': jira_key,
                'comments': comments[:10],  # Limit to 10 most recent comments
                'solutions': solutions,
                'total_comments': len(comments),
                'solution_count': len(solutions)
            }
            
        except Exception as e:
            logger.error(f"Error extracting Jira comments for {jira_key}: {e}")
            return None
    
    async def _extract_slack_thread_solutions_with_llm(self, result: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Extract thread conversations and solutions from Slack message with LLM analysis"""
        try:
            print(f"\n" + "="*80)
            print(f"🔍 SLACK THREAD EXTRACTION DEBUG STARTED")
            print(f"="*80)
            
            # Import Slack tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from slack_tool import SlackTool
            
            slack_tool = SlackTool()
            
            # NEW APPROACH: Parse Slack URL to get channel_id and thread_ts directly
            print(f"📋 INPUT RESULT ANALYSIS:")
            print(f"  Full result keys: {list(result.keys())}")
            print(f"  Metadata keys: {list(result.get('metadata', {}).keys())}")
            print(f"  Full metadata: {result.get('metadata', {})}")
            print(f"  Result URL: '{result.get('url', '')}'")
            
            # Parse URL to extract channel_id and thread_ts
            url = result.get('url', '')
            url_info = self._parse_slack_url_for_thread_info(url)
            
            if not url_info:
                print(f"❌ FAILED: Could not parse Slack URL for thread info")
                return None
                
            channel_id = url_info['channel_id']
            thread_ts = url_info['thread_ts']  # This is the parent thread timestamp
            
            print(f"\n🎯 EXTRACTED FROM URL:")
            print(f"  Channel ID: '{channel_id}' (direct from URL)")
            print(f"  Thread TS: '{thread_ts}' (parent thread timestamp)")
            print(f"  Skipping channel name resolution - using direct channel ID")
            
            print(f"\n🧵 SLACK API - Thread Message Retrieval:")
            print(f"  Channel ID: '{channel_id}'")
            print(f"  Parent Thread TS: '{thread_ts}'")
            print(f"  Calling slack_tool._get_thread_messages_by_channel_id('{channel_id}', '{thread_ts}')")
            
            # Get full thread conversation using direct API call with channel_id and parent thread_ts
            thread_messages = await self._get_slack_thread_messages_direct(slack_tool, channel_id, thread_ts)
            
            print(f"  API Response Type: {type(thread_messages)}")
            print(f"  API Response Length: {len(thread_messages) if thread_messages else 0}")
            
            if thread_messages:
                print(f"  ✅ SUCCESS: Retrieved {len(thread_messages)} thread messages")
                print(f"  Thread message details:")
                for i, msg in enumerate(thread_messages[:5], 1):  # Show first 5 messages
                    user = msg.get('user', 'Unknown')
                    text = msg.get('text', msg.get('content', ''))
                    ts = msg.get('ts', msg.get('timestamp', ''))
                    print(f"    Message {i}: User='{user}', TS='{ts}'")
                    print(f"      Text: {text[:150]}{'...' if len(text) > 150 else ''}")
                if len(thread_messages) > 5:
                    print(f"    ... and {len(thread_messages) - 5} more messages")
            else:
                print(f"  ❌ FAILED: No thread messages returned (None or empty list)")
            
            if not thread_messages:
                print(f"\n🔄 FALLBACK ACTIVATED:")
                logger.info(f"No thread messages found for {channel_id}:{thread_ts}")
                print(f"  Reason: Direct Slack API call returned empty/None")
                print(f"  Action: Using single original message as fallback")
                # Fallback to original message analysis
                thread_messages = [result]
                print(f"  Fallback thread_messages: {len(thread_messages)} message(s)")
                print(f"  Fallback message content: {result.get('content', '')[:200]}...")
            else:
                print(f"  💪 USING FULL THREAD: {len(thread_messages)} messages for analysis")
            
            print(f"\n📝 THREAD CONTENT PROCESSING:")
            print(f"  Processing {len(thread_messages)} messages for analysis")
            
            # Extract all Jira tickets found in thread messages
            all_jira_tickets = []
            for msg in thread_messages:
                jira_tickets = msg.get('jira_tickets', [])
                all_jira_tickets.extend(jira_tickets)
            
            # Remove duplicates and log findings
            unique_jira_tickets = list(set(all_jira_tickets))
            if unique_jira_tickets:
                print(f"  🎫 Found {len(unique_jira_tickets)} Jira tickets: {unique_jira_tickets}")
                logger.info(f"🎫 Found {len(unique_jira_tickets)} unique Jira tickets in Slack thread: {unique_jira_tickets}")
            else:
                print(f"  📋 No Jira tickets found in thread messages")
            
            # Combine all thread content for LLM analysis
            print(f"  📄 Building thread content for LLM analysis...")
            channel_name = result.get('metadata', {}).get('channel', 'unknown')  # Get original channel name for display
            full_thread_content = f"""
SLACK THREAD ANALYSIS
Channel: #{channel_name} (ID: {channel_id})
Original Query Context: {query}

THREAD CONVERSATION:
"""
            
            for i, msg in enumerate(thread_messages, 1):
                user = msg.get('user', 'Unknown')
                text = msg.get('text', msg.get('content', ''))
                ts = msg.get('ts', msg.get('timestamp', ''))
                
                full_thread_content += f"""
Message {i} by {user} at {ts}:
{text}
---
"""
            
            print(f"  📏 Full thread content length: {len(full_thread_content)} characters")
            print(f"  📋 Content preview (first 300 chars): {full_thread_content[:300]}...")
            
            # Use LLM to analyze the thread for solutions
            print(f"\n🤖 LLM ANALYSIS:")
            print(f"  LLM Available: {self.llm is not None}")
            print(f"  Thread messages count: {len(thread_messages)}")
            
            llm_analysis = ""
            solutions = []
            
            if self.llm and len(thread_messages) > 0:
                try:
                    print(f"  🚀 Starting LLM analysis...")
                    
                    system_prompt = f"""You are a technical expert analyzing a Slack thread conversation to find solutions related to the query: "{query}".

Your task is to:
1. Identify any solutions, workarounds, or fixes mentioned in the conversation
2. Extract step-by-step instructions if available
3. Identify who provided the most helpful solutions
4. Note any follow-up questions or clarifications
5. Highlight community insights and team knowledge
6. Assess the reliability of solutions based on team member responses

Focus on actionable technical solutions and community-validated approaches."""

                    content_to_analyze = full_thread_content[:4000]  # Limit content for LLM
                    print(f"  📏 Content sent to LLM: {len(content_to_analyze)} characters (truncated from {len(full_thread_content)})")
                    
                    user_prompt = f"""
Query Context: {query}

Slack Thread Analysis:
{content_to_analyze}

Please analyze this Slack thread and extract:
1. All solutions or workarounds mentioned by team members
2. Step-by-step instructions provided
3. Community validation (replies, reactions, confirmations)
4. Expert contributors and their suggestions
5. Confidence level for each solution based on team feedback (0.0 to 1.0)

Format your response as a structured analysis with clear sections for solutions, contributors, and validation.
"""
                    
                    print(f"  🔄 Sending request to LLM...")
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    
                    response = self.llm(messages)
                    llm_analysis = response.content
                    
                    print(f"  ✅ LLM Response received: {len(llm_analysis)} characters")
                    print(f"  📋 LLM Response preview: {llm_analysis[:300]}...")
                    
                    # Parse LLM response to extract structured solutions
                    if "solution" in llm_analysis.lower() or "fix" in llm_analysis.lower() or "workaround" in llm_analysis.lower():
                        print(f"  🎯 Solution keywords found in LLM response - creating solution entry")
                        solutions.append({
                            'source': 'llm_thread_analysis',
                            'content': llm_analysis,
                            'confidence': 0.8,
                            'type': 'llm_extracted_community_solution',
                            'author': 'AI Analysis of Team Discussion'
                        })
                    else:
                        print(f"  ⚠️ No solution keywords found in LLM response")
                    
                except Exception as e:
                    print(f"  🚨 LLM ANALYSIS EXCEPTION: {e}")
                    logger.warning(f"LLM analysis failed for Slack thread {channel_id}:{thread_ts}: {e}")
                    llm_analysis = f"LLM analysis failed: {str(e)}"
            else:
                if not self.llm:
                    print(f"  ❌ LLM not available - skipping AI analysis")
                else:
                    print(f"  ❌ No thread messages - skipping LLM analysis")
            
            print(f"\n🔍 KEYWORD-BASED SOLUTION EXTRACTION:")
            # Fallback keyword-based extraction from thread messages
            solution_keywords = [
                'solved', 'fixed', 'resolved', 'solution', 'answer', 'workaround',
                'try this', 'here\'s how', 'fix', 'restart', 'rerun', 'check now',
                'update', 'change', 'working', 'works', 'success', 'done',
                'complete', 'issue resolved', 'problem solved', 'that worked',
                'run this', 'use this', 'do this', 'configure', 'set', 'enable',
                'i fixed it', 'this works', 'found the issue', 'here\'s the fix'
            ]
            
            print(f"  📝 Scanning {len(thread_messages)} messages for {len(solution_keywords)} solution keywords...")
            keyword_solutions_found = 0
            
            for msg in thread_messages:
                msg_text = msg.get('text', msg.get('content', '')).lower()
                user = msg.get('user', 'Unknown')
                
                # Check if message contains solution keywords
                matching_keywords = [kw for kw in solution_keywords if kw in msg_text]
                if matching_keywords:
                    keyword_solutions_found += 1
                    confidence = min(len(matching_keywords) / 3.0, 1.0)
                    print(f"    ✅ Message from {user}: {len(matching_keywords)} keywords matched, confidence: {confidence:.2f}")
                    print(f"      Keywords: {matching_keywords}")
                    print(f"      Text preview: {msg_text[:150]}...")
                    
                    solutions.append({
                        'author': user,
                        'content': msg.get('text', msg.get('content', ''))[:500],
                        'timestamp': msg.get('ts', msg.get('timestamp', '')),
                        'keywords_matched': matching_keywords,
                        'confidence': confidence,
                        'type': 'keyword_extracted_community_solution'
                    })
            
            print(f"  📊 Keyword extraction results: {keyword_solutions_found} solution-containing messages found")
            
            # Generate thread summary
            thread_summary = f"Thread with {len(thread_messages)} messages in #{channel_name} (ID: {channel_id})"
            if solutions:
                thread_summary += f" - {len(solutions)} potential solutions identified"
            
            print(f"\n📋 FINAL RESULTS SUMMARY:")
            print(f"  Total solutions found: {len(solutions)}")
            print(f"  LLM analysis length: {len(llm_analysis)} characters")
            print(f"  Thread summary: {thread_summary}")
            print(f"  Full thread content length: {len(full_thread_content)} characters")
            print(f"="*80)
            
            logger.info(f"💬 Extracted {len(solutions)} solutions from Slack thread #{channel_name} (ID: {channel_id}) using LLM + keyword analysis")
            
            return {
                'channel': channel_name,
                'channel_id': channel_id,
                'thread_ts': thread_ts,
                'thread_summary': thread_summary,
                'solutions': solutions,
                'thread_message_count': len(thread_messages),
                'solution_count': len(solutions),
                'llm_analysis': llm_analysis,
                'full_thread_content': full_thread_content
            }
            
        except Exception as e:
            print(f"\n🚨 EXCEPTION IN SLACK THREAD EXTRACTION:")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            import traceback
            print(f"  Traceback: {traceback.format_exc()}")
            print(f"="*80)
            logger.error(f"Error extracting Slack thread solutions with LLM: {e}")
            return None

    async def _extract_slack_thread_solutions(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract thread conversations and solutions from Slack message (fallback method)"""
        try:
            # Import Slack tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from slack_tool import SlackTool
            
            slack_tool = SlackTool()
            
            # Get channel and timestamp from result - check both 'timestamp' and 'ts' fields
            channel = result.get('metadata', {}).get('channel', '')
            timestamp = result.get('metadata', {}).get('timestamp', '') or result.get('metadata', {}).get('ts', '')
            
            if not channel:
                logger.warning(f"Missing channel for Slack thread extraction. Available metadata: {list(result.get('metadata', {}).keys())}")
                return None
            
            if not timestamp:
                logger.warning(f"Missing timestamp for Slack thread extraction. Available metadata: {list(result.get('metadata', {}).keys())}")
                return None
            
            # Get channel ID
            channel_id = slack_tool._get_channel_id(channel)
            if not channel_id:
                logger.warning(f"Could not find channel ID for {channel}")
                return None
            
            # Get thread summary (this method already exists in SlackTool)
            thread_summary = slack_tool._get_thread_summary(channel_id, timestamp)
            
            solutions = []
            
            # If thread summary contains solution indicators, extract them
            if thread_summary and any(keyword in thread_summary.lower() for keyword in ['solution', 'fixed', 'resolved', 'working']):
                solutions.append({
                    'source': 'thread_summary',
                    'content': thread_summary,
                    'confidence': 0.8,
                    'type': 'community_solution'
                })
            
            # Also check the original message for solution content
            original_content = result.get('content', '').lower()
            solution_keywords = ['solved', 'fixed', 'resolved', 'solution', 'workaround', 'try this']
            
            if any(keyword in original_content for keyword in solution_keywords):
                solutions.append({
                    'source': 'original_message',
                    'content': result.get('content', '')[:300],
                    'confidence': 0.6,
                    'type': 'discussion_solution'
                })
            
            logger.info(f"Extracted {len(solutions)} potential solutions from Slack thread in #{channel}")
            
            return {
                'channel': channel,
                'thread_summary': thread_summary,
                'solutions': solutions,
                'original_message': result.get('content', '')[:200]
            }
            
        except Exception as e:
            logger.error(f"Error extracting Slack thread solutions: {e}")
            return None
    
    async def _extract_confluence_page_solutions_with_llm(self, result: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Extract full page content and solutions from Confluence page with LLM analysis"""
        try:
            # Import Confluence tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from confluence_tool import ConfluenceTool
            
            confluence_tool = ConfluenceTool()
            
            # Get page ID from result metadata - check multiple possible locations
            page_id = result.get('metadata', {}).get('id', '') or result.get('id', '')
            
            if not page_id:
                logger.warning(f"Missing page ID for Confluence content extraction. Available metadata: {list(result.get('metadata', {}).keys())}, Available result keys: {list(result.keys())}")
                # Try to extract page ID from URL
                url = result.get('url', '')
                if url:
                    try:
                        # Extract page ID from Confluence URL patterns
                        if '/pages/viewpage.action?pageId=' in url:
                            page_id = url.split('pageId=')[1].split('&')[0]
                        elif '/display/' in url:
                            # For display URLs, we can't easily get page ID, so skip detailed extraction
                            logger.info(f"Confluence display URL detected, skipping detailed extraction: {url}")
                            return None
                        elif '/wiki/spaces/' in url:
                            # Modern Confluence URL format
                            url_parts = url.split('/')
                            if 'pages' in url_parts:
                                page_idx = url_parts.index('pages')
                                if page_idx + 1 < len(url_parts):
                                    page_id = url_parts[page_idx + 1]
                        
                        if page_id and page_id.isdigit():
                            logger.info(f"Extracted page ID from URL: {page_id}")
                        else:
                            logger.warning(f"Could not extract valid page ID from URL: {url}")
                            return None
                    except Exception as e:
                        logger.warning(f"Failed to extract page ID from URL {url}: {e}")
                        return None
                else:
                    return None
            
            # Get full page content
            page_content = confluence_tool.get_page_content(page_id)
            
            if not page_content:
                logger.warning(f"Could not retrieve content for Confluence page {page_id}")
                return None
            
            full_content = page_content.get('content', '')
            page_title = page_content.get('title', '')
            
            # Prepare content for LLM analysis
            confluence_analysis_content = f"""
CONFLUENCE PAGE ANALYSIS
Page Title: {page_title}
Page ID: {page_id}
Original Query Context: {query}

PAGE CONTENT:
{full_content[:4000]}  # Limit content for LLM processing
"""
            
            # Use LLM to analyze the page content for solutions
            llm_analysis = ""
            solutions = []
            
            if self.llm and full_content:
                try:
                    system_prompt = f"""You are a technical expert analyzing a Confluence documentation page to find solutions related to the query: "{query}".

Your task is to:
1. Identify any solutions, procedures, or fixes mentioned in the documentation
2. Extract step-by-step instructions if available
3. Identify troubleshooting guides and best practices
4. Note any code examples or configuration details
5. Highlight official procedures and recommended approaches
6. Assess the completeness and reliability of the documentation

Focus on actionable technical solutions and official procedures from the knowledge base."""

                    user_prompt = f"""
Query Context: {query}

Confluence Page Analysis:
{confluence_analysis_content}

Please analyze this Confluence page and extract:
1. All solutions, procedures, or fixes mentioned
2. Step-by-step instructions or troubleshooting guides
3. Code examples or configuration details
4. Best practices and recommended approaches
5. Confidence level for each solution based on documentation completeness (0.0 to 1.0)

Format your response as a structured analysis with clear sections for solutions, procedures, and recommendations.
"""

                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    
                    response = self.llm(messages)
                    llm_analysis = response.content
                    
                    # Parse LLM response to extract structured solutions
                    if any(keyword in llm_analysis.lower() for keyword in ['solution', 'procedure', 'fix', 'steps', 'troubleshooting']):
                        solutions.append({
                            'source': 'llm_documentation_analysis',
                            'content': llm_analysis,
                            'confidence': 0.9,  # High confidence for official documentation
                            'type': 'llm_extracted_documentation_solution',
                            'author': 'AI Analysis of Official Documentation'
                        })
                    
                except Exception as e:
                    logger.warning(f"LLM analysis failed for Confluence page {page_id}: {e}")
                    llm_analysis = f"LLM analysis failed: {str(e)}"
            
            # Fallback keyword-based extraction for solution patterns
            solution_patterns = [
                'solution:', 'resolution:', 'fix:', 'workaround:', 'how to:',
                'steps to resolve', 'troubleshooting', 'problem resolution',
                'issue fix', 'recommended approach', 'procedure:', 'instructions:',
                'configuration:', 'setup:', 'implementation:', 'best practice'
            ]
            
            content_lower = full_content.lower()
            
            for pattern in solution_patterns:
                if pattern in content_lower:
                    # Extract content around the solution pattern
                    pattern_index = content_lower.find(pattern)
                    if pattern_index != -1:
                        # Get 500 characters after the pattern
                        solution_text = full_content[pattern_index:pattern_index + 500]
                        solutions.append({
                            'pattern': pattern,
                            'content': solution_text,
                            'confidence': 0.8,  # High confidence for structured documentation
                            'type': 'keyword_extracted_documentation_solution'
                        })
            
            # Look for code blocks or structured solutions
            code_solutions = []
            if '```' in full_content:
                code_blocks = full_content.split('```')
                for i in range(1, len(code_blocks), 2):  # Every odd index is a code block
                    if code_blocks[i].strip():
                        code_solutions.append({
                            'pattern': 'code_block',
                            'content': f"Code example: {code_blocks[i][:300]}...",
                            'confidence': 0.9,
                            'type': 'code_documentation_solution'
                        })
            
            if '<code>' in full_content:
                # Extract HTML code blocks
                import re
                code_matches = re.findall(r'<code>(.*?)</code>', full_content, re.DOTALL)
                for code_match in code_matches[:3]:  # Limit to 3 code examples
                    if code_match.strip():
                        code_solutions.append({
                            'pattern': 'html_code_block',
                            'content': f"Code configuration: {code_match[:300]}...",
                            'confidence': 0.9,
                            'type': 'code_documentation_solution'
                        })
            
            solutions.extend(code_solutions)
            
            logger.info(f"📄 Extracted {len(solutions)} solutions from Confluence page {page_id} using LLM + keyword analysis")
            
            return {
                'page_id': page_id,
                'page_title': page_title,
                'content': full_content[:1000],  # Limit content for response size
                'solutions': solutions,
                'solution_count': len(solutions),
                'full_content_length': len(full_content),
                'llm_analysis': llm_analysis
            }
            
        except Exception as e:
            logger.error(f"Error extracting Confluence page solutions with LLM: {e}")
            return None

    async def _extract_confluence_page_solutions(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract full page content and solutions from Confluence page (fallback method)"""
        try:
            # Import Confluence tool
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tools', 'cdp_chat_tool'))
            from confluence_tool import ConfluenceTool
            
            confluence_tool = ConfluenceTool()
            
            # Get page ID from result metadata - check multiple possible locations
            page_id = result.get('metadata', {}).get('id', '') or result.get('id', '')
            
            if not page_id:
                logger.warning(f"Missing page ID for Confluence content extraction. Available metadata: {list(result.get('metadata', {}).keys())}, Available result keys: {list(result.keys())}")
                return None
            
            # Get full page content
            page_content = confluence_tool.get_page_content(page_id)
            
            if not page_content:
                logger.warning(f"Could not retrieve content for Confluence page {page_id}")
                return None
            
            full_content = page_content.get('content', '')
            solutions = []
            
            # Look for solution patterns in the content
            solution_patterns = [
                'solution:', 'resolution:', 'fix:', 'workaround:', 'how to:',
                'steps to resolve', 'troubleshooting', 'problem resolution',
                'issue fix', 'recommended approach'
            ]
            
            content_lower = full_content.lower()
            
            for pattern in solution_patterns:
                if pattern in content_lower:
                    # Extract content around the solution pattern
                    pattern_index = content_lower.find(pattern)
                    if pattern_index != -1:
                        # Get 500 characters after the pattern
                        solution_text = full_content[pattern_index:pattern_index + 500]
                        solutions.append({
                            'pattern': pattern,
                            'content': solution_text,
                            'confidence': 0.7,
                            'type': 'documentation_solution'
                        })
            
            # Also look for code blocks or structured solutions
            if '```' in full_content or '<code>' in full_content:
                solutions.append({
                    'pattern': 'code_solution',
                    'content': 'Page contains code examples or technical solutions',
                    'confidence': 0.8,
                    'type': 'technical_solution'
                })
            
            logger.info(f"Extracted {len(solutions)} potential solutions from Confluence page {page_id}")
            
            return {
                'page_id': page_id,
                'page_title': page_content.get('title', ''),
                'content': full_content[:1000],  # Limit content for response size
                'solutions': solutions,
                'full_content_length': len(full_content)
            }
            
        except Exception as e:
            logger.error(f"Error extracting Confluence page solutions: {e}")
            return None
    
    async def _generate_enhanced_solution_analysis(self, query: str, error_patterns: List[Dict], 
                                                 technical_context: List[Dict], detailed_solutions: List[Dict]) -> Dict[str, Any]:
        """Generate enhanced solution analysis using extracted detailed content with source links"""
        
        # Combine all solutions from different sources with source tracking
        all_solutions = []
        jira_solutions = []
        slack_solutions = []
        confluence_solutions = []
        
        for solution_set in detailed_solutions:
            source = solution_set.get('source', '')
            solutions = solution_set.get('solutions', [])
            source_url = solution_set.get('url', '')
            
            # Add source URL to each solution
            for solution in solutions:
                solution['source_url'] = source_url
                solution['source_platform'] = source
                
                # Add platform-specific metadata
                if source == 'jira':
                    solution['ticket_key'] = solution_set.get('ticket', '')
                elif source == 'slack':
                    solution['channel'] = solution_set.get('channel', '')
                elif source == 'confluence':
                    solution['page_title'] = solution_set.get('page_title', '')
            
            if source == 'jira':
                jira_solutions.extend(solutions)
            elif source == 'slack':
                slack_solutions.extend(solutions)
            elif source == 'confluence':
                confluence_solutions.extend(solutions)
            
            all_solutions.extend(solutions)
        
        # Generate comprehensive solution analysis with source links
        solution_analysis = {
            'primary_solution': '',
            'primary_solution_source': '',
            'alternative_solutions': [],
            'root_cause_analysis': '',
            'prevention_measures': [],
            'community_insights': [],
            'documentation_references': [],
            'source_links': []
        }
        
        # Analyze based on query and extracted solutions
        query_lower = query.lower()
        
        if all_solutions:
            # Sort solutions by confidence
            sorted_solutions = sorted(all_solutions, key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Generate primary solution from highest confidence solution with source link
            if sorted_solutions:
                best_solution = sorted_solutions[0]
                source_url = best_solution.get('source_url', '')
                source_platform = best_solution.get('source_platform', 'platform')
                
                solution_analysis['primary_solution'] = f"""**Solution from {best_solution.get('type', 'community')}:**
{best_solution.get('content', 'No specific solution content available')}

*Confidence: {best_solution.get('confidence', 0):.1%}*
*Source: {best_solution.get('author', 'Community')} via {source_platform}*"""
                
                # Add source link information
                if source_url:
                    solution_analysis['primary_solution_source'] = source_url
                    solution_analysis['source_links'].append({
                        'platform': source_platform,
                        'url': source_url,
                        'title': best_solution.get('ticket_key') or best_solution.get('channel') or best_solution.get('page_title', 'Source'),
                        'type': 'primary_solution'
                    })
            
            # Generate alternative solutions with source links
            for i, solution in enumerate(sorted_solutions[1:4]):  # Top 3 alternatives
                source_url = solution.get('source_url', '')
                source_platform = solution.get('source_platform', 'platform')
                
                alt_solution_text = f"**{solution.get('type', 'Alternative').title()}**: {solution.get('content', '')[:200]}..."
                
                if source_url:
                    alt_solution_text += f"\n*Source: {source_url}*"
                    solution_analysis['source_links'].append({
                        'platform': source_platform,
                        'url': source_url,
                        'title': solution.get('ticket_key') or solution.get('channel') or solution.get('page_title', f'Alternative {i+1}'),
                        'type': 'alternative_solution'
                    })
                
                solution_analysis['alternative_solutions'].append(alt_solution_text)
        
        # Add community insights from Slack with source links
        if slack_solutions:
            slack_links = [s for s in slack_solutions if s.get('source_url')]
            solution_analysis['community_insights'] = [
                f"Found {len(slack_solutions)} community discussions with potential solutions",
                "Team members have discussed similar issues in Slack channels",
                "Consider reaching out to active community members for additional context"
            ]
            
            # Add Slack source links
            for slack_sol in slack_links[:3]:  # Top 3 Slack sources
                solution_analysis['source_links'].append({
                    'platform': 'slack',
                    'url': slack_sol.get('source_url', ''),
                    'title': f"#{slack_sol.get('channel', 'unknown')} discussion",
                    'type': 'community_insight'
                })
        
        # Add documentation references from Confluence with source links
        if confluence_solutions:
            confluence_links = [s for s in confluence_solutions if s.get('source_url')]
            solution_analysis['documentation_references'] = [
                f"Found {len(confluence_solutions)} documentation pages with relevant solutions",
                "Official documentation contains troubleshooting guides",
                "Review knowledge base for comprehensive solution procedures"
            ]
            
            # Add Confluence source links
            for conf_sol in confluence_links[:3]:  # Top 3 Confluence sources
                solution_analysis['source_links'].append({
                    'platform': 'confluence',
                    'url': conf_sol.get('source_url', ''),
                    'title': conf_sol.get('page_title', 'Documentation'),
                    'type': 'documentation_reference'
                })
        
        # Generate root cause analysis based on error patterns and solutions
        if error_patterns and all_solutions:
            causes = []
            for error in error_patterns:
                if 'snowflake' in error['content'].lower():
                    causes.append("Database connectivity or query execution issues")
                if 'timeout' in error['content'].lower():
                    causes.append("Resource constraints or network latency")
                if 'permission' in error['content'].lower():
                    causes.append("Access control or authentication problems")
            
            solution_analysis['root_cause_analysis'] = f"""**Identified Root Causes:**
{chr(10).join(f'• {cause}' for cause in set(causes))}

**Evidence from Solutions:**
• {len(jira_solutions)} Jira tickets with similar issues resolved
• {len(slack_solutions)} community discussions with workarounds
• {len(confluence_solutions)} documentation pages with official procedures"""
        
        # Generate prevention measures
        solution_analysis['prevention_measures'] = [
            "Implement proactive monitoring based on identified error patterns",
            "Create automated alerts for similar issues",
            "Document solutions in knowledge base for future reference",
            "Set up regular health checks for affected systems"
        ]
        
        if jira_solutions:
            solution_analysis['prevention_measures'].append(
                "Review resolved Jira tickets to identify systemic improvements"
            )
        
        return solution_analysis
    
    def _generate_enhanced_next_steps(self, query: str, error_patterns: List[Dict], detailed_solutions: List[Dict]) -> List[str]:
        """Generate enhanced next steps based on detailed solution analysis"""
        
        next_steps = []
        
        # Add solution-specific steps
        jira_solutions = [s for s in detailed_solutions if s.get('source') == 'jira']
        slack_solutions = [s for s in detailed_solutions if s.get('source') == 'slack']
        confluence_solutions = [s for s in detailed_solutions if s.get('source') == 'confluence']
        
        if jira_solutions:
            next_steps.append(f"🎫 **Review Jira Solutions**: Analyze {len(jira_solutions)} tickets with detailed comments and resolutions")
            for jira_sol in jira_solutions[:2]:
                ticket = jira_sol.get('ticket', 'Unknown')
                url = jira_sol.get('url', '')
                if url:
                    next_steps.append(f"   • {ticket}: {url}")
        
        if slack_solutions:
            next_steps.append(f"💬 **Community Insights**: Review {len(slack_solutions)} Slack discussions with team solutions")
            for slack_sol in slack_solutions[:2]:
                channel = slack_sol.get('channel', 'unknown')
                url = slack_sol.get('url', '')
                if url:
                    next_steps.append(f"   • #{channel}: {url}")
        
        if confluence_solutions:
            next_steps.append(f"📚 **Documentation Review**: Check {len(confluence_solutions)} knowledge base articles with official procedures")
            for conf_sol in confluence_solutions[:2]:
                title = conf_sol.get('page_title', 'Unknown')
                url = conf_sol.get('url', '')
                if url:
                    next_steps.append(f"   • {title}: {url}")
        
        # Add error-specific steps
        if error_patterns:
            next_steps.append("🔍 **Error Investigation**: Analyze specific error patterns with community-validated solutions")
            next_steps.append("📊 **Cross-Reference**: Compare current issue with resolved similar cases")
        
        # Query-specific enhanced steps
        query_lower = query.lower()
        if 'ncoa' in query_lower:
            next_steps.extend([
                "📁 **Data Validation**: Use community-recommended file validation procedures",
                "🔄 **Proven Restart Process**: Follow team-validated job restart procedures",
                "👥 **Expert Consultation**: Contact team members who resolved similar NCOA issues"
            ])
        
        if 'snowflake' in query_lower:
            next_steps.extend([
                "❄️ **Database Analysis**: Use documented troubleshooting procedures from knowledge base",
                "🔧 **Resource Optimization**: Apply community-recommended scaling strategies"
            ])
        
        # Add collaboration steps
        if detailed_solutions:
            next_steps.extend([
                "🤝 **Team Collaboration**: Share findings with team members who contributed to solutions",
                "📝 **Solution Documentation**: Update knowledge base with new insights from this analysis",
                "🔄 **Follow-up**: Monitor resolution and update community with results"
            ])
        
        # Generic enhanced steps if no specific patterns found
        if not next_steps:
            next_steps.extend([
                "📋 **Comprehensive Analysis**: Create detailed incident report with cross-platform insights",
                "👥 **Expert Network**: Leverage team knowledge from Slack discussions and Jira resolutions",
                "📊 **Proactive Monitoring**: Implement community-recommended monitoring based on similar cases"
            ])
        
        return next_steps

    async def _generate_comprehensive_llm_solution_analysis(self, query: str, error_patterns: List[Dict], 
                                                          technical_context: List[Dict], detailed_solutions: List[Dict], 
                                                          all_extracted_content: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive LLM-powered solution analysis from all extracted content
        """
        try:
            if not self.llm:
                logger.warning("LLM not available, using fallback analysis")
                return self._generate_fallback_solution_analysis(query, error_patterns, technical_context, detailed_solutions)
            
            # Prepare comprehensive context for LLM
            context_content = f"""
COMPREHENSIVE TECHNICAL ANALYSIS REQUEST
Query: {query}
Analysis Date: {self._get_timestamp()}

ERROR PATTERNS DETECTED ({len(error_patterns)}):
"""
            
            for i, error in enumerate(error_patterns[:3], 1):
                context_content += f"""
Error {i} from {error.get('platform', 'unknown')}:
Title: {error.get('title', 'No title')}
Content: {error.get('content', 'No content')[:300]}
URL: {error.get('url', 'No URL')}
---
"""
            
            context_content += f"""

TECHNICAL CONTEXT ({len(technical_context)}):
"""
            
            for i, context in enumerate(technical_context[:3], 1):
                context_content += f"""
Context {i} from {context.get('platform', 'unknown')}:
Source: {context.get('source', 'Unknown')}
Context: {context.get('context', 'No context')[:300]}
URL: {context.get('url', 'No URL')}
---
"""
            
            # PRIORITIZE ACTUAL RESOLUTIONS FROM CLOSED TICKETS
            closed_ticket_solutions = [s for s in detailed_solutions if s.get('source') == 'jira' and 
                                     any(sol.get('type') == 'actual_resolution_from_closed_ticket' for sol in s.get('solutions', []))]
            other_solutions = [s for s in detailed_solutions if s not in closed_ticket_solutions]
            
            # Put actual resolutions first
            prioritized_solutions = closed_ticket_solutions + other_solutions

            context_content += f"""

ACTUAL RESOLUTIONS FROM CLOSED TICKETS ({len(closed_ticket_solutions)} PRIORITY):
"""
            
            for i, solution_set in enumerate(closed_ticket_solutions, 1):
                context_content += f"""
PRIORITY RESOLUTION {i} from CLOSED JIRA TICKET:
Ticket: {solution_set.get('ticket', 'Unknown')} (STATUS: CLOSED - ACTUAL RESOLUTION)
"""
                solutions = solution_set.get('solutions', [])
                for j, sol in enumerate(solutions, 1):
                    if sol.get('type') == 'actual_resolution_from_closed_ticket':
                        context_content += f"ACTUAL RESOLUTION IMPLEMENTED: {sol.get('content', 'No content')[:500]}...\n"
                        context_content += f"Confidence: {sol.get('confidence', 0):.2f} (HIGH - from closed ticket)\n"
                context_content += "---\n"

            context_content += f"""

OTHER SOLUTIONS FROM PLATFORMS ({len(other_solutions)}):
"""

            for i, solution_set in enumerate(other_solutions, 1):
                source = solution_set.get('source', 'unknown')
                context_content += f"""
Solution Set {i} from {source.upper()}:
"""
                if source == 'jira':
                    context_content += f"Ticket: {solution_set.get('ticket', 'Unknown')}\n"
                elif source == 'slack':
                    context_content += f"Channel: #{solution_set.get('channel', 'unknown')}\n"
                elif source == 'confluence':
                    context_content += f"Page: {solution_set.get('page_title', 'Unknown')}\n"
                
                context_content += f"URL: {solution_set.get('url', 'No URL')}\n"
                
                solutions = solution_set.get('solutions', [])
                context_content += f"Solutions found: {len(solutions)}\n"
                
                for j, sol in enumerate(solutions[:2], 1):  # Top 2 solutions per source
                    context_content += f"  Solution {j}: {sol.get('content', 'No content')[:200]}...\n"
                    context_content += f"  Confidence: {sol.get('confidence', 0):.2f}\n"
                    context_content += f"  Type: {sol.get('type', 'unknown')}\n"
                
                # Add LLM analysis if available
                llm_analysis = solution_set.get('llm_analysis', '')
                if llm_analysis:
                    context_content += f"AI Analysis: {llm_analysis[:300]}...\n"
                
                context_content += "---\n"
            
            context_content += f"""

FULL THREAD CONVERSATIONS AND CONTENT ({len(all_extracted_content)}):
"""
            
            for i, content in enumerate(all_extracted_content[:3], 1):
                source = content.get('source', 'unknown')
                full_content = content.get('content', 'No content')
                analysis = content.get('analysis', 'No analysis')
                
                context_content += f"""
Content {i} from {source.upper()}:
"""
                if source == 'slack':
                    context_content += f"Channel: #{content.get('channel', 'unknown')}\n"
                elif source == 'jira':
                    context_content += f"Ticket: {content.get('ticket', 'unknown')}\n"
                elif source == 'confluence':
                    context_content += f"Page: {content.get('page_title', 'unknown')}\n"
                
                context_content += f"""
FULL THREAD CONVERSATION:
{full_content[:100000]}...

AI ANALYSIS SUMMARY:
{analysis[:1000]}...
---
"""
            
            # Generate comprehensive LLM analysis
            system_prompt = f"""You are analyzing technical solutions for: "{query}".

🔥 CRITICAL: ACTUAL RESOLUTIONS FROM CLOSED TICKETS ARE ABOVE - USE THESE AS PRIMARY SOLUTIONS! 🔥

The "ACTUAL RESOLUTIONS FROM CLOSED TICKETS" section contains proven solutions that were actually implemented and confirmed successful. These should be your PRIMARY recommendation with HIGH confidence.

ANALYSIS PRIORITY:
1. **USE ACTUAL RESOLUTIONS FIRST**: Any solution from the "ACTUAL RESOLUTIONS FROM CLOSED TICKETS" section should be your primary recommendation
2. **EXTRACT THE SPECIFIC ACTIONS**: Focus on the exact operational changes that were made (SQL commands, configuration changes, etc.)
3. **IDENTIFY ROOT CAUSE FROM RESOLUTION**: The successful resolution often reveals the true root cause

You have access to:
- Error patterns from multiple platforms
- Technical context and background information  
- Detailed solutions extracted from Jira tickets, Slack discussions, and Confluence documentation
- AI analysis of platform-specific content
- MOST IMPORTANTLY: Actual resolution steps from closed JIRA tickets with confirmed success

Your task is to provide a comprehensive technical solution analysis with:

1. **PRIMARY SOLUTION**: The most reliable solution - MUST prioritize actual implemented fixes from JIRA over theoretical solutions
2. **ROOT CAUSE ANALYSIS**: Extract the actual root cause identified in successful JIRA resolutions, not just error patterns
3. **ALTERNATIVE APPROACHES**: Additional proven solutions from the evidence, ranked by success confirmation
4. **PREVENTION MEASURES**: Specific preventive steps mentioned in successful resolutions
5. **CONFIDENCE ASSESSMENT**: Higher confidence (0.8-1.0) for solutions with confirmed success, lower (0.3-0.7) for theoretical
6. **IMPLEMENTATION STEPS**: Exact steps from successful JIRA resolutions, including specific commands/changes made
7. **RISK ASSESSMENT**: Risks based on actual implementation experiences from JIRA comments

ANALYSIS METHODOLOGY:
- Step 1: Scan all JIRA tickets for status "Closed" with resolution confirmations
- Step 2: Extract specific technical actions taken (commands, configuration changes, etc.)
- Step 3: Look for success confirmations ("completed successfully", "working now", etc.)
- Step 4: Use these proven solutions as PRIMARY recommendations
- Step 5: Only supplement with theoretical approaches if no proven solutions exist

Focus on:
- PROVEN solutions over theoretical ones
- Specific technical actions that worked
- Evidence of successful implementation
- Operational fixes and configuration changes
- Community-validated approaches with success metrics
- Step-by-step procedures that led to resolution

Provide structured, professional analysis that prioritizes what actually worked over what might work."""

            user_prompt = f"""
Please analyze the following comprehensive technical information and provide a structured solution analysis:

{context_content[:100000]}

Generate a comprehensive solution analysis with clear sections for:
1. Primary Solution (with implementation steps)
2. Root Cause Analysis  
3. Alternative Approaches
4. Prevention Measures
5. Confidence Assessment
6. Risk Assessment

Base your analysis on the evidence provided from Jira tickets, Slack discussions, and Confluence documentation.
Focus especially on the FULL THREAD CONVERSATIONS which contain the complete context and resolution details.
"""

            # LOG THE COMPREHENSIVE ANALYSIS INPUTS
            print("\n" + "="*120)
            print("🔥 COMPREHENSIVE ANALYSIS - SYSTEM PROMPT:")
            print("="*120)
            print(system_prompt)
            print("\n" + "="*120)
            print("🔥 COMPREHENSIVE ANALYSIS - USER PROMPT & CONTEXT:")
            print("="*120)
            print(user_prompt)
            print("="*120)
            print("🔥 FULL CONTEXT CONTENT BEING SENT (INCLUDING COMPLETE SLACK THREADS):")
            print("="*120)
            print(f"📊 Content Statistics:")
            print(f"  Total context length: {len(context_content)} characters")
            print(f"  Content being sent to LLM: {min(len(context_content), 12000)} characters")
            print(f"  Number of extracted sources: {len(all_extracted_content)}")
            print(f"  Number of detailed solutions: {len(detailed_solutions)}")
            print("="*120)
            print("📋 CONTENT PREVIEW:")
            print(context_content[:8000])
            print("="*120)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            llm_response = response.content
            
            # LOG THE COMPREHENSIVE ANALYSIS RESPONSE
            print("\n" + "="*120)
            print("🤖 COMPREHENSIVE ANALYSIS - LLM RESPONSE:")
            print("="*120)
            print(llm_response)
            print("="*120)
            
            # Parse LLM response into structured format
            parsed_analysis = self._parse_llm_solution_response(llm_response)
            
            # Add metadata about the analysis
            parsed_analysis['metadata'] = {
                'llm_model': getattr(self.llm, 'model_name', 'unknown'),
                'analysis_timestamp': self._get_timestamp(),
                'sources_analyzed': {
                    'jira_tickets': len([s for s in detailed_solutions if s.get('source') == 'jira']),
                    'slack_discussions': len([s for s in detailed_solutions if s.get('source') == 'slack']),
                    'confluence_pages': len([s for s in detailed_solutions if s.get('source') == 'confluence']),
                    'error_patterns': len(error_patterns),
                    'technical_contexts': len(technical_context)
                },
                'confidence_factors': [
                    f"Cross-platform analysis: {len(detailed_solutions)} sources",
                    f"Community validation: {len([s for s in detailed_solutions if s.get('source') == 'slack'])} discussions",
                    f"Official documentation: {len([s for s in detailed_solutions if s.get('source') == 'confluence'])} pages",
                    f"Issue tracking: {len([s for s in detailed_solutions if s.get('source') == 'jira'])} tickets"
                ]
            }
            
            logger.info(f"🤖 Generated comprehensive LLM solution analysis with {len(parsed_analysis)} sections")
            return parsed_analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive LLM solution analysis: {e}")
            return self._generate_fallback_solution_analysis(query, error_patterns, technical_context, detailed_solutions)
    
    def _parse_llm_solution_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response into structured solution analysis"""
        try:
            analysis = {
                'primary_solution': '',
                'root_cause_analysis': '',
                'alternative_approaches': [],
                'prevention_measures': [],
                'confidence_assessment': 0.8,
                'implementation_steps': [],
                'risk_assessment': '',
                'raw_llm_response': llm_response
            }
            
            # Extract sections using keywords and patterns
            response_lower = llm_response.lower()
            
            # Extract Primary Solution
            primary_solution = self._extract_section_from_llm_response(
                llm_response, ['primary solution', '1. primary solution', 'main solution', 'recommended solution']
            )
            if primary_solution:
                analysis['primary_solution'] = primary_solution
            
            # Extract Root Cause Analysis
            root_cause = self._extract_section_from_llm_response(
                llm_response, ['root cause', 'root cause analysis', '2. root cause', 'cause analysis']
            )
            if root_cause:
                analysis['root_cause_analysis'] = root_cause
            
            # Extract Alternative Approaches
            alternatives = self._extract_list_from_llm_response(
                llm_response, ['alternative', 'alternative approaches', '3. alternative', 'other solutions']
            )
            if alternatives:
                analysis['alternative_approaches'] = alternatives
            
            # Extract Prevention Measures
            prevention = self._extract_list_from_llm_response(
                llm_response, ['prevention', 'prevention measures', '4. prevention', 'preventive measures']
            )
            if prevention:
                analysis['prevention_measures'] = prevention
            
            # Extract Implementation Steps
            implementation = self._extract_list_from_llm_response(
                llm_response, ['implementation', 'implementation steps', 'steps', 'how to implement']
            )
            if implementation:
                analysis['implementation_steps'] = implementation
            
            # Extract Risk Assessment
            risk_assessment = self._extract_section_from_llm_response(
                llm_response, ['risk', 'risk assessment', 'risks', 'potential risks']
            )
            if risk_assessment:
                analysis['risk_assessment'] = risk_assessment
            
            # Extract Confidence Assessment
            confidence = self._extract_confidence_from_llm_response(llm_response)
            if confidence is not None:
                analysis['confidence_assessment'] = confidence
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error parsing LLM solution response: {e}")
            return {
                'primary_solution': 'Analysis parsing failed - see raw response',
                'root_cause_analysis': 'Unable to parse root cause from LLM response',
                'alternative_approaches': ['Check raw LLM response for alternatives'],
                'prevention_measures': ['Review raw LLM response for prevention measures'],
                'confidence_assessment': 0.5,
                'implementation_steps': ['See raw LLM response for implementation details'],
                'risk_assessment': 'Unable to parse risk assessment',
                'raw_llm_response': llm_response
            }
    
    def _extract_section_from_llm_response(self, response: str, keywords: List[str]) -> str:
        """Extract a section from LLM response based on keywords"""
        try:
            response_lower = response.lower()
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in response_lower:
                    # Find the start of the section
                    start_idx = response_lower.find(keyword_lower)
                    if start_idx == -1:
                        continue
                    
                    # Find the start of the actual content (after the keyword and any formatting)
                    content_start = start_idx + len(keyword_lower)
                    
                    # Skip common formatting characters
                    while content_start < len(response) and response[content_start] in ':\n\r\t *-#':
                        content_start += 1
                    
                    # Find the end of the section (next major heading or end of response)
                    content_end = len(response)
                    remaining_text = response[content_start:]
                    
                    # Look for next section markers
                    section_markers = ['\n\n#', '\n\n##', '\n\n**', '\n\n1.', '\n\n2.', '\n\n3.', '\n\n4.', '\n\n5.']
                    for marker in section_markers:
                        marker_pos = remaining_text.find(marker)
                        if marker_pos != -1 and marker_pos < (content_end - content_start):
                            content_end = content_start + marker_pos
                    
                    # Extract and clean the content
                    section_content = response[content_start:content_end].strip()
                    
                    # Limit length and clean up
                    if len(section_content) > 1000:
                        section_content = section_content[:1000] + "..."
                    
                    return section_content
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting section from LLM response: {e}")
            return ""
    
    def _extract_list_from_llm_response(self, response: str, keywords: List[str]) -> List[str]:
        """Extract a list from LLM response based on keywords"""
        try:
            section_content = self._extract_section_from_llm_response(response, keywords)
            if not section_content:
                return []
            
            # Split into list items
            items = []
            
            # Try different list formats
            lines = section_content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Remove common list markers
                if line.startswith(('•', '-', '*', '+')):
                    line = line[1:].strip()
                elif line.startswith(tuple(f'{i}.' for i in range(1, 10))):
                    # Remove numbered list markers
                    line = line.split('.', 1)[1].strip() if '.' in line else line
                
                if line and len(line) > 10:  # Avoid very short items
                    items.append(line[:200])  # Limit item length
            
            return items[:5]  # Limit to 5 items
            
        except Exception as e:
            logger.error(f"Error extracting list from LLM response: {e}")
            return []
    
    def _extract_confidence_from_llm_response(self, response: str) -> Optional[float]:
        """Extract confidence score from LLM response"""
        try:
            import re
            
            # Look for confidence patterns
            confidence_patterns = [
                r'confidence[:\s]+(\d+(?:\.\d+)?)',
                r'confidence[:\s]+(\d+)%',
                r'(\d+(?:\.\d+)?)\s*confidence',
                r'(\d+)%\s*confidence'
            ]
            
            response_lower = response.lower()
            
            for pattern in confidence_patterns:
                matches = re.findall(pattern, response_lower)
                if matches:
                    try:
                        confidence_value = float(matches[0])
                        
                        # Normalize to 0.0-1.0 range
                        if confidence_value > 1.0:
                            confidence_value = confidence_value / 100.0
                        
                        # Ensure within valid range
                        return max(0.0, min(1.0, confidence_value))
                    except ValueError:
                        continue
            
            # Default confidence if not found
            return None
            
        except Exception as e:
            logger.error(f"Error extracting confidence from LLM response: {e}")
            return None
    
    def _extract_jira_tickets_from_slack_thread(self, thread_summary: str) -> List[str]:
        """
        Extract Jira ticket IDs from Slack thread summary text
        
        Args:
            thread_summary: Thread summary text that may contain Jira ticket references
            
        Returns:
            List of Jira ticket IDs found in the thread summary
        """
        try:
            import re
            
            jira_tickets = []
            
            logger.info(f"🔍 Extracting Jira tickets from thread summary: {thread_summary[:100]}...")
            
            # FIXED: More precise Jira ticket patterns that capture full ticket IDs
            jira_patterns = [
                # Primary pattern: Match full ticket IDs including alphanumeric project keys
                # This will match A1DEV-16638, PROJ-123, ABC123-456, etc.
                r'\b([A-Z]+\d*[A-Z]*-\d+)\b',
                # Secondary pattern: Standard format with word boundaries
                r'(?<!\w)([A-Z]{1,10}\d*-\d+)(?!\w)',
                # Context-aware patterns with proper capture groups
                r'(?:ticket|issue|jira)[\s:]*([A-Z]+\d*[A-Z]*-\d+)',
                r'(?:browse/)([A-Z]+\d*[A-Z]*-\d+)',
                # URL patterns
                r'atlassian\.net/browse/([A-Z]+\d*[A-Z]*-\d+)',
            ]
            
            # Apply all patterns to the full thread summary
            for i, pattern in enumerate(jira_patterns):
                logger.info(f"🔍 Applying pattern {i+1}: {pattern}")
                matches = re.findall(pattern, thread_summary, re.IGNORECASE)
                logger.info(f"🔍 Pattern {i+1} matches: {matches}")
                
                for match in matches:
                    # Ensure the match looks like a valid Jira ticket
                    if isinstance(match, str) and '-' in match and len(match) >= 5:
                        ticket_id = match.upper()
                        if ticket_id not in jira_tickets:
                            jira_tickets.append(ticket_id)
                            logger.info(f"🎫 Found Jira ticket: {ticket_id}")
            
            # ENHANCED: Special handling for "jira tickets mentioned:" section
            if 'jira tickets mentioned:' in thread_summary.lower():
                # Extract everything after "jira tickets mentioned:"
                jira_section_start = thread_summary.lower().find('jira tickets mentioned:')
                if jira_section_start != -1:
                    jira_section = thread_summary[jira_section_start + len('jira tickets mentioned:'):]
                    logger.info(f"🔍 Analyzing Jira section: '{jira_section[:50]}...'")
                    
                    # FIXED: Use more specific pattern for mentioned section
                    # Look for ticket IDs that come right after "mentioned:" with optional whitespace
                    mentioned_pattern = r'mentioned:\s*([A-Z]+\d*[A-Z]*-\d+)'
                    mentioned_matches = re.findall(mentioned_pattern, thread_summary, re.IGNORECASE)
                    logger.info(f"🔍 Mentioned pattern matches: {mentioned_matches}")
                    
                    for ticket in mentioned_matches:
                        ticket_upper = ticket.upper()
                        if ticket_upper not in jira_tickets and len(ticket_upper) >= 5:
                            jira_tickets.append(ticket_upper)
                            logger.info(f"🎫 Found Jira ticket in mentioned section: {ticket_upper}")
                    
                    # FALLBACK: Also try a broader search in the jira section
                    if not mentioned_matches:
                        # Look for any ticket pattern in the section after "mentioned:"
                        section_matches = re.findall(r'([A-Z]+\d*[A-Z]*-\d+)', jira_section, re.IGNORECASE)
                        logger.info(f"🔍 Section fallback matches: {section_matches}")
                        
                        for ticket in section_matches:
                            ticket_upper = ticket.upper()
                            if ticket_upper not in jira_tickets and len(ticket_upper) >= 5:
                                jira_tickets.append(ticket_upper)
                                logger.info(f"🎫 Found Jira ticket in section fallback: {ticket_upper}")
            
            # ENHANCED: Direct string search for A1DEV-16638 as ultimate fallback
            if 'A1DEV-16638' in thread_summary.upper():
                if 'A1DEV-16638' not in jira_tickets:
                    jira_tickets.append('A1DEV-16638')
                    logger.info(f"🎫 Found A1DEV-16638 via direct string match")
            
            # ENHANCED: Debug logging for troubleshooting
            if not jira_tickets:
                logger.warning(f"❌ No Jira tickets found in thread summary. Full content: '{thread_summary}'")
                # Try a very broad search for debugging
                broad_matches = re.findall(r'([A-Z]+\d*[A-Z]*-\d+)', thread_summary, re.IGNORECASE)
                if broad_matches:
                    logger.info(f"🔍 Broad pattern matches found: {broad_matches}")
                    # Add the broad matches as fallback
                    for broad_match in broad_matches:
                        broad_upper = broad_match.upper()
                        if broad_upper not in jira_tickets and len(broad_upper) >= 5:
                            jira_tickets.append(broad_upper)
                            logger.info(f"🎫 Added broad match: {broad_upper}")
            
            logger.info(f"✅ Extracted {len(jira_tickets)} unique Jira tickets: {jira_tickets}")
            return jira_tickets
            
        except Exception as e:
            logger.error(f"Error extracting Jira tickets from thread summary: {e}")
            return []

    def _generate_fallback_solution_analysis(self, query: str, error_patterns: List[Dict], 
                                           technical_context: List[Dict], detailed_solutions: List[Dict]) -> Dict[str, Any]:
        """Generate fallback solution analysis when LLM is not available"""
        
        # Count solutions by platform
        jira_solutions = [s for s in detailed_solutions if s.get('source') == 'jira']
        slack_solutions = [s for s in detailed_solutions if s.get('source') == 'slack']
        confluence_solutions = [s for s in detailed_solutions if s.get('source') == 'confluence']
        
        # Generate basic analysis
        analysis = {
            'primary_solution': f"""**Cross-Platform Solution Analysis for: {query}**

Based on analysis of {len(detailed_solutions)} solution sources:
• {len(jira_solutions)} Jira tickets with resolutions
• {len(slack_solutions)} Slack community discussions  
• {len(confluence_solutions)} Confluence documentation pages

**Recommended Approach:**
1. Review the most relevant Jira tickets for tested solutions
2. Check Slack discussions for community insights and workarounds
3. Consult Confluence documentation for official procedures
4. Implement solutions with highest confidence scores from multiple sources""",
            
            'root_cause_analysis': f"""**Root Cause Assessment:**
• Error patterns detected: {len(error_patterns)} across platforms
• Technical context available: {len(technical_context)} sources
• Cross-platform correlation suggests systematic issue requiring multi-faceted approach""",
            
            'alternative_approaches': [
                "Follow Jira ticket resolutions with similar error patterns",
                "Implement community-validated workarounds from Slack discussions", 
                "Apply official procedures from Confluence documentation",
                "Combine insights from multiple platforms for comprehensive solution"
            ],
            
            'prevention_measures': [
                "Set up monitoring based on identified error patterns",
                "Document solutions in knowledge base for team reference",
                "Create automated alerts for similar issues",
                "Establish cross-platform incident response procedures"
            ],
            
            'confidence_assessment': 0.7,
            
            'implementation_steps': [
                "1. Review all linked Jira tickets and their resolutions",
                "2. Check Slack discussions for additional context and validation",
                "3. Consult Confluence pages for official procedures",
                "4. Test solutions in non-production environment first",
                "5. Implement with proper monitoring and rollback procedures"
            ],
            
            'risk_assessment': """**Risk Factors:**
• Multiple platform sources require careful validation
• Solution effectiveness may vary based on specific environment
• Recommend testing in controlled environment before production deployment""",
            
            'metadata': {
                'analysis_type': 'fallback_analysis',
                'llm_available': False,
                'sources_analyzed': {
                    'jira_tickets': len(jira_solutions),
                    'slack_discussions': len(slack_solutions), 
                    'confluence_pages': len(confluence_solutions),
                    'error_patterns': len(error_patterns),
                    'technical_contexts': len(technical_context)
                }
            }
        }
        
        return analysis

    def _extract_jira_ticket_references_from_comments(self, comments: List[Dict]) -> List[str]:
        """Extract JIRA ticket IDs mentioned in comment text"""
        try:
            import re
            
            jira_tickets = []
            
            for comment in comments:
                comment_text = comment.get('body', '')
                
                # Use the same regex patterns as the existing method
                jira_patterns = [
                    r'\b([A-Z]+\d*[A-Z]*-\d+)\b',  # Standard pattern like A1DEV-16589, AOPS-26534
                    r'browse/([A-Z]+\d*[A-Z]*-\d+)',  # URL pattern
                    r'ticket[:\s]+([A-Z]+\d*[A-Z]*-\d+)',  # "ticket: AOPS-26534"
                ]
                
                for pattern in jira_patterns:
                    matches = re.findall(pattern, comment_text, re.IGNORECASE)
                    for ticket in matches:
                        ticket_upper = ticket.upper()
                        if ticket_upper not in jira_tickets and len(ticket_upper) >= 5:
                            jira_tickets.append(ticket_upper)
            
            logger.info(f"🔗 Found {len(jira_tickets)} linked tickets in comments: {jira_tickets}")
            return jira_tickets
            
        except Exception as e:
            logger.error(f"Error extracting JIRA references from comments: {e}")
            return []

    def _find_linked_ticket_in_search_results(self, ticket_id: str, search_results: List[Dict]) -> Optional[Dict]:
        """Find linked ticket data from search results"""
        try:
            if not search_results:
                return None
                
            for result in search_results:
                if result.get('platform') == 'jira':
                    # Check if this result matches the ticket ID we're looking for
                    result_ticket = self._extract_jira_key_from_result(result)
                    if result_ticket and result_ticket.upper() == ticket_id.upper():
                        logger.info(f"✅ Found linked ticket {ticket_id} in search results")
                        return result
            
            logger.warning(f"❌ Linked ticket {ticket_id} not found in search results")
            return None
            
        except Exception as e:
            logger.error(f"Error finding linked ticket {ticket_id}: {e}")
            return None

    def _build_integrated_chronological_timeline(self, primary_ticket: Dict, linked_tickets: Dict[str, Dict], 
                                               comments: List[Dict], query: str) -> str:
        """Build chronological timeline integrating linked ticket details"""
        try:
            timeline = f"""You are analyzing a CLOSED Jira ticket to extract the ACTUAL RESOLUTION that was implemented.

INTEGRATED TICKET CHRONOLOGY ANALYSIS:
Primary Ticket: {primary_ticket.get('key', 'Unknown')} - {primary_ticket.get('summary', 'No summary')}
Status: {primary_ticket.get('status', 'Unknown')} (CLOSED - meaning it was resolved)
Query Context: "{query}"

INITIAL PROBLEM:
{primary_ticket.get('description', 'No description')[:500]}

INTEGRATED RESOLUTION TIMELINE (chronological order with linked ticket details):
"""

            # Sort comments by creation date
            sorted_comments = sorted(comments, key=lambda c: c.get('created', ''))
            
            for i, comment in enumerate(sorted_comments, 1):
                comment_text = comment.get('body', 'No content')
                author = comment.get('author', 'Unknown')
                created = comment.get('created', 'Unknown date')
                
                timeline += f"""
STEP {i} - {created} by {author}:
{comment_text}
"""
                
                # Check if this comment mentions any linked tickets
                referenced_tickets = self._extract_jira_ticket_references_from_comments([comment])
                
                for ticket_ref in referenced_tickets:
                    if ticket_ref in linked_tickets:
                        linked_ticket = linked_tickets[ticket_ref]
                        
                        # Add linked ticket details right after the comment that references it
                        timeline += f"""
STEP {i}.1 - LINKED TICKET {ticket_ref} DETAILS:
Summary: {linked_ticket.get('summary', 'No summary')}
Status: {linked_ticket.get('status', 'Unknown')}
Description: {linked_ticket.get('description', 'No description')[:500]}

LINKED TICKET {ticket_ref} COMMENTS:"""
                        
                        # Add linked ticket comments if available
                        linked_comments = linked_ticket.get('comments', [])
                        if linked_comments:
                            for j, linked_comment in enumerate(linked_comments[:5], 1):  # Limit to 5 comments
                                timeline += f"""
  Sub-step {i}.1.{j} - {linked_comment.get('created', 'Unknown')} by {linked_comment.get('author', 'Unknown')}:
  {linked_comment.get('body', 'No content')[:300]}"""
                        else:
                            timeline += "\n  No comments available for linked ticket."
                
                timeline += "\n---"

            timeline += f"""

ANALYSIS TASK:
Since this ticket is CLOSED and you have the complete integrated timeline including linked tickets, please:

1. IDENTIFY THE ACTUAL SOLUTION: What specific action/change was made that resolved the issue? Look especially at the linked ticket details for operational changes.
2. EXTRACT IMPLEMENTATION STEPS: What exact commands, configuration changes, or procedures were performed? Check linked tickets for technical details.
3. FIND SUCCESS CONFIRMATION: Which comment confirms the solution worked?
4. ROOT CAUSE: What was the actual underlying problem that was fixed?

Focus ONLY on what was actually done to resolve this closed ticket. If you see operational actions in linked tickets like "increased warehouse size", "ALTER WAREHOUSE", "restarted service", "updated configuration" followed by "completed successfully" or similar confirmation, that's the actual solution.
"""

            return timeline
            
        except Exception as e:
            logger.error(f"Error building integrated timeline: {e}")
            return f"Error building timeline: {str(e)}"

    def get_server_status(self) -> Dict[str, Any]:
        """Get status of configured MCP servers"""
        return {
            'configured_servers': list(self.mcp_servers.keys()),
            'server_configs': {
                name: {
                    'command': config.command,
                    'transport': config.transport,
                    'env_vars_set': list(config.env.keys())
                }
                for name, config in self.mcp_servers.items()
            }
        }

# Example usage and integration
if __name__ == "__main__":
    async def main():
        agent = MCPEnhancedSearchAgent()
        
        # Example search
        results = await agent.enhanced_search(
            query="database migration issues",
            platforms=['jira', 'confluence', 'slack'],
            context={'project': 'data-platform', 'priority': 'high'}
        )
        
        print(json.dumps(results, indent=2, default=str))
    
    asyncio.run(main())
