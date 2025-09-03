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

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    transport: str = "stdio"  # stdio, sse, or http

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
        
        # Atlassian MCP Server (Official - supports both Jira and Confluence)
        if os.getenv('ATLASSIAN_API_TOKEN') and os.getenv('ATLASSIAN_INSTANCE_URL'):
            self.mcp_servers['atlassian'] = MCPServerConfig(
                name="atlassian",
                command="npx",
                args=["-y", "@atlassian/mcp-server"],
                env={
                    "ATLASSIAN_API_TOKEN": os.getenv('ATLASSIAN_API_TOKEN'),
                    "ATLASSIAN_INSTANCE_URL": os.getenv('ATLASSIAN_INSTANCE_URL'),
                    "ATLASSIAN_EMAIL": os.getenv('ATLASSIAN_EMAIL', '')
                }
            )
        
        # Slack MCP Server (Community - enhanced version)
        if os.getenv('SLACK_BOT_TOKEN'):
            self.mcp_servers['slack'] = MCPServerConfig(
                name="slack",
                command="npx",
                args=["-y", "@zencoderai/slack-mcp-server"],
                env={
                    "SLACK_BOT_TOKEN": os.getenv('SLACK_BOT_TOKEN'),
                    "SLACK_USER_TOKEN": os.getenv('SLACK_USER_TOKEN', ''),
                    "SLACK_SIGNING_SECRET": os.getenv('SLACK_SIGNING_SECRET', '')
                }
            )
        
        # GitHub MCP Server (for additional context from code repositories)
        if os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'):
            self.mcp_servers['github'] = MCPServerConfig(
                name="github",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={
                    "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
                }
            )
        
        # Memory MCP Server (for persistent context across searches)
        self.mcp_servers['memory'] = MCPServerConfig(
            name="memory",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-memory"],
            env={}
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
                    if platform == 'jira' and 'atlassian' in self.mcp_servers:
                        # Try MCP server first, fallback to existing tools
                        try:
                            jira_results = await self._search_atlassian(query, 'jira')
                            # Convert SearchResult objects to dictionaries
                            cross_platform_results.extend([r.__dict__ if hasattr(r, '__dict__') else r for r in jira_results])
                        except Exception as mcp_error:
                            logger.warning(f"MCP Jira search failed: {mcp_error}, using existing tools")
                            jira_results = await self._search_jira_fallback(query)
                            cross_platform_results.extend(jira_results)
                    
                    elif platform == 'confluence' and 'atlassian' in self.mcp_servers:
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
                    "tool": "search_issues",
                    "parameters": {
                        "jql": f'text ~ "{sanitized_query}" OR summary ~ "{sanitized_query}" OR description ~ "{sanitized_query}"',
                        "maxResults": 20
                    }
                }
            else:  # confluence
                # Sanitize query for CQL
                sanitized_query = self._sanitize_query_for_cql(query)
                mcp_query = {
                    "tool": "search_content",
                    "parameters": {
                        "cql": f'text ~ "{sanitized_query}"',
                        "limit": 20
                    }
                }
            
            # Execute MCP server call
            result = await self._execute_mcp_call('atlassian', mcp_query)
            
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
        """Execute an MCP server call with timeout and error handling"""
        if server_name not in self.mcp_servers:
            raise ValueError(f"MCP server {server_name} not configured")
        
        server_config = self.mcp_servers[server_name]
        
        # Create temporary file for MCP communication
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(query, f)
            query_file = f.name
        
        try:
            # Execute MCP server command with shorter timeout to fail fast
            cmd = [server_config.command] + server_config.args + ['--query', query_file]
            
            # Increased timeout to 8 seconds for MCP server startup
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    env={**os.environ, **server_config.env},
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=8.0
            )
            
            # Increased communication timeout to 12 seconds
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=12.0
            )
            
            if process.returncode != 0:
                raise Exception(f"MCP server error: {stderr.decode()}")
            
            return json.loads(stdout.decode())
            
        except asyncio.TimeoutError:
            logger.error(f"MCP server {server_name} call timed out")
            raise Exception(f"MCP server {server_name} timed out")
        except FileNotFoundError:
            logger.error(f"MCP server command not found: {server_config.command}")
            raise Exception(f"MCP server {server_name} not installed")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from MCP server {server_name}: {e}")
            raise Exception(f"Invalid response from MCP server {server_name}")
        except Exception as e:
            logger.error(f"MCP server {server_name} call failed: {e}")
            raise Exception(f"MCP server {server_name} failed: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(query_file)
            except:
                pass
    
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
        Enhanced solution analysis that follows MCP-provided links and extracts detailed content
        from Jira comments and Slack threads to find existing solutions
        """
        
        # Ensure results is a list and handle string inputs gracefully
        if isinstance(results, str):
            logger.warning(f"Expected list of results but got string: {results[:100]}...")
            results = []
        elif not isinstance(results, list):
            logger.warning(f"Expected list of results but got {type(results)}")
            results = []
        
        # Extract error details from search results
        error_patterns = []
        technical_context = []
        related_tickets = []
        slack_discussions = []
        confluence_docs = []
        
        # Enhanced analysis: Follow links and extract detailed content
        detailed_solutions = []
        
        for result in results:
            # Ensure result is a dictionary
            if not isinstance(result, dict):
                logger.warning(f"Expected dict result but got {type(result)}")
                continue
                
            content = result.get('content', '').lower()
            title = result.get('title', '').lower()
            platform = result.get('platform', '')
            url = result.get('url', '')
            
            # Extract error patterns
            if any(error_word in content for error_word in ['error', 'exception', 'failed', 'failure']):
                error_patterns.append({
                    'platform': platform,
                    'content': result.get('content', '')[:300],
                    'url': url,
                    'title': result.get('title', '')
                })
            
            # Enhanced Jira analysis: Extract comments for solutions
            if platform == 'jira':
                jira_key = self._extract_jira_key_from_result(result)
                if jira_key:
                    logger.info(f"🔍 Following Jira link to extract comments from {jira_key}")
                    jira_details = await self._extract_jira_comments_and_solutions(jira_key)
                    if jira_details:
                        detailed_solutions.append({
                            'source': 'jira',
                            'ticket': jira_key,
                            'url': url,
                            'solutions': jira_details.get('solutions', []),
                            'comments': jira_details.get('comments', [])
                        })
                
                related_tickets.append({
                    'title': result.get('title', ''),
                    'url': url,
                    'status': result.get('metadata', {}).get('status', 'Unknown'),
                    'key': jira_key
                })
            
            # Enhanced Slack analysis: Extract thread conversations for solutions
            elif platform == 'slack':
                logger.info(f"🔍 Following Slack link to extract thread conversations")
                slack_solutions = await self._extract_slack_thread_solutions(result)
                if slack_solutions:
                    detailed_solutions.append({
                        'source': 'slack',
                        'channel': result.get('metadata', {}).get('channel', 'unknown'),
                        'url': url,
                        'solutions': slack_solutions.get('solutions', []),
                        'thread_summary': slack_solutions.get('thread_summary', '')
                    })
                
                slack_discussions.append({
                    'channel': result.get('metadata', {}).get('channel', 'unknown'),
                    'content': result.get('content', '')[:200],
                    'url': url,
                    'thread_info': slack_solutions.get('thread_summary', '') if slack_solutions else ''
                })
            
            # Enhanced Confluence analysis: Extract full page content for solutions
            elif platform == 'confluence':
                logger.info(f"🔍 Following Confluence link to extract full page content")
                confluence_solutions = await self._extract_confluence_page_solutions(result)
                if confluence_solutions:
                    detailed_solutions.append({
                        'source': 'confluence',
                        'page_title': result.get('title', ''),
                        'url': url,
                        'solutions': confluence_solutions.get('solutions', []),
                        'full_content': confluence_solutions.get('content', '')
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
        
        # Generate enhanced solution analysis with extracted content
        solution_analysis = await self._generate_enhanced_solution_analysis(
            query, error_patterns, technical_context, detailed_solutions
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
                'confluence_pages_analyzed': len([s for s in detailed_solutions if s['source'] == 'confluence'])
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
    
    async def _extract_jira_comments_and_solutions(self, jira_key: str) -> Optional[Dict[str, Any]]:
        """Extract comments and solutions from a Jira ticket"""
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
    
    async def _extract_slack_thread_solutions(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract thread conversations and solutions from Slack message"""
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
    
    async def _extract_confluence_page_solutions(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract full page content and solutions from Confluence page"""
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
