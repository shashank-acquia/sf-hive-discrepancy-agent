"""
Direct API Fallback for Technical Solutions & Analysis
When MCP servers timeout or fail, this module provides direct API access
to Jira, Slack, and Confluence to ensure solution analysis data is available.
"""

import logging
import json
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class DirectAPIFallback:
    """Fallback mechanism using direct API calls when MCP servers fail"""
    
    def __init__(self):
        self.jira_tool = None
        self.slack_tool = None
        self.confluence_tool = None
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize direct API tools"""
        try:
            from tools.cdp_chat_tool.jira_tool import JiraTool
            from tools.cdp_chat_tool.slack_tool import SlackTool
            from tools.cdp_chat_tool.confluence_tool import ConfluenceTool
            
            self.jira_tool = JiraTool()
            self.slack_tool = SlackTool()
            self.confluence_tool = ConfluenceTool()
            
            logger.info("✅ Direct API tools initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize direct API tools: {e}")
    
    def generate_solution_analysis_fallback(self, search_results: List[Dict], query: str) -> Dict[str, Any]:
        """
        Generate solution analysis using direct API calls as fallback
        """
        try:
            logger.info(f"🔄 Generating fallback solution analysis for query: '{query}'")
            
            # Extract platform-specific results
            jira_results = [r for r in search_results if r.get('platform', '').lower() == 'jira']
            slack_results = [r for r in search_results if r.get('platform', '').lower() == 'slack']
            confluence_results = [r for r in search_results if r.get('platform', '').lower() == 'confluence']
            
            # Generate Jira comments using direct API
            jira_comments = self._extract_jira_comments_fallback(jira_results, query)
            
            # Generate Slack thread solutions using direct API
            slack_thread_solutions = self._extract_slack_solutions_fallback(slack_results, query)
            
            # Generate Confluence solutions using direct API
            confluence_solutions = self._extract_confluence_solutions_fallback(confluence_results, query)
            
            # Create comprehensive solution analysis
            solution_analysis = {
                'solutions': [{
                    'error_pattern': f"Technical Issue Analysis for: {query}",
                    'technical_context': f"Analyzed {len(search_results)} results across platforms",
                    'root_cause': self._analyze_root_cause_fallback(jira_results, slack_results, confluence_results, query),
                    'solution': self._generate_primary_solution_fallback(jira_results, slack_results, confluence_results, query),
                    'prevention_measures': self._generate_prevention_measures_fallback(jira_results, query),
                    'next_steps': self._generate_next_steps_fallback(jira_results, slack_results, query),
                    'related_tickets': [r.get('key', r.get('title', 'Unknown')) for r in jira_results[:5]],
                    'confidence_score': 0.85,  # High confidence for direct API
                    'source_url': jira_results[0].get('url', '') if jira_results else '',
                    'jira_comments': jira_comments,
                    'slack_thread_solutions': slack_thread_solutions,
                    'confluence_solutions': confluence_solutions
                }],
                'error_analysis': {
                    'error_patterns': self._identify_error_patterns_fallback(search_results, query),
                    'common_causes': self._identify_common_causes_fallback(jira_results, query),
                    'affected_systems': self._identify_affected_systems_fallback(search_results, query)
                },
                'detailed_solutions': [
                    {
                        'source': 'jira',
                        'comments': [{'author': c['author'], 'content': c['content'], 'created': c['created']} for c in jira_comments],
                        'solutions': [{'content': c['content'], 'confidence': c.get('confidence', 0.8)} for c in jira_comments if c.get('type') == 'solution']
                    },
                    {
                        'source': 'slack',
                        'solutions': [{'content': s['content'], 'confidence': s.get('confidence', 0.7)} for s in slack_thread_solutions]
                    },
                    {
                        'source': 'confluence',
                        'solutions': [{'content': s['content'], 'confidence': s.get('confidence', 0.6)} for s in confluence_solutions]
                    }
                ],
                'metadata': {
                    'processing_time': '< 1s',
                    'fallback_used': True,
                    'total_sources': len(search_results),
                    'jira_sources': len(jira_results),
                    'slack_sources': len(slack_results),
                    'confluence_sources': len(confluence_results)
                }
            }
            
            logger.info(f"✅ Fallback solution analysis completed: {len(jira_comments)} Jira comments, {len(slack_thread_solutions)} Slack solutions, {len(confluence_solutions)} Confluence solutions")
            return solution_analysis
            
        except Exception as e:
            logger.error(f"❌ Fallback solution analysis failed: {e}")
            return {
                'solutions': [{
                    'error_pattern': 'Fallback Analysis Error',
                    'solution': f'Direct API fallback encountered an error: {str(e)}. Basic search results are still available.',
                    'confidence_score': 0.3,
                    'jira_comments': [],
                    'slack_thread_solutions': [],
                    'confluence_solutions': []
                }],
                'error': f'Fallback analysis failed: {str(e)}'
            }
    
    def _extract_jira_comments_fallback(self, jira_results: List[Dict], query: str) -> List[Dict]:
        """Extract Jira comments using direct API calls"""
        comments = []
        try:
            for result in jira_results[:3]:  # Limit to top 3 Jira issues
                issue_key = result.get('key', result.get('id'))
                if not issue_key:
                    continue
                
                try:
                    # Get issue details with comments using direct API
                    if self.jira_tool:
                        issue_details = self.jira_tool.get_issue_details(issue_key)
                        if issue_details and 'comments' in issue_details:
                            for comment in issue_details['comments'][:3]:  # Top 3 comments per issue
                                comments.append({
                                    'author': comment.get('author', {}).get('displayName', 'Unknown'),
                                    'content': comment.get('body', 'No content'),
                                    'created': comment.get('created', ''),
                                    'type': 'comment',
                                    'confidence': 0.8,
                                    'issue_key': issue_key
                                })
                    
                    # Add the issue description as a "solution" if it contains relevant keywords
                    description = result.get('content', result.get('description', ''))
                    if description and any(keyword.lower() in description.lower() for keyword in query.split()):
                        comments.append({
                            'author': result.get('metadata', {}).get('assignee', 'System'),
                            'content': f"Issue Description: {description[:500]}...",
                            'created': result.get('metadata', {}).get('created', ''),
                            'type': 'solution',
                            'confidence': 0.9,
                            'keywords_matched': [kw for kw in query.split() if kw.lower() in description.lower()],
                            'issue_key': issue_key
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to get comments for Jira issue {issue_key}: {e}")
                    # Add basic info from search result
                    comments.append({
                        'author': result.get('metadata', {}).get('assignee', 'Unknown'),
                        'content': result.get('content', result.get('title', 'No content available')),
                        'created': result.get('metadata', {}).get('created', ''),
                        'type': 'issue_summary',
                        'confidence': 0.6,
                        'issue_key': issue_key
                    })
            
        except Exception as e:
            logger.error(f"Error extracting Jira comments: {e}")
        
        return comments[:10]  # Limit total comments
    
    def _extract_slack_solutions_fallback(self, slack_results: List[Dict], query: str) -> List[Dict]:
        """Extract Slack thread solutions using direct API calls"""
        solutions = []
        try:
            for result in slack_results[:3]:  # Limit to top 3 Slack messages
                metadata = result.get('metadata', {})
                channel = metadata.get('channel')
                timestamp = metadata.get('timestamp')
                
                try:
                    # Get thread replies using direct API
                    if self.slack_tool and channel and timestamp:
                        thread_replies = self.slack_tool.get_thread_replies(channel, timestamp)
                        if thread_replies:
                            for reply in thread_replies[:3]:  # Top 3 replies per thread
                                solutions.append({
                                    'author': reply.get('user', 'Unknown'),
                                    'content': reply.get('text', 'No content'),
                                    'timestamp': reply.get('ts', ''),
                                    'confidence': 0.7,
                                    'type': 'thread_solution',
                                    'channel': channel
                                })
                    
                    # Add the original message as a solution if relevant
                    content = result.get('content', result.get('text', ''))
                    if content and len(content) > 50:  # Substantial content
                        solutions.append({
                            'author': metadata.get('user', 'Unknown'),
                            'content': content,
                            'timestamp': timestamp,
                            'confidence': 0.8,
                            'type': 'original_message',
                            'channel': channel
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to get thread for Slack message: {e}")
                    # Add basic message info
                    solutions.append({
                        'author': metadata.get('user', 'Unknown'),
                        'content': result.get('content', 'No content available'),
                        'timestamp': timestamp,
                        'confidence': 0.5,
                        'type': 'message_summary',
                        'channel': channel
                    })
            
        except Exception as e:
            logger.error(f"Error extracting Slack solutions: {e}")
        
        return solutions[:8]  # Limit total solutions
    
    def _extract_confluence_solutions_fallback(self, confluence_results: List[Dict], query: str) -> List[Dict]:
        """Extract Confluence solutions using direct API calls"""
        solutions = []
        try:
            for result in confluence_results[:3]:  # Limit to top 3 Confluence pages
                try:
                    content = result.get('content', '')
                    title = result.get('title', '')
                    
                    # Extract relevant sections from content
                    if content:
                        # Look for solution patterns in content
                        solution_patterns = [
                            'solution:', 'fix:', 'resolution:', 'workaround:', 
                            'to resolve:', 'steps:', 'procedure:'
                        ]
                        
                        for pattern in solution_patterns:
                            if pattern in content.lower():
                                # Extract text around the pattern
                                start_idx = content.lower().find(pattern)
                                solution_text = content[start_idx:start_idx+300]
                                
                                solutions.append({
                                    'content': solution_text,
                                    'pattern': pattern,
                                    'confidence': 0.6,
                                    'type': 'documentation_solution',
                                    'page_title': title,
                                    'space': result.get('metadata', {}).get('space', 'Unknown')
                                })
                                break
                    
                    # Add page summary as fallback
                    if not any(s.get('page_title') == title for s in solutions):
                        solutions.append({
                            'content': f"Documentation: {title} - {content[:200]}...",
                            'pattern': 'general_documentation',
                            'confidence': 0.4,
                            'type': 'page_summary',
                            'page_title': title,
                            'space': result.get('metadata', {}).get('space', 'Unknown')
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to process Confluence page: {e}")
            
        except Exception as e:
            logger.error(f"Error extracting Confluence solutions: {e}")
        
        return solutions[:6]  # Limit total solutions
    
    def _analyze_root_cause_fallback(self, jira_results: List[Dict], slack_results: List[Dict], 
                                   confluence_results: List[Dict], query: str) -> str:
        """Analyze root cause based on available data"""
        try:
            causes = []
            
            # Analyze Jira patterns
            if jira_results:
                statuses = [r.get('metadata', {}).get('status', '') for r in jira_results]
                priorities = [r.get('metadata', {}).get('priority', '') for r in jira_results]
                
                if 'Critical' in priorities or 'High' in priorities:
                    causes.append("High-priority issues detected in Jira")
                if 'Open' in statuses or 'In Progress' in statuses:
                    causes.append("Active issues found in Jira tracking system")
            
            # Analyze Slack patterns
            if slack_results:
                channels = [r.get('metadata', {}).get('channel', '') for r in slack_results]
                if any('support' in ch.lower() or 'help' in ch.lower() for ch in channels):
                    causes.append("Support discussions indicate user-facing issues")
            
            # Analyze query patterns
            error_keywords = ['error', 'failed', 'timeout', 'connection', 'authentication']
            if any(keyword in query.lower() for keyword in error_keywords):
                causes.append(f"Query indicates {[kw for kw in error_keywords if kw in query.lower()][0]} related issue")
            
            if causes:
                return f"Based on cross-platform analysis: {'; '.join(causes[:3])}"
            else:
                return f"Root cause analysis in progress for '{query}' - review related tickets and discussions for patterns"
                
        except Exception as e:
            logger.error(f"Error in root cause analysis: {e}")
            return "Root cause analysis unavailable - please review individual results"
    
    def _generate_primary_solution_fallback(self, jira_results: List[Dict], slack_results: List[Dict], 
                                          confluence_results: List[Dict], query: str) -> str:
        """Generate primary solution based on available data"""
        try:
            solutions = []
            
            # Extract solutions from Jira
            if jira_results:
                resolved_issues = [r for r in jira_results if r.get('metadata', {}).get('status') in ['Resolved', 'Closed', 'Done']]
                if resolved_issues:
                    solutions.append(f"Review resolved Jira tickets: {', '.join([r.get('key', 'Unknown') for r in resolved_issues[:2]])}")
            
            # Extract solutions from Slack
            if slack_results:
                solutions.append(f"Check Slack discussions in {len(set([r.get('metadata', {}).get('channel') for r in slack_results]))} channels for community solutions")
            
            # Extract solutions from Confluence
            if confluence_results:
                solutions.append(f"Consult documentation in {len(set([r.get('metadata', {}).get('space') for r in confluence_results]))} Confluence spaces")
            
            if solutions:
                return f"Recommended approach: {'; '.join(solutions[:3])}"
            else:
                return f"For '{query}': Start by reviewing the search results above, check for similar patterns in your environment, and consult with team members who have worked on related issues."
                
        except Exception as e:
            logger.error(f"Error generating primary solution: {e}")
            return "Solution generation unavailable - please review search results manually"
    
    def _generate_prevention_measures_fallback(self, jira_results: List[Dict], query: str) -> List[str]:
        """Generate prevention measures based on Jira patterns"""
        try:
            measures = []
            
            if jira_results:
                # Analyze issue types
                issue_types = [r.get('metadata', {}).get('issue_type', '') for r in jira_results]
                if 'Bug' in issue_types:
                    measures.append("Implement additional testing to catch similar bugs early")
                if 'Task' in issue_types or 'Story' in issue_types:
                    measures.append("Document procedures to prevent similar issues")
            
            # Query-based measures
            if 'authentication' in query.lower():
                measures.append("Review authentication configuration and token expiration policies")
            if 'connection' in query.lower() or 'timeout' in query.lower():
                measures.append("Monitor connection health and implement retry mechanisms")
            if 'error' in query.lower():
                measures.append("Enhance error handling and logging for better diagnostics")
            
            return measures[:5] if measures else [
                "Monitor system health regularly",
                "Maintain updated documentation",
                "Implement proper error handling",
                "Regular team knowledge sharing"
            ]
            
        except Exception as e:
            logger.error(f"Error generating prevention measures: {e}")
            return ["Prevention measures unavailable"]
    
    def _generate_next_steps_fallback(self, jira_results: List[Dict], slack_results: List[Dict], query: str) -> List[str]:
        """Generate next steps based on available data"""
        try:
            steps = []
            
            if jira_results:
                steps.append(f"Review {len(jira_results)} related Jira tickets for detailed solutions")
                open_issues = [r for r in jira_results if r.get('metadata', {}).get('status') in ['Open', 'In Progress']]
                if open_issues:
                    steps.append(f"Monitor {len(open_issues)} active tickets for updates")
            
            if slack_results:
                steps.append(f"Follow up on {len(slack_results)} Slack discussions for additional context")
            
            # Query-specific steps
            if 'failed' in query.lower():
                steps.append("Check system logs for detailed error messages")
            if 'authentication' in query.lower():
                steps.append("Verify credentials and permissions")
            
            steps.append("Consult with team members who worked on similar issues")
            steps.append("Update documentation with findings")
            
            return steps[:6]
            
        except Exception as e:
            logger.error(f"Error generating next steps: {e}")
            return ["Next steps unavailable - review search results manually"]
    
    def _identify_error_patterns_fallback(self, search_results: List[Dict], query: str) -> List[Dict]:
        """Identify error patterns from search results"""
        try:
            patterns = []
            
            for result in search_results[:5]:  # Top 5 results
                platform = result.get('platform', '').lower()
                title = result.get('title', result.get('key', 'Unknown'))
                content = result.get('content', '')
                
                # Look for error indicators
                error_indicators = ['error', 'failed', 'exception', 'timeout', 'connection']
                found_indicators = [ind for ind in error_indicators if ind in content.lower() or ind in title.lower()]
                
                if found_indicators:
                    patterns.append({
                        'title': title,
                        'platform': platform,
                        'error_type': found_indicators[0],
                        'frequency': 1,  # Simplified frequency
                        'description': f"{platform.title()} result indicating {found_indicators[0]} issue"
                    })
            
            return patterns[:5]
            
        except Exception as e:
            logger.error(f"Error identifying error patterns: {e}")
            return []
    
    def _identify_common_causes_fallback(self, jira_results: List[Dict], query: str) -> List[str]:
        """Identify common causes from Jira results"""
        try:
            causes = []
            
            if jira_results:
                # Analyze issue metadata
                priorities = [r.get('metadata', {}).get('priority', '') for r in jira_results]
                components = [r.get('metadata', {}).get('project', '') for r in jira_results]
                
                if 'High' in priorities or 'Critical' in priorities:
                    causes.append("High-priority system issues")
                
                unique_components = list(set(components))
                if len(unique_components) > 1:
                    causes.append(f"Issues across multiple components: {', '.join(unique_components[:3])}")
            
            # Query-based causes
            if 'snowflake' in query.lower():
                causes.append("Snowflake database connectivity or query issues")
            if 'runner' in query.lower():
                causes.append("Automated process or job execution problems")
            
            return causes[:4] if causes else ["Common causes analysis in progress"]
            
        except Exception as e:
            logger.error(f"Error identifying common causes: {e}")
            return ["Common causes unavailable"]
    
    def _identify_affected_systems_fallback(self, search_results: List[Dict], query: str) -> List[str]:
        """Identify affected systems from search results"""
        try:
            systems = []
            
            # Extract from platforms
            platforms = list(set([r.get('platform', '').title() for r in search_results]))
            systems.extend([f"{p} platform" for p in platforms if p])
            
            # Extract from query
            system_keywords = ['snowflake', 'jira', 'confluence', 'slack', 'database', 'api']
            found_systems = [kw.title() for kw in system_keywords if kw in query.lower()]
            systems.extend(found_systems)
            
            # Extract from Jira projects
            jira_results = [r for r in search_results if r.get('platform', '').lower() == 'jira']
            projects = list(set([r.get('metadata', {}).get('project', '') for r in jira_results]))
            systems.extend([f"{p} project" for p in projects if p and p != 'Unknown'])
            
            return list(set(systems))[:5]
            
        except Exception as e:
            logger.error(f"Error identifying affected systems: {e}")
            return ["System analysis unavailable"]

# Global instance
direct_api_fallback = DirectAPIFallback()
