import os
import re
from jira import JIRA
from typing import List, Dict, Optional
import logging
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

class JiraTool:
    def __init__(self):
        self.server = os.getenv('JIRA_SERVER')
        self.username = os.getenv('JIRA_USERNAME')
        self.api_token = os.getenv('JIRA_API_TOKEN')
        
        if self.server and self.username and self.api_token:
            try:
                self.jira = JIRA(
                    server=self.server,
                    basic_auth=(self.username, self.api_token)
                )
            except Exception as e:
                logger.error(f"Failed to initialize JIRA client: {e}")
                self.jira = None
        else:
            logger.warning("JIRA credentials not found in environment variables")
            self.jira = None
    
    def search_issues(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search for JIRA issues using JQL or text search"""
        if not self.jira:
            return []
        
        try:
            # If query doesn't look like JQL, create a text search with proper escaping
            if not any(keyword in query.lower() for keyword in ['project', 'status', 'assignee', 'reporter']):
                # Text search in summary and description with proper JQL escaping
                escaped_query = self._escape_jql_string(query)
                jql_query = f'text ~ {escaped_query} OR summary ~ {escaped_query} OR description ~ {escaped_query}'
            else:
                jql_query = query
            
            issues = self.jira.search_issues(jql_query, maxResults=max_results, expand='changelog')
            
            results = []
            for issue in issues:
                results.append({
                    'key': issue.key,
                    'summary': issue.fields.summary,
                    'description': getattr(issue.fields, 'description', '') or '',
                    'status': issue.fields.status.name,
                    'priority': getattr(issue.fields.priority, 'name', 'None') if issue.fields.priority else 'None',
                    'assignee': issue.fields.assignee.displayName if issue.fields.assignee else 'Unassigned',
                    'reporter': issue.fields.reporter.displayName if issue.fields.reporter else 'Unknown',
                    'created': str(issue.fields.created),
                    'updated': str(issue.fields.updated),
                    'url': f"{self.server}/browse/{issue.key}",
                    'project': issue.fields.project.name,
                    'issue_type': issue.fields.issuetype.name
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching JIRA issues: {e}")
            return []
    
    def get_similar_issues(self, text: str, max_results: int = 10) -> List[Dict]:
        """Find similar issues based on text content"""
        if not self.jira:
            return []
        
        # Extract key terms from the text for better searching
        keywords = self._extract_keywords(text)
        
        all_results = []
        
        # Search using different strategies with proper JQL escaping
        search_strategies = [
            f'text ~ {self._escape_jql_string(text[:100])}',  # Direct text search (truncated)
            f'summary ~ {self._escape_jql_string(keywords[:50])}',  # Summary search with keywords
            f'description ~ {self._escape_jql_string(keywords[:50])}'  # Description search with keywords
        ]
        
        for strategy in search_strategies:
            try:
                issues = self.jira.search_issues(strategy, maxResults=max_results//len(search_strategies) + 1)
                for issue in issues:
                    issue_data = {
                        'key': issue.key,
                        'summary': issue.fields.summary,
                        'description': getattr(issue.fields, 'description', '') or '',
                        'status': issue.fields.status.name,
                        'priority': getattr(issue.fields.priority, 'name', 'None') if issue.fields.priority else 'None',
                        'assignee': issue.fields.assignee.displayName if issue.fields.assignee else 'Unassigned',
                        'url': f"{self.server}/browse/{issue.key}",
                        'similarity_score': self._calculate_similarity(text, issue.fields.summary + ' ' + (getattr(issue.fields, 'description', '') or ''))
                    }
                    
                    # Avoid duplicates
                    if not any(result['key'] == issue_data['key'] for result in all_results):
                        all_results.append(issue_data)
                        
            except Exception as e:
                logger.error(f"Error with search strategy '{strategy}': {e}")
                continue
        
        # Sort by similarity score and return top results
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return all_results[:max_results]
    
    def get_issue_details(self, issue_key: str) -> Dict:
        """Get detailed information about a specific issue"""
        if not self.jira:
            return {}
        
        try:
            issue = self.jira.issue(issue_key, expand='changelog,comments')
            
            # Get comments
            comments = []
            for comment in issue.fields.comment.comments:
                comments.append({
                    'author': comment.author.displayName,
                    'body': comment.body,
                    'created': str(comment.created)
                })
            
            # Get attachments
            attachments = []
            for attachment in issue.fields.attachment:
                attachments.append({
                    'filename': attachment.filename,
                    'size': attachment.size,
                    'created': str(attachment.created),
                    'url': attachment.content
                })
            
            return {
                'key': issue.key,
                'summary': issue.fields.summary,
                'description': getattr(issue.fields, 'description', '') or '',
                'status': issue.fields.status.name,
                'priority': getattr(issue.fields.priority, 'name', 'None') if issue.fields.priority else 'None',
                'assignee': issue.fields.assignee.displayName if issue.fields.assignee else 'Unassigned',
                'reporter': issue.fields.reporter.displayName if issue.fields.reporter else 'Unknown',
                'created': str(issue.fields.created),
                'updated': str(issue.fields.updated),
                'url': f"{self.server}/browse/{issue.key}",
                'project': issue.fields.project.name,
                'issue_type': issue.fields.issuetype.name,
                'comments': comments,
                'attachments': attachments
            }
        except Exception as e:
            logger.error(f"Error getting issue details for {issue_key}: {e}")
            return {}
    
    def _extract_keywords(self, text: str) -> str:
        """Extract key terms from text for better searching"""
        # Simple keyword extraction - remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(keywords[:10])  # Return top 10 keywords
    
    def _escape_jql_string(self, text: str) -> str:
        """Escape special characters and reserved words in JQL strings"""
        if not text:
            return '""'
        
        # JQL reserved words that need to be handled carefully
        jql_reserved_words = {
            'and', 'or', 'not', 'empty', 'null', 'order', 'by', 'asc', 'desc',
            'in', 'is', 'was', 'changed', 'after', 'before', 'during', 'on',
            'from', 'to', 'by', 'with', 'without', 'contains', 'does', 'not'
        }
        
        # Clean the text and remove problematic characters
        # Replace quotes and backslashes
        cleaned_text = text.replace('"', '\\"').replace('\\', '\\\\')
        
        # Split into words and handle reserved words
        words = cleaned_text.split()
        processed_words = []
        
        for word in words:
            # Remove special characters that cause JQL issues
            clean_word = re.sub(r'[^\w\s\-\.]', ' ', word.lower())
            if clean_word.strip() and clean_word.strip() not in jql_reserved_words:
                processed_words.append(clean_word.strip())
        
        # Join words and wrap in quotes
        if processed_words:
            final_text = ' '.join(processed_words)
            return f'"{final_text}"'
        else:
            # Fallback: use original text but escape quotes
            escaped_text = text.replace('"', '\\"')
            return f'"{escaped_text}"'
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_project_issues(self, project_key: str, max_results: int = 50) -> List[Dict]:
        """Get recent issues from a specific project"""
        if not self.jira:
            return []
        
        try:
            jql_query = f'project = "{project_key}" ORDER BY updated DESC'
            issues = self.jira.search_issues(jql_query, maxResults=max_results)
            
            results = []
            for issue in issues:
                results.append({
                    'key': issue.key,
                    'summary': issue.fields.summary,
                    'status': issue.fields.status.name,
                    'priority': getattr(issue.fields.priority, 'name', 'None') if issue.fields.priority else 'None',
                    'assignee': issue.fields.assignee.displayName if issue.fields.assignee else 'Unassigned',
                    'updated': str(issue.fields.updated),
                    'url': f"{self.server}/browse/{issue.key}"
                })
            
            return results
        except Exception as e:
            logger.error(f"Error getting project issues for {project_key}: {e}")
            return []
