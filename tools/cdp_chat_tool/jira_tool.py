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
        self.project_keys = self._get_project_keys()
        
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
    
    def _get_project_keys(self) -> List[str]:
        """Get project keys from environment variable"""
        project_key_env = os.getenv('JIRA_PROJECT_KEY', '')
        if project_key_env:
            # Split by comma and strip whitespace
            keys = [key.strip() for key in project_key_env.split(',') if key.strip()]
            logger.info(f"JIRA search restricted to projects: {keys}")
            return keys
        else:
            logger.warning("JIRA_PROJECT_KEY not found in environment variables")
            return []
    
    def _build_project_filter(self) -> str:
        """Build JQL project filter clause"""
        if not self.project_keys:
            return ""
        
        if len(self.project_keys) == 1:
            return f'project = "{self.project_keys[0]}"'
        else:
            project_list = ', '.join([f'"{key}"' for key in self.project_keys])
            return f'project in ({project_list})'
    
    def search_issues(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search for JIRA issues using JQL or text search"""
        if not self.jira:
            return []
        
        try:
            # Build project filter
            project_filter = self._build_project_filter()
            
            # If query doesn't look like JQL, create a text search with proper escaping
            if not any(keyword in query.lower() for keyword in ['project', 'status', 'assignee', 'reporter']):
                # Text search in summary and description with proper JQL escaping
                escaped_query = self._escape_jql_string(query)
                search_clause = f'text ~ {escaped_query} OR summary ~ {escaped_query} OR description ~ {escaped_query}'
                
                # Combine with project filter
                if project_filter:
                    jql_query = f'({project_filter}) AND ({search_clause})'
                else:
                    jql_query = search_clause
            else:
                # User provided JQL query - add project filter if not already present
                if project_filter and 'project' not in query.lower():
                    jql_query = f'({project_filter}) AND ({query})'
                else:
                    jql_query = query
            
            logger.info(f"Executing JQL query: {jql_query}")
            issues = self.jira.search_issues(jql_query, maxResults=max_results, expand='changelog')
            
            results = []
            for issue in issues:
                try:
                    # Safe access to all issue fields with proper error handling
                    issue_data = {
                        'key': issue.key,
                        'summary': getattr(issue.fields, 'summary', 'No summary'),
                        'description': getattr(issue.fields, 'description', '') or '',
                        'status': getattr(issue.fields.status, 'name', 'Unknown') if hasattr(issue.fields, 'status') and issue.fields.status else 'Unknown',
                        'priority': self._safe_get_priority(issue),
                        'assignee': getattr(issue.fields.assignee, 'displayName', 'Unassigned') if hasattr(issue.fields, 'assignee') and issue.fields.assignee else 'Unassigned',
                        'reporter': getattr(issue.fields.reporter, 'displayName', 'Unknown') if hasattr(issue.fields, 'reporter') and issue.fields.reporter else 'Unknown',
                        'created': str(getattr(issue.fields, 'created', 'Unknown')),
                        'updated': str(getattr(issue.fields, 'updated', 'Unknown')),
                        'url': f"{self.server}/browse/{issue.key}",
                        'project': getattr(issue.fields.project, 'name', 'Unknown') if hasattr(issue.fields, 'project') and issue.fields.project else 'Unknown',
                        'issue_type': getattr(issue.fields.issuetype, 'name', 'Unknown') if hasattr(issue.fields, 'issuetype') and issue.fields.issuetype else 'Unknown'
                    }
                    results.append(issue_data)
                except Exception as field_error:
                    logger.warning(f"Error processing issue {issue.key}: {field_error}")
                    # Add minimal issue data even if some fields fail
                    results.append({
                        'key': issue.key,
                        'summary': 'Error loading summary',
                        'description': '',
                        'status': 'Unknown',
                        'priority': 'Unknown',
                        'assignee': 'Unknown',
                        'reporter': 'Unknown',
                        'created': 'Unknown',
                        'updated': 'Unknown',
                        'url': f"{self.server}/browse/{issue.key}",
                        'project': 'Unknown',
                        'issue_type': 'Unknown'
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching JIRA issues: {e}")
            return []
    
    def get_similar_issues(self, text: str, max_results: int = 10) -> List[Dict]:
        """Find similar issues based on text content"""
        if not self.jira:
            return []
        
        # Build project filter
        project_filter = self._build_project_filter()
        
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
                # Add project filter to each search strategy
                if project_filter:
                    jql_query = f'({project_filter}) AND ({strategy})'
                else:
                    jql_query = strategy
                
                logger.info(f"Executing similarity search JQL: {jql_query}")
                issues = self.jira.search_issues(jql_query, maxResults=max_results//len(search_strategies) + 1)
                for issue in issues:
                    try:
                        # Safe access to all issue fields with proper error handling
                        issue_data = {
                            'key': issue.key,
                            'summary': getattr(issue.fields, 'summary', 'No summary'),
                            'description': getattr(issue.fields, 'description', '') or '',
                            'status': getattr(issue.fields.status, 'name', 'Unknown') if hasattr(issue.fields, 'status') and issue.fields.status else 'Unknown',
                            'priority': self._safe_get_priority(issue),
                            'assignee': getattr(issue.fields.assignee, 'displayName', 'Unassigned') if hasattr(issue.fields, 'assignee') and issue.fields.assignee else 'Unassigned',
                            'url': f"{self.server}/browse/{issue.key}",
                            'similarity_score': self._calculate_similarity(text, getattr(issue.fields, 'summary', '') + ' ' + (getattr(issue.fields, 'description', '') or ''))
                        }
                    except Exception as field_error:
                        logger.warning(f"Error processing similarity issue {issue.key}: {field_error}")
                        # Add minimal issue data even if some fields fail
                        issue_data = {
                            'key': issue.key,
                            'summary': 'Error loading summary',
                            'description': '',
                            'status': 'Unknown',
                            'priority': 'Unknown',
                            'assignee': 'Unknown',
                            'url': f"{self.server}/browse/{issue.key}",
                            'similarity_score': 0.0
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
        """Get detailed information about a specific issue, including all comments (no duplicates, no pagination)."""
        if not self.jira:
            return {}
        try:
            issue = self.jira.issue(issue_key, expand='changelog')
            # Get all comments using jira.comments(issue_key)
            comments = []
            try:
                all_comments = self.jira.comments(issue_key)
                logger.info(f"API returned {len(all_comments)} comments for issue {issue_key}")
                for idx, comment in enumerate(all_comments):
                    logger.debug(f"Comment {idx+1}: {getattr(comment, 'body', '')[:200]}")
                    comments.append({
                        'author': getattr(comment.author, 'displayName', 'Unknown') if hasattr(comment, 'author') and comment.author else 'Unknown',
                        'body': getattr(comment, 'body', ''),
                        'created': str(getattr(comment, 'created', 'Unknown'))
                    })
                logger.info(f"Processed {len(comments)} comments for issue {issue_key}")
            except Exception as comments_error:
                logger.error(f"Error accessing comments for issue {issue_key}: {comments_error}")
            # Get attachments with safe field access
            attachments = []
            try:
                if hasattr(issue.fields, 'attachment') and issue.fields.attachment:
                    for attachment in issue.fields.attachment:
                        try:
                            attachments.append({
                                'filename': getattr(attachment, 'filename', 'Unknown'),
                                'size': getattr(attachment, 'size', 0),
                                'created': str(getattr(attachment, 'created', 'Unknown')),
                                'url': getattr(attachment, 'content', '')
                            })
                        except Exception as attachment_error:
                            logger.warning(f"Error processing attachment in issue {issue_key}: {attachment_error}")
                            attachments.append({
                                'filename': 'Error loading attachment',
                                'size': 0,
                                'created': 'Unknown',
                                'url': ''
                            })
            except Exception as attachments_error:
                logger.error(f"Error accessing attachments for issue {issue_key}: {attachments_error}")
            # Safe access to all issue fields with proper error handling
            return {
                'key': issue.key,
                'summary': getattr(issue.fields, 'summary', 'No summary'),
                'description': getattr(issue.fields, 'description', '') or '',
                'status': getattr(issue.fields.status, 'name', 'Unknown') if hasattr(issue.fields, 'status') and issue.fields.status else 'Unknown',
                'priority': self._safe_get_priority(issue),
                'assignee': getattr(issue.fields.assignee, 'displayName', 'Unassigned') if hasattr(issue.fields, 'assignee') and issue.fields.assignee else 'Unassigned',
                'reporter': getattr(issue.fields.reporter, 'displayName', 'Unknown') if hasattr(issue.fields, 'reporter') and issue.fields.reporter else 'Unknown',
                'created': str(getattr(issue.fields, 'created', 'Unknown')),
                'updated': str(getattr(issue.fields, 'updated', 'Unknown')),
                'url': f"{self.server}/browse/{issue.key}",
                'project': getattr(issue.fields.project, 'name', 'Unknown') if hasattr(issue.fields, 'project') and issue.fields.project else 'Unknown',
                'issue_type': getattr(issue.fields.issuetype, 'name', 'Unknown') if hasattr(issue.fields, 'issuetype') and issue.fields.issuetype else 'Unknown',
                'comments': comments,
                'attachments': attachments
            }
        except Exception as e:
            logger.error(f"Error getting issue details for {issue_key}: {e}")
            return {}
    
    def _clean_text(self, text: str) -> str:
        """Remove markdown/html-like formatting from text."""
        # Remove {panel}, asterisks, underscores, h3./h4., and extra whitespace
        text = re.sub(r'\{panel.*?\}', '', text)
        text = re.sub(r'\*', '', text)
        text = re.sub(r'_+', '', text)
        text = re.sub(r'h3\. ', '', text)
        text = re.sub(r'h4\. ', '', text)
        text = re.sub(r'\|', '', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def _parse_remediation_plan(self, remediation_text: str) -> Dict:
        """Split remediation plan into Short Term, Medium Term, Long Term."""
        plan = {}
        short_match = re.search(r'Short Term\n(.*?)(Medium Term:|Long Term:|$)', remediation_text, re.DOTALL)
        medium_match = re.search(r'Medium Term:\n(.*?)(Long Term:|$)', remediation_text, re.DOTALL)
        long_match = re.search(r'Long Term:\n(.*)', remediation_text, re.DOTALL)
        if short_match:
            plan['Short Term'] = self._clean_text(short_match.group(1))
        if medium_match:
            plan['Medium Term'] = self._clean_text(medium_match.group(1))
        if long_match:
            plan['Long Term'] = self._clean_text(long_match.group(1))
        return plan

    def _parse_postmortem_description(self, description: str) -> Dict:
        """Parse a postmortem description into selected structured sections and clean formatting."""
        sections = [
            ('Incident Summary', r'\*Incident Summary\*\n(.*?)----'),
            ('Root Cause', r'\*Root Cause\*\n(.*?)----'),
            ('Remediation Plan', r'\*Remediation Plan\*\n(.*?)----'),
            ('Timeline', r'\*Timeline\*\n(.*?)----')
        ]
        parsed = {}
        for name, pattern in sections:
            match = re.search(pattern, description, re.DOTALL)
            if match:
                cleaned = self._clean_text(match.group(1))
                if name == 'Remediation Plan':
                    parsed[name] = self._parse_remediation_plan(cleaned)
                else:
                    parsed[name] = cleaned
        return parsed
    
    def get_issue_with_links_and_comments(self, issue_key: str) -> Dict:
        """Given a JIRA issue key, return its summary, description, all comments, and details of directly linked issues (A1DEV/AOPS only, filtered by type, no recursion)."""
        if not self.jira:
            return {}
        try:
            main_details = self.get_issue_details(issue_key)
            main_comments = main_details.get('comments', [])  # All comments
            linked_issues_details = []
            try:
                issue_obj = self.jira.issue(issue_key, expand='issuelinks')
                if hasattr(issue_obj.fields, 'issuelinks') and issue_obj.fields.issuelinks:
                    print(f"Found {len(issue_obj.fields.issuelinks)} linked issues for {issue_key}")
                    all_linked = []
                    for link in issue_obj.fields.issuelinks:
                        linked_issue_key = None
                        if hasattr(link, 'outwardIssue') and link.outwardIssue:
                            linked_issue_key = getattr(link.outwardIssue, 'key', None)
                        elif hasattr(link, 'inwardIssue') and link.inwardIssue:
                            linked_issue_key = getattr(link.inwardIssue, 'key', None)
                        if linked_issue_key:
                            try:
                                linked_issue_obj = self.jira.issue(linked_issue_key)
                                linked_project_key = getattr(linked_issue_obj.fields.project, 'key', '') if hasattr(linked_issue_obj.fields, 'project') else ''
                                linked_issue_type = getattr(linked_issue_obj.fields.issuetype, 'name', '') if hasattr(linked_issue_obj.fields, 'issuetype') else ''
                                all_linked.append((linked_issue_key, linked_project_key, linked_issue_type))
                                if linked_project_key in ['A1DEV', 'AOPS'] and linked_issue_type in ['Bug', 'Task', 'Story', 'Postmortem']:
                                    linked_details = self.get_issue_details(linked_issue_key)
                                    if linked_issue_type == 'Postmortem':
                                        parsed_postmortem = self._parse_postmortem_description(linked_details.get('description', ''))
                                        linked_issue_info = {
                                            'key': linked_issue_key,
                                            'project': linked_project_key,
                                            'type': linked_issue_type,
                                            'summary': linked_details.get('summary', ''),
                                            'parsed_postmortem': parsed_postmortem
                                        }
                                    else:
                                        linked_issue_info = {
                                            'key': linked_issue_key,
                                            'project': linked_project_key,
                                            'type': linked_issue_type,
                                            'summary': linked_details.get('summary', ''),
                                            'description': linked_details.get('description', ''),
                                            'comments': linked_details.get('comments', [])
                                        }
                                    linked_issues_details.append(linked_issue_info)
                            except Exception as linked_obj_error:
                                print(f"Error fetching linked issue object for {linked_issue_key}: {linked_obj_error}")
                    print(f"All linked issues for {issue_key}: {all_linked}")
            except Exception as link_error:
                print(f"Error fetching linked tickets for {issue_key}: {link_error}")
            print(f"Final linked_issues_details for {issue_key}: {linked_issues_details}")
            return {
                'key': issue_key,
                'summary': main_details.get('summary', ''),
                'description': main_details.get('description', ''),
                'comments': main_comments,
                'linked_issues_details': linked_issues_details
            }
        except Exception as e:
            print(f"Error getting issue with links and comments for {issue_key}: {e}")
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
    
    def _safe_get_priority(self, issue) -> str:
        """Safely get priority from issue with proper error handling"""
        try:
            if hasattr(issue.fields, 'priority') and issue.fields.priority:
                return getattr(issue.fields.priority, 'name', 'Unknown')
            else:
                return 'None'
        except Exception as e:
            logger.warning(f"Error accessing priority for issue {getattr(issue, 'key', 'unknown')}: {e}")
            return 'Unknown'
    
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
                try:
                    # Safe access to all issue fields with proper error handling
                    results.append({
                        'key': issue.key,
                        'summary': getattr(issue.fields, 'summary', 'No summary'),
                        'status': getattr(issue.fields.status, 'name', 'Unknown') if hasattr(issue.fields, 'status') and issue.fields.status else 'Unknown',
                        'priority': self._safe_get_priority(issue),
                        'assignee': getattr(issue.fields.assignee, 'displayName', 'Unassigned') if hasattr(issue.fields, 'assignee') and issue.fields.assignee else 'Unassigned',
                        'updated': str(getattr(issue.fields, 'updated', 'Unknown')),
                        'url': f"{self.server}/browse/{issue.key}"
                    })
                except Exception as field_error:
                    logger.warning(f"Error processing project issue {issue.key}: {field_error}")
                    # Add minimal issue data even if some fields fail
                    results.append({
                        'key': issue.key,
                        'summary': 'Error loading summary',
                        'status': 'Unknown',
                        'priority': 'Unknown',
                        'assignee': 'Unknown',
                        'updated': 'Unknown',
                        'url': f"{self.server}/browse/{issue.key}"
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error getting project issues for {project_key}: {e}")
            return []

# call main for the method get_issue_with_links_and_comments
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jira_tool = JiraTool()
    test_issue_key = "AOPS-26612"  # Replace with a valid issue key for testing
    issue_data = jira_tool.get_issue_with_links_and_comments(test_issue_key)
    print(issue_data)
