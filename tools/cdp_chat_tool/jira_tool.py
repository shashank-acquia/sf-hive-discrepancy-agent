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
                    basic_auth=(self.username, self.api_token),
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
    
    def search_issues(self, query: str, max_results: int = 20, jira_ids: Optional[List[str]] = None) -> List[Dict]:
        """Search for JIRA issues using JQL or text search"""
        if not self.jira:
            return []
        
        try:
            # Build project filter
            project_filter = self._build_project_filter()

            id_clause = ""
            if jira_ids:
                sanitized_ids = [self._escape_jql_string(jid).strip('"') for jid in jira_ids]
                id_list_str = ', '.join(f'"{jid}"' for jid in sanitized_ids)
                id_clause = f'key in ({id_list_str})'
                print(f"[DEBUG] Constructed Jira ID clause for JQL: {id_clause}")

            text_search_clause = ""
            if query and query.strip():
                # If query doesn't look like advanced JQL, treat it as a text search
                if not any(keyword in query.lower() for keyword in ['project =', 'status =', 'assignee =']):
                    escaped_query = self._escape_jql_string(query)
                    field_search_clauses = [
                        f'summary ~ {escaped_query}',
                        f'description ~ {escaped_query}',
                        f'comment ~ {escaped_query}'
                    ]
                    text_search_clause = f"({' OR '.join(field_search_clauses)})"
                    print(f"[DEBUG] Constructed text search clause for fields [summary, description, comment].")
                else:  # Assume user provided a valid JQL snippet
                    text_search_clause = f'({query})'

            # Combine clauses into the final JQL query
            jql_clauses = []

            # Main search logic: (ID search) OR (text search)
            main_search_parts = []
            if id_clause:
                main_search_parts.append(f'({id_clause})')
            if text_search_clause:
                main_search_parts.append(f'({text_search_clause})')

            if not main_search_parts:
                print("[WARN] No valid search criteria for Jira (no query or IDs). Aborting search.")
                return []

            jql_clauses.append(f"({' OR '.join(main_search_parts)})")

            # Project filter is always combined with AND at the start
            if project_filter:
                jql_clauses.insert(0, f'({project_filter})')

            jql_query = ' AND '.join(jql_clauses)
            jql_query += " ORDER BY updated DESC"  # Always sort by relevance/recency

            print(f"[DEBUG] Executing final JQL query: {jql_query}")
            issues = self.jira.search_issues(jql_query, maxResults=max_results, expand='changelog')

            print(f"[INFO] JIRA API returned {len(issues)} issues.")

            results = [self._format_issue(issue) for issue in issues]
            return results
        except Exception as e:
            print(f"[ERROR] Error searching JIRA issues: {e}")
            return []

    def _format_issue(self, issue) -> Dict:
        """Helper function to safely format a JIRA issue object into a dictionary."""
        try:
            issue_data = {
                'key': issue.key,
                'summary': getattr(issue.fields, 'summary', 'No summary'),
                'description': getattr(issue.fields, 'description', '') or '',
                'status': getattr(issue.fields.status, 'name', 'Unknown') if hasattr(issue.fields,
                                                                                     'status') and issue.fields.status else 'Unknown',
                'priority': self._safe_get_priority(issue),
                'assignee': getattr(issue.fields.assignee, 'displayName', 'Unassigned') if hasattr(issue.fields,
                                                                                                   'assignee') and issue.fields.assignee else 'Unassigned',
                'reporter': getattr(issue.fields.reporter, 'displayName', 'Unknown') if hasattr(issue.fields,
                                                                                                'reporter') and issue.fields.reporter else 'Unknown',
                'created': str(getattr(issue.fields, 'created', 'Unknown')),
                'updated': str(getattr(issue.fields, 'updated', 'Unknown')),
                'url': f"{self.server}/browse/{issue.key}",
                'project': getattr(issue.fields.project, 'name', 'Unknown') if hasattr(issue.fields,
                                                                                       'project') and issue.fields.project else 'Unknown',
                'issue_type': getattr(issue.fields.issuetype, 'name', 'Unknown') if hasattr(issue.fields,
                                                                                            'issuetype') and issue.fields.issuetype else 'Unknown',
                # --- NEW: Extracting additional rich fields ---
                'components': [comp.name for comp in getattr(issue.fields, 'components', [])],
                'labels': getattr(issue.fields, 'labels', []),
                'issuelinks': [
                    {
                        'type': link.type.name,
                        'direction': 'outward' if hasattr(link, 'outwardIssue') else 'inward',
                        'linked_issue_key': getattr(link, 'outwardIssue', getattr(link, 'inwardIssue', None)).key
                    }
                    for link in getattr(issue.fields, 'issuelinks', []) if
                    hasattr(link.type, 'name') and (hasattr(link, 'outwardIssue') or hasattr(link, 'inwardIssue'))
                ]
            }

            print(f'[INFO] Formatted issue data: {issue_data}')
            return issue_data
        except Exception as field_error:
            print(f"[WARNING] Error processing issue {issue.key}: {field_error}")
            return {'key': issue.key, 'summary': 'Error loading details', 'url': f"{self.server}/browse/{issue.key}"}

    def get_linked_issues(self, issue_key: str) -> List[Dict]:
        """
        NEW: Fetches and returns full details for all issues linked to the given issue_key.
        """
        if not self.jira:
            return []
        try:
            print(f"[DEBUG] Fetching linked issues for {issue_key}...")
            # First, get the main issue to find its links
            main_issue = self.jira.issue(issue_key, expand='issuelinks')
            if not hasattr(main_issue.fields, 'issuelinks') or not main_issue.fields.issuelinks:
                print(f"[DEBUG] No issue links found for {issue_key}.")
                return []

            linked_issue_keys = []
            for link in main_issue.fields.issuelinks:
                if hasattr(link, 'outwardIssue'):
                    linked_issue_keys.append(link.outwardIssue.key)
                elif hasattr(link, 'inwardIssue'):
                    linked_issue_keys.append(link.inwardIssue.key)

            if not linked_issue_keys:
                return []

            print(f"[INFO] Found {len(linked_issue_keys)} linked issue keys: {linked_issue_keys}")
            # Now, fetch these issues in a single batch query
            # We don't need to re-apply the project filter here, as linked issues can be in any project
            jql_query = f'key in ({", ".join(linked_issue_keys)})'
            linked_issues_result = self.jira.search_issues(jql_query, maxResults=len(linked_issue_keys))

            # Format them using our standard helper
            formatted_linked_issues = [self._format_issue(issue) for issue in linked_issues_result]
            return formatted_linked_issues

        except Exception as e:
            print(f"[ERROR] Failed to get linked issues for {issue_key}: {e}")
            return []

    def _print_issue_fields(self, issue):
        """Print all fields and their values from a JIRA issue object"""
        print(f"\n----- Fields for {issue.key} -----")

        # Get all attribute names
        field_names = dir(issue.fields)

        # Filter out private/special attributes and methods
        field_names = [f for f in field_names if not f.startswith('_') and f != 'raw']

        # Print each field and its value
        for field in field_names:
            try:
                value = getattr(issue.fields, field)
                # Handle nested objects
                if hasattr(value, '__dict__'):
                    print(f"{field}: (object)")
                    for k, v in vars(value).items():
                        if not k.startswith('_'):
                            print(f"  - {k}: {v}")
                else:
                    print(f"{field}: {value}")
            except Exception as e:
                print(f"{field}: Error retrieving value - {e}")

        print("--------------------------\n")


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
        """Get detailed information about a specific issue"""
        if not self.jira:
            return {}
        
        try:
            issue = self.jira.issue(issue_key, expand='changelog,comments')
            
            # Get comments with safe field access
            comments = []
            try:
                if hasattr(issue.fields, 'comment') and issue.fields.comment and hasattr(issue.fields.comment, 'comments'):
                    for comment in issue.fields.comment.comments:
                        try:
                            comments.append({
                                'author': getattr(comment.author, 'displayName', 'Unknown') if hasattr(comment, 'author') and comment.author else 'Unknown',
                                'body': getattr(comment, 'body', ''),
                                'created': str(getattr(comment, 'created', 'Unknown'))
                            })
                        except Exception as comment_error:
                            logger.warning(f"Error processing comment in issue {issue_key}: {comment_error}")
                            comments.append({
                                'author': 'Unknown',
                                'body': 'Error loading comment',
                                'created': 'Unknown'
                            })
            except Exception as comments_error:
                logger.warning(f"Error accessing comments for issue {issue_key}: {comments_error}")
            
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
                logger.warning(f"Error accessing attachments for issue {issue_key}: {attachments_error}")
            
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
