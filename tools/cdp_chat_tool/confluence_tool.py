import os
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Optional
import logging
import time
from atlassian import Confluence

logger = logging.getLogger(__name__)

class ConfluenceTool:
    def __init__(self):
        self.server = os.getenv('CONFLUENCE_SERVER')
        self.username = os.getenv('CONFLUENCE_USERNAME')
        self.api_token = os.getenv('CONFLUENCE_API_TOKEN')
        self.spaces = self._parse_spaces(os.getenv('CONFLUENCE_SPACES', ''))
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        if self.server and self.username and self.api_token:
            try:
                # Create a session with retry strategy
                session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"],  # Updated parameter name
                    backoff_factor=1
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                
                self.confluence = Confluence(
                    url=self.server,
                    username=self.username,
                    password=self.api_token,
                    cloud=True,  # Set to False for Confluence Server
                    session=session,
                    timeout=30,  # 30 second timeout
                    verify_ssl=False
                )
                
                # Test the connection
                self._test_connection()
                
            except Exception as e:
                logger.error(f"Failed to initialize Confluence client: {e}")
                self.confluence = None
        else:
            logger.warning("Confluence credentials not found in environment variables")
            self.confluence = None
    
    def _test_connection(self):
        """Test the Confluence connection"""
        try:
            if self.confluence:
                # Try a simple CQL query as a connection test
                test_results = self.confluence.cql('type = page', limit=1)
                logger.info(f"Successfully connected to Confluence - test query returned {len(test_results.get('results', []))} results")
                return True
        except Exception as e:
            logger.error(f"Confluence connection test failed: {e}")
            return False
        return False
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check if it's a retryable error
                if any(error_type in error_msg for error_type in [
                    'connection aborted', 'remote disconnected', 'timeout', 
                    'connection reset', 'connection refused', 'read timeout'
                ]):
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                # If it's not retryable or we've exhausted retries, raise the exception
                logger.error(f"Non-retryable error or max retries exceeded: {e}")
                raise e
        
        # If we get here, all retries failed
        raise last_exception
    
    def search_content(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for content in Confluence"""
        if not self.confluence:
            logger.warning("Confluence client not initialized")
            return []
        
        def _search():
            # Use CQL (Confluence Query Language) for search
            cql_query = f'text ~ "{query}"'
            logger.info(f"Executing Confluence search with query: {cql_query}")
            
            results = self.confluence.cql(cql_query, limit=limit)
            
            content_list = []
            if 'results' in results:
                for result in results['results']:
                    content = result.get('content', {})
                    content_list.append({
                        'id': content.get('id', ''),
                        'title': content.get('title', ''),
                        'type': content.get('type', ''),
                        'space': content.get('space', {}).get('name', ''),
                        'url': f"{self.server}/wiki{content.get('_links', {}).get('webui', '')}",
                        'excerpt': result.get('excerpt', ''),
                        'last_modified': content.get('version', {}).get('when', ''),
                        'author': content.get('version', {}).get('by', {}).get('displayName', '')
                    })
            
            return content_list
        
        try:
            return self._execute_with_retry(_search)
        except Exception as e:
            logger.error(f"Error searching Confluence content after retries: {e}")
            return []
    
    def search_in_space(self, query: str, space_key: str, limit: int = 10) -> List[Dict]:
        """Search for content in a specific Confluence space"""
        if not self.confluence:
            return []
        
        try:
            cql_query = f'space = "{space_key}" AND text ~ "{query}"'
            
            results = self.confluence.cql(cql_query, limit=limit)
            
            content_list = []
            if 'results' in results:
                for result in results['results']:
                    content = result.get('content', {})
                    content_list.append({
                        'id': content.get('id', ''),
                        'title': content.get('title', ''),
                        'type': content.get('type', ''),
                        'space': content.get('space', {}).get('name', ''),
                        'url': f"{self.server}/wiki{content.get('_links', {}).get('webui', '')}",
                        'excerpt': result.get('excerpt', ''),
                        'last_modified': content.get('version', {}).get('when', ''),
                        'author': content.get('version', {}).get('by', {}).get('displayName', '')
                    })
            
            return content_list
        except Exception as e:
            logger.error(f"Error searching in Confluence space {space_key}: {e}")
            return []
    
    def get_page_content(self, page_id: str) -> Dict:
        """Get full content of a Confluence page"""
        if not self.confluence:
            return {}
        
        try:
            page = self.confluence.get_page_by_id(
                page_id, 
                expand='body.storage,version,space'
            )
            
            return {
                'id': page.get('id', ''),
                'title': page.get('title', ''),
                'type': page.get('type', ''),
                'space': page.get('space', {}).get('name', ''),
                'url': f"{self.server}/wiki{page.get('_links', {}).get('webui', '')}",
                'content': page.get('body', {}).get('storage', {}).get('value', ''),
                'last_modified': page.get('version', {}).get('when', ''),
                'author': page.get('version', {}).get('by', {}).get('displayName', ''),
                'version': page.get('version', {}).get('number', 1)
            }
        except Exception as e:
            logger.error(f"Error getting page content for {page_id}: {e}")
            return {}
        
    def search_by_url(self, url: str) -> List[Dict]:
        """Search for a specific page by its URL"""
        if not self.confluence:
            return []
        
        try:
            # Extract page ID from URL
            # URL format: https://acquia.atlassian.net/wiki/spaces/DEV/pages/1080590357/Support+Runbook+-+Pinterest+Connector
            import re
            page_id_match = re.search(r'/pages/(\d+)', url)
            
            if page_id_match:
                page_id = page_id_match.group(1)
                print(f"[DEBUG] Extracted page ID {page_id} from URL: {url}")
                
                # Get the page by ID
                page_content = self.get_page_content(page_id)
                if page_content:
                    return [page_content]
            
            # If we can't extract page ID, try searching by the page title from URL
            title_match = re.search(r'/pages/\d+/([^/?]+)', url)
            if title_match:
                title = title_match.group(1).replace('+', ' ').replace('%20', ' ')
                print(f"[DEBUG] Extracted title '{title}' from URL, searching...")
                
                # Search by title
                cql_query = f'title ~ "{title}"'
                results = self.confluence.cql(cql_query, limit=5)
                
                content_list = []
                if 'results' in results:
                    for result in results['results']:
                        content = result.get('content', {})
                        content_list.append({
                            'id': content.get('id', ''),
                            'title': content.get('title', ''),
                            'type': content.get('type', ''),
                            'space': content.get('space', {}).get('name', ''),
                            'url': f"{self.server}/wiki{content.get('_links', {}).get('webui', '')}",
                            'excerpt': result.get('excerpt', ''),
                            'last_modified': content.get('version', {}).get('when', ''),
                            'author': content.get('version', {}).get('by', {}).get('displayName', '')
                        })
            
                return content_list
        
            return []
        
        except Exception as e:
            logger.error(f"Error searching by URL {url}: {e}")
            return []
    
    def search_similar_content(self, text: str, limit: int = 10) -> List[Dict]:
        """Find similar content based on text"""
        if not self.confluence:
            return []
        
        # Check if this is a URL - if so, redirect to search_by_url
        if text.startswith('http') and 'atlassian.net/wiki' in text:
            return self.search_by_url(text)
        
        # Extract keywords for better searching
        keywords = self._extract_keywords(text)
        
        all_results = []
        
        # Try different search strategies
        search_queries = [
            f'text ~ "{text[:100]}"',  # Direct text search
            f'title ~ "{keywords[:50]}"',  # Title search with keywords
            f'text ~ "{keywords[:50]}"'  # Content search with keywords
        ]
        
        for query in search_queries:
            try:
                results = self.confluence.cql(query, limit=limit//len(search_queries) + 1)

                
                if 'results' in results:
                    for result in results['results']:
                        content = result.get('content', {})
                        content_data = {
                            'id': content.get('id', ''),
                            'title': content.get('title', ''),
                            'type': content.get('type', ''),
                            'space': content.get('space', {}).get('name', ''),
                            'url': f"{self.server}/wiki{content.get('_links', {}).get('webui', '')}",
                            'excerpt': result.get('excerpt', ''),
                            'last_modified': content.get('version', {}).get('when', ''),
                            'similarity_score': self._calculate_similarity(text, content.get('title', '') + ' ' + result.get('excerpt', ''))
                        }
                        
                        # Avoid duplicates
                        if not any(item['id'] == content_data['id'] for item in all_results):
                            all_results.append(content_data)
                            
            except Exception as e:
                logger.error(f"Error with search query '{query}': {e}")
                continue
        
        # Sort by similarity score
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return all_results[:limit]
    
    def get_space_content(self, space_key: str, limit: int = 50) -> List[Dict]:
        """Get recent content from a specific space"""
        if not self.confluence:
            return []
        
        try:
            cql_query = f'space = "{space_key}" ORDER BY lastModified DESC'
            
            results = self.confluence.cql(cql_query, limit=limit)
            
            content_list = []
            if 'results' in results:
                for result in results['results']:
                    content = result.get('content', {})
                    content_list.append({
                        'id': content.get('id', ''),
                        'title': content.get('title', ''),
                        'type': content.get('type', ''),
                        'url': f"{self.server}/wiki{content.get('_links', {}).get('webui', '')}",
                        'last_modified': content.get('version', {}).get('when', ''),
                        'author': content.get('version', {}).get('by', {}).get('displayName', '')
                    })
            
            return content_list
        except Exception as e:
            logger.error(f"Error getting space content for {space_key}: {e}")
            return []
    
    def search_by_labels(self, labels: List[str], limit: int = 20) -> List[Dict]:
        """Search content by labels"""
        if not self.confluence or not labels:
            return []
        
        try:
            # Build CQL query for labels
            label_conditions = ' AND '.join([f'label = "{label}"' for label in labels])
            cql_query = f'{label_conditions} ORDER BY lastModified DESC'
            
            results = self.confluence.cql(cql_query, limit=limit)
            
            content_list = []
            if 'results' in results:
                for result in results['results']:
                    content = result.get('content', {})
                    content_list.append({
                        'id': content.get('id', ''),
                        'title': content.get('title', ''),
                        'type': content.get('type', ''),
                        'space': content.get('space', {}).get('name', ''),
                        'url': f"{self.server}/wiki{content.get('_links', {}).get('webui', '')}",
                        'last_modified': content.get('version', {}).get('when', ''),
                        'author': content.get('version', {}).get('by', {}).get('displayName', '')
                    })
            
            return content_list
        except Exception as e:
            logger.error(f"Error searching by labels {labels}: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> str:
        """Extract key terms from text for better searching"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(keywords[:10])
    
    def _parse_spaces(self, spaces_str: str) -> List[str]:
        """Parse comma-separated spaces from environment variable"""
        if not spaces_str:
            return []
        return [space.strip() for space in spaces_str.split(',') if space.strip()]
    
    def search_in_configured_spaces(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for content in all configured Confluence spaces"""
        if not self.confluence or not self.spaces:
            logger.warning("No spaces configured or Confluence client not initialized")
            return []
        
        all_results = []
        results_per_space = max(1, limit // len(self.spaces))
        
        for space in self.spaces:
            try:
                space_results = self.search_in_space(query, space, results_per_space)
                all_results.extend(space_results)
                logger.info(f"Found {len(space_results)} results in space '{space}'")
            except Exception as e:
                logger.error(f"Error searching in space '{space}': {e}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_results = []
        seen_ids = set()
        
        for result in all_results:
            if result.get('id') not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.get('id'))
        
        return unique_results[:limit]
    
    def search_in_multiple_spaces(self, query: str, space_keys: List[str], limit: int = 20) -> List[Dict]:
        """Search for content in multiple specified Confluence spaces"""
        if not self.confluence or not space_keys:
            return []
        
        all_results = []
        results_per_space = max(1, limit // len(space_keys))
        
        for space_key in space_keys:
            try:
                space_results = self.search_in_space(query, space_key, results_per_space)
                all_results.extend(space_results)
                logger.info(f"Found {len(space_results)} results in space '{space_key}'")
            except Exception as e:
                logger.error(f"Error searching in space '{space_key}': {e}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_results = []
        seen_ids = set()
        
        for result in all_results:
            if result.get('id') not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.get('id'))
        
        return unique_results[:limit]
    
    def get_configured_spaces(self) -> List[str]:
        """Get the list of configured spaces"""
        return self.spaces.copy()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
