import os
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Optional
import logging
from atlassian import Confluence

logger = logging.getLogger(__name__)

class ConfluenceTool:
    def __init__(self):
        self.server = os.getenv('CONFLUENCE_SERVER')
        self.username = os.getenv('CONFLUENCE_USERNAME')
        self.api_token = os.getenv('CONFLUENCE_API_TOKEN')
        
        if self.server and self.username and self.api_token:
            try:
                self.confluence = Confluence(
                    url=self.server,
                    username=self.username,
                    password=self.api_token,
                    cloud=True  # Set to False for Confluence Server
                )
            except Exception as e:
                logger.error(f"Failed to initialize Confluence client: {e}")
                self.confluence = None
        else:
            logger.warning("Confluence credentials not found in environment variables")
            self.confluence = None
    
    def search_content(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for content in Confluence"""
        if not self.confluence:
            return []
        
        try:
            # Use CQL (Confluence Query Language) for search
            cql_query = f'text ~ "{query}"'
            
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
            logger.error(f"Error searching Confluence content: {e}")
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
    
    def search_similar_content(self, text: str, limit: int = 10) -> List[Dict]:
        """Find similar content based on text"""
        if not self.confluence:
            return []
        
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
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity calculation based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
