"""
MCP Server Configuration
Handles configuration and initialization of MCP servers for cross-platform search
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for MCP servers"""
    name: str
    command: str
    args: List[str]
    env: Optional[Dict[str, str]] = None
    enabled: bool = True

class MCPConfigManager:
    """Manages MCP server configurations"""
    
    def __init__(self):
        self.servers = {}
        self._load_default_configs()
    
    def _extract_site_name_from_url(self, url: Optional[str]) -> Optional[str]:
        """Extract site name from Atlassian URL"""
        if not url:
            return None
        
        # Remove protocol and trailing slashes
        url = url.replace('https://', '').replace('http://', '').rstrip('/')
        
        # Extract site name from patterns like:
        # mycompany.atlassian.net -> mycompany
        # mycompany.atlassian.com -> mycompany
        if '.atlassian.' in url:
            return url.split('.atlassian.')[0]
        
        # If it's just the site name already, return it
        if '.' not in url:
            return url
        
        return None
    
    def _load_default_configs(self):
        """Load default MCP server configurations"""
        
        # Atlassian MCP Server (Jira + Confluence)
        # Use existing environment variables from .env
        atlassian_env = {}
        
        # Check for both direct ATLASSIAN_ vars and fallback to JIRA_/CONFLUENCE_
        atlassian_token = os.getenv('ATLASSIAN_API_TOKEN') or os.getenv('JIRA_API_TOKEN') or os.getenv('CONFLUENCE_API_TOKEN')
        atlassian_site_name = os.getenv('ATLASSIAN_SITE_NAME') or self._extract_site_name_from_url(
            os.getenv('ATLASSIAN_INSTANCE_URL') or os.getenv('JIRA_SERVER') or os.getenv('CONFLUENCE_SERVER')
        )
        atlassian_email = os.getenv('ATLASSIAN_USER_EMAIL') or os.getenv('ATLASSIAN_EMAIL') or os.getenv('JIRA_USERNAME') or os.getenv('CONFLUENCE_USERNAME')
        
        if atlassian_token and atlassian_site_name and atlassian_email:
            atlassian_env.update({
                'ATLASSIAN_API_TOKEN': atlassian_token,
                'ATLASSIAN_SITE_NAME': atlassian_site_name,
                'ATLASSIAN_USER_EMAIL': atlassian_email,
                # Keep legacy variables for backward compatibility
                'ATLASSIAN_INSTANCE_URL': os.getenv('ATLASSIAN_INSTANCE_URL', ''),
                'ATLASSIAN_EMAIL': os.getenv('ATLASSIAN_EMAIL', ''),
                'JIRA_SERVER': os.getenv('JIRA_SERVER', ''),
                'CONFLUENCE_SERVER': os.getenv('CONFLUENCE_SERVER', ''),
                'JIRA_PROJECT_KEY': os.getenv('JIRA_PROJECT_KEY', ''),
                'CONFLUENCE_SPACES': os.getenv('CONFLUENCE_SPACES', '')
            })
        
        # Split into separate Jira and Confluence servers for better reliability
        self.servers['jira'] = MCPServerConfig(
            name='jira',
            command='npx',
            args=['-y', '@aashari/mcp-server-atlassian-jira'],
            env=atlassian_env,
            enabled=bool(atlassian_token and atlassian_site_name and atlassian_email)
        )
        
        self.servers['confluence'] = MCPServerConfig(
            name='confluence',
            command='npx',
            args=['-y', '@aashari/mcp-server-atlassian-confluence'],
            env=atlassian_env,
            enabled=bool(atlassian_token and atlassian_site_name and atlassian_email)
        )
        
        # Slack MCP Server
        slack_env = {}
        # Check for available Slack tokens
        slack_user_token = os.getenv('SLACK_USER_TOKEN')  # xoxp- token from .env
        slack_bot_token = os.getenv('SLACK_BOT_TOKEN')    # xoxb- token from .env
        slack_app_token = os.getenv('SLACK_APP_TOKEN')    # xapp- token from .env
        
        # The @modelcontextprotocol/server-slack expects SLACK_USER_TOKEN
        slack_auth_available = slack_user_token or slack_bot_token
        
        if slack_auth_available:
            slack_env.update({
                # Primary authentication for @modelcontextprotocol/server-slack
                'SLACK_USER_TOKEN': slack_user_token or slack_bot_token,
                'SLACK_BOT_TOKEN': slack_bot_token or '',
                'SLACK_APP_TOKEN': slack_app_token or '',
                'SLACK_SIGNING_SECRET': os.getenv('SLACK_SIGNING_SECRET', ''),
                'SLACK_SEARCH_CHANNELS': os.getenv('SLACK_SEARCH_CHANNELS', ''),
                'SLACK_TEAM_ID': os.getenv('SLACK_TEAM_ID', ''),
                # Additional configuration
                'SLACK_PORT': os.getenv('SLACK_PORT', '3000'),
                'SLACK_HOST': os.getenv('SLACK_HOST', '127.0.0.1')
            })
        
        self.servers['slack'] = MCPServerConfig(
            name='slack',
            command='npx',
            args=['-y', '@modelcontextprotocol/server-slack'],
            env=slack_env,
            enabled=bool(slack_auth_available)
        )
        
        # GitHub MCP Server (for additional context)
        github_env = {}
        if os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'):
            github_env.update({
                'GITHUB_PERSONAL_ACCESS_TOKEN': os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
            })
        
        self.servers['github'] = MCPServerConfig(
            name='github',
            command='npx',
            args=['-y', '@modelcontextprotocol/server-github'],
            env=github_env,
            enabled=bool(os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'))
        )
        
        # Memory MCP Server (for persistent context)
        self.servers['memory'] = MCPServerConfig(
            name='memory',
            command='npx',
            args=['-y', '@modelcontextprotocol/server-memory'],
            env={},
            enabled=True  # Always enabled as it doesn't require external credentials
        )
        
        # Google Docs MCP Server
        google_docs_env = {}
        if os.getenv('MCP_GOOGLE_DOC_ENABLED', 'false').lower() == 'true':
            google_docs_env.update({
                'GOOGLE_CLIENT_ID': os.getenv('GOOGLE_CLIENT_ID', ''),
                'GOOGLE_CLIENT_SECRET': os.getenv('GOOGLE_CLIENT_SECRET', ''),
                'GOOGLE_REFRESH_TOKEN': os.getenv('GOOGLE_REFRESH_TOKEN', ''),
                'GOOGLE_DOCS_FOLDER_ID': os.getenv('GOOGLE_DOCS_FOLDER_ID', ''),
                'GOOGLE_DOCS_FOLDER_NAME': os.getenv('GOOGLE_DOCS_FOLDER_NAME', ''),
                'GOOGLE_DOCS_RECURSIVE_SEARCH': os.getenv('GOOGLE_DOCS_RECURSIVE_SEARCH', 'true'),
                'GOOGLE_DOCS_MAX_RESULTS': os.getenv('GOOGLE_DOCS_MAX_RESULTS', '50')
            })
        
        self.servers['google_docs'] = MCPServerConfig(
            name='google_docs',
            command='npx',
            args=['-y', 'mcp-google-drive'],
            env=google_docs_env,
            enabled=bool(os.getenv('MCP_GOOGLE_DOC_ENABLED', 'false').lower() == 'true' and 
                        os.getenv('GOOGLE_CLIENT_ID') and 
                        os.getenv('GOOGLE_CLIENT_SECRET'))
        )
    
    def get_enabled_servers(self) -> Dict[str, MCPServerConfig]:
        """Get all enabled MCP server configurations"""
        return {name: config for name, config in self.servers.items() if config.enabled}
    
    def get_server_config(self, name: str) -> Optional[MCPServerConfig]:
        """Get configuration for a specific server"""
        return self.servers.get(name)
    
    def is_server_enabled(self, name: str) -> bool:
        """Check if a server is enabled"""
        config = self.servers.get(name)
        return config.enabled if config else False
    
    def validate_configuration(self) -> Dict[str, str]:
        """Validate MCP server configurations and return status"""
        status = {}
        
        for name, config in self.servers.items():
            if not config.enabled:
                status[name] = "Disabled - missing required environment variables"
                continue
            
            # Check if required environment variables are set
            if name in ['jira', 'confluence']:
                # Use the same fallback logic as in _load_default_configs
                atlassian_token = os.getenv('ATLASSIAN_API_TOKEN') or os.getenv('JIRA_API_TOKEN') or os.getenv('CONFLUENCE_API_TOKEN')
                atlassian_site_name = os.getenv('ATLASSIAN_SITE_NAME') or self._extract_site_name_from_url(
                    os.getenv('ATLASSIAN_INSTANCE_URL') or os.getenv('JIRA_SERVER') or os.getenv('CONFLUENCE_SERVER')
                )
                atlassian_email = os.getenv('ATLASSIAN_USER_EMAIL') or os.getenv('ATLASSIAN_EMAIL') or os.getenv('JIRA_USERNAME') or os.getenv('CONFLUENCE_USERNAME')
                
                if not (atlassian_token and atlassian_site_name and atlassian_email):
                    status[name] = "Disabled - missing required environment variables"
                    config.enabled = False
                else:
                    status[name] = "Ready"
            
            elif name == 'slack':
                # Check for available Slack tokens (matching the _load_default_configs logic)
                slack_user_token = os.getenv('SLACK_USER_TOKEN')
                slack_bot_token = os.getenv('SLACK_BOT_TOKEN')
                
                slack_auth_available = slack_user_token or slack_bot_token
                
                if not slack_auth_available:
                    status[name] = "Missing Slack authentication - need SLACK_USER_TOKEN or SLACK_BOT_TOKEN"
                    config.enabled = False
                else:
                    status[name] = "Ready"
            
            elif name == 'github':
                if not os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'):
                    status[name] = "Missing GITHUB_PERSONAL_ACCESS_TOKEN"
                    config.enabled = False
                else:
                    status[name] = "Ready"
            
            elif name == 'google_docs':
                if not os.getenv('MCP_GOOGLE_DOC_ENABLED', 'false').lower() == 'true':
                    status[name] = "Disabled - MCP_GOOGLE_DOC_ENABLED not set to true"
                    config.enabled = False
                elif not (os.getenv('GOOGLE_CLIENT_ID') and os.getenv('GOOGLE_CLIENT_SECRET')):
                    status[name] = "Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET"
                    config.enabled = False
                elif not (os.getenv('GOOGLE_DOCS_FOLDER_ID', '').strip() or os.getenv('GOOGLE_DOCS_FOLDER_NAME', '').strip()):
                    status[name] = "Missing GOOGLE_DOCS_FOLDER_ID or GOOGLE_DOCS_FOLDER_NAME"
                    config.enabled = False
                else:
                    status[name] = "Ready"
            
            else:
                status[name] = "Ready"
        
        return status

# Global configuration manager instance
mcp_config = MCPConfigManager()
