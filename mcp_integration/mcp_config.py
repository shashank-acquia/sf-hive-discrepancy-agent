"""
MCP Server Configuration
Handles configuration and initialization of MCP servers for cross-platform search
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

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
    
    def _load_default_configs(self):
        """Load default MCP server configurations"""
        
        # Atlassian MCP Server (Jira + Confluence)
        # Use existing environment variables from .env
        atlassian_env = {}
        jira_token = os.getenv('JIRA_API_TOKEN')
        confluence_token = os.getenv('CONFLUENCE_API_TOKEN')
        
        if jira_token or confluence_token:
            atlassian_env.update({
                'ATLASSIAN_API_TOKEN': jira_token or confluence_token,
                'ATLASSIAN_DOMAIN': 'acquia.atlassian.net',  # Extracted from JIRA_SERVER
                'ATLASSIAN_EMAIL': os.getenv('JIRA_USERNAME', os.getenv('CONFLUENCE_USERNAME', '')),
                'JIRA_SERVER': os.getenv('JIRA_SERVER', ''),
                'CONFLUENCE_SERVER': os.getenv('CONFLUENCE_SERVER', ''),
                'JIRA_PROJECT_KEY': os.getenv('JIRA_PROJECT_KEY', ''),
                'CONFLUENCE_SPACES': os.getenv('CONFLUENCE_SPACES', '')
            })
        
        self.servers['atlassian'] = MCPServerConfig(
            name='atlassian',
            command='npx',
            args=['-y', '@modelcontextprotocol/server-atlassian'],
            env=atlassian_env,
            enabled=bool(jira_token or confluence_token)
        )
        
        # Slack MCP Server
        slack_env = {}
        if os.getenv('SLACK_BOT_TOKEN'):
            slack_env.update({
                'SLACK_BOT_TOKEN': os.getenv('SLACK_BOT_TOKEN'),
                'SLACK_APP_TOKEN': os.getenv('SLACK_APP_TOKEN', ''),
                'SLACK_SIGNING_SECRET': os.getenv('SLACK_SIGNING_SECRET', ''),
                'SLACK_USER_TOKEN': os.getenv('SLACK_USER_TOKEN', ''),
                'SLACK_SEARCH_CHANNELS': os.getenv('SLACK_SEARCH_CHANNELS', ''),
                'SLACK_TEAM_ID': os.getenv('SLACK_TEAM_ID', '')
            })
        
        self.servers['slack'] = MCPServerConfig(
            name='slack',
            command='npx',
            args=['-y', '@zencoderai/mcp-server-slack'],
            env=slack_env,
            enabled=bool(os.getenv('SLACK_BOT_TOKEN'))
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
            if name == 'atlassian':
                # Check for either JIRA or Confluence tokens (using existing env vars)
                jira_token = os.getenv('JIRA_API_TOKEN')
                confluence_token = os.getenv('CONFLUENCE_API_TOKEN')
                jira_username = os.getenv('JIRA_USERNAME')
                confluence_username = os.getenv('CONFLUENCE_USERNAME')
                
                if not (jira_token or confluence_token):
                    status[name] = "Disabled - missing required environment variables"
                    config.enabled = False
                elif not (jira_username or confluence_username):
                    status[name] = "Missing username configuration"
                    config.enabled = False
                else:
                    status[name] = "Ready"
            
            elif name == 'slack':
                required_vars = ['SLACK_BOT_TOKEN']
                missing = [var for var in required_vars if not os.getenv(var)]
                if missing:
                    status[name] = f"Missing environment variables: {', '.join(missing)}"
                    config.enabled = False
                else:
                    status[name] = "Ready"
            
            elif name == 'github':
                if not os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'):
                    status[name] = "Missing GITHUB_PERSONAL_ACCESS_TOKEN"
                    config.enabled = False
                else:
                    status[name] = "Ready"
            
            else:
                status[name] = "Ready"
        
        return status

# Global configuration manager instance
mcp_config = MCPConfigManager()
