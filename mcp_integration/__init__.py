"""
MCP Enhanced Cross-Platform Search Integration

This module provides Model Context Protocol (MCP) server integration
for enhanced cross-platform search across Slack, Jira, and Confluence.

Key Components:
- MCPEnhancedSearchAgent: Main search agent with MCP capabilities
- MCPConfigManager: Configuration management for MCP servers
- Flask Integration: REST API endpoints for MCP-enhanced search

Usage:
    from mcp_integration import MCPEnhancedSearchAgent
    
    # Initialize the search agent
    agent = MCPEnhancedSearchAgent()
    
    # Perform enhanced search
    results = await agent.enhanced_search("authentication error")
"""

__version__ = "1.0.0"
__author__ = "SF-Hive Discrepancy Agent Team"

# Import main components for easy access
try:
    from .mcp_enhanced_search_agent import MCPEnhancedSearchAgent, SearchResult
    from .mcp_config import MCPConfigManager, MCPServerConfig, mcp_config
    from .flask_integration import (
        register_mcp_routes, 
        enhance_existing_search_results,
        async_enhance_search_results
    )
    
    __all__ = [
        'MCPEnhancedSearchAgent',
        'SearchResult',
        'MCPConfigManager', 
        'MCPServerConfig',
        'mcp_config',
        'register_mcp_routes',
        'enhance_existing_search_results',
        'async_enhance_search_results'
    ]
    
    # Module is fully available
    MCP_MODULE_AVAILABLE = True
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import logging
    logging.warning(f"MCP integration dependencies not available: {e}")
    
    # Provide stub implementations
    class MCPEnhancedSearchAgent:
        def __init__(self):
            raise ImportError("MCP dependencies not installed. Run: pip install mcp langchain-openai scikit-learn")
    
    class MCPConfigManager:
        def __init__(self):
            raise ImportError("MCP dependencies not installed")
    
    def register_mcp_routes(app):
        logging.warning("MCP integration not available - routes not registered")
        return False
    
    def enhance_existing_search_results(results, query):
        logging.warning("MCP enhancement not available - returning original results")
        return results
    
    async def async_enhance_search_results(results, query):
        logging.warning("MCP enhancement not available - returning original results")
        return results
    
    __all__ = [
        'MCPEnhancedSearchAgent',
        'MCPConfigManager',
        'register_mcp_routes',
        'enhance_existing_search_results',
        'async_enhance_search_results'
    ]
    
    # Module has limited functionality
    MCP_MODULE_AVAILABLE = False

# Version and availability info
def get_version():
    """Get the current version of the MCP integration module."""
    return __version__

def is_available():
    """Check if MCP integration is fully available."""
    return MCP_MODULE_AVAILABLE

def get_status():
    """Get detailed status of MCP integration availability."""
    if MCP_MODULE_AVAILABLE:
        try:
            # Try to get configuration status
            status = mcp_config.validate_configuration()
            enabled_servers = list(mcp_config.get_enabled_servers().keys())
            
            return {
                'available': True,
                'version': __version__,
                'enabled_servers': enabled_servers,
                'server_status': status,
                'message': 'MCP integration fully available'
            }
        except Exception as e:
            return {
                'available': True,
                'version': __version__,
                'enabled_servers': [],
                'server_status': {},
                'message': f'MCP integration available but configuration error: {e}'
            }
    else:
        return {
            'available': False,
            'version': __version__,
            'enabled_servers': [],
            'server_status': {},
            'message': 'MCP integration not available - missing dependencies'
        }
