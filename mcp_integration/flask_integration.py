"""
Flask Integration for MCP-Enhanced Search
Provides Flask routes and integration for the MCP-enhanced search functionality
"""

from flask import Blueprint, request, jsonify, render_template
import asyncio
import logging
from typing import Dict, Any, Optional
import json

from .mcp_enhanced_search_agent import MCPEnhancedSearchAgent
from .mcp_config import mcp_config

logger = logging.getLogger(__name__)

# Create Blueprint for MCP search routes
mcp_search_bp = Blueprint('mcp_search', __name__, url_prefix='/api/mcp')

# Global search agent instance
search_agent = None

def init_mcp_search_agent():
    """Initialize the MCP search agent"""
    global search_agent
    try:
        search_agent = MCPEnhancedSearchAgent()
        logger.info("MCP Enhanced Search Agent initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize MCP Enhanced Search Agent: {e}")
        return False

# Initialize search agent on module import
try:
    init_mcp_search_agent()
except Exception as e:
    logger.warning(f"Failed to initialize MCP search agent on import: {e}")

@mcp_search_bp.route('/status', methods=['GET'])
def get_mcp_status():
    """Get status of MCP servers and configuration"""
    try:
        status = mcp_config.validate_configuration()
        enabled_servers = list(mcp_config.get_enabled_servers().keys())
        
        return jsonify({
            'success': True,
            'agent_initialized': search_agent is not None,
            'enabled_servers': enabled_servers,
            'server_status': status,
            'total_servers': len(mcp_config.servers),
            'enabled_count': len(enabled_servers)
        })
    except Exception as e:
        logger.error(f"Error getting MCP status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mcp_search_bp.route('/search', methods=['POST'])
def enhanced_search():
    """Perform enhanced search across platforms using MCP servers"""
    if not search_agent:
        return jsonify({
            'success': False,
            'error': 'MCP Enhanced Search Agent not initialized'
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': 'Query parameter is required'
            }), 400
        
        query = data['query']
        platforms = data.get('platforms', None)  # None means search all enabled platforms
        context = data.get('context', {})
        
        # Run async search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                search_agent.enhanced_search(query, platforms, context)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced search: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mcp_search_bp.route('/platforms', methods=['GET'])
def get_available_platforms():
    """Get list of available search platforms"""
    try:
        enabled_servers = mcp_config.get_enabled_servers()
        platforms = []
        
        for server_name, config in enabled_servers.items():
            if server_name == 'atlassian':
                platforms.extend(['jira', 'confluence'])
            elif server_name in ['slack', 'github']:
                platforms.append(server_name)
            # Memory server is used for context, not direct search
        
        return jsonify({
            'success': True,
            'platforms': platforms,
            'server_configs': {name: {
                'name': config.name,
                'enabled': config.enabled
            } for name, config in enabled_servers.items()}
        })
        
    except Exception as e:
        logger.error(f"Error getting platforms: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mcp_search_bp.route('/search/history', methods=['GET'])
def get_search_history():
    """Get recent search history from memory server"""
    if not search_agent:
        return jsonify({
            'success': False,
            'error': 'MCP Enhanced Search Agent not initialized'
        }), 503
    
    try:
        # This would use the memory server to retrieve search history
        # Implementation depends on memory server capabilities
        return jsonify({
            'success': True,
            'history': [],
            'message': 'Search history feature requires memory server implementation'
        })
        
    except Exception as e:
        logger.error(f"Error getting search history: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@mcp_search_bp.route('/config/validate', methods=['POST'])
def validate_mcp_config():
    """Validate and update MCP configuration"""
    try:
        data = request.get_json() or {}
        
        # Update environment variables if provided
        import os
        for key, value in data.items():
            if key.startswith(('ATLASSIAN_', 'SLACK_', 'GITHUB_')):
                os.environ[key] = value
        
        # Reload configuration
        mcp_config._load_default_configs()
        status = mcp_config.validate_configuration()
        
        # Reinitialize search agent if needed
        global search_agent
        if any(config.enabled for config in mcp_config.servers.values()):
            init_mcp_search_agent()
        
        return jsonify({
            'success': True,
            'status': status,
            'agent_reinitialized': search_agent is not None
        })
        
    except Exception as e:
        logger.error(f"Error validating MCP config: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def register_mcp_routes(app):
    """Register MCP search routes with Flask app"""
    app.register_blueprint(mcp_search_bp)
    
    # Initialize search agent on startup
    with app.app_context():
        init_success = init_mcp_search_agent()
        if init_success:
            logger.info("MCP Enhanced Search integration registered successfully")
        else:
            logger.warning("MCP Enhanced Search Agent initialization failed - check configuration")

# Utility functions for integration with existing search functionality

def enhance_existing_search_results(existing_results: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Enhance existing search results with MCP-powered insights"""
    if not search_agent:
        return existing_results
    
    try:
        # Run MCP search to get additional context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            mcp_results = loop.run_until_complete(
                search_agent.enhanced_search(query, context={'existing_results': existing_results})
            )
        finally:
            loop.close()
        
        # Merge results
        enhanced_results = existing_results.copy()
        enhanced_results['mcp_enhanced'] = True
        enhanced_results['additional_insights'] = mcp_results.get('summary', '')
        enhanced_results['cross_platform_results'] = mcp_results.get('results', [])
        enhanced_results['semantic_score'] = mcp_results.get('metadata', {}).get('avg_relevance_score', 0)
        enhanced_results['platform_insights'] = mcp_results.get('platform_insights', {})
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Error enhancing search results: {e}")
        return existing_results

async def async_enhance_search_results(existing_results: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Async version of enhance_existing_search_results"""
    if not search_agent:
        return existing_results
    
    try:
        mcp_results = await search_agent.enhanced_search(
            query, 
            context={'existing_results': existing_results}
        )
        
        enhanced_results = existing_results.copy()
        enhanced_results['mcp_enhanced'] = True
        enhanced_results['additional_insights'] = mcp_results.get('summary', '')
        enhanced_results['cross_platform_results'] = mcp_results.get('results', [])
        enhanced_results['semantic_score'] = mcp_results.get('metadata', {}).get('avg_relevance_score', 0)
        
        return enhanced_results
        
    except Exception as e:
        logger.error(f"Error enhancing search results: {e}")
        return existing_results
