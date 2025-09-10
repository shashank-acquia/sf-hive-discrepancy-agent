import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import subprocess
import tempfile
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    transport: str = "stdio"

class MCPProtocolClient:
    """
    Proper MCP Protocol client that communicates with MCP servers using stdio transport
    """
    
    def __init__(self):
        self.active_servers = {}
        
    async def start_server(self, config: MCPServerConfig) -> bool:
        """Start an MCP server process"""
        try:
            logger.info(f"Starting MCP server {config.name} with command: {config.command} {' '.join(config.args)}")
            
            # Start the MCP server process
            process = await asyncio.create_subprocess_exec(
                config.command,
                *config.args,
                env={**os.environ, **config.env},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send initialization message
            init_message = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "sf-hive-discrepancy-agent",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Send the message
            message_str = json.dumps(init_message) + "\n"
            process.stdin.write(message_str.encode())
            await process.stdin.drain()
            
            # Wait for response with timeout
            try:
                response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
                response_str = response_line.decode().strip()
                
                if response_str:
                    response = json.loads(response_str)
                    if response.get("result"):
                        logger.info(f"✅ MCP server {config.name} initialized successfully")
                        self.active_servers[config.name] = {
                            'process': process,
                            'config': config,
                            'capabilities': response.get("result", {}).get("capabilities", {})
                        }
                        return True
                    else:
                        logger.error(f"❌ MCP server {config.name} initialization failed: {response}")
                        await self._cleanup_process(process)
                        return False
                else:
                    logger.error(f"❌ No response from MCP server {config.name}")
                    await self._cleanup_process(process)
                    return False
                    
            except asyncio.TimeoutError:
                logger.error(f"❌ MCP server {config.name} initialization timed out")
                await self._cleanup_process(process)
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start MCP server {config.name}: {e}")
            return False
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call a tool on an MCP server"""
        if server_name not in self.active_servers:
            logger.error(f"MCP server {server_name} not active")
            return None
            
        try:
            server_info = self.active_servers[server_name]
            process = server_info['process']
            
            # Create tool call message
            tool_message = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Send the message
            message_str = json.dumps(tool_message) + "\n"
            process.stdin.write(message_str.encode())
            await process.stdin.drain()
            
            # Wait for response
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=30.0)
            response_str = response_line.decode().strip()
            
            if response_str:
                response = json.loads(response_str)
                if "result" in response:
                    logger.info(f"✅ MCP tool call {tool_name} on {server_name} successful")
                    return response["result"]
                elif "error" in response:
                    logger.error(f"❌ MCP tool call {tool_name} on {server_name} failed: {response['error']}")
                    return None
                else:
                    logger.warning(f"⚠️ Unexpected response from {server_name}: {response}")
                    return None
            else:
                logger.error(f"❌ No response from MCP server {server_name} for tool {tool_name}")
                return None
                
        except asyncio.TimeoutError:
            logger.error(f"❌ MCP tool call {tool_name} on {server_name} timed out")
            return None
        except Exception as e:
            logger.error(f"❌ Error calling tool {tool_name} on {server_name}: {e}")
            return None
    
    async def list_tools(self, server_name: str) -> Optional[List[Dict[str, Any]]]:
        """List available tools on an MCP server"""
        if server_name not in self.active_servers:
            logger.error(f"MCP server {server_name} not active")
            return None
            
        try:
            server_info = self.active_servers[server_name]
            process = server_info['process']
            
            # Create list tools message
            list_message = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list",
                "params": {}
            }
            
            # Send the message
            message_str = json.dumps(list_message) + "\n"
            process.stdin.write(message_str.encode())
            await process.stdin.drain()
            
            # Wait for response
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            response_str = response_line.decode().strip()
            
            if response_str:
                response = json.loads(response_str)
                if "result" in response:
                    tools = response["result"].get("tools", [])
                    logger.info(f"✅ Listed {len(tools)} tools from MCP server {server_name}")
                    return tools
                else:
                    logger.error(f"❌ Failed to list tools from {server_name}: {response}")
                    return None
            else:
                logger.error(f"❌ No response from MCP server {server_name} for tools list")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error listing tools from {server_name}: {e}")
            return None
    
    async def _cleanup_process(self, process):
        """Clean up a subprocess"""
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except:
            try:
                process.kill()
                await process.wait()
            except:
                pass
    
    async def shutdown_server(self, server_name: str):
        """Shutdown an MCP server"""
        if server_name in self.active_servers:
            server_info = self.active_servers[server_name]
            process = server_info['process']
            await self._cleanup_process(process)
            del self.active_servers[server_name]
            logger.info(f"🔌 Shut down MCP server {server_name}")
    
    async def shutdown_all(self):
        """Shutdown all active MCP servers"""
        for server_name in list(self.active_servers.keys()):
            await self.shutdown_server(server_name)
        logger.info("🔌 All MCP servers shut down")
    
    def get_active_servers(self) -> List[str]:
        """Get list of active server names"""
        return list(self.active_servers.keys())
    
    def is_server_active(self, server_name: str) -> bool:
        """Check if a server is active"""
        return server_name in self.active_servers
