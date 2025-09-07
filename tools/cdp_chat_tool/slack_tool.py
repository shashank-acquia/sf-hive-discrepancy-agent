import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SlackTool:
    def __init__(self):
        self.client = WebClient(token=os.getenv('SLACK_BOT_TOKEN'))
        self.user_token = os.getenv('SLACK_USER_TOKEN')  # For reading messages from channels
        self.user_client = WebClient(token=self.user_token) if self.user_token else None
        
    def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None) -> Dict:
        """Send a message to a Slack channel"""
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=text,
                thread_ts=thread_ts
            )
            return {
                'success': True,
                'message': 'Message sent successfully',
                'ts': response['ts']
            }
        except SlackApiError as e:
            logger.error(f"Error sending message: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def search_messages(self, query: str, count: int = 20) -> List[Dict]:
        """Search for messages across Slack workspace"""
        try:
            if not self.user_client:
                return []
                
            response = self.user_client.search_messages(
                query=query,
                count=count
            )
            
            messages = []
            if response['ok'] and 'messages' in response:
                for match in response['messages']['matches']:
                    messages.append({
                        'text': match.get('text', ''),
                        'user': match.get('user', ''),
                        'channel': match.get('channel', {}).get('name', ''),
                        'ts': match.get('ts', ''),
                        'permalink': match.get('permalink', ''),
                        'score': match.get('score', 0)
                    })
            
            return messages
        except SlackApiError as e:
            logger.error(f"Error searching messages: {e}")
            return []
    
    def get_channel_history(self, channel: str, limit: int = 50) -> List[Dict]:
        """Get recent messages from a specific channel"""
        try:
            if not self.user_client:
                return []
                
            response = self.user_client.conversations_history(
                channel=channel,
                limit=limit
            )
            
            messages = []
            if response['ok']:
                for message in response['messages']:
                    messages.append({
                        'text': message.get('text', ''),
                        'user': message.get('user', ''),
                        'ts': message.get('ts', ''),
                        'thread_ts': message.get('thread_ts', '')
                    })
            
            return messages
        except SlackApiError as e:
            logger.error(f"Error getting channel history: {e}")
            return []
    
    def get_user_info(self, user_id: str) -> Dict:
        """Get user information"""
        try:
            response = self.client.users_info(user=user_id)
            if response['ok']:
                user = response['user']
                return {
                    'name': user.get('name', ''),
                    'real_name': user.get('real_name', ''),
                    'email': user.get('profile', {}).get('email', '')
                }
            return {}
        except SlackApiError as e:
            logger.error(f"Error getting user info: {e}")
            return {}
    
    def get_channel_info(self, channel: str) -> Dict:
        """Get channel information"""
        try:
            response = self.client.conversations_info(channel=channel)
            if response['ok']:
                channel_info = response['channel']
                return {
                    'name': channel_info.get('name', ''),
                    'purpose': channel_info.get('purpose', {}).get('value', ''),
                    'topic': channel_info.get('topic', {}).get('value', ''),
                    'is_private': channel_info.get('is_private', False)
                }
            return {}
        except SlackApiError as e:
            logger.error(f"Error getting channel info: {e}")
            return {}
    
    def search_in_channels(self, query: str, channels: List[str], limit: int = 10) -> List[Dict]:
        """Search for messages in specific channels using conversation history"""
        all_results = []
        
        # If no user token available, use conversation history as fallback
        if not self.user_client:
            logger.warning("No user token available for search. Using conversation history as fallback.")
            return self._search_via_conversation_history(query, channels, limit)
        
        for channel in channels:
            try:
                # Build search query for specific channel
                channel_query = f"{query} in:#{channel}"
                
                response = self.user_client.search_messages(
                    query=channel_query,
                    count=limit
                )
                
                if response['ok'] and 'messages' in response:
                    for match in response['messages']['matches']:
                        all_results.append({
                            'text': match.get('text', ''),
                            'user': match.get('user', ''),
                            'channel': match.get('channel', {}).get('name', ''),
                            'ts': match.get('ts', ''),
                            'permalink': match.get('permalink', ''),
                            'score': match.get('score', 0)
                        })
                        
            except SlackApiError as e:
                logger.error(f"Error searching in channel {channel}: {e}")
                # Fallback to conversation history for this channel
                fallback_results = self._search_via_conversation_history(query, [channel], limit)
                all_results.extend(fallback_results)
                continue
        
        # Sort by score (relevance)
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:limit * len(channels)]
    
    def _search_via_conversation_history(self, query: str, channels: List[str], limit: int = 10) -> List[Dict]:
        """Fallback search using conversation history when search API is not available"""
        all_results = []
        query_words = query.lower().split()
        
        for channel in channels:
            try:
                # Get channel ID from name
                channel_id = self._get_channel_id(channel)
                if not channel_id:
                    continue
                
                # Get recent messages from channel
                response = self.client.conversations_history(
                    channel=channel_id,
                    limit=100  # Get more messages to search through
                )
                
                if response['ok']:
                    for message in response['messages']:
                        text = message.get('text', '').lower()
                        
                        # Simple keyword matching
                        if any(word in text for word in query_words):
                            # Calculate simple relevance score
                            score = sum(1 for word in query_words if word in text)
                            
                            # Get thread replies if this message has replies or is part of a thread
                            thread_summary = ""
                            reply_count = message.get('reply_count', 0)
                            
                            # Check if this message has replies or is a thread parent
                            if reply_count > 0 or message.get('thread_ts'):
                                thread_ts = message.get('thread_ts') or message.get('ts', '')
                                thread_summary = self._get_thread_summary(channel_id, thread_ts)
                                logger.info(f"Thread summary for message {message.get('ts', '')}: {thread_summary}")
                            
                            all_results.append({
                                'text': message.get('text', ''),
                                'user': message.get('user', ''),
                                'channel': channel,
                                'ts': message.get('ts', ''),
                                'permalink': f"https://slack.com/archives/{channel_id}/p{message.get('ts', '').replace('.', '')}",
                                'score': score,
                                'thread_summary': thread_summary,
                                'reply_count': reply_count
                            })
                            
            except SlackApiError as e:
                logger.error(f"Error getting conversation history for {channel}: {e}")
                continue
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return all_results[:limit]
    
    def _get_channel_id(self, channel_name: str) -> Optional[str]:
        """Get channel ID from channel name"""
        try:
            # Remove # if present
            channel_name = channel_name.lstrip('#')
            
            response = self.client.conversations_list(
                types="public_channel,private_channel"
            )
            
            if response['ok']:
                for channel in response['channels']:
                    if channel.get('name') == channel_name:
                        return channel.get('id')
            
            return None
            
        except SlackApiError as e:
            logger.error(f"Error getting channel ID for {channel_name}: {e}")
            return None
    
    def _get_thread_summary(self, channel_id: str, thread_ts: str) -> str:
        """Get a summary of thread replies"""
        try:
            if not thread_ts:
                logger.debug("No thread_ts provided")
                return ""
            
            logger.info(f"Getting thread replies for ts: {thread_ts} in channel: {channel_id}")
            
            response = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=20  # Get more replies
            )
            
            logger.info(f"Thread replies response: ok={response.get('ok')}, message_count={len(response.get('messages', []))}")
            
            if response['ok'] and len(response['messages']) > 1:
                # Skip the first message (original) and get replies
                replies = response['messages'][1:]
                logger.info(f"Found {len(replies)} thread replies")
                
                # Expanded solution keywords - more comprehensive
                solution_keywords = [
                    'solved', 'fixed', 'resolved', 'solution', 'answer', 'try this', 'here\'s how',
                    'workaround', 'fix', 'restart', 'rerun', 'check', 'update', 'change',
                    'working', 'works', 'success', 'done', 'complete', 'issue resolved',
                    'problem solved', 'that worked', 'thanks', 'perfect', 'great',
                    'run this', 'use this', 'do this', 'configure', 'set', 'enable'
                ]
                
                solutions = []
                all_replies = []
                
                for i, reply in enumerate(replies):
                    text = reply.get('text', '')
                    text_lower = text.lower()
                    all_replies.append(f"Reply {i+1}: {text[:100]}...")
                    
                    # Check for solution keywords
                    matching_keywords = [kw for kw in solution_keywords if kw in text_lower]
                    if matching_keywords:
                        logger.info(f"Found solution keywords {matching_keywords} in reply: {text[:50]}...")
                        solutions.append(text[:300])  # Get more text for solutions
                
                logger.info(f"Found {len(solutions)} potential solutions")
                
                if solutions:
                    # Return the most relevant solutions
                    solution_text = ' | '.join(solutions[:2])
                    return f"Solutions found: {solution_text}"
                else:
                    # Return summary with some actual content
                    reply_count = len(replies)
                    sample_replies = ' | '.join(all_replies[:2])
                    return f"{reply_count} replies discussing: {sample_replies}"
            else:
                logger.info("No thread replies found or API call failed")
            
            return ""
            
        except SlackApiError as e:
            logger.error(f"Error getting thread replies for {thread_ts}: {e}")
            return f"Error accessing thread (may need permissions): {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error getting thread replies: {e}")
            return f"Error processing thread: {str(e)}"
