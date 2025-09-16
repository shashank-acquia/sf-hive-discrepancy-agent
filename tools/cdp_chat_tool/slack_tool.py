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
    
    def search_in_channels(self, query: str, channels: List[str],  limit: int = 10, slack_ids: Optional[List[str]] = None, message_text: Optional[str] = None) -> List[Dict]:
        """Search for messages in specific channels with enhanced thread detection and direct Slack URL lookup"""
        all_results = []

        # Extract Slack URLs from message_text if provided
        if message_text:
            import re
            slack_url_pattern = r'https://acquia\.slack\.com/archives/[A-Z0-9]+/p[0-9]+'
            extracted_urls = re.findall(slack_url_pattern, message_text)
            if slack_ids:
                slack_ids = list(set(slack_ids + extracted_urls))
            else:
                slack_ids = extracted_urls

        # If slack_ids are provided, fetch those threads/messages directly
        if slack_ids:
            for url in slack_ids:
                # Example URL: https://acquia.slack.com/archives/C012J3T0S9H/p1756995100810879
                import re
                match = re.match(r'https://acquia\.slack\.com/archives/([A-Z0-9]+)/p(\d+)', url)
                if match:
                    channel_id = match.group(1)
                    ts_raw = match.group(2)
                    # Slack timestamps are like 1756995100810879, need to convert to 1756995100.810879
                    ts = ts_raw[:10] + '.' + ts_raw[10:] if len(ts_raw) > 10 else ts_raw
                    try:
                        response = self.client.conversations_replies(
                            channel=channel_id,
                            ts=ts,
                            limit=20
                        )
                        if response['ok']:
                            for message in response['messages']:
                                all_results.append({
                                    'text': message.get('text', ''),
                                    'user': message.get('user', ''),
                                    'channel': channel_id,
                                    'ts': message.get('ts', ''),
                                    'permalink': url,
                                    'score': 100,  # High score for direct match
                                    'thread_summary': '',
                                    'reply_count': message.get('reply_count', 0)
                                })
                    except SlackApiError as e:
                        error_msg = str(e)
                        logger.error(f"Error fetching thread for Slack URL {url}: {e}")
                        
                        # Add fallback for channel_not_found error
                        if "channel_not_found" in error_msg:
                            logger.warning(f"Channel {channel_id} not found or bot doesn't have access. Adding placeholder result.")
                            # Add a placeholder result with error information
                            all_results.append({
                                'text': f"⚠️ Unable to access this message. The channel may be private, archived, or the bot lacks access permissions.",
                                'user': "SYSTEM",
                                'channel': channel_id,
                                'ts': ts,
                                'permalink': url,
                                'score': 50,  # Lower score for error placeholder
                                'thread_summary': "Access error: channel_not_found",
                                'reply_count': 0,
                                'error': True
                            })
                        continue

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
                        # Enhanced result processing with thread detection
                        result = {
                            'text': match.get('text', ''),
                            'user': match.get('user', ''),
                            'channel': match.get('channel', {}).get('name', ''),
                            'ts': match.get('ts', ''),
                            'permalink': match.get('permalink', ''),
                            'score': match.get('score', 0),
                            'thread_summary': '',
                            'reply_count': 0
                        }
                        
                        # Try to get thread information using search API for thread messages
                        thread_summary = self._get_thread_summary_from_search(channel, match.get('ts', ''), query)
                        if thread_summary:
                            result['thread_summary'] = thread_summary
                            # Estimate reply count from thread summary
                            if 'replies discussing' in thread_summary:
                                import re
                                reply_match = re.search(r'(\d+) replies', thread_summary)
                                if reply_match:
                                    result['reply_count'] = int(reply_match.group(1))
                        
                        all_results.append(result)
                        
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
    
    def _get_thread_summary_from_search(self, channel: str, thread_ts: str, original_query: str) -> str:
        """Get thread summary using search API by analyzing all search results for thread relationships"""
        try:
            if not thread_ts or not self.user_client:
                return ""
            
            logger.info(f"Getting thread summary via search analysis for ts: {thread_ts} in channel: {channel}")
            
            # Instead of searching for thread specifically, search broadly in the channel
            # and then filter for messages that are part of this thread
            broad_query = f"in:#{channel}"
            
            # Add some keywords from the original query to narrow down results
            query_words = original_query.split()[:3]  # Take first 3 words
            if query_words:
                broad_query += f" {' '.join(query_words)}"
            
            logger.info(f"Using broad search query: '{broad_query}'")
            
            try:
                response = self.user_client.search_messages(
                    query=broad_query,
                    count=50  # Get more results to find thread messages
                )
                
                if response['ok'] and 'messages' in response:
                    all_matches = response['messages']['matches']
                    logger.info(f"Found {len(all_matches)} total messages in broad search")
                    
                    # Filter for messages that are part of this thread
                    # Look for messages with thread_ts in their permalink or that reference the thread
                    thread_messages = []
                    original_message = None
                    
                    for match in all_matches:
                        match_ts = match.get('ts', '')
                        permalink = match.get('permalink', '')
                        
                        # Check if this is the original message
                        if match_ts == thread_ts:
                            original_message = match
                            logger.info(f"Found original message: {match.get('text', '')[:50]}...")
                        
                        # Check if this message is part of the thread (has thread_ts in permalink)
                        elif f"thread_ts={thread_ts}" in permalink:
                            thread_messages.append(match)
                            logger.info(f"Found thread message: {match.get('text', '')[:50]}...")
                    
                    logger.info(f"Found {len(thread_messages)} thread messages for thread {thread_ts}")
                    
                    if thread_messages:
                        # Process thread messages for solutions and Jira tickets
                        solutions = []
                        jira_tickets = []
                        
                        # Enhanced Jira ticket detection patterns
                        import re
                        jira_patterns = [
                            r'\b[A-Z0-9]{2,10}-\d+\b',
                            r'acquia\.atlassian\.net/browse/([A-Z0-9]+-\d+)',
                            r'atlassian\.net/browse/([A-Z0-9]+-\d+)',
                            r'/browse/([A-Z0-9]+-\d+)',
                            r'ticket\s*[:#]?\s*([A-Z0-9]+-\d+)',
                            r'issue\s*[:#]?\s*([A-Z0-9]+-\d+)',
                            r'jira\s*[:#]?\s*([A-Z0-9]+-\d+)',
                        ]
                        
                        solution_keywords = [
                            'solved', 'fixed', 'resolved', 'solution', 'answer', 'try this', 'here\'s how',
                            'workaround', 'fix', 'restart', 'rerun', 'check', 'update', 'change',
                            'working', 'works', 'success', 'done', 'complete', 'issue resolved',
                            'problem solved', 'that worked', 'thanks', 'perfect', 'great',
                            'run this', 'use this', 'do this', 'configure', 'set', 'enable',
                            'created', 'opened'  # Added for Jira ticket creation messages
                        ]
                        
                        # Process each thread message
                        for i, match in enumerate(thread_messages, 1):
                            text = match.get('text', '')
                            text_lower = text.lower()
                            user = match.get('user', 'Unknown')
                            match_ts = match.get('ts', '')
                            
                            logger.info(f"Processing thread message {i}: {text[:80]}...")
                            
                            # Enhanced Jira ticket detection
                            for pattern in jira_patterns:
                                pattern_matches = re.findall(pattern, text, re.IGNORECASE)
                                for pattern_match in pattern_matches:
                                    ticket_key = pattern_match if isinstance(pattern_match, str) else pattern_match[0] if pattern_match else None
                                    if ticket_key and ticket_key not in jira_tickets:
                                        jira_tickets.append(ticket_key)
                                        logger.info(f"🎫 Found Jira ticket reference: {ticket_key} in thread reply by {user} at {match_ts}")
                            
                            # Check for solution keywords
                            matching_keywords = [kw for kw in solution_keywords if kw in text_lower]
                            if matching_keywords:
                                logger.info(f"Found solution keywords {matching_keywords} in reply: {text[:50]}...")
                                solutions.append(text[:300])
                        
                        logger.info(f"Thread analysis complete: {len(solutions)} solutions, {len(jira_tickets)} Jira tickets")
                        
                        # Build enhanced summary
                        summary_parts = []
                        
                        if jira_tickets:
                            summary_parts.append(f"🎫 Jira tickets mentioned: {', '.join(jira_tickets[:3])}")
                        
                        if solutions:
                            solution_text = ' | '.join(solutions[:2])
                            summary_parts.append(f"Solutions found: {solution_text}")
                        else:
                            reply_count = len(thread_messages)
                            summary_parts.append(f"{reply_count} replies discussing the issue")
                        
                        final_summary = ' | '.join(summary_parts) if summary_parts else ""
                        logger.info(f"Final thread summary: {final_summary}")
                        return final_summary
                    else:
                        logger.info("No thread messages found in search results")
                else:
                    logger.warning(f"Broad search failed: {response.get('error', 'Unknown error')}")
                
            except SlackApiError as search_error:
                logger.warning(f"Search API failed for thread, trying fallback: {search_error}")
                # Fallback: try to get thread via conversations API if we have access
                return self._get_thread_summary_fallback(channel, thread_ts)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error getting thread summary via search: {e}")
            return ""
    
    def _get_thread_summary_fallback(self, channel: str, thread_ts: str) -> str:
        """Fallback method to get thread summary via conversations API"""
        try:
            # Get channel ID
            channel_id = self._get_channel_id(channel)
            if not channel_id:
                return ""
            
            return self._get_thread_summary(channel_id, thread_ts)
            
        except Exception as e:
            logger.error(f"Fallback thread summary failed: {e}")
            return ""
    
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
                jira_tickets = []  # Track Jira ticket references
                
                # Enhanced Jira ticket detection patterns for Acquia Atlassian
                import re
                jira_patterns = [
                    r'\b[A-Z0-9]{2,10}-\d+\b',  # Standard format: A1DEV-16638, PROJ-123
                    r'acquia\.atlassian\.net/browse/([A-Z0-9]+-\d+)',  # Acquia Jira URLs
                    r'atlassian\.net/browse/([A-Z0-9]+-\d+)',  # Generic Atlassian URLs
                    r'/browse/([A-Z0-9]+-\d+)',  # Any browse URL
                    r'ticket\s*[:#]?\s*([A-Z0-9]+-\d+)',  # "ticket: A1DEV-16638"
                    r'issue\s*[:#]?\s*([A-Z0-9]+-\d+)',   # "issue: A1DEV-16638"
                    r'jira\s*[:#]?\s*([A-Z0-9]+-\d+)',    # "jira: A1DEV-16638"
                ]
                
                for i, reply in enumerate(replies):
                    text = reply.get('text', '')
                    text_lower = text.lower()
                    user = reply.get('user', 'Unknown')
                    
                    all_replies.append(f"Reply {i+1}: {text[:100]}...")
                    
                    # Enhanced Jira ticket detection in thread replies
                    for pattern in jira_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            # Extract ticket key (handle tuple results from groups)
                            ticket_key = match if isinstance(match, str) else match[0] if match else None
                            if ticket_key and ticket_key not in jira_tickets:
                                jira_tickets.append(ticket_key)
                                logger.info(f"🎫 Found Jira ticket reference: {ticket_key} in thread reply by {user}")
                    
                    # Check for solution keywords
                    matching_keywords = [kw for kw in solution_keywords if kw in text_lower]
                    if matching_keywords:
                        logger.info(f"Found solution keywords {matching_keywords} in reply: {text[:50]}...")
                        solutions.append(text[:300])  # Get more text for solutions
                
                logger.info(f"Found {len(solutions)} potential solutions and {len(jira_tickets)} Jira ticket references")
                
                # Build enhanced summary with Jira ticket information
                summary_parts = []
                
                if jira_tickets:
                    summary_parts.append(f"🎫 Jira tickets mentioned: {', '.join(jira_tickets[:3])}")
                
                if solutions:
                    # Return the most relevant solutions
                    solution_text = ' | '.join(solutions[:2])
                    summary_parts.append(f"Solutions found: {solution_text}")
                else:
                    # Return summary with some actual content
                    reply_count = len(replies)
                    sample_replies = ' | '.join(all_replies[:2])
                    summary_parts.append(f"{reply_count} replies discussing: {sample_replies}")
                
                return ' | '.join(summary_parts) if summary_parts else ""
            else:
                logger.info("No thread replies found or API call failed")
            
            return ""
            
        except SlackApiError as e:
            error_msg = str(e)
            logger.error(f"Error getting thread replies for {thread_ts}: {e}")
            
            # Provide more specific error message for channel_not_found
            if "channel_not_found" in error_msg:
                logger.warning(f"Channel {channel_id} not found or bot doesn't have access.")
                return "⚠️ Unable to access this thread. The channel may be private, archived, or the bot lacks access permissions."
            
            return f"Error accessing thread (may need permissions): {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error getting thread replies: {e}")
            return f"Error processing thread: {str(e)}"
    
    def _get_thread_messages(self, channel_id: str, thread_ts: str) -> List[Dict]:
        """Get full thread messages for detailed analysis with enhanced Jira ticket detection"""
        try:
            if not thread_ts:
                return []
            
            response = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=50  # Get more messages for comprehensive analysis
            )
            
            if response['ok']:
                messages = response['messages']
                
                # Enhanced processing: Add Jira ticket detection to each message
                import re
                jira_patterns = [
                    r'\b[A-Z0-9]{2,10}-\d+\b',  # Standard format: A1DEV-16638, PROJ-123
                    r'acquia\.atlassian\.net/browse/([A-Z0-9]+-\d+)',  # Acquia Jira URLs
                    r'atlassian\.net/browse/([A-Z0-9]+-\d+)',  # Generic Atlassian URLs
                    r'/browse/([A-Z0-9]+-\d+)',  # Any browse URL
                    r'ticket\s*[:#]?\s*([A-Z0-9]+-\d+)',  # "ticket: A1DEV-16638"
                    r'issue\s*[:#]?\s*([A-Z0-9]+-\d+)',   # "issue: A1DEV-16638"
                    r'jira\s*[:#]?\s*([A-Z0-9]+-\d+)',    # "jira: A1DEV-16638"
                ]
                
                # Add Jira ticket detection to each message
                for message in messages:
                    text = message.get('text', '')
                    jira_tickets = []
                    
                    for pattern in jira_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        for match in matches:
                            ticket_key = match if isinstance(match, str) else match[0] if match else None
                            if ticket_key and ticket_key not in jira_tickets:
                                jira_tickets.append(ticket_key)
                    
                    # Add jira_tickets to message metadata
                    message['jira_tickets'] = jira_tickets
                    if jira_tickets:
                        logger.info(f"🎫 Found Jira tickets {jira_tickets} in message by {message.get('user', 'Unknown')}")
                
                return messages
            else:
                logger.warning(f"Failed to get thread messages: {response.get('error', 'Unknown error')}")
                return []
                
        except SlackApiError as e:
            error_msg = str(e)
            logger.error(f"Error getting thread messages for {thread_ts}: {e}")
            
            # Add specific handling for channel_not_found error
            if "channel_not_found" in error_msg:
                logger.warning(f"Channel {channel_id} not found or bot doesn't have access.")
                # Return a placeholder message to indicate the access issue
                return [{
                    'text': f"⚠️ Unable to access this thread. The channel may be private, archived, or the bot lacks access permissions.",
                    'user': "SYSTEM",
                    'ts': thread_ts,
                    'error': True,
                    'jira_tickets': []
                }]
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting thread messages: {e}")
            return []
