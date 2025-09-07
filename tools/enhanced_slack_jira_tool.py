#!/usr/bin/env python3

import os
import re
import logging
from typing import List, Dict, Optional, Set
from tools.cdp_chat_tool.slack_tool import SlackTool
from tools.cdp_chat_tool.jira_tool import JiraTool

logger = logging.getLogger(__name__)

class EnhancedSlackJiraTool:
    """Enhanced tool that finds JIRA tickets referenced in Slack messages"""
    
    def __init__(self):
        self.slack_tool = SlackTool()
        self.jira_tool = JiraTool()
        
        # JIRA ticket patterns - comprehensive list
        self.jira_patterns = [
            r'\b[A-Z]{2,10}-\d+\b',  # Standard JIRA format (e.g., PROJ-123, A1DEV-16623)
            r'\b[A-Z]+\s*#\s*\d+\b',  # Alternative format (e.g., PROJ #123)
            r'(?:ticket|issue|bug|story|task)[\s:]*([A-Z]{2,10}-\d+)',  # Contextual references
            r'(?:jira|atlassian)\.net/browse/([A-Z]{2,10}-\d+)',  # URL references
        ]
    
    def search_with_jira_extraction(self, query: str, channels: List[str] = None, limit: int = 10) -> Dict:
        """
        Enhanced search that finds Slack messages and extracts JIRA references from them
        """
        results = {
            'slack_messages': [],
            'jira_tickets_from_slack': [],
            'direct_jira_search': [],
            'total_jira_tickets': []
        }
        
        # Step 1: Search Slack messages
        if channels:
            slack_results = self.slack_tool.search_in_channels(query, channels, limit)
        else:
            slack_results = self.slack_tool.search_messages(query, limit)
        
        results['slack_messages'] = slack_results
        logger.info(f"Found {len(slack_results)} Slack messages")
        
        # Step 2: Extract JIRA references from Slack messages
        jira_tickets_found = set()
        
        for msg in slack_results:
            # Extract JIRA tickets from message text
            msg_text = msg.get('text', '')
            tickets = self.extract_jira_references(msg_text)
            
            if tickets:
                logger.info(f"Found JIRA tickets {tickets} in Slack message from #{msg.get('channel')}")
                jira_tickets_found.update(tickets)
                
                # Add ticket info to the message
                msg['jira_tickets'] = tickets
        
        # Step 3: Get details for all found JIRA tickets
        jira_details = []
        for ticket in jira_tickets_found:
            details = self.jira_tool.get_issue_details(ticket)
            if details:
                details['source'] = 'slack_reference'
                jira_details.append(details)
                logger.info(f"Retrieved details for JIRA ticket {ticket}")
            else:
                logger.warning(f"Could not retrieve details for JIRA ticket {ticket}")
        
        results['jira_tickets_from_slack'] = jira_details
        
        # Step 4: Also do direct JIRA search
        direct_jira = self.jira_tool.get_similar_issues(query, max_results=5)
        for issue in direct_jira:
            issue['source'] = 'direct_search'
        results['direct_jira_search'] = direct_jira
        
        # Step 5: Combine and deduplicate JIRA results
        all_jira = {}
        
        # Add tickets from Slack references (higher priority)
        for ticket in jira_details:
            key = ticket.get('key')
            if key:
                ticket['priority_score'] = 100  # High priority for Slack references
                all_jira[key] = ticket
        
        # Add direct search results (lower priority, don't overwrite)
        for ticket in direct_jira:
            key = ticket.get('key')
            if key and key not in all_jira:
                ticket['priority_score'] = ticket.get('similarity_score', 0) * 50
                all_jira[key] = ticket
        
        # Sort by priority score
        sorted_jira = sorted(all_jira.values(), key=lambda x: x.get('priority_score', 0), reverse=True)
        results['total_jira_tickets'] = sorted_jira
        
        logger.info(f"Total unique JIRA tickets found: {len(sorted_jira)}")
        
        return results
    
    def extract_jira_references(self, text: str) -> List[str]:
        """Extract JIRA ticket references from text using multiple patterns"""
        found_tickets = set()
        
        for pattern in self.jira_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Handle different capture groups
                if isinstance(match, tuple):
                    # Pattern with capture group
                    ticket = match[0] if match[0] else match[1] if len(match) > 1 else ''
                else:
                    # Direct match
                    ticket = match
                
                if ticket:
                    # Clean up the ticket reference
                    clean_ticket = re.sub(r'\s*#\s*', '-', ticket.upper().strip())
                    # Validate format (PROJECT-NUMBER)
                    if re.match(r'^[A-Z]{2,10}-\d+$', clean_ticket):
                        found_tickets.add(clean_ticket)
        
        return list(found_tickets)
    
    def analyze_slack_thread_for_solutions(self, message: Dict) -> Dict:
        """
        Analyze a Slack message and its thread for solutions, including JIRA tickets
        """
        analysis = {
            'has_thread': False,
            'thread_summary': '',
            'jira_tickets': [],
            'solutions_found': [],
            'solution_score': 0
        }
        
        # Check if message has thread information
        thread_summary = message.get('thread_summary', '')
        reply_count = message.get('reply_count', 0)
        
        if thread_summary or reply_count > 0:
            analysis['has_thread'] = True
            analysis['thread_summary'] = thread_summary
            
            # Extract JIRA tickets from thread summary
            if thread_summary:
                jira_tickets = self.extract_jira_references(thread_summary)
                analysis['jira_tickets'] = jira_tickets
                
                # Look for solution indicators
                solution_keywords = [
                    'solved', 'fixed', 'resolved', 'solution', 'answer', 'try this', 
                    'here\'s how', 'workaround', 'fix', 'working', 'works', 'success',
                    'that worked', 'thanks', 'perfect', 'great', 'run this', 'use this'
                ]
                
                thread_lower = thread_summary.lower()
                found_solutions = [kw for kw in solution_keywords if kw in thread_lower]
                
                if found_solutions:
                    analysis['solutions_found'] = found_solutions
                    analysis['solution_score'] = len(found_solutions) * 10
                    
                    # Higher score if JIRA tickets are also present
                    if jira_tickets:
                        analysis['solution_score'] += 50
        
        # Also check main message text for JIRA references
        main_text = message.get('text', '')
        main_jira = self.extract_jira_references(main_text)
        if main_jira:
            analysis['jira_tickets'].extend(main_jira)
            analysis['jira_tickets'] = list(set(analysis['jira_tickets']))  # Remove duplicates
        
        return analysis
    
    def get_comprehensive_search_results(self, query: str, channels: List[str] = None) -> Dict:
        """
        Get comprehensive search results with enhanced JIRA detection and solution analysis
        """
        # Get enhanced search results
        search_results = self.search_with_jira_extraction(query, channels)
        
        # Analyze each Slack message for solutions and JIRA tickets
        enhanced_slack_messages = []
        for msg in search_results['slack_messages']:
            analysis = self.analyze_slack_thread_for_solutions(msg)
            msg['analysis'] = analysis
            enhanced_slack_messages.append(msg)
        
        # Sort messages by solution score (messages with solutions first)
        enhanced_slack_messages.sort(key=lambda x: x['analysis']['solution_score'], reverse=True)
        
        search_results['slack_messages'] = enhanced_slack_messages
        
        # Get detailed information for all JIRA tickets found
        all_jira_with_details = []
        for jira_ticket in search_results['total_jira_tickets']:
            # Add solution context if this ticket was found in a Slack thread with solutions
            jira_ticket['slack_context'] = []
            
            for msg in enhanced_slack_messages:
                if jira_ticket['key'] in msg['analysis']['jira_tickets']:
                    context = {
                        'channel': msg.get('channel'),
                        'permalink': msg.get('permalink'),
                        'has_solutions': len(msg['analysis']['solutions_found']) > 0,
                        'solution_keywords': msg['analysis']['solutions_found'],
                        'thread_summary': msg['analysis']['thread_summary']
                    }
                    jira_ticket['slack_context'].append(context)
            
            all_jira_with_details.append(jira_ticket)
        
        search_results['total_jira_tickets'] = all_jira_with_details
        
        return search_results
    
    def format_results_for_display(self, results: Dict) -> str:
        """Format the comprehensive results for display"""
        output = []
        
        # Summary
        slack_count = len(results['slack_messages'])
        jira_count = len(results['total_jira_tickets'])
        jira_from_slack = len(results['jira_tickets_from_slack'])
        
        output.append(f"🔍 **Search Results Summary**")
        output.append(f"• Found {slack_count} Slack messages")
        output.append(f"• Found {jira_count} total JIRA tickets ({jira_from_slack} from Slack references)")
        output.append("")
        
        # JIRA Tickets (prioritized)
        if results['total_jira_tickets']:
            output.append("🎫 **JIRA Tickets Found**")
            for i, ticket in enumerate(results['total_jira_tickets'][:5], 1):
                output.append(f"{i}. **{ticket['key']}**: {ticket['summary']}")
                output.append(f"   Status: {ticket['status']} | Source: {ticket['source']}")
                output.append(f"   🔗 {ticket['url']}")
                
                # Show Slack context if available
                if ticket.get('slack_context'):
                    for ctx in ticket['slack_context']:
                        if ctx['has_solutions']:
                            output.append(f"   💡 **Solution found in Slack**: #{ctx['channel']}")
                            output.append(f"      Keywords: {', '.join(ctx['solution_keywords'])}")
                            if ctx['permalink']:
                                output.append(f"      🔗 {ctx['permalink']}")
                output.append("")
        
        # Slack Messages with Solutions
        solution_messages = [msg for msg in results['slack_messages'] if msg['analysis']['solution_score'] > 0]
        if solution_messages:
            output.append("💬 **Slack Messages with Solutions**")
            for i, msg in enumerate(solution_messages[:3], 1):
                output.append(f"{i}. **#{msg['channel']}** (Score: {msg['analysis']['solution_score']})")
                output.append(f"   {msg['text'][:150]}...")
                if msg['analysis']['solutions_found']:
                    output.append(f"   💡 Solution keywords: {', '.join(msg['analysis']['solutions_found'])}")
                if msg['analysis']['jira_tickets']:
                    output.append(f"   🎫 JIRA tickets: {', '.join(msg['analysis']['jira_tickets'])}")
                if msg.get('permalink'):
                    output.append(f"   🔗 {msg['permalink']}")
                output.append("")
        
        return "\n".join(output)

if __name__ == "__main__":
    # Test the enhanced tool
    tool = EnhancedSlackJiraTool()
    
    query = "Timestamp '-1-11-28 17:00:00.000' is not recognized"
    channels = ['a1_engineering', 'a1_ops-ask-resource-prod']
    
    print("Testing Enhanced Slack-JIRA Tool")
    print("=" * 50)
    
    results = tool.get_comprehensive_search_results(query, channels)
    formatted_output = tool.format_results_for_display(results)
    
    print(formatted_output)
