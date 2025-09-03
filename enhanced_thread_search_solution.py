#!/usr/bin/env python3

import os
import sys
import re
import logging
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.cdp_chat_tool.slack_tool import SlackTool
from tools.cdp_chat_tool.jira_tool import JiraTool

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedThreadSearchSolution:
    """Enhanced solution to find JIRA tickets when thread access is limited"""
    
    def __init__(self):
        self.slack_tool = SlackTool()
        self.jira_tool = JiraTool()
    
    def search_with_thread_workaround(self, query: str, channels: list = None) -> dict:
        """
        Enhanced search that works around thread access limitations
        """
        results = {
            'slack_messages': [],
            'related_jira_tickets': [],
            'search_strategies': []
        }
        
        # Strategy 1: Find the original Slack message
        if channels:
            slack_results = self.slack_tool.search_in_channels(query, channels, limit=10)
        else:
            slack_results = self.slack_tool.search_messages(query, limit=10)
        
        results['slack_messages'] = slack_results
        print(f"Found {len(slack_results)} Slack messages")
        
        for msg in slack_results:
            print(f"\nAnalyzing message from #{msg.get('channel')} at {msg.get('ts')}")
            
            # Strategy 2: Extract error details and search JIRA for similar errors
            error_details = self.extract_error_details(msg.get('text', ''))
            if error_details:
                print(f"Extracted error details: {error_details}")
                results['search_strategies'].append(f"Error-based search: {error_details}")
                
                # Search JIRA for tickets containing these error details
                for error_term in error_details:
                    jira_tickets = self.jira_tool.get_similar_issues(error_term, max_results=3)
                    for ticket in jira_tickets:
                        ticket['search_strategy'] = f'error_term: {error_term}'
                        ticket['relevance_score'] = self.calculate_relevance(ticket, error_details)
                        results['related_jira_tickets'].append(ticket)
            
            # Strategy 3: Search for tickets related to the job/workflow
            job_details = self.extract_job_details(msg.get('text', ''))
            if job_details:
                print(f"Extracted job details: {job_details}")
                results['search_strategies'].append(f"Job-based search: {job_details}")
                
                for job_term in job_details:
                    jira_tickets = self.jira_tool.get_similar_issues(job_term, max_results=3)
                    for ticket in jira_tickets:
                        ticket['search_strategy'] = f'job_term: {job_term}'
                        ticket['relevance_score'] = self.calculate_relevance(ticket, job_details)
                        results['related_jira_tickets'].append(ticket)
            
            # Strategy 4: Search for tickets in the same time frame
            timestamp = msg.get('ts', '')
            if timestamp:
                time_based_tickets = self.search_jira_by_timeframe(timestamp)
                for ticket in time_based_tickets:
                    ticket['search_strategy'] = 'time_based'
                    ticket['relevance_score'] = self.calculate_time_relevance(ticket, timestamp)
                    results['related_jira_tickets'].append(ticket)
        
        # Strategy 5: Broader search for the specific error message
        direct_search = self.jira_tool.get_similar_issues(query, max_results=5)
        for ticket in direct_search:
            ticket['search_strategy'] = 'direct_query'
            ticket['relevance_score'] = ticket.get('similarity_score', 0) * 100
            results['related_jira_tickets'].append(ticket)
        
        # Deduplicate and sort by relevance
        unique_tickets = {}
        for ticket in results['related_jira_tickets']:
            key = ticket.get('key')
            if key:
                if key not in unique_tickets or ticket['relevance_score'] > unique_tickets[key]['relevance_score']:
                    unique_tickets[key] = ticket
        
        sorted_tickets = sorted(unique_tickets.values(), key=lambda x: x.get('relevance_score', 0), reverse=True)
        results['related_jira_tickets'] = sorted_tickets
        
        return results
    
    def extract_error_details(self, text: str) -> list:
        """Extract specific error details from Slack message"""
        error_terms = []
        
        # Extract specific error messages
        error_patterns = [
            r'error_message\s*([^]]+)',  # error_message content
            r'SnowflakeExecutorException:\s*([^|]+)',  # Snowflake exceptions
            r'Timestamp\s+\'([^\']+)\'\s+is\s+not\s+recognized',  # Timestamp errors
            r'error-code\s+(\d+)',  # Error codes
            r'queryId\s+([a-f0-9-]+)',  # Query IDs
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            error_terms.extend(matches)
        
        # Also extract the main error components
        if 'SF_DW_MAPPER_DEFAULT' in text:
            error_terms.append('SF_DW_MAPPER_DEFAULT')
        
        if 'JCrew' in text:
            error_terms.append('JCrew')
        
        if '1086' in text:
            error_terms.append('1086')
        
        return [term.strip() for term in error_terms if term.strip()]
    
    def extract_job_details(self, text: str) -> list:
        """Extract job/workflow details from Slack message"""
        job_terms = []
        
        # Extract job-related information
        job_patterns = [
            r'Job ID\s*:\s*([^\n]+)',  # Job IDs
            r'SF_DW_MAPPER_DEFAULT',   # Job name
            r'JCrew\s*\((\d+)\)',      # Tenant info
            r'Status Detail:\s*([^\n]+)',  # Status details
        ]
        
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            job_terms.extend(matches)
        
        return [term.strip() for term in job_terms if term.strip()]
    
    def search_jira_by_timeframe(self, slack_timestamp: str) -> list:
        """Search for JIRA tickets created around the same time as the Slack message"""
        try:
            # Convert Slack timestamp to approximate date
            import datetime
            timestamp_float = float(slack_timestamp)
            message_date = datetime.datetime.fromtimestamp(timestamp_float)
            
            # Search for tickets created within 24 hours of the message
            date_str = message_date.strftime('%Y-%m-%d')
            next_date = (message_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Use proper JQL syntax for date range
            jql_query = f'created >= "{date_str}" AND created < "{next_date}"'
            
            return self.jira_tool.search_issues(jql_query, max_results=10)
        except Exception as e:
            print(f"Error in time-based search: {e}")
            return []
    
    def calculate_relevance(self, ticket: dict, search_terms: list) -> float:
        """Calculate relevance score based on how many search terms appear in the ticket"""
        score = 0
        ticket_text = (ticket.get('summary', '') + ' ' + ticket.get('description', '')).lower()
        
        for term in search_terms:
            if term.lower() in ticket_text:
                score += 10
        
        # Bonus for closed tickets (likely resolved)
        if ticket.get('status', '').lower() in ['closed', 'resolved', 'done']:
            score += 5
        
        return score
    
    def calculate_time_relevance(self, ticket: dict, slack_timestamp: str) -> float:
        """Calculate relevance based on time proximity"""
        try:
            import datetime
            slack_time = datetime.datetime.fromtimestamp(float(slack_timestamp))
            
            # Parse ticket creation time
            created_str = ticket.get('created', '')
            if created_str:
                # Simplified time comparison
                return 3.0  # Base score for time-based matches
        except:
            pass
        
        return 1.0

def test_enhanced_thread_search():
    """Test the enhanced thread search solution"""
    
    query = "Timestamp '-1-11-28 17:00:00.000' is not recognized"
    channels = ['a1_engineering', 'a1_ops-ask-resource-prod']
    
    print(f"\n{'='*80}")
    print(f"ENHANCED THREAD SEARCH FOR: {query}")
    print(f"{'='*80}")
    
    solution = EnhancedThreadSearchSolution()
    results = solution.search_with_thread_workaround(query, channels)
    
    print(f"\n{'-'*60}")
    print("SEARCH STRATEGIES USED:")
    print(f"{'-'*60}")
    for strategy in results['search_strategies']:
        print(f"  • {strategy}")
    
    print(f"\n{'-'*60}")
    print("RELATED JIRA TICKETS (PRIORITIZED):")
    print(f"{'-'*60}")
    
    for i, ticket in enumerate(results['related_jira_tickets'][:5], 1):
        print(f"\n{i}. {ticket['key']}: {ticket.get('summary', 'No summary')}")
        print(f"   Status: {ticket.get('status', 'Unknown')}")
        print(f"   Relevance Score: {ticket.get('relevance_score', 0):.1f}")
        print(f"   Search Strategy: {ticket.get('search_strategy', 'unknown')}")
        print(f"   URL: {ticket.get('url', 'N/A')}")
        
        # Show description snippet
        description = ticket.get('description', '')
        if description:
            print(f"   Description: {description[:150]}...")
        
        # Show if it contains solution indicators
        if ticket.get('status', '').lower() in ['closed', 'resolved']:
            print(f"   ✅ This ticket is RESOLVED - likely contains solution!")

if __name__ == "__main__":
    test_enhanced_thread_search()
