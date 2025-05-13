#!/usr/bin/env python3

"""
Example script showing how to use the script expansion functionality
in the SF-Hive discrepancy agent pipeline.
"""

import os
import logging
import json
from dotenv import load_dotenv
from agents.extract_agent import lookup as extract_discrepancy_ids
from agents.discrepancy_agent_with_expansion import lookup_with_expanded_scripts
from agents.suggester_agent_with_expansion import generate_discrepancy_suggestions_with_scripts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('script_expansion.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Main function demonstrating the script expansion pipeline."""
    # 1. Set up configuration
    table_name = os.getenv("TARGET_TABLE", "CUSTOMER")
    metadata_dir = os.getenv("METADATA_DIR", "script-testing/src/test/resources/evolutions/t0/prod-eu")
    
    # Ensure the metadata directory exists
    if not os.path.isdir(metadata_dir):
        logger.warning(f"Metadata directory not found: {metadata_dir}")
        logger.warning("Script expansion may not work correctly without metadata.")
    
    # 2. Extract discrepancy IDs for the table
    logger.info(f"Extracting discrepancy IDs for table: {table_name}")
    try:
        discrepancy_ids = extract_discrepancy_ids(table_name)
        if not discrepancy_ids:
            logger.warning(f"No discrepancy IDs found for table: {table_name}")
            return
            
        # Parse the IDs if they're returned as a string
        if isinstance(discrepancy_ids, str):
            # Try parsing as JSON array
            try:
                discrepancy_ids = json.loads(discrepancy_ids)
            except json.JSONDecodeError:
                # If not JSON, try splitting by commas or whitespace
                discrepancy_ids = [id.strip() for id in discrepancy_ids.replace(",", " ").split()]
        
        logger.info(f"Found {len(discrepancy_ids)} discrepancy IDs")
        
        # Limit to a few IDs for testing
        discrepancy_ids = discrepancy_ids[:5]
        logger.info(f"Using sample IDs: {discrepancy_ids}")
        
    except Exception as e:
        logger.error(f"Error extracting discrepancy IDs: {e}", exc_info=True)
        return
    
    # 3. Get discrepancies with expanded scripts
    logger.info("Getting discrepancies with expanded scripts")
    try:
        result = lookup_with_expanded_scripts(table_name, discrepancy_ids)
        discrepancies = result['discrepancies']
        expanded_scripts = result['expanded_scripts']
        
        logger.info(f"Found {len(expanded_scripts)} expanded scripts")
        
    except Exception as e:
        logger.error(f"Error getting discrepancies with expanded scripts: {e}", exc_info=True)
        return
    
    # 4. Generate suggestions using expanded scripts
    logger.info("Generating discrepancy suggestions with expanded scripts")
    try:
        suggestions = generate_discrepancy_suggestions_with_scripts(
            table_name, 
            discrepancies,
            expanded_scripts
        )
        
        # Save suggestions to file
        output_file = f"{table_name}_suggestions.json"
        with open(output_file, 'w') as f:
            f.write(suggestions)
            
        logger.info(f"Saved suggestions to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print(f"DISCREPANCY ANALYSIS FOR TABLE: {table_name}")
        print("="*50)
        print(f"Analyzed {len(discrepancy_ids)} IDs with discrepancies")
        print(f"Expanded {len(expanded_scripts)} SQL scripts")
        print(f"Generated suggestions saved to: {output_file}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main()