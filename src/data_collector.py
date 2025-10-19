"""
Financial Data Collector
==========================================

This script downloads SEC filings for training our model.

Author: Anagha
Date: 19-10-2025
"""


import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import requests
import pandas as pd
from pathlib import Path

# Load API key
load_dotenv()
API_KEY = os.getenv('SEC_API_KEY')

class SECDataCollector:

    """
    Collects SEC filings for model training

    What this class does:

    1. Searches for company filings (10-K, 10-Q)
    2. Downloads the full text
    3. Saves organized data for training
    """

    def __init__(self, api_key):

        """
        Initialize the collector

        Arg:
            api_key (str): Your SEC API key
        """

        self.api_key = api_key
        self.base_url = "https://api.sec-api.io"

        # Rate limiting
        self.request_delay = 0.2 # Wait for 0.2 seconds between requests

        print("SEC Data Collector initialized")


    def  search_filings(self, companies, form_types=["10-K", "10-Q"], max_per_company=5):

        """
        Search for company filings

        Args:

            companies (list): List of comapny tickers (e.g., ['AAPL', 'MSFT'])
            form_types (list): List of form types to search for (e.g., ['10-K', '10-Q'])
            max_per_company (int): Maximum number of filings per company

        Returns:
            list: Found filings with metadata
        """    

        all_filings = []

        print(f"\n Search for filings...")
        print(f"Companies: {companies}")
        print(f"Form Types: {form_types}")

        for company_ticker in companies:
            print(f"\n Searching {company_ticker} ...")

            for form_type in form_types:
                # Build search query
                # This is like typing in a search box
                
                query = {
                "query": f'ticker:{company_ticker} AND formType:"{form_type}"',
                "from": "0",
                "size": str(max_per_company),
                "sort": [{"filedAt": {"order": "desc"}}]
                }

                try:
                    # Make API request
                    response = requests.post(
                        f"{self.base_url}",
                        json=query,
                        headers = {"Authorization": self.api_key}
                    )

                    if response.status_code == 200:
                        data = response.json()
                        filings = data.get('filings', [])
                        
                        print(f"Found {len(filings)} {form_type} filings")
                        all_filings.extend(filings)

                    else:
                        print(f"Error: {response.status_code}: {response.text}")

                    # Wait before next request
                    time.sleep(self.request_delay)

                except Exception as e:
                    print(f"Failed: {e}")
                    continue

        print(f"\n Total filings found: {len(all_filings)}")
        return all_filings




        

       