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


    
    def download_filing_content(self, filing_url):

        """
        Download the actual text content of a filing

        Args:
            filing_url (str) : URL to the filing

        Returns:
            str: Full text content of the filing

        """

        try:
            # Download filing content directly from SEC EDGAR
            # The filing_url should be a direct link to the SEC filing
            response = requests.get(
                filing_url,
                headers = {
                    "User-Agent": "Financial Compliance Project (your-email@example.com)",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                }
            )

            if response.status_code == 200:
                return response.text
            else:
                print(f"Download failed with status {response.status_code}: {response.text}")
                return None

        except Exception as e:
            print(f"Download failed: {e}")
            return None


    def extract_key_sections(self, content):
        """
        Extract important sections from filing

        Args:
            content (str): Full filing text

        Returns:
            dict: Dictionary with extracted sections
        """

        sections = {}

        if not content:
            return sections

        content_lower = content.lower()

        # Define section markers to look for
        section_markers = {
            'risk_factors' : [
                'item 1a', 'risk factors',
                'item 1a. risk factors'
            ],
            'business_overview': [
                'item 1.', 'item 1', 'business'
            ],
            'md_and_a': [
                'item 7', "management's discussion",
                "md&a"
            ],
            'controls': [
                'item 9a', 'controls and procedures',
                'internal control'
            ]
        }

        # Extract each section
        for section_name, markers in section_markers.items():
            for marker in markers:
                start_idx = content_lower.find(marker)

                if start_idx != -1:
                    # Found the section
                    # Extract next 3000 characters
                    section_text =  content[start_idx: start_idx + 3000]
                    sections[section_name] = section_text
                    break # Found it, move to next section

        return sections


    def save_dataset(self, filings, output_dir = "data/raw"):

        """
        Save collected data in organized format

        Args:
            filings (list): List of filing metadata
            output_dir (str): Where to save the data

        Note: The data will be saved in multiple formats:

        - CSV for easy viewing in Excel
        - JSON for complete data with all details
        """

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n Saving dataset to {output_dir}...")

        # Prepare data for saving
        dataset = []

        for i, filing in enumerate(filings):
            print(f"Processing {i+1}/{len(filings)}: {filing.get('companyName')}...")

            # Extract basic info
            filing_data = {
                'company_name': filing.get('companyName'),
                'ticker': filing.get('ticker'),
                'form_type': filing.get('formType'),
                'filed_date': filing.get('accessionNo'),
                'filing_url': filing.get('linkToFilingDetails')
            }

            # Download content
            if filing_data['filing_url']:
                print(f"Downloading content...")
                content = self.download_filing_content(filing_data['filing_url'])

                if content:
                    filing_data['full_content'] = content
                    filing_data['content_length'] = len(content)

                    # Extract key sections
                    sections = self.extract_key_sections(content)
                    filing_data.update(sections)

                    print(f"Downloaded {len(content)} characters")
                    print(f"Extracted {len(sections)} sections")

                else:
                    print(f"Content download failed")

                # Wait between downloads   
                time.sleep(self.request_delay)

            dataset.append(filing_data)

        # Save as CSV (easy to view)
        df = pd.DataFrame(dataset)
        csv_path = f"{output_dir}/sec_filings.csv"
        df.to_csv(csv_path, index = False)
        print(f"Saved CSV: {csv_path}")

        # Save as JSON (complete data)
        json_path = f"{output_dir}/sec_filings.json"
        with open(json_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved JSON: {json_path}")

        # Print summary
        print(f"\n Dataset Summary:")
        print(f"Total filings: {len(df)}")
        print(f"\nFilings by type:")
        print(df['form_type'].value_counts())
        print(f"\nCompanies included:")
        print(df['company_name'].value_counts())

        return df

def main():

    """
    Main function to run data collection

    """

    print("=" * 60)
    print("SEC FINANCIAL DOCUMENT DATA COLLECTOR")
    print("=" * 60)

    # Check API key
    if not API_KEY or API_KEY =='your_actual_api_key_here':
        print("ERROR: Please set your SEC_API_KEY in .env file")
        print("\nSteps to fix:")
        print("1. Get free API key from https://sec-api.io")
        print("2. Create .env file in project root")
        print("3. Add line: SEC_API_KEY=your_key_here")
        return

    # Initialize collector
    collector = SECDataCollector(API_KEY)

    # Define companies to collect (starting small at first)
    # Checking the code by starting with major, well-documented companies
    companies = [
        'AAPL', # Apple
        'MSFT', # Microsoft
        'GOOGL', # Google
        'AMZN', # Amazon
        'TSLA', # Tesla
    ]

    print(f"\n Will collect data for: {', '.join(companies)}")
    print(f"This will take about 5-10 minutes...")

    # Search for filings
    filings = collector.search_filings(
        companies = companies,
        form_types = ['10-K', '10-Q'],
        max_per_company = 3 # Start with 3 per company (For testing purposes)
    )

    if not filings:
        print("No filings found. Check your API key and internet connection.")
        return

    # Save dataset
    df = collector.save_dataset(filings)

    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"\nNext steps:")
    print("1. Open data/raw/sec_filings.csv in Excel to view")
    print("2. Check data/raw/sec_filings.json for full content")
    print("3. Move to data preprocessing step")


if __name__ == "__main__":
    main()


        

       