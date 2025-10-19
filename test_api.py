import os
from dotenv import load_dotenv
import requests

# Load API key from .env file
load_dotenv()
api_key = os.getenv('SEC_API_KEY')

print(f"API Key loaded: {api_key[:10]}...") # Show first 10 chars only

# Test API request
test_url = "https://api.sec-api.io"
headers = {"Authorization": api_key}

# Simple test query - search for Apple's recent filings
query = {
    "query": "formType:\"10-Q\"",
    "from": "0",
    "size": "50",
    "sort": [{ "filedAt": { "order": "desc" }}]
}


try:
    response = requests.post(test_url, json=query, headers=headers)
    
    if response.status_code == 200:
        print("✅ API working! Found filings:")
        data = response.json()
        for filing in data.get('filings', [])[:3]:
            print(f"  - {filing.get('companyName')}: {filing.get('formType')}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Message: {response.text}")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Check your internet connection and API key")