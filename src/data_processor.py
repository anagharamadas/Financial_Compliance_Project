"""

Financial Document Preprocessor
==================================
Cleans and structures SEC filings for model training

What this does:
1. Removes HTML and formatting noise
2. Extracts relevant sections
3. Creates training examples
4. Labels data with compliance categories
"""

import re
import pandas as pd
import numpy as np
import json
from pathlib import Path
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Download required NLTK data (first time only)
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class FinancialDocumentProcessor:

    """

    Processes financial documents for training

    It takes messy documents and makes them neat and organized for the AI model to learn from
    """

    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize processor

        Args:
            model_name (str): Name of model (determines tokenizer)
        """

        # Load tokenizer
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load compliance rules
        with open('configs/compliance_rules.json', 'r') as f:
            self.compliance_rules = json.load(f)

        print(f"Processor initialized")
        print(f"Compliance rules loaded: {len(self.compliance_rules)}")


    def clean_text(self, text):

        """
        Clean raw document text

        Args:
            text (str): Raw text from SEC filing

        Returns:
            str: Cleaned text
        """


        if not text or not isinstance(text, str):
            return ""

        # Step 1: Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Step 2: remove HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)

        # Step 3: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Step 4: Remove special characters (keep basic punctuation such as letter, numbers, basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()\-\'\"$%]', '', text)

        # Step 5: Normalize dollar amounts
        # Example -> $1,234,567 -> $1234567
        text = re.sub(r'\$\s*(\d+),?(\d+)*', r'$\1\2', text)

        # Step 6: Remove extra periods in acronyms
        # Example: U.S.A -> usa
        text = re.sub(r'\.(?=[A-Z])', '', text)

        # Final cleanup
        text = text.strip()

        return text


    def extract_sentences(self, text, max_sentences=50):
        """

        Split text into sentences

        Args:
            text (str): Cleaned text
            max_sentences (int): Maximum sentences to extract

        Returns:
            list: List of sentences

        Note: Instead of processing huge paragraphs, this function breaks the text into sentences which makes it easier for the model

        """

        sentences = sent_tokenize(text)

        # Filtering out very short sentences (likely noise)
        sentences = [s for s in sentences if len(s.split()) > 5]

        # Limit the number of sentences
        sentences =  sentences[:max_sentences]

        return sentences

    def detect_compliance_issues(self, text):
        """

        Detect potential compliance issues in text
        
        Args:
            text (str): Document text

        Returns:
            dict: Detected issues with categories

        Note: This function is like a spell checker, but for compliance violations. It looks for red flag keywords.
        """

        text_lower = text.lower()
        detected_issues = {}

        for rule_id, rule_info in self.compliance_rules.items():
            # Count how many keywords from this rule appear
            keyword_matches = []

            for keyword in rule_info['keywords']:
                if keyword.lower() in text_lower:
                    keyword_matches.append(keyword)

            # If we found matches, record them
            if keyword_matches:
                detected_issues[rule_id] = {
                    'category': rule_info['category'],
                    'severity': rule_info['severity'],
                    'matched_keywords': keyword_matches,
                    'count': len(keyword_matches)
                }

        return detected_issues

    def create_training_example(self, text, metadata):
        """

        Create a training example in instruction format

        Args:
            text (str): Document excerpt
            metadata (dict): Document metadata (company, date, etc.)

        Returns:
            dict: Training example with instruction and response

        Note: This function formats data in a way the model can learn from

        Format:
            Instruction: "Analyze this document for compliance..."
            Response: "This document shows HIGH RISK because..."

        """

        # Detect issues in text
        issues = self.detect_compliance_issues(text)

        # Create instruction
        instruction = f"""As a financial expert, analyze this SEC filing excerpt and identify any compliance concerns.

        Company: {metadata.get('company_name', 'Unknown')}
        Filing Type: {metadata.get('form_type', 'Unknown')}
        Filed Date: {metadata.get('filed_date', 'Unknown')}

        Document Excerpt:
        {text}

        Provide a detailed compliance analysis including:
        1. Risk level assessment (HIGH, MEDIUM, or LOW)
        2. Specific compliance concerns identified
        3. Relevant SEC regulations
        4. Recommendations for improvement"""

        # Generate response based on detected issues
        response = self._generate_response(issues, text)

        return {
            'instruction': instruction,
            'response': response,
            'metadata': metadata,
            'detected_issues': issues
        }

    def _generate_response(self, issues, text):
        """

        Generate appropriate response based on detected issues

        Note: This creates the answer part of our training examples.

        """

        # Determine the risk level
        if not issues:
            risk_level = "LOW"
        elif any(issue['severity'] == 'HIGH' for issue in issues.values()):
            risk_level = 'HIGH'
        else:
            risk_level = 'MEDIUM'

        "Start response"
        response_parts = [
            f"## Compliance Analysis\n",
            f" **Risk level: {risk_level}**\n"
        ]

        if not issues:
            response_parts.append("""
            ### Assessment
            This document excerpt shows no significant comopliance red flags. The language appears transparent and follows standard disclosure practises.

            ### Findings
            No material disclosure deficiencies detected
            Risk language appears adequate
            No obvious compliance violations

            ### Recommendations
            - Continue monitoring for completeness
            - Ensure all material events are disclosed timely
            - Maintain current disclosure standards
            """
            )

        else:
            response_parts.append("\n### Compliance concerns identified\n")

            for rule_id, issue in issues.items():
                response_parts.append(f"""
                **{issue['category']}** (Severity: {issue['severity']})
                - Found {issue['count']} relevant indicators
                - Keywords detected: {','.join(issue['matched_keywords'][:3])}
                - Requires detailed review and potential remediation
                """
                )

            response_parts.append(
            """
            ### Recommendations
            **Immediate Actions:**
            1. Conduct thorough legal review of flagged sections
            2. Verify all material information is disclosed
            3. Ensure compliance with relevant SEC regulations
            4. Consider consulting with compliance counsel

            **Documentation:**
            - Document all compliance review steps
            - Maintain audit trail of changes
            - Update disclosure controls as needed
            """
            )

        return ''.join(response_parts)

    def process_dataset(self, input_csv, output_dir = 'data/processed'):
        """
        Process entire dataset

        Args:
            input_csv (str): Path to raw data CSV
            output_dir (str): Where to save processed data

        Returns:
            dict: Processed datasets (train, val, test)
        
        Note: This is the main function that processes all collected documents and creates training data.

        """

        print("\n" + "="*60)
        print("PROCESSING FINANCIAL DOCUMENTS")
        print("="*60)

        # Load raw data
        print(f"\n Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        print(f"Loaded {len(df)} documents")

        # Process each document
        training_examples = []

        print(f"\n Processing documents...")
        for idx, row in df.iterrows():
            print(f" Processing  {idx+1}/{len(df)}: {row['company_name']}...", end ='\r' )

            # Get risk factors section (most relevant for compliance)
            text = row.get('risk_factors', row.get('full_content', ''))
            if pd.isna(text) or len(str(text)) < 100:
                continue

            # Clean text
            cleaned_text = self.clean_text(str(text))

            # Split into manageable chunks
            sentences = self.extract_sentences(cleaned_text, max_sentences=30)

            if not sentences:
                continue

            # Combine into chunk (max 1500 words)
            chunk_text = ' '.join(sentences)
            words = chunk_text.split()
            if len(words) > 1500:
                chunk_text = ' '.join(words[:1500])

            # Create metadata
            metadata = {
                'company_name': row['company_name'],
                'ticker': row['ticker'],
                'form_type': row['form_type'],
                'filed_date': row['filed_date']
            }

            # Create training example
            example = self.create_training_example(chunk_text, metadata)
            training_examples.append(example)

        print(f"\n Created {len(training_examples)} training_examples")

        # Split into train/validation/test
        print(f"\n Splitting dataset...")
        np.random.shuffle(training_examples)

        n = len(training_examples)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)

        splits = {
            'train': training_examples[:train_size],
            'validation': training_examples[train_size:train_size+val_size],
            'test': training_examples[train_size+val_size:]
        }

        print(f" Training: {len(splits['train'])} examples")
        print(f" Validation: {len(splits['validation'])} examples")
        print(f" Test: {len(splits['test'])} examples")

        # Save processed data
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for split_name, examples in splits.items():
            output_file = f"{output_dir}/{split_name}.json"

            with open(output_file, 'w') as f:
                json.dump(examples, f, indent=2)

            print(f"Saved {split_name}: {output_file}")

        # Create summary
        self._create_summary(splits, output_dir)

        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)

        return splits

    def _create_summary(self, splits, output_dir):
        """

        Create Summary Statistics

        """

        summary = {
            'total_examples': sum(len(split) for split in splits.values()),
            'splits': {name: len(split) for name, split in splits.items()},
            'compliance_rules': list(self.compliance_rules.keys()),
            'average_instruction_length': np.mean([
                len(ex['instruction'].split())
                for split in splits.values()
                for ex in split
            ]),
            'average_response_length': np.mean([
                len(ex['response'].split())
                for split in splits.values()
                for ex in split
            ])
        }

        # Save summary
        with open(f"{output_dir}/summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nDataset Summary:")
        print(f" Total examples: {summary['total_examples']}")
        print(f" Avg instruction length: {summary['average_instruction_length']:.0f} words")
        print(f"   Avg response length: {summary['average_response_length']:.0f} words")

def main():
    """Run preprocessing"""

    print("Starting data preprocessing")

    # Initialise processor
    processor = FinancialDocumentProcessor()

    # Process dataset
    splits = processor.process_dataset(
        input_csv = 'data/raw/sec_filings.csv',
        output_dir = 'data/processed'
    )

    print("\n Preprocessing complete!")
    print("\nNext Steps:")
    print("1. Review data/processed/ folder")
    print("2. Check train.json, validation.json, test.json")
    print("3. Ready for model training!")

if __name__ == "__main__":
    main()

