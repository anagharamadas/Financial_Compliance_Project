"""

Model Setup and LoRA Configuration
====================================
Sets up FinBERT with LoRA adapters for fine-tuning

Note: This file is like preparing a student (the model) for specialized training. We're adding "adapters" that let it learn efficiently.

"""

import torch
from transformers import(
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

import json

class FinancialComplianceModel:
    """

    Sets up model for compliance checking

    Note: This class handles all the technical setup.
    
    """

    def __init__(self, model_name="ProsusAI/finbert"):
        """
        Initialize model setup

        Args: 
            model_name (str): Which pre-trained model to use

        Note: We use FinBERT because it already knows financial language.
        """

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n{'='*60}")
        print("MODEL SETUP")
        print("="*60)
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")

        if self.device == "cpu":
            print("⚠️  Warning: No GPU detected. Training will be slow.")
            print("   Consider using Google Colab (free GPU) or AWS.")
        else:
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")

    def create_quantization_config(self):
        """
        Create 4-bit quantization config for memory efficiency

        Note:
        Quantization = Using less precise numbers

        Normal: 32-bit numbers (high precision, lots of memory)
        Quantized: 4-bit

        """
