# config.py
import os
import torch
from dotenv import load_dotenv

load_dotenv()

# --- File Paths ---
# CSV is now mainly for reference or potential input, not answer retrieval
CSV_DATA_PATH = "faq_data.csv" # Keep if needed for context, but not for answers

# Directory containing the saved intent model and vectorizer
MODEL_DIR = "models/"
INTENT_MODEL_FILE = os.path.join(MODEL_DIR, "trained_model.joblib")
INTENT_VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.joblib")

# --- Model Names ---
# Meditron model for answer generation
MEDICAL_MODEL_NAME = "malhajar/meditron-7b-chat"

# T5 model for question generation/rewriting
T5_MODEL_NAME = "google/flan-t5-base"

# --- API Keys ---
# HF_TOKEN might still be needed if models are private or rate limits are hit,
# but often not strictly necessary for public models like meditron/flan-t5
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Device Setup ---
# Check for CUDA availability (used by T5 and Meditron)
try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
        # Check if the installed PyTorch version supports bfloat16
        # Often beneficial for newer models if GPU supports it
        # if torch.cuda.is_bf16_supported():
        #     print("Device supports bfloat16.")
        #     # Can potentially add torch_dtype=torch.bfloat16 in model loading
        # else:
        #     print("Device does not support bfloat16, using float16 or float32.")
    else:
        DEVICE = "cpu"
except ImportError:
    print("PyTorch not found. Defaulting DEVICE to 'cpu'. Performance will be affected.")
    DEVICE = "cpu"
except Exception as e:
    print(f"Error checking for CUDA: {e}. Defaulting DEVICE to 'cpu'.")
    DEVICE = "cpu"

print(f"Configuration: Using device '{DEVICE}' for models.")

