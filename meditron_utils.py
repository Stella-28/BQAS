# meditron_utils.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import streamlit as st
import re
from config import MEDICAL_MODEL_NAME, DEVICE

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner="Loading Medical AI Model...")
def load_medical_model():
    """Load the medical model and tokenizer with optimizations."""
    logger.info(f"Attempting to load model: {MEDICAL_MODEL_NAME} onto device: {DEVICE}")
    model = None
    tokenizer = None

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        device_map = "auto"
        if DEVICE == 'cpu':
            logger.warning("4-bit quantization is primarily designed for GPU.")

        tokenizer = AutoTokenizer.from_pretrained(
            MEDICAL_MODEL_NAME,
            trust_remote_code=True,
            use_fast=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set tokenizer pad_token to eos_token")

        model = AutoModelForCausalLM.from_pretrained(
            MEDICAL_MODEL_NAME,
            quantization_config=quantization_config,
            device_map=device_map,
            trust_remote_code=True,
        )

        logger.info("Medical model and tokenizer loaded successfully.")

    except ImportError as ie:
        logger.error(f"ImportError: {ie}. Make sure 'bitsandbytes' and 'accelerate' are installed.")
        st.error(f"Failed to load model dependencies: {ie}. Please install required packages.")
    except Exception as e:
        logger.error(f"Error loading medical model {MEDICAL_MODEL_NAME}: {e}", exc_info=True)
        st.error(f"Error loading medical model: {e}")
        model, tokenizer = None, None

    return model, tokenizer


class MedicalQuestionAnswerer:
    """Class to answer medical questions using the loaded Meditron model."""

    def __init__(self, model, tokenizer):
        if not model or not tokenizer:
            raise ValueError("Model and Tokenizer must be provided and loaded.")

        self.model = model
        self.tokenizer = tokenizer

        try:
            self.device = next(model.parameters()).device
            logger.info(f"Answerer using device: {self.device}")
        except Exception:
            logger.warning("Could not determine model device automatically. Assuming input needs manual placement if using single device.")
            self.device = DEVICE

        self.model.eval()

        # Updated system message with strict instructions
        self.sys_message = (
            "You are Meditron, an AI Medical Assistant. "
            "CRITICAL RULES:\n"
            "1. Only answer MEDICAL questions.\n"
            "2. For non-medical queries: 'Please ask a medical question.'\n"
            "3. For medical questions: Provide concise, evidence-based, and complete answers.\n"
            "4. If the answer is long, summarize the main points clearly and finish your answer.\n"
            "5. NEVER include:\n"
            "   - Closing remarks (e.g., 'Is there anything else?')\n"
            "   - Voice assistant instructions\n"
            "   - Meta-comments about your capabilities\n"
            "   - Any XML/HTML tags (like <p>, </p>, <body>)\n"
            "Only return the Generated Answer dont return anything else"
        )

    def answer_question(self, question,intent=None):
        """Generate an answer for a given medical question."""
        if not self.model or not self.tokenizer:
            logger.error("Attempted to answer question but model/tokenizer not loaded.")
            return "Error: Medical AI Model is not available."
        if not question or not isinstance(question, str):
            logger.warning("Received empty or invalid question.")
            return "Please provide a valid question."

        prompt = (
            f"<|system|>\n{self.sys_message}\n**Intent:** {intent}\n\n"
            f"<|user|>\n{question}\n</s>\n"
            "<|assistant|>\n"
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=350,  # Increased for completeness
                    temperature=0.3,
                    top_p=0.9,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

            # Extract the answer after <|assistant|>
            if "<|assistant|>" in full_response:
                answer = full_response.split("<|assistant|>", 1)[-1].strip()
            else:
                answer = full_response.replace(prompt, "").strip()

            # Clean artifacts and truncate at stop markers
            answer = re.sub(r'(<s>|</s>|<\|.*?\|>)', '', answer)
            answer = re.sub(r'\n+', '\n', answer).strip()

            stop_phrases = [
                "Is there anything else",
                "If you are using",
                "Best,",
                "I've answered",
                "Note:",
                "<|end|>",
                "<|user|>"
            ]
            for phrase in stop_phrases:
                if phrase in answer:
                    answer = answer.split(phrase)[0].strip()
                    break

            answer = answer.split("</s>")[0].strip()
            answer = answer.split("<|end|>")[0].strip()

            # If answer ends with a comma, hyphen, or is clearly cut off, append a note
            if answer and (answer[-1] in [',', '-', ':'] or answer.endswith('such as an')):
                answer += "\n\nFor more detailed information, please consult your healthcare provider."

            non_medical_triggers = ["non-medical", "ask a medical", "not medical"]
            if any(trigger in answer.lower() for trigger in non_medical_triggers):
                return "Please ask a medical question."

            logger.info(f"Generated answer: {answer[:100]}...")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return f"Error during answer generation: {e}"
