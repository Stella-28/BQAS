import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

T5_MODEL_NAME = "google/flan-t5-base"
DEVICE = torch.device("cpu")

def load_t5_model_and_tokenizer():
    print(f"Loading T5 model ({T5_MODEL_NAME}) and tokenizer onto CPU...")
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME, model_max_length=512)
    model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)
    model.eval()
    print("T5 model loaded successfully.")
    return model, tokenizer

def post_process_question(question):
    question = question.strip()
    if not re.search(r'[?.!]$', question):
        question += '?'
    if question:
        question = question[0].upper() + question[1:]
    return question

def generate_question(original_question, style="faq"):
    model, tokenizer = load_t5_model_and_tokenizer()
    if not original_question or not isinstance(original_question, str) or original_question.strip() == "":
        return "Error: Original question is empty."

    # PROMPT DESIGN
    if style == "faq":
        prompt = (
            "Reformulate this question with correct spelling, using professional medical terminology, convert layman terms into profession medical terms, "
            "and keep it concise (15 to 20 tokens). Ensure the revised question is 80 to 90 percent similar to the original, but shorter and not in layman terms cover the full question and then summize it\n"
            f"{original_question}"
        )
        max_length = 20
    elif style == "detailed":
        prompt = (
            "Reformulate this question with correct spelling, using professional medical terminology, convert layman terms into profession medical terms,"
            "and keep it concise (up to 40 tokens). Ensure the revised question is 80 to 90 percent similar to the original, but shorter and not in layman terms cover the full question and then summize it:\n"
            f"{original_question}"
        )
        max_length = 40
    else:
        prompt = (
            "Reformulate this question with correct spelling, using professional medical terminology, "
            "and keep it concise. Ensure the revised question is 80 to 90 percent similar to the original, but shorter and not in layman terms:\n"
            f"{original_question}"
        )
        max_length = 25

    print(f"Generating '{style}' question for: '{original_question[:50]}...'")

    try:
        with torch.no_grad():
            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).input_ids.to(DEVICE)

            output = model.generate(
                input_ids,
                max_length=max_length,
                min_length=5,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                num_return_sequences=1,
            )

        generated_question = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_question = post_process_question(generated_question)
        print(f"Generated question: {generated_question}")
        return generated_question

    except Exception as e:
        return f"Error during T5 question generation: {e}"

# Example usage
if __name__ == "__main__":
    orig_q = "whatt are the best treatmnt option for diabtes patints?"
    faq_q = generate_question(orig_q, style="faq")
    detailed_q = generate_question(orig_q, style="detailed")
    print("\nFAQ style:", faq_q)
    print("Detailed style:", detailed_q)
