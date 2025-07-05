# app.py
import streamlit as st
import time
import torch # For checking cuda availability status if needed

# Import functions from other modules
import config # Import config first
from meditron_utils import load_medical_model, MedicalQuestionAnswerer # NEW: Import Meditron utils
from intent_classifier import load_intent_model_and_vectorizer, predict_intent
from question_generator import generate_question

# --- Page Setup ---
st.set_page_config(page_title="Medical AI Assistant", layout="wide")
st.title("⚕️ Medical AI Assistant")
st.markdown("Using T5 for question style, Joblib for intent classification, and Meditron-7B for answer generation.")

# --- Load Models and Data (Cached) ---
@st.cache_resource(show_spinner=False) # Master cache for all components
def load_all_components():
    """Loads Meditron, Intent Model, and T5 (via question_generator)."""
    components = {}
    st.write("---")
    st.info("🚀 Initializing AI components... This might take a minute, especially the first time.")
    
    with st.spinner("Loading Intent Classifier..."):
        intent_model, intent_vectorizer = load_intent_model_and_vectorizer()
        if not intent_model or not intent_vectorizer:
             st.warning("Intent classification model/vectorizer failed to load. Intent prediction disabled.")
        components['intent_model'] = intent_model
        components['intent_vectorizer'] = intent_vectorizer
    
   
    meditron_model, meditron_tokenizer = load_medical_model()
    
    meditron_answerer = None
    if meditron_model and meditron_tokenizer:
        with st.spinner("Initializing Medical Question Answerer..."):
            try:
                meditron_answerer = MedicalQuestionAnswerer(meditron_model, meditron_tokenizer)
                st.success("✅ Medical AI Model Ready!")
            except ValueError as ve:
                 st.error(f"Failed to initialize Medical Answerer: {ve}")
            except Exception as e:
                 st.error(f"An unexpected error occurred initializing Medical Answerer: {e}")
    else:
        st.error("❌ Failed to load the Medical AI Model. Answer generation is disabled.")
        
    components['meditron_answerer'] = meditron_answerer

    # T5 model is loaded lazily by question_generator when first called (due to its own @st.cache_resource)
    # No explicit loading needed here.

    st.success("👍 AI Components Initialized.")
    st.write("---")
    return components

# Load components using the cached function
components = load_all_components()
meditron_answerer = components.get('meditron_answerer')
intent_model = components.get('intent_model')
intent_vectorizer = components.get('intent_vectorizer')

# --- Helper Function for Displaying Results ---
def display_results(generated_q, predicted_intent, generated_answer):
    """Helper to display the generated question, intent, and Meditron answer."""

    st.subheader("Revised Question:")
    if "Error" in generated_q:
        st.error(generated_q)
    else:
        st.info(generated_q) # Use info box for generated content

    st.subheader("Predicted Intent:")
    if "Error" in predicted_intent:
         st.error(predicted_intent)
    elif "No input" in predicted_intent or "not loaded" in predicted_intent.lower():
         st.warning(predicted_intent)
    else:
         st.success(f"`{predicted_intent}`") # Use success box for successful prediction

    st.subheader("Generated Answer:")
    if generated_answer:
        if "Error" in generated_answer:
             st.error(generated_answer)
        else:
            st.markdown(generated_answer) # Display the model's answer
    else:
        st.warning("No answer was generated.")


# --- UI Tabs ---
tab1, tab2 = st.tabs(["❓ FAQ Style Question", "🔍 Detailed Style Question"])

# --- Tab 1: FAQ ---
with tab1:
    st.header("Ask a Question (FAQ Style)")
    st.markdown("Enter your medical question below. We will:")
    st.markdown("1. Rewrite it in a typical FAQ style (using FLAN-T5).")
    st.markdown("2. Predict the intent of the rewritten question (using a trained model).")
    st.markdown("3. Generate an answer using the Meditron-7B AI model.")

    user_question_faq = st.text_input("Your Question:", key="faq_input", placeholder="e.g., what causes high blood pressure?")

    if st.button("Get FAQ Answer", key="faq_button") and user_question_faq:
        if not meditron_answerer:
             st.error("Medical AI Model is not available. Cannot generate answer.")
        else:
            st.markdown("---")
            generated_faq_q = "Processing..."
            predicted_faq_intent = "Processing..."
            generated_answer = "Processing..."

            # 1. Generate FAQ Question
            with st.spinner("Styling question (FAQ)..."):
                generated_faq_q = generate_question(user_question_faq, style="faq")

            # 2. Predict Intent of FAQ Question
            if "Error" not in generated_faq_q:
                with st.spinner("Predicting intent..."):
                    predicted_faq_intent = predict_intent(generated_faq_q, intent_model, intent_vectorizer)
            else:
                 predicted_faq_intent = "Skipped due to question generation error."


            # 3. Generate Answer using Meditron with the generated FAQ question
            if "Error" not in generated_faq_q:
                 with st.spinner("🤖 Thinking... Generating answer..."):
                     start_time = time.time()
                     generated_answer = meditron_answerer.answer_question(generated_faq_q,predicted_faq_intent)
                     end_time = time.time()
                     st.caption(f"Answer generated in {end_time - start_time:.2f} seconds.")
            else:
                generated_answer = "Skipped due to question generation error."

            # 4. Display Results
            display_results(generated_faq_q, predicted_faq_intent, generated_answer)
            st.markdown("---")

# --- Tab 2: Detailed Question ---
with tab2:
    st.header("Ask a Question (Detailed Style)")
    st.markdown("Enter your medical question below. We will:")
    st.markdown("1. Rewrite it in a more detailed/specific style (using FLAN-T5).")
    st.markdown("2. Predict the intent of the rewritten question (using a trained model).")
    st.markdown("3. Generate an answer using the Meditron-7B AI model.")

    user_question_detail = st.text_input("Your Question:", key="detail_input", placeholder="e.g., describe the renin-angiotensin system")

    if st.button("Get Detailed Answer", key="detail_button") and user_question_detail:
        if not meditron_answerer:
             st.error("Medical AI Model is not available. Cannot generate answer.")
        else:
            st.markdown("---")
            generated_detail_q = "Processing..."
            predicted_detail_intent = "Processing..."
            generated_answer = "Processing..."

            # 1. Generate Detailed Question
            with st.spinner("Styling question (Detailed)..."):
                generated_detail_q = generate_question(user_question_detail, style="detailed")

            # 2. Predict Intent of Detailed Question
            if "Error" not in generated_detail_q:
                with st.spinner("Predicting intent..."):
                    predicted_detail_intent = predict_intent(generated_detail_q, intent_model, intent_vectorizer)
            else:
                 predicted_detail_intent = "Skipped due to question generation error."

            # 3. Generate Answer using Meditron with the generated detailed question
            if "Error" not in generated_detail_q:
                with st.spinner("🤖 Thinking... Generating answer..."):
                     start_time = time.time()
                     generated_answer = meditron_answerer.answer_question(generated_detail_q,predicted_detail_intent)
                     end_time = time.time()
                     st.caption(f"Answer generated in {end_time - start_time:.2f} seconds.")
            else:
                 generated_answer = "Skipped due to question generation error."


            # 4. Display Results
            display_results(generated_detail_q, predicted_detail_intent, generated_answer)
            st.markdown("---")


# --- Sidebar ---
st.sidebar.header("AI Models Used")
st.sidebar.markdown(f"**Answer Generation:**")
st.sidebar.code(config.MEDICAL_MODEL_NAME, language=None)
st.sidebar.markdown(f"**Question Styling:**")
st.sidebar.code(config.T5_MODEL_NAME, language=None)
st.sidebar.markdown(f"**Intent Classification:**")
st.sidebar.code(f"{config.INTENT_MODEL_FILE}\n{config.INTENT_VECTORIZER_FILE}", language=None)

st.sidebar.header("Setup Info")
st.sidebar.markdown(f"**Processing Device:** `{config.DEVICE.upper()}`")
if config.DEVICE == 'cuda' and not torch.cuda.is_available():
     st.sidebar.warning("Configured for CUDA, but no GPU detected by PyTorch!")
elif config.DEVICE == 'cpu':
     st.sidebar.warning("Running on CPU. Operations will be significantly slower.")

hf_token_status = "Set ✔️" if config.HF_TOKEN else "Not Set"
st.sidebar.markdown(f"**Hugging Face Token:** {hf_token_status}")

st.sidebar.markdown("---")
st.sidebar.caption("Ensure `models/` directory contains intent model and vectorizer.")
st.sidebar.caption("4-bit model loading requires `bitsandbytes` and `accelerate`.")
