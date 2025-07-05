# intent_classifier.py
# (Keep the previous version - it correctly loads and uses the joblib model)
import joblib
import os
import streamlit as st
from config import INTENT_MODEL_FILE, INTENT_VECTORIZER_FILE

@st.cache_resource # Cache the loaded model and vectorizer
def load_intent_model_and_vectorizer():
    """Loads the intent classification model and vectorizer using joblib."""
    model = None
    vectorizer = None
    print("Loading intent model and vectorizer...")
    # Ensure model directory exists
    model_dir = os.path.dirname(INTENT_MODEL_FILE)
    if not os.path.exists(model_dir):
         st.warning(f"Model directory not found: {model_dir}")
         # Optionally create it: os.makedirs(model_dir, exist_ok=True)
         
    try:
        if os.path.exists(INTENT_MODEL_FILE):
            model = joblib.load(INTENT_MODEL_FILE)
            print(f"Intent model loaded from {INTENT_MODEL_FILE}.")
        else:
            st.error(f"Intent model file not found: {INTENT_MODEL_FILE}")

        if os.path.exists(INTENT_VECTORIZER_FILE):
            vectorizer = joblib.load(INTENT_VECTORIZER_FILE)
            print(f"Intent vectorizer loaded from {INTENT_VECTORIZER_FILE}.")
        else:
            st.error(f"Intent vectorizer file not found: {INTENT_VECTORIZER_FILE}")

    except ImportError as ie:
         st.error(f"Error loading model/vectorizer: {ie}. Make sure scikit-learn is installed (`pip install scikit-learn`).")
         model, vectorizer = None, None
    except Exception as e:
        st.error(f"Error loading intent model/vectorizer: {e}")
        model, vectorizer = None, None # Ensure both are None on error

    return model, vectorizer

def predict_intent(text, _model, _vectorizer):
    """Predicts the intent for a given text using the loaded model and vectorizer."""
    if not _model or not _vectorizer:
        st.warning("Intent model or vectorizer not loaded. Cannot predict intent.")
        return "Error: Model/Vectorizer not loaded"

    if not text: # Handle empty input text
         st.info("Input text for intent prediction is empty.")
         return "No input provided"

    if not isinstance(text, list):
         text = [text] # Model expects a list/iterable

    try:
        # Ensure text is properly formatted (e.g., handle potential NaN or None)
        processed_text = [str(t) if t is not None else "" for t in text]

        text_tfidf = _vectorizer.transform(processed_text)
        prediction = _model.predict(text_tfidf)
        # Return the first prediction assuming single input
        intent = prediction[0] if prediction.size > 0 else "Prediction failed"
        print(f"Predicted intent for '{text[0][:50]}...': {intent}")
        return intent
    except AttributeError as ae:
         st.error(f"Error during intent prediction: {ae}. Check if the vectorizer/model loaded correctly and matches the training format.")
         return f"Error: Attribute error during prediction"
    except Exception as e:
        st.error(f"Error during intent prediction: {e}")
        return f"Error: {e}"
