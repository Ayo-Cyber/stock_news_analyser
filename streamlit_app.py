import streamlit as st
import pandas as pd
import os
from src.predictor import StockSentimentPredictor
from src.ner_extractor import NERExtractor
import spacy
from spacy import displacy

# Define paths for model and vectorizer
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

# Load the predictor and NER extractor
@st.cache_resource
def load_models():
    predictor = None
    if os.path.exists(os.path.join(MODELS_DIR, 'best_model.pkl')) and os.path.exists(VECTORIZER_PATH):
        predictor = StockSentimentPredictor(MODELS_DIR, VECTORIZER_PATH)
    ner_extractor = NERExtractor()
    return predictor, ner_extractor

predictor, ner_extractor = load_models()

st.set_page_config(page_title="Financial Insight Engine", layout="wide")

# --- UI Enhancements ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #2E86C1;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #5D6D7E;
    }
    .stButton>button {
        background-color: #2E86C1;
        color: white;
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #21618C;
    }
    .results-card {
        background-color: #F8F9F9;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #E5E7E9;
    }
    .entity-style {
        padding: 0.3em 0.6em;
        margin: 0 0.25em;
        line-height: 1;
        display: inline-block;
        border-radius: 0.35em;
        border: 1px solid;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Financial Insight Engine 🤖</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Analyze stock headlines, predict sentiment, and extract financial entities.</p>', unsafe_allow_html=True)


if predictor is None or predictor.model is None or predictor.feature_extractor is None:
    st.error("Model or vectorizer not found. Please train the model first by running `python main.py` in your terminal.")
else:
    user_input = st.text_area("Enter stock headline(s) (one per line):", "", height=150)

    if st.button("Analyze"):
        if user_input:
            headlines = [h.strip() for h in user_input.split('\n') if h.strip()]
            if headlines:
                with st.spinner('🧠 Performing analysis...'):
                    predictions = predictor.predict_sentiment(headlines)
                    
                    st.subheader("Analysis Results")

                    for i, headline in enumerate(headlines):
                        sentiment = predictions[i]
                        sentiment_label = "Up" if sentiment == 1 else "Down/Same"
                        sentiment_icon = "🔼" if sentiment == 1 else "🔽"


                        
                        col1, col2 = st.columns([4, 1])
                        
                        with col1:
                            st.markdown(f"**Headline:** {headline}")
                        
                        with col2:
                            st.markdown(f"**Sentiment:** {sentiment_label} {sentiment_icon}")

                        with st.expander("Named Entity Recognition"):
                            doc = ner_extractor.nlp(headline)
                            ent_html = displacy.render(doc, style="ent", jupyter=False)
                            st.markdown(ent_html, unsafe_allow_html=True)
                            
                        st.markdown(f'</div>', unsafe_allow_html=True)

            else:
                st.warning("Please enter at least one headline.")
        else:
            st.warning("Please enter some text to analyze.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a machine learning model to classify the sentiment of stock headlines and extracts named entities. "
    "It's built as a demonstration of how NLP can be applied to financial news. "
)

st.sidebar.header("Disclaimer")
st.sidebar.warning(
    "This tool is for educational and demonstrative purposes only and should not be used for actual financial decision-making."
)

st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1. **Train the Model:** Run `python main.py` to train the model.
    2. **Run the App:** Execute `streamlit run streamlit_app.py`.
    3. **Analyze:** Enter headlines and click 'Analyze'.
    """
)

