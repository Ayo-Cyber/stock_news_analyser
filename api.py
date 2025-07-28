
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predictor import StockSentimentPredictor
from src.ner_extractor import NERExtractor
import spacy
from spacy import displacy
from mangum import Mangum

# Initializing FastAPI application
app = FastAPI()
handler = Mangum(app)

# Defining paths for model and vectorizer
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

# Loading the predictor and NER extractor
predictor = StockSentimentPredictor(MODELS_DIR, VECTORIZER_PATH)
ner_extractor = NERExtractor()

class HeadlineRequest(BaseModel):
    headline: str

class HeadlineResponse(BaseModel):
    headline: str
    is_finance_related: bool
    sentiment: str = None
    entities: list = None


@app.post("/analyze", response_model=HeadlineResponse)
async def analyze_headline(request: HeadlineRequest):
    headline = request.headline
    
    if not headline:
        raise HTTPException(status_code=400, detail="Headline cannot be empty.")

    is_finance = ner_extractor.is_finance_related(headline)
    
    if not is_finance:
        return HeadlineResponse(
            headline=headline,
            is_finance_related=False
        )

    # Perform sentiment prediction
    prediction = predictor.predict_sentiment([headline])[0]
    sentiment = "Bullish" if prediction == 1 else "Bearish"
    
    # Perform named entity recognition
    doc = ner_extractor.nlp(headline)
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
    
    return HeadlineResponse(
        headline=headline,
        is_finance_related=True,
        sentiment=sentiment,
        entities=entities
    )

class HeadlinesRequest(BaseModel):
    headlines: list[str]

@app.post("/analyze_batch", response_model=list[HeadlineResponse])
async def analyze_headlines(request: HeadlinesRequest):
    headlines = request.headlines
    
    if not headlines:
        raise HTTPException(status_code=400, detail="Headlines list cannot be empty.")

    responses = []
    for headline in headlines:
        is_finance = ner_extractor.is_finance_related(headline)
        
        if not is_finance:
            responses.append(HeadlineResponse(
                headline=headline,
                is_finance_related=False
            ))
            continue

        prediction = predictor.predict_sentiment([headline])[0]
        sentiment = "Bullish" if prediction == 1 else "Bearish"
        
        doc = ner_extractor.nlp(headline)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        responses.append(HeadlineResponse(
            headline=headline,
            is_finance_related=True,
            sentiment=sentiment,
            entities=entities
        ))
    
    return responses

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Financial Insight API"}
