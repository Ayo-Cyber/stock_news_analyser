import joblib
import os
import json
import pandas as pd
from .preprocessor import TextPreprocessor
from .feature_extractor import FeatureExtractor

class StockSentimentPredictor:
    def __init__(self, models_dir, vectorizer_path):
        self.model = self._load_model(models_dir)
        self.feature_extractor = self._load_vectorizer(vectorizer_path)
        self.preprocessor = TextPreprocessor()

    def _load_model(self, models_dir):
        """
        Loads the best trained machine learning model from the specified directory.
        """
        try:
            model_path = os.path.join(models_dir, 'best_model.pkl')
            model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}")
            return None
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

    def _load_vectorizer(self, vectorizer_path):
        """
        Loads the fitted vectorizer.
        """
        try:
            # Use the FeatureExtractor's class method to load the vectorizer
            self.feature_extractor = FeatureExtractor.load_vectorizer(vectorizer_path)
            print(f"Vectorizer loaded successfully from {vectorizer_path}")
            return self.feature_extractor
        except FileNotFoundError:
            print(f"Error: Vectorizer file not found at {vectorizer_path}")
            return None
        except Exception as e:
            print(f"Error loading vectorizer from {vectorizer_path}: {e}")
            return None

    def predict_sentiment(self, headlines):
        """
        Predicts the sentiment (stock movement) for new headlines.

        Args:
            headlines (list or pd.Series): A list or pandas Series of raw headlines.

        Returns:
            list: Predicted labels (0 for down/same, 1 for up).
        """
        if self.model is None or self.feature_extractor is None:
            print("Model or vectorizer not loaded. Cannot make predictions.")
            return []

        # Ensure headlines is a DataFrame for preprocessing
        if isinstance(headlines, list):
            # Create a DataFrame with a single row and multiple columns
            headlines_df = pd.DataFrame([headlines], columns=[f'Top{i+1}' for i in range(len(headlines))])
        elif isinstance(headlines, pd.Series):
            headlines_df = pd.DataFrame([headlines.tolist()], columns=[f'Top{i+1}' for i in range(len(headlines))])
        elif isinstance(headlines, pd.DataFrame):
            headlines_df = headlines
        else:
            raise ValueError("Input 'headlines' must be a list, pandas Series, or pandas DataFrame.")

        # Preprocess the headlines
        processed_headlines = self.preprocessor.preprocess_headlines(headlines_df)

        # Transform headlines into features using the loaded vectorizer
        features = self.feature_extractor.vectorizer.transform(processed_headlines).toarray()

        # Make prediction
        predictions = self.model.predict(features)
        return predictions.tolist()

if __name__ == '__main__':
    import pandas as pd
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    # Determine which vectorizer to use based on the training script
    # For this example, we assume tfidf was used as in the new model_trainer
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')

    # Dummy data for demonstration
    dummy_headlines = [
        "Company X announced record profits, stock expected to soar.",
        "Market analysts predict a downturn due to global economic concerns.",
        "New product launch boosts investor confidence."
    ]

    # To run this example, you need to have trained and saved the model and vectorizer first
    if os.path.exists(os.path.join(models_dir, 'best_model.pkl')) and os.path.exists(vectorizer_path):
        # Pass the directory for models, and the specific path for the vectorizer
        predictor = StockSentimentPredictor(models_dir=models_dir, vectorizer_path=vectorizer_path)
        
        if predictor.model and predictor.feature_extractor:
            predictions = predictor.predict_sentiment(dummy_headlines)
            print(f"Predictions for dummy headlines: {predictions}")
    else:
        print("Best model or vectorizer not found. Please run model_trainer.py first.")
