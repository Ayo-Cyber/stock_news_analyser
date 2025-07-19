from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
import os
import numpy as np
import json

class ModelTrainer:
    def __init__(self, model_names=['RandomForestClassifier', 'XGBClassifier']):
        self.models = self._get_models(model_names)
        self.best_model = None
        self.best_model_name = None

    def _get_models(self, model_names):
        models = {}
        if 'RandomForestClassifier' in model_names:
            models['RandomForestClassifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        if 'XGBClassifier' in model_names:
            models['XGBClassifier'] = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        if not models:
            raise ValueError("Unsupported model name(s).")
        return models

    def train_and_evaluate_with_cv(self, X_train, y_train, cv=5):
        """
        Trains and evaluates models using cross-validation, then selects the best one.

        Args:
            X_train (numpy.ndarray): Training features.
            y_train (pd.Series): Training labels.
            cv (int): Number of cross-validation folds.
        """
        best_score = -1
        
        for name, model in self.models.items():
            print(f"--- Evaluating {name} with {cv}-fold cross-validation ---")
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            mean_cv_score = np.mean(cv_scores)
            
            print(f"CV Accuracy scores: {cv_scores}")
            print(f"Mean CV Accuracy: {mean_cv_score:.4f}")
            
            if mean_cv_score > best_score:
                best_score = mean_cv_score
                self.best_model_name = name
        
        print(f"\nBest model selected: {self.best_model_name} with average accuracy: {best_score:.4f}")
        
        # Train the best model on the entire training data
        self.best_model = self.models[self.best_model_name]
        print(f"Training the final {self.best_model_name} model on the full training data...")
        self.best_model.fit(X_train, y_train)
        print("Final model training complete.")

    def evaluate_on_test(self, X_test, y_test):
        """
        Evaluates the best model on the test set.

        Args:
            X_test (numpy.ndarray): Testing features.
            y_test (pd.Series): Testing labels.

        Returns:
            dict: A dictionary containing performance metrics.
        """
        if self.best_model is None:
            raise RuntimeError("No best model selected. Run train_and_evaluate_with_cv first.")

        print(f"\n--- Evaluating best model ({self.best_model_name}) on test data ---")
        y_pred = self.best_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

    def save_best_model(self, model_dir):
        """
        Saves the best trained model and its metadata to a directory.

        Args:
            model_dir (str): The directory to save the model artifacts.
        """
        if self.best_model is None:
            raise RuntimeError("No best model to save.")

        # Create directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Save the model file
        model_path = os.path.join(model_dir, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"Best model ({self.best_model_name}) saved to {model_path}")

        # Save metadata
        metadata = {'best_model_name': self.best_model_name}
        metadata_path = os.path.join(model_dir, 'best_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        print(f"Model metadata saved to {metadata_path}")

    @classmethod
    def load_model(cls, model_dir):
        """
        Loads the best model from a directory.

        Args:
            model_dir (str): The directory where the model is saved.

        Returns:
            A loaded model object.
        """
        model_path = os.path.join(model_dir, 'best_model.pkl')
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model

if __name__ == '__main__':
    from data_loader import load_and_split_data
    from preprocessor import TextPreprocessor
    from feature_extractor import FeatureExtractor
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, 'Data', 'Stock Headlines.csv')
    models_dir = os.path.join(project_root, 'models')
    
    # Using TfidfVectorizer for this example
    vectorizer_type = 'tfidf' 
    vectorizer_path = os.path.join(models_dir, f'{vectorizer_type}_vectorizer.pkl')

    train_df, y_train_series, test_df, y_test_series = load_and_split_data(data_file_path)

    if train_df is not None:
        preprocessor = TextPreprocessor()
        processed_train_headlines = preprocessor.preprocess_headlines(train_df)
        processed_test_headlines = preprocessor.preprocess_headlines(test_df)

        # Initialize and use the feature extractor
        feature_extractor = FeatureExtractor(vectorizer_type=vectorizer_type)
        X_train = feature_extractor.fit_transform(processed_train_headlines)
        X_test = feature_extractor.transform(processed_test_headlines)
        
        # Save the vectorizer
        feature_extractor.save_vectorizer(vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

        # Initialize model trainer with a list of models to compare
        model_trainer = ModelTrainer(model_names=['RandomForestClassifier', 'XGBClassifier'])
        
        # Train, evaluate with CV, and select the best model
        model_trainer.train_and_evaluate_with_cv(X_train, y_train_series)
        
        # Evaluate the chosen best model on the held-out test set
        model_trainer.evaluate_on_test(X_test, y_test_series)
        
        # Save the best model and its metadata
        model_trainer.save_best_model(models_dir)

        # Example of loading the best model
        loaded_model = ModelTrainer.load_model(models_dir)
        print("\nLoaded model type:", type(loaded_model))
        dummy_prediction = loaded_model.predict(X_test[0].reshape(1, -1))
        print(f"Dummy prediction for first test sample with loaded model: {dummy_prediction[0]}")
    else:
        print("Failed to load data for model training example.")
