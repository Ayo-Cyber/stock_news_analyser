import os
from src.data_loader import load_and_split_data
from src.preprocessor import TextPreprocessor
from src.feature_extractor import FeatureExtractor
from src.model_trainer import ModelTrainer

def train_model_pipeline():
    """
    Orchestrates the entire model training pipeline:
    1. Loads and splits data.
    2. Preprocesses text.
    3. Extracts features.
    4. Trains and evaluates the model.
    5. Saves the trained model and vectorizer.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(project_root, 'Data', 'Stock Headlines.csv')
    models_dir = os.path.join(project_root, 'models')
    
    # Using TfidfVectorizer for this example
    vectorizer_type = 'tfidf' 
    vectorizer_path = os.path.join(models_dir, f'{vectorizer_type}_vectorizer.pkl')

    print("Starting model training pipeline...")

    # 1. Load and split data
    train_df, y_train_series, test_df, y_test_series = load_and_split_data(data_file_path)

    if train_df is None:
        print("Failed to load data. Exiting.")
        return

    # 2. Preprocess text
    preprocessor = TextPreprocessor()
    print("Preprocessing training headlines...")
    processed_train_headlines = preprocessor.preprocess_headlines(train_df)
    print("Preprocessing testing headlines...")
    processed_test_headlines = preprocessor.preprocess_headlines(test_df)

    # 3. Extract features
    feature_extractor = FeatureExtractor(vectorizer_type=vectorizer_type)
    print("Fitting and transforming training features...")
    X_train = feature_extractor.fit_transform(processed_train_headlines)
    print("Transforming testing features...")
    X_test = feature_extractor.transform(processed_test_headlines)

    # Save the fitted vectorizer
    feature_extractor.save_vectorizer(vectorizer_path)

    # 4. Train and evaluate the model
    model_trainer = ModelTrainer(model_names=['RandomForestClassifier', 'XGBClassifier'])
    model_trainer.train_and_evaluate_with_cv(X_train, y_train_series)
    model_trainer.evaluate_on_test(X_test, y_test_series)

    # Save the best model
    model_trainer.save_best_model(models_dir)

    print("Model training pipeline completed successfully.")

if __name__ == '__main__':
    train_model_pipeline()
