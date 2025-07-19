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
    model_save_path = os.path.join(project_root, 'models', 'random_forest_model.pkl')
    vectorizer_save_path = os.path.join(project_root, 'models', 'count_vectorizer.pkl')

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
    feature_extractor = FeatureExtractor()
    print("Fitting and transforming training features...")
    X_train = feature_extractor.fit_transform(processed_train_headlines)
    print("Transforming testing features...")
    X_test = feature_extractor.transform(processed_test_headlines)

    # Save the fitted vectorizer
    feature_extractor.save_vectorizer(vectorizer_save_path)

    # 4. Train and evaluate the model
    model_trainer = ModelTrainer()
    model_trainer.train(X_train, y_train_series)
    model_trainer.evaluate(X_test, y_test_series)

    # Save the trained model
    model_trainer.save_model(model_save_path)

    print("Model training pipeline completed successfully.")

if __name__ == '__main__':
    # Adjust the current working directory to the project root for correct path resolution
    # This assumes main.py is directly under the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) # Go up one level from 'src'
    os.chdir(project_root)

    train_model_pipeline()
