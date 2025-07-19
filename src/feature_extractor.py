from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import os

class FeatureExtractor:
    def __init__(self, vectorizer_type='count', max_features=10000, ngram_range=(2, 2)):
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        elif vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        else:
            raise ValueError("Unsupported vectorizer_type. Choose 'count' or 'tfidf'.")

    def fit(self, corpus):
        """
        Fits the vectorizer on the provided text corpus.
        Args:
            corpus (list): A list of preprocessed text documents.
        """
        self.vectorizer.fit(corpus)

    def transform(self, corpus):
        """
        Transforms the text corpus into a feature matrix using the fitted vectorizer.
        Args:
            corpus (list): A list of preprocessed text documents.
        Returns:
            numpy.ndarray: The feature matrix.
        """
        return self.vectorizer.transform(corpus).toarray()

    def fit_transform(self, corpus):
        """
        Fits the vectorizer and then transforms the text corpus.
        Args:
            corpus (list): A list of preprocessed text documents.
        Returns:
            numpy.ndarray: The feature matrix.
        """
        return self.vectorizer.fit_transform(corpus).toarray()

    def save_vectorizer(self, file_path):
        """
        Saves the fitted vectorizer to a file.
        Args:
            file_path (str): The absolute path to save the vectorizer.
        """
        joblib.dump(self.vectorizer, file_path)
        print(f"Vectorizer saved to {file_path}")

    @classmethod
    def load_vectorizer(cls, file_path):
        """
        Loads a fitted vectorizer from a file.
        Args:
            file_path (str): The absolute path to the saved vectorizer.
        Returns:
            CountVectorizer or TfidfVectorizer: The loaded vectorizer.
        """
        vectorizer = joblib.load(file_path)
        print(f"Vectorizer loaded from {file_path}")
        # We don't know the original vectorizer_type, so we create a dummy instance
        # and then replace its vectorizer.
        instance = cls(vectorizer_type='count') # or 'tfidf', it doesn't matter
        instance.vectorizer = vectorizer
        return instance

if __name__ == '__main__':
    # Example usage:
    from data_loader import load_and_split_data
    from preprocessor import TextPreprocessor
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, 'Data', 'Stock Headlines.csv')
    
    # --- Example with CountVectorizer ---
    print("--- Using CountVectorizer ---")
    vectorizer_path_count = os.path.join(project_root, 'models', 'count_vectorizer.pkl')
    train_df, y_train_series, test_df, y_test_series = load_and_split_data(data_file_path)

    if train_df is not None:
        preprocessor = TextPreprocessor()
        processed_train_headlines = preprocessor.preprocess_headlines(train_df)
        processed_test_headlines = preprocessor.preprocess_headlines(test_df)

        feature_extractor_count = FeatureExtractor(vectorizer_type='count')
        X_train_count = feature_extractor_count.fit_transform(processed_train_headlines)
        X_test_count = feature_extractor_count.transform(processed_test_headlines)

        print("Feature extraction with CountVectorizer complete.")
        print(f"X_train_count shape: {X_train_count.shape}")
        print(f"X_test_count shape: {X_test_count.shape}")

        feature_extractor_count.save_vectorizer(vectorizer_path_count)
        loaded_fe_count = FeatureExtractor.load_vectorizer(vectorizer_path_count)
        print("CountVectorizer loaded successfully.")
    else:
        print("Failed to load data for CountVectorizer example.")

    # --- Example with TfidfVectorizer ---
    print("--- Using TfidfVectorizer ---")
    vectorizer_path_tfidf = os.path.join(project_root, 'models', 'tfidf_vectorizer.pkl')
    
    # No need to reload data if it's already loaded
    if train_df is not None:
        # We can reuse the preprocessed headlines
        feature_extractor_tfidf = FeatureExtractor(vectorizer_type='tfidf')
        X_train_tfidf = feature_extractor_tfidf.fit_transform(processed_train_headlines)
        X_test_tfidf = feature_extractor_tfidf.transform(processed_test_headlines)

        print("Feature extraction with TfidfVectorizer complete.")
        print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
        print(f"X_test_tfidf shape: {X_test_tfidf.shape}")

        feature_extractor_tfidf.save_vectorizer(vectorizer_path_tfidf)
        loaded_fe_tfidf = FeatureExtractor.load_vectorizer(vectorizer_path_tfidf)
        print("TfidfVectorizer loaded successfully.")
    else:
        print("Failed to load data for TfidfVectorizer example.")
