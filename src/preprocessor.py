import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stopwords_set = set(stopwords.words('english'))

    def preprocess_headlines(self, df_headlines):
        """
        Applies a series of preprocessing steps to the headlines DataFrame.

        Args:
            df_headlines (pd.DataFrame): DataFrame containing the headlines.
                                         Expected to have columns like 'Top1', 'Top2', etc.

        Returns:
            list: A list of preprocessed and stemmed headlines.
        """
        processed_headlines = []
        temp_df = df_headlines.copy()

        # Remove punctuation and special characters
        temp_df.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)

        # Convert to lowercase
        for col in temp_df.columns:
            temp_df[col] = temp_df[col].astype(str).str.lower()

        # Join all headlines for each row
        combined_headlines = []
        for row_index in range(temp_df.shape[0]):
            combined_headlines.append(' '.join(str(x) for x in temp_df.iloc[row_index, :]))

        # Apply stemming and remove stopwords
        for headline in combined_headlines:
            words = headline.split()
            words = [word for word in words if word not in self.stopwords_set]
            words = [self.ps.stem(word) for word in words]
            processed_headlines.append(' '.join(words))

        return processed_headlines

if __name__ == '__main__':
    # Example usage:
    from data_loader import load_and_split_data
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(project_root, 'Data', 'Stock Headlines.csv')

    train_df, y_train_series, test_df, y_test_series = load_and_split_data(data_file_path)

    if train_df is not None:
        preprocessor = TextPreprocessor()
        processed_train_headlines = preprocessor.preprocess_headlines(train_df)
        processed_test_headlines = preprocessor.preprocess_headlines(test_df)

        print("Text preprocessing complete.")
        print(f"First 5 processed train headlines: {processed_train_headlines[:5]}")
        print(f"First 5 processed test headlines: {processed_test_headlines[:5]}")
    else:
        print("Failed to load data for preprocessing example.")
