# Financial Insight Engine: Stock Sentiment & NER Analysis

This project is a web-based application that analyzes financial news headlines to predict stock market sentiment and performs Named Entity Recognition (NER) to identify key financial entities. The application is built with Python, using a Random Forest Classifier for sentiment analysis and spaCy for NER, with an interactive user interface powered by Streamlit.

## ✨ Features

- **Sentiment Analysis**: Predicts whether a stock headline implies a positive (Up) or neutral/negative (Down/Same) sentiment.
- **Named Entity Recognition (NER)**: Identifies and highlights entities like organizations, people, and locations within the headlines.
- **Interactive UI**: A user-friendly web interface built with Streamlit for easy interaction and analysis.
- **Modular Codebase**: The project is structured into logical modules for data loading, preprocessing, model training, and prediction.

## 🚀 How It Works

The application follows a standard machine learning pipeline:

1.  **Data Loading**: Loads stock headline data from the provided CSV file.
2.  **Preprocessing**: Cleans the text data by removing punctuation, converting to lowercase, and applying stemming.
3.  **Feature Extraction**: Uses `CountVectorizer` to convert the text data into a numerical format.
4.  **Model Training**: A `RandomForestClassifier` is trained on the preprocessed data to predict sentiment.
5.  **Prediction**: The trained model predicts the sentiment of new headlines provided by the user.
6.  **NER**: spaCy's pre-trained model is used to extract and display named entities from the headlines.

## 📂 Project Structure

```
/project_test_api/
├─── Data/
│    └─── Stock Headlines.csv
├─── models/
│    ├─── count_vectorizer.pkl
│    └─── random_forest_model.pkl
├─── src/
│    ├─── data_loader.py
│    ├─── feature_extractor.py
│    ├─── model_trainer.py
│    ├─── ner_extractor.py
│    ├─── predictor.py
│    └─── preprocessor.py
├─── main.py
├─── requirements.txt
└─── streamlit_app.py
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd project_test_api
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

## ▶️ Usage

1.  **Train the model:**
    Before running the application, you need to train the model by running the main pipeline script:
    ```bash
    python main.py
    ```
    This will process the data, train the Random Forest model and the CountVectorizer, and save them to the `/models` directory.

2.  **Run the Streamlit application:**
    ```bash
    streamlit run streamlit_app.py
    ```
    Open your web browser and navigate to the local URL provided by Streamlit.

## Dependencies

- pandas
- numpy
- scikit-learn
- nltk
- joblib
- streamlit
- spacy

## Disclaimer

This tool is for educational and demonstrative purposes only. The sentiment analysis model is based on historical data and should not be used for making actual financial decisions.
