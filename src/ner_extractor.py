import spacy
from spacy import displacy
# from transformers import pipeline

class NERExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # self.sentiment_classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

    def extract_entities(self, text):
        doc = self.nlp(text)
        return doc

    def is_finance_related(self, text):
        """
        Checks if a headline is finance-related based on a pre-trained model, NER, and keywords.
        """
        # 1. Use FinBERT for classification
        # results = self.sentiment_classifier(text)
        # # FinBERT returns 'positive', 'negative', 'neutral'. We can assume all are finance-related.
        # # We can set a confidence threshold if needed.
        # if results[0]['score'] > 0.6:
        #     return True

        # 2. Check for financial named entities as a fallback
        doc = self.nlp(text)
        financial_ents = {'ORG', 'MONEY', 'PERCENT', 'PRODUCT', 'BUSINESS', 'TRADE', 'GPE', 'NORP'}
        if any(ent.label_ in financial_ents for ent in doc.ents):
            return True

        # 3. Fallback to keyword matching
        financial_keywords = [
            'stock', 'market', 'shares', 'earnings', 'dividend', 'equity',
            'portfolio', 'investor', 'trading', 'nasdaq', 'nyse', 'dow jones',
            'inflation', 'recession', 'bull', 'bear', 'growth', 'loss', 'scale',
            'finance', 'economic', 'corporate', 'investment', 'profit', 'revenue'
        ]
        if any(keyword in text.lower() for keyword in financial_keywords):
            return True

        return False

    def get_entities_html(self, doc):
        return displacy.render(doc, style="ent", jupyter=False)