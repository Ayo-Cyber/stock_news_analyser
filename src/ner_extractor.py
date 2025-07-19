import spacy
from spacy import displacy

class NERExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def extract_entities(self, text):
        doc = self.nlp(text)
        return doc

    def get_entities_html(self, doc):
        return displacy.render(doc, style="ent", jupyter=False)