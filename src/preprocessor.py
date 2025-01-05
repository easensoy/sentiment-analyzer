from sklearn.feature_extraction.text import TfidfVectorizer
import re

class TextPreprocessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def clean_text(self, text):
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        return text.strip()

    def fit_transform(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.fit_transform(cleaned_texts)

    def transform(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts)
