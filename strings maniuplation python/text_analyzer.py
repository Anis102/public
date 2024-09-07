# text_analyzer.py

import re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

class TextAnalyzer:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english')

    def clean_text(self, text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def get_sentiment(self, text):
        return TextBlob(text).sentiment.polarity  # Returns a value between -1 and 1

    def extract_keywords(self, texts):
        X = self.vectorizer.fit_transform(texts)
        keywords = self.vectorizer.get_feature_names_out()
        keyword_counts = X.sum(axis=0).A1
        return dict(zip(keywords, keyword_counts))
