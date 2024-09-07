# review_analyzer.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from text_analyzer import TextAnalyzer

class ReviewAnalyzer:
    def __init__(self, data_file):
        self.data_file = data_file
        self.text_analyzer = TextAnalyzer()
        self.reviews = pd.read_csv(data_file)
        self.keyword_freq = {}

    def load_data(self):
        # Load and clean data
        
        self.reviews['cleaned_text'] = self.reviews['Review'].apply(self.text_analyzer.clean_text)

    def analyze_sentiments(self):
        # Apply sentiment analysis
        self.reviews['sentiment'] = self.reviews['cleaned_text'].apply(self.text_analyzer.get_sentiment)

    def analyze_keywords(self):
        # Extract keywords and their frequencies
        self.keyword_freq = self.text_analyzer.extract_keywords(self.reviews['cleaned_text'])

    def visualize_results(self):
        # Visualize sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.reviews['sentiment'], bins=30)
        plt.title('Sentiment Distribution')
        plt.show()

        # Visualize top keywords
        sorted_keywords = sorted(self.keyword_freq.items(), key=lambda x: x[1], reverse=True)
        top_keywords = dict(sorted_keywords[:10])
        plt.figure(figsize=(10, 6))
        plt.bar(top_keywords.keys(), top_keywords.values())
        plt.title('Top 10 Keywords')
        plt.xticks(rotation=45)
        plt.show()
