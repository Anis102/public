# main.py

from review_analyzer import ReviewAnalyzer

if __name__ == "__main__":
    # Initialize and use the ReviewAnalyzer class
    review_analyzer = ReviewAnalyzer('reviews.csv')
    review_analyzer.load_data()
    review_analyzer.analyze_sentiments()
    review_analyzer.analyze_keywords()
    review_analyzer.visualize_results()
