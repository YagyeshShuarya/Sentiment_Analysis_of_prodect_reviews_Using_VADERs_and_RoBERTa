import joblib
from Analysis import SentimentAnalyzer
analyzer = SentimentAnalyzer()
joblib.dump(analyzer, 'sentiment_analyzer.joblib')
analyzer = joblib.load('sentiment_analyzer.joblib')

def produce():
    review = input("Enter the comment(Product Review): ",)
    analyzer.analyze_sentiment(review)

produce()
