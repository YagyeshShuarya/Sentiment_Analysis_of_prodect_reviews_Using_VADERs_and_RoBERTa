import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

nltk.download('vader_lexicon')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    def polarity_scores_vaders(self, text):
        vader_result = self.sia.polarity_scores(text)
        if vader_result['compound'] >= 0.05:
            return "Positive"
        elif vader_result['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def polarity_scores_roberta(self, text):
        encoded_text = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_text)
        scores = softmax(output.logits.detach().numpy(), axis=1)
        scores_dict = {
            'roberta_neg': scores[0][0],
            'roberta_neu': scores[0][1],
            'roberta_pos': scores[0][2]
        }
        if scores_dict['roberta_pos'] > scores_dict['roberta_neg'] and scores_dict['roberta_pos'] > scores_dict[
            'roberta_neu']:
            return "Positive"
        elif scores_dict['roberta_neg'] > scores_dict['roberta_neu']:
            return "Negative"
        else:
            return "Neutral"

    def analyze_sentiment(self, text):
        vaders_score = self.polarity_scores_vaders(text)
        roberta_score = self.polarity_scores_roberta(text)

        best_model = "VADER" if vaders_score else "RoBERTa"
        best_sentiment = vaders_score if vaders_score else roberta_score

        print('Model used:', best_model)
        print('Sentiment:', best_sentiment)