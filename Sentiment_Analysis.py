import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from scipy.special import softmax
import joblib

df = pd.read_csv('Dataset-SA.csv')

df.head()

df.info()

df.insert(0, 'ID', range(1, len(df) + 1))

df.info()

df.isnull().sum()

df.dropna(inplace = True)

df.isnull().sum()

df.head()

df.describe()

# Distribution of the Sentiment column
df['Sentiment'].value_counts()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette=['green', 'red', 'blue'], legend=False)
plt.title('Sentiment Distribution')
plt.show()

# Analyzing the relationship between Rate and Sentiment
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Rate', hue='Sentiment',palette=['red', 'green', 'blue'])
plt.title('Sentiment Distribution by Rate')
plt.show()

df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df = df.dropna(subset=['Rate'])

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Rate', hue='Sentiment',palette=['red', 'green', 'blue'])
plt.title('Sentiment Distribution by Rate')
plt.show()

df['Summary'] = df['Summary'].fillna('').astype(str)
df = df.head(500)

# VADERS
example = df['Summary'][50]
print(example)

tokens = nltk.word_tokenize(example)
tokens[:10]

tagged = nltk.pos_tag(tokens)
tagged[:10]

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

sia.polarity_scores('many issues')
sia.polarity_scores('good')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)

res = {}
for i, row in df.iterrows():
    text = row['Summary']
    myid = row['ID']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'ID'})
vaders = vaders.merge(df, how='left')

vaders.head()

ax = sns.barplot(data=vaders, x='Rate', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Rate', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Rate', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Rate', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

def polarity_scores_vaders(text):
    vader_result = sia.polarity_scores(text)
    if vader_result['compound'] >= 0.05:
        return "Positive"
    elif vader_result['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

polarity_scores_vaders(example)

# RoBERTa Pretrained Model
tokenizer = RobertaTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
model = RobertaForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

print(example)
sia.polarity_scores(example)

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(example,'\n')
print(scores_dict)


def polarity_scores_roberta_batch(reviews):
    encoded_batch = tokenizer(reviews, return_tensors='pt', truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        output = model(**encoded_batch)

    scores = output.logits.detach().numpy()
    scores = softmax(scores, axis=1)

    results = []
    for score in scores:
        score = softmax(score)
        if score[2] > score[0] and score[2] > score[1]:
            results.append("positive")
        elif score[0] > score[1] and score[0] > score[2]:
            results.append("negative")
        else:
            results.append("neutral")
    return results[0]

polarity_scores_roberta_batch('Good')

res = {}
for i, row in df.iterrows():
    try:
        text = row['Summary']
        myid = row['ID']

        vader_result = polarity_scores_vaders(text)
        roberta_sentiment = polarity_scores_roberta_batch(text)

        both = {'vader sentiment': vader_result, 'roberta_sentiment': roberta_sentiment}
        res[myid] = both

    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'ID'})
results_df = results_df.merge(df, how='left')

results_df.head()

results_df.columns

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

#RoBERTa Sentiment by Rate
sns.countplot(data=results_df, x='Rate', hue='roberta_sentiment', palette=['red', 'blue', 'green'], ax=axs[0])
axs[0].set_title('RoBERTa Sentiment by Rate')
axs[0].legend(title='RoBERTa Sentiment')

#VADER Sentiment by Rate
sns.countplot(data=results_df, x='Rate', hue='vader sentiment', palette=['red', 'green', 'blue'], ax=axs[1])
axs[1].set_title('VADER Sentiment by Rate')
axs[1].legend(title='VADER Sentiment')

#True Sentiment by Rate
sns.countplot(data=results_df, x='Rate', hue='Sentiment', palette=['red', 'green', 'blue'], ax=axs[2])
axs[2].set_title('True Sentiment by Rate')
axs[2].legend(title='True Sentiment')

plt.tight_layout()
plt.show()

print(results_df.query('Rate == 1').sort_values('roberta_sentiment', ascending=False)['Summary'].values[0])
print(results_df.query('Rate == 1').sort_values('vader sentiment', ascending=False)['Summary'].values[0])
print(results_df.query('Rate == 5').sort_values('roberta_sentiment', ascending=False)['Summary'].values[0])
print(results_df.query('Rate == 5').sort_values('vader sentiment', ascending=False)['Summary'].values[0])

df.head()
#df = pd.read_csv('Dataset-SA.csv')
#df.insert(0, 'ID', range(1, len(df) + 1))
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df = df.dropna(subset=['Rate'])

test_data = df
test_inputs = test_data['Summary'].tolist()
true_sentiments = test_data['Sentiment'].tolist()

predicted_sentiment_roberta = []
predicted_sentiment_vaders = []

for review in test_inputs:
    if isinstance(review, str):  # Check if review is a string
        sentiment_roberta = polarity_scores_roberta_batch(review)
        sentiment_vaders = polarity_scores_vaders(review)
        predicted_sentiment_roberta.append(sentiment_roberta)
        predicted_sentiment_vaders.append(sentiment_vaders)
    else:
        predicted_sentiment_roberta.append("Invalid")
        predicted_sentiment_vaders.append("Invalid")

true_sentiments_lower = [sentiment.lower() for sentiment in true_sentiments]
predicted_sentiment_roberta_lower = [sentiment.lower() for sentiment in predicted_sentiment_roberta]
predicted_sentiment_vaders_lower = [sentiment.lower() for sentiment in predicted_sentiment_vaders]

vaders_accuracy = accuracy_score(true_sentiments_lower, predicted_sentiment_vaders_lower)
roberta_accuracy = accuracy_score(true_sentiments_lower, predicted_sentiment_roberta_lower)

print(f"RoBERTa Accuracy: {roberta_accuracy * 100:.2f}%")
print(f"VADER Accuracy: {vaders_accuracy * 100:.2f}%")


def Analyze_Sentiment():
    review = input("Enter the comment(Product Review): ", )
    if roberta_accuracy > vaders_accuracy:
        best_model = "RoBERTa"
        best_function = polarity_scores_roberta
    else:
        best_model = "VADER"
        best_function = polarity_scores_vaders

    sentiment = best_function(review)
    print('Model used:', best_model)
    print('Sentiment:', sentiment)

#Analyze_Sentiment()


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

analyzer = SentimentAnalyzer()
joblib.dump(analyzer, 'sentiment_analyzer.joblib')
print("Analyze_Sentiment function and dependencies saved to sentiment_analyzer.joblib")


analyzer = joblib.load('sentiment_analyzer.joblib')

def produce():
    review = input("Enter the comment(Product Review): ",)
    analyzer.analyze_sentiment(review)

produce()