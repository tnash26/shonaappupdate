import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load Shona stopwords
shona_stopwords = ['kuita', 'ona', 'iyi', 'uyu', 'ose', ' iwe ',' neni', ' iwa ' , ' ayo ' , ' uko ' , ' icho ' , ' ichi ' ,
                   ' ari ' , ' wena ' , ' inga ' ,' nhasi ' , ' uko ' , ' kwavo ' , ' tanga ' , ' tangoti ' , 'kuzonzi' ,
                   'here', 'ita', 'kuti' , 'kudai', 'kana', 'ndiyo', 'ndiye', 'uko', 'kunge', 'kuti', 'chaiko','chete',
                   'saka' , 'ndi', 'ne', 'yemunhu', 'wangu', 'wako', 'wake', ' vamwe ' , ' avo ' , 'waya', 'vachiri',
                   'vatiri', 'vamwe', 'vavo', 'nhasi', 'masikati', 'mubvunzo', 'kumashure', 'kumagumo', 'kuchikoro',
                   'kuchauya', 'kwazvo', 'kwese', 'chero', 'chete ', 'ari', 'avo', 'ayo', 'ichi', 'icho', 'inga',
                   'iwa', 'iwe', 'kwavo', 'neni', 'tanga', 'tangoti', 'wena']

def remove_stop_words(text):
    if isinstance(text, str):
        tokens = text.split()
        clean_tokens = [token for token in tokens if token.lower() not in shona_stopwords]
        clean_text = ' '.join(clean_tokens)
        return clean_text

# Load Shona to English translation data
translation_data = pd.read_csv('C:/Users/TinasheMunyanyiwa/Videos/HIT400/Project/shonapp2-main/shonapp2-main/ShonaSentimentApp/ShonaToEnglishTranslation.csv')


# Load English Sentiment Dictionary from file
english_sentiments = {}
with open('C:/Users/TinasheMunyanyiwa/Videos/HIT400/Project/shonapp2-main/shonapp2-main/ShonaSentimentApp/EnglishSentimentDictionary.txt', 'r') as f:
    for line in f:
        word, sentiment = line.strip().split('\t')
        english_sentiments[word] = int(sentiment)

# Define a function to score a sentence based on English Sentiment Dictionary
def score_sentence(sentence):
    sentiment_score = 0
    for word in sentence.split():
        if word in english_sentiments:
            sentiment_score += english_sentiments[word]
    return sentiment_score

# Preprocess data by removing stop words
translation_data['shona'] = translation_data['shona'].apply(remove_stop_words)
translation_data['english'] = translation_data['english'].apply(remove_stop_words)

# Split dataset into training and testing sets
train_data, test_data = train_test_split(translation_data, test_size=0.2, random_state=42)

# Set up model pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# Train model on training set
model.fit(train_data['english'], train_data['sentiment'])

# Evaluate model on testing set
accuracy = model.score(test_data['english'], test_data['sentiment'])
print('Accuracy:', accuracy)
