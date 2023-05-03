import pandas as pd
import re
from shona_sentiment_neg_5 import shona_sentiment_neg_5
from shona_sentiment_pos_5 import shona_sentiment_pos_5
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define your own list of Shona stop words
stopwords = ['kuita', 'ona', 'iyi', 'uyu', 'ose', 'kuzonzi', 'kwete', 'here', 'ita', 'kuti', 'kudai', 'kana', 'ndiyo', 'ndiye', 'uko', 'kunge', 'kuti', 'chaiko', 'chete', 'chokwadi', 'saka', 'saka', 'ndi', 'ne', 'yemunhu', 'wangu', 'wako', 'wake', 'wedenga', 'wekumusha', 'waya', 'vachiri', 'vatiri', 'vamwe', 'vavo', 'nhasi', 'masikati', 'mubvunzo', 'kumashure', 'kumagumo', 'kuchikoro', 'kufambidzana', 'kuchauya', 'kwakanaka', 'kwazvo', 'kwese', 'chero', 'chete']

# Read in the Shona to English translation dictionary
shona_dict = pd.read_csv('shona_dataset.csv', encoding='utf-8')

# Convert Shona column to lowercase
shona_dict['shona'] = shona_dict['shona'].str.lower()

# Check if the English column contains non-alphabetic characters and remove them if necessary
if not shona_dict['english'].str.isalpha().all():
    shona_dict['english'] = shona_dict['english'].apply(lambda x: re.sub(r'[^a-zA-Z ]+', '', x))

# Create a DataFrame from the shona_dict Series
shona_dict_df = shona_dict[['shona', 'english']].copy()

# Read in the Shona text with sentiment labels
shona_text_sentiment = pd.read_csv('shona_text_with_sentiment_labels.csv', encoding='utf-8')

# Fill in missing English translations using the shona_dict DataFrame
shona_text_sentiment = shona_text_sentiment.merge(shona_dict_df, on='shona', how='left')
shona_text_sentiment['english'] = shona_text_sentiment.apply(lambda x: x['english_y'] if pd.isna(x['english_x']) else x['english_x'], axis=1)
shona_text_sentiment.drop(['english_x', 'english_y'], axis=1, inplace=True)

# Preprocess the Shona text by removing stop words and tokenizing
shona_text_sentiment['shona_processed'] = shona_text_sentiment['shona'].apply(lambda x: [word for word in word_tokenize(x) if word.lower() not in stopwords])

# Convert the Shona text to lowercase
shona_text_sentiment['shona_processed'] = shona_text_sentiment['shona_processed'].apply(lambda x: [word.lower() for word in x])

# Join the processed Shona text back into a single string
shona_text_sentiment['shona_processed'] = shona_text_sentiment['shona_processed'].apply(lambda x: ' '.join(x))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(shona_text_sentiment['shona_processed'], shona_text_sentiment['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data using the TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a logistic regression model on the vectorized training data
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Use the trained model to make predictions on the vectorized test data
y_pred = clf.predict(X_test_vec)

# Evaluate the performance of the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
confusion_mat = confusion_matrix(y_test, y_pred)

# Print out the results
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
print('Confusion Matrix:\n', confusion_mat)

# Define a function to classify a single sentence
def classify_sentence(sentence):
    # Preprocess the Shona text by removing stop words and tokenizing
    sentence_processed = [word for word in word_tokenize(sentence) if word.lower() not in stopwords]

    # Convert the Shona text to lowercase
    sentence_processed = [word.lower() for word in sentence_processed]

    # Join the processed Shona text back into a single string
    sentence_processed = ' '.join(sentence_processed)

    # Vectorize the processed text using the TfidfVectorizer
    sentence_vec = vectorizer.transform([sentence_processed])

    # Use the trained model to predict the sentiment of the sentence
    sentiment = clf.predict(sentence_vec)

    # Return the predicted sentiment
    return sentiment[0]
