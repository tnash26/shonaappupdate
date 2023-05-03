"""
import pandas as pd
import json
import re
import nltk
import csv

# Load CSV file into DataFrame
shona_to_english_df = pd.read_csv('ShonaToEnglishTranslation.csv', encoding='utf-8')

# Filter for relevant part of speech tags
relevant_tags = ['ADJ', 'JJ', 'VB', 'NN', 'NUM', 'INT', 'DT', 'CD', 'CONJ', 'ADV']
shona_to_english_df = shona_to_english_df[shona_to_english_df['part_of_speech_tag'].isin(relevant_tags)]

# Print the filtered DataFrame
print(shona_to_english_df.head())


# Load JSON file into dictionary
with open('shona_sentiment_pos_5.json', 'r') as f:
    shona_sentiment_pos_dict = json.load(f)

# Print the dictionary
print(shona_sentiment_pos_dict)


# Load JSON file into dictionary
with open('shona_sentiment_neg_5.json', 'r') as f:
    shona_sentiment_neg_dict = json.load(f)

# Print the dictionary
print(shona_sentiment_neg_dict)


# AFINN lexicon to assign scores to the words in the englishSentimentDictionary.txt file
english_sentiment_scores = {}

with open('englishSentimentDictionary.txt', 'r') as f:
    for line in f:
        word, score = line.strip().split('\t')
        english_sentiment_scores[word] = int(score)

# Print the dictionary of words and their sentiment scores
print(english_sentiment_scores)


# Text cleaning
def clean_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text



def tokenize_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Split text into tokens
    tokens = text.split()
    return tokens

shona_stopwords = ['kuita', 'ona', 'iyi', 'uyu', 'ose', ' iwe ',' neni', ' iwa ' , ' ayo ' , ' uko ' , ' icho ' , ' ichi ' , 
                   ' ari ' , ' wena ' , ' inga ' ,' nhasi ' , ' uko ' , ' kwavo ' , ' tanga ' , ' tangoti ' , 'kuzonzi' , 
                   'here', 'ita', 'kuti' , 'kudai', 'kana', 'ndiyo', 'ndiye', 'uko', 'kunge', 'kuti', 'chaiko','chete', 
                   'saka' , 'ndi', 'ne', 'yemunhu', 'wangu', 'wako', 'wake', ' vamwe ' , ' avo ' , 'waya', 'vachiri', 
                   'vatiri', 'vamwe', 'vavo', 'nhasi', 'masikati', 'mubvunzo', 'kumashure', 'kumagumo', 'kuchikoro',
                   'kuchauya', 'kwazvo', 'kwese', 'chero', 'chete ', 'ari', 'avo', 'ayo', 'ichi', 'icho', 'inga', 
                   'iwa', 'iwe', 'kwavo', 'neni', 'tanga', 'tangoti', 'wena']

def remove_stop_words(text):
    tokens = text.split()
    clean_tokens = [token for token in tokens if token.lower() not in shona_stopwords]
    clean_text = ' '.join(clean_tokens)
    return clean_text




# Load custom dictionary into DataFrame
shona_to_eng_df = pd.read_csv("ShonaToEnglishTranslation.csv", encoding="utf-8")

# Create dictionary mapping Shona words to their corresponding tags
shona_word_tags = dict(zip(shona_to_eng_df["shona"], shona_to_eng_df["part_of_speech_tag"]))

# Tokenize and tag Shona text
def tag_shona_text(text_to_tag):
    tokens = nltk.word_tokenize(text_to_tag)
    tagged_tokens = []
    for token in tokens:
        if token in shona_word_tags:
            tagged_tokens.append((token, shona_word_tags[token]))
        else:
            eng_translation = shona_to_eng_df.loc[shona_to_eng_df["shona"] == token, "english"].values
            if len(eng_translation) > 0:
                eng_token = eng_translation[0]
                eng_tagged_tokens = nltk.pos_tag(nltk.word_tokenize(eng_token))
                if len(eng_tagged_tokens) > 0:
                    eng_tag = eng_tagged_tokens[0][1]
                    tagged_tokens.append((token, eng_tag))
                else:
                    tagged_tokens.append((token, "UNKNOWN"))
            else:
                tagged_tokens.append((token, "UNKNOWN"))
    return tagged_tokens


# Load Shona-to-English translation dictionary into DataFrame
shona_to_eng_df = pd.read_csv("ShonaToEnglishTranslation.csv", encoding="utf-8")

# Load Shona positive sentiment dictionary
with open("shona_sentiment_pos_5.json", "r") as f:
    shona_sentiment_pos = json.load(f)

# Load Shona negative sentiment dictionary
with open("shona_sentiment_neg_5.json", "r") as f:
    shona_sentiment_neg = json.load(f)

# Load English sentiment dictionary
with open("englishSentimentDictionary.txt", "r") as f:
    english_sentiment_dict = {}
    for line in f:
        word, score = line.strip().split("\t")
        english_sentiment_dict[word] = float(score)

# Function to assign sentiment score to a word based on Shona sentiment dictionaries
def get_shona_sentiment_score(word):
    if word in shona_sentiment_pos:
        return shona_sentiment_pos[word]
    elif word in shona_sentiment_neg:
        return shona_sentiment_neg[word]
    else:
        return None

# Function to assign sentiment score to a word based on English sentiment dictionary
def get_english_sentiment_score(word):
    if word in english_sentiment_dict:
        return english_sentiment_dict[word]
    else:
        return None

import json

def get_sentiment_score(word):
    if not word:
        return 0
    
    # Load Shona sentiment dictionaries
    with open('shona_sentiment_pos_5.json', 'r', encoding='utf-8') as f:
        shona_sentiment_pos = json.load(f)
    with open('shona_sentiment_neg_5.json', 'r', encoding='utf-8') as f:
        shona_sentiment_neg = json.load(f)
    
    # Check if word is in Shona sentiment dictionaries
    if word in shona_sentiment_pos:
        return shona_sentiment_pos[word]
    elif word in shona_sentiment_neg:
        return shona_sentiment_neg[word]
    
    # Load Shona to English dictionary
    with open('ShonaToEnglishTranslation.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    shona_to_english = {}
    for line in lines:
        shona, pos_tag, english = line.strip().split(',')
        shona_to_english[shona] = english
    
    # Check if word is in Shona to English dictionary
    if word in shona_to_english:
        english_word = shona_to_english[word]
        # Load English sentiment dictionary
        with open('englishSentimentDictionary.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        english_sentiment = {}
        for line in lines:
            word, score = line.strip().split('\t')
            english_sentiment[word] = int(score)
        # Check if English word is in English sentiment dictionary
        if english_word in english_sentiment:
            return english_sentiment[english_word]
    
    return 0

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def get_sentiment(text):
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Tokenize text and get sentiment scores
    tokens = nltk.word_tokenize(text)
    scores = sia.polarity_scores(text)

    # Determine sentiment based on compound score
    if scores['compound'] > 5:
        return 'positive'
    elif scores['compound'] < -5:
        return 'negative'
    else:
        return 'neutral'

"""

import pandas as pd
import json
import re
import nltk
import csv

# Load CSV file into DataFrame
shona_to_english_df = pd.read_csv('ShonaToEnglishTranslation.csv', encoding='utf-8')

# Filter for relevant part of speech tags
relevant_tags = ['ADJ', 'JJ', 'VB', 'NN', 'NUM', 'INT', 'DT', 'CD', 'CONJ', 'ADV']
shona_to_english_df = shona_to_english_df[shona_to_english_df['part_of_speech_tag'].isin(relevant_tags)]

# Print the filtered DataFrame
print(shona_to_english_df.head())


# Load JSON file into dictionary
with open('shona_sentiment_pos_5.json', 'r') as f:
    shona_sentiment_pos_dict = json.load(f)

# Print the dictionary
print(shona_sentiment_pos_dict)


# Load JSON file into dictionary
with open('shona_sentiment_neg_5.json', 'r') as f:
    shona_sentiment_neg_dict = json.load(f)

# Print the dictionary
print(shona_sentiment_neg_dict)


# AFINN lexicon to assign scores to the words in the englishSentimentDictionary.txt file
english_sentiment_scores = {}

with open('englishSentimentDictionary.txt', 'r') as f:
    for line in f:
        word, score = line.strip().split('\t')
        english_sentiment_scores[word] = int(score)

# Print the dictionary of words and their sentiment scores
print(english_sentiment_scores)


# Text cleaning
def clean_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text


def tokenize_text(text):
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Split text into tokens
    tokens = text.split()
    return tokens


shona_stopwords = ['kuita', 'ona', 'iyi', 'uyu', 'ose', ' iwe ',' neni', ' iwa ' , ' ayo ' , ' uko ' , ' icho ' , ' ichi ' , 
                   ' ari ' , ' wena ' , ' inga ' ,' nhasi ' , ' uko ' , ' kwavo ' , ' tanga ' , ' tangoti ' , 'kuzonzi' , 
                   'here', 'ita', 'kuti' , 'kudai', 'kana', 'ndiyo', 'ndiye', 'uko', 'kunge', 'kuti', 'chaiko','chete', 
                   'saka' , 'ndi', 'ne', 'yemunhu', 'wangu', 'wako', 'wake', ' vamwe ' , ' avo ' , 'waya', 'vachiri', 
                   'vatiri', 'vamwe', 'vavo', 'nhasi', 'masikati', 'mubvunzo', 'kumashure', 'kumagumo', 'kuchikoro',
                   'kuchauya', 'kwazvo', 'kwese', 'chero', 'chete ', 'ari', 'avo', 'ayo', 'ichi', 'icho', 'inga', 
                   'iwa', 'iwe', 'kwavo', 'neni', 'tanga', 'tangoti', 'wena']

def remove_stop_words(text):
    tokens = text.split()
    clean_tokens = [token for token in tokens if token.lower() not in shona_stopwords]
    clean_text = ' '.join(clean_tokens)
    return clean_text

def calculate_sentiment_score(sentence):
    # Clean and tokenize the sentence
    tokens = tokenize_text(clean_text(sentence))
    
    # Initialize the sentiment score
    sentiment_score = 0
    
    # Iterate over the tokens and add up the sentiment scores
    for token in tokens:
        if token in english_sentiment_scores:
            sentiment_score += english_sentiment_scores[token]
        elif token in shona_sentiment_pos_dict:
            sentiment_score += shona_sentiment_pos_dict[token]
        elif token in shona_sentiment_neg_dict:
            sentiment_score += shona_sentiment_neg_dict[token]
    
    # Return the sentiment score
    return sentiment_score

# Prompt user to enter a piece of Shona text
text = input("Enter a piece of Shona text: ")

# Clean and tokenize the text
cleaned_text = clean_text(text)
tokens = tokenize_text(cleaned_text)

def calculate_shona_sentiment_score(text):
    # Clean and tokenize the text
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    
    # Initialize sentiment score
    sentiment_score = 0
    
    # Iterate over tokens and add up sentiment scores
    for token in tokens:
        # Check if token has a sentiment score in the English lexicon
        if token in english_sentiment_scores:
            # Add English sentiment score to the total sentiment score
            sentiment_score += english_sentiment_scores[token]
        # Check if token has a positive sentiment score in the Shona dictionary
        elif token in shona_sentiment_pos_dict:
            # Add Shona sentiment score to the total sentiment score
            sentiment_score += shona_sentiment_pos_dict[token]
        # Check if token has a negative sentiment score in the Shona dictionary
        elif token in shona_sentiment_neg_dict:
            # Subtract Shona sentiment score from the total sentiment score
            sentiment_score -= shona_sentiment_neg_dict[token]
    
    # Return the final sentiment score
    return sentiment_score

# Prompt user to enter text
text = input("Enter a piece of Shona text: ")

# Calculate sentiment score
sentiment_score = calculate_shona_sentiment_score(text)

# Print sentiment score
print("The sentiment score of the text is:", sentiment_score)

