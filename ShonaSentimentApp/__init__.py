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

import os
import pandas as pd

# set the current working directory to the project root
os.chdir(r'./myenv')

# read the CSV file
shona_to_english_df = pd.read_csv('../ShonaSentimentApp/ShonaToEnglishTranslation.csv', encoding='utf-8')



# Filter for relevant part of speech tags
relevant_tags = ['ADJ', 'JJ', 'VB', 'NN', 'NUM', 'INT', 'DT', 'CD', 'CONJ', 'ADV']
shona_to_english_df = shona_to_english_df[shona_to_english_df['part_of_speech_tag'].isin(relevant_tags)]

# Print the filtered DataFrame
print(shona_to_english_df.head())

# Load JSON files into dictionaries
with open('../ShonaSentimentApp/shona_sentiment_pos_5.json', 'r') as f:
    shona_sentiment_pos_dict = json.load(f)

with open('../ShonaSentimentApp/shona_sentiment_neg_5.json', 'r') as f:
    shona_sentiment_neg_dict = json.load(f)

# Print the dictionaries
print(shona_sentiment_pos_dict)
print(shona_sentiment_neg_dict)

# Load English sentiment scores into a dictionary
english_sentiment_scores = {}

with open('../ShonaSentimentApp/englishSentimentDictionary.txt', 'r') as f:
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

def calculate_shona_sentiment_score(text):
    sentiment_score = 0
    tokens = tokenize_text(text)
    for token in tokens:
        if token in shona_sentiment_pos_dict:
            sentiment_score += shona_sentiment_pos_dict[token]
        elif token in shona_sentiment_neg_dict:
            sentiment_score += shona_sentiment_neg_dict[token]
        elif token in shona_to_english_df['shona'].values:
            english_word = shona_to_english_df.loc[shona_to_english_df['shona']==token, 'english'].values[0]
            if english_word in english_sentiment_scores:
                sentiment_score += english_sentiment_scores[english_word]
        else:
            pass
    return sentiment_score

# Prompt user for input
input_text = input("Enter a piece of Shona text: ")

# Calculate sentiment score
sentiment_score = calculate_shona_sentiment_score(input_text)

def calculate_shona_sentiment_score(text):
    # Tokenize input text
    tokens = nltk.word_tokenize(text)

    # Convert Shona tokens to English
    english_tokens = []
    for token in tokens:
        if token.lower() in shona_to_english_df.index:
            english_tokens.append(shona_to_english_df.loc[token.lower()]['english'])
        else:
            english_tokens.append(token)

    # Calculate sentiment score for English tokens
    sentiment_score = 0
    for token in english_tokens:
        if token in english_sentiment_scores:
            sentiment_score += english_sentiment_scores[token]

    # Calculate sentiment score for Shona tokens
    for token in tokens:
        if token.lower() in shona_sentiment_pos_dict:
            sentiment_score += shona_sentiment_pos_dict[token.lower()]
        elif token.lower() in shona_sentiment_neg_dict:
            sentiment_score += shona_sentiment_neg_dict[token.lower()]

    return sentiment_score

def compound_shona_sentiment_score(text):
    tokens = nltk.word_tokenize(text)
    sentiment_scores = []
    for token in tokens:
        if token in shona_sentiment_pos_dict:
            sentiment_scores.append(shona_sentiment_pos_dict[token])
        elif token in shona_sentiment_neg_dict:
            sentiment_scores.append(shona_sentiment_neg_dict[token])
        elif token in shona_to_english_df['shona'].values:
            english_word = shona_to_english_df.loc[shona_to_english_df['shona']==token, 'english'].values[0]
            if english_word in english_sentiment_scores:
                sentiment_scores.append(english_sentiment_scores[english_word])
        else:
            sentiment_scores.append(0)
    compound_score = sum(sentiment_scores)
    return compound_score

# Prompt user for input
input_text = input("Enter a piece of Shona text: ")

# Calculate sentiment score
sentiment_score = calculate_shona_sentiment_score(input_text)

# Calculate compound sentiment score of each word
compound_score = compound_shona_sentiment_score(input_text)

def compound_shona_sentiment_score(text):
    tokens = nltk.word_tokenize(text)
    sentiment_scores = []
    for token in tokens:
        if token in shona_sentiment_pos_dict:
            sentiment_scores.append(shona_sentiment_pos_dict[token])
        elif token in shona_sentiment_neg_dict:
            sentiment_scores.append(shona_sentiment_neg_dict[token])
        elif token in shona_to_english_df['shona'].values:
            english_word = shona_to_english_df.loc[shona_to_english_df['shona']==token, 'english'].values[0]
            if english_word in english_sentiment_scores:
                sentiment_scores.append(english_sentiment_scores[english_word])
        else:
            sentiment_scores.append(0)
    compound_score = sum(sentiment_scores)
    return compound_score


# Determine sentiment label based on compound score
if compound_score < 0:
    sentiment_label = "Negative"
    color = "red"
elif compound_score == 0:
    sentiment_label = "Neutral"
    color = "grey"
else:
    sentiment_label = "Positive"
    color = "green"

# Print sentiment score and sentiment label with color
print(f"The sentiment score for '{input_text}' is {sentiment_score}.")
print(f"The sentiment label is {sentiment_label}.")

# Output sentiment label with color
from IPython.display import display, HTML
color_code = f"background-color: {color}; padding: 5px; border-radius: 5px; color: white; font-weight: bold;"
html = f"<div style='{color_code}'>{sentiment_label}</div>"
display(HTML(html))
