from flask import Flask, render_template, request,jsonify
import ShonaSentimentApp
from ShonaSentimentApp import logistic_regression

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    text = request.form['text']
    score = ShonaSentimentApp.calculate_shona_sentiment_score(text)
    english_sentiment = logistic_regression.CountVectorizer(text)
    return render_template('index.html', sentiment_score=score, english_sentiment= english_sentiment)

if __name__ == '__main__':
    app.run(debug=True)