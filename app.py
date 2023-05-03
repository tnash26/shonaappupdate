from flask import Flask, render_template, request,jsonify
import ShonaSentimentApp


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    text = request.form['text']
    score = ShonaSentimentApp.calculate_shona_sentiment_score(text)
    return render_template('index.html', sentiment_score=score)

if __name__ == '__main__':
    app.run(debug=True)