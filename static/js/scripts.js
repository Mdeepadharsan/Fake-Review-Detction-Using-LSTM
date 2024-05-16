from flask import Flask, render_template, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np

nltk.download('stopwords')

app = Flask(__name__)
model = load_model('fake_review.h5')  # Load the model at the start to save loading time per request.

def preprocess(reviews):
    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    processed_reviews = []
    for review in reviews:
        words = []
        sentences = sent_tokenize(review.lower())
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            filtered_words = [w for w in tokens if w not in stop_words and len(w) > 1]
            words.extend(filtered_words)
        processed_reviews.append(words)
    return processed_reviews

def convert_text_to_no(reviews):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(reviews)
    sequences = tokenizer.texts_to_sequences(reviews)
    maxlen = 100
    return pad_sequences(sequences, maxlen=maxlen)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = preprocess([review])
    sequence_review = convert_text_to_no(processed_review)
    sequence_review = np.array(sequence_review, dtype=np.float32)
    prediction = model.predict(sequence_review)
    result = "Original Review" if prediction[0] >= 0.2 else "Fake Review"
    return jsonify(result=result)

if __name__ == '__main__':
    app.run(debug=True)
