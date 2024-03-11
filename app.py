from flask import Flask, render_template, request, jsonify
import joblib
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Load the trained model and the CountVectorizer
model = joblib.load('my_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Load the vectorizer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    # Transform the input text using the loaded vectorizer
    text_counts = vectorizer.transform([text])
    prediction = model.predict(text_counts)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
