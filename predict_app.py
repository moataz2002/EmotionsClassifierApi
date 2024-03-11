from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and the CountVectorizer
model = joblib.load('my_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')  # Load the vectorizer

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    text = json_['text']
    # Transform the input text using the loaded vectorizer
    text_counts = vectorizer.transform([text])
    prediction = model.predict(text_counts)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
