from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load trained model, vectorizer, and label encoder
mlp_model = joblib.load("mlp_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data["news"]

        # Transform text into vectorized form
        text_vectorized = vectorizer.transform([news_text])

        # Predict category
        prediction = mlp_model.predict(text_vectorized)
        category = label_encoder.inverse_transform(prediction)[0]

        return jsonify({"category": category})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
