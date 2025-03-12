from flask import Flask, request, jsonify
import pickle
from preprocessing import preprocess_text
import pandas as pd

# Load trained model & vectorizer
model = pickle.load(open("spam_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Load or create feedback dataset
try:
    feedback_data = pd.read_csv("feedback.csv")
except FileNotFoundError:
    feedback_data = pd.DataFrame(columns=["message", "label"])

app = Flask(__name__)

@app.route('/')
def home():
    """Check if the API is running."""
    return jsonify({"message": "Spam Classifier API is live!"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a message is spam or ham."""
    data = request.get_json()
    message = data.get("message", "")

    processed_text = preprocess_text(message)
    vectorized_text = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(vectorized_text)[0]
    
    return jsonify({"message": message, "prediction": "Spam" if prediction == 1 else "Not Spam"})

@app.route('/feedback', methods=['POST'])
def feedback():
    """Store feedback but do not retrain instantly."""
    data = request.get_json()
    message = data.get("message", "")
    correct_label = data.get("label", "").lower()

    if correct_label not in ["ham", "spam"]:
        return jsonify({"error": "Invalid label. Use 'ham' or 'spam'."}), 400

    label = 0 if correct_label == "ham" else 1

    # Save feedback
    global feedback_data
    feedback_data = feedback_data.append({"message": message, "label": label}, ignore_index=True)
    feedback_data.to_csv("feedback.csv", index=False)

    return jsonify({"message": "Feedback received! Model will learn from this later."})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)  # Render requires specifying a port
