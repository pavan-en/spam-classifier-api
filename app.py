from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict_message

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to classify SMS messages as spam or not spam."""
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    prediction = predict_message(message)
    return jsonify({"message": message, "prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
