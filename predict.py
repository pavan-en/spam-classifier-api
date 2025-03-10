import pickle
from preprocessing import preprocess_text

# Load trained model and vectorizer
model = pickle.load(open("spam_classifier.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def predict_message(msg):
    """Predicts if a given message is spam or not."""
    msg_clean = preprocess_text(msg)
    msg_vectorized = vectorizer.transform([msg_clean]).toarray()
    prediction = model.predict(msg_vectorized)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example Tests
if __name__ == "__main__":
    print(predict_message("Congratulations! You won a free iPhone! Click here now."))
    print(predict_message("Hey, let's catch up for coffee this evening."))
