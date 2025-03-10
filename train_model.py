import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from load_data import load_dataset

# Load dataset
df = load_dataset()

# Convert text into numerical format using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['message_clean']).toarray()
y = df['label']

# Split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model & vectorizer
pickle.dump(model, open("spam_classifier.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
