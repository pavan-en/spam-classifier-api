import nltk
import re
from nltk.corpus import stopwords

# Download stopwords if not present
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Cleans and preprocesses the text by removing stopwords and special characters."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
