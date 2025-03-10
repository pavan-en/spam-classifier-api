import pandas as pd
from preprocessing import preprocess_text

def load_dataset():
    """Loads the SMS Spam dataset from a given URL and preprocesses it."""
    url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    df = pd.read_csv(url, encoding='latin-1')[['v1', 'v2']]
    
    # Rename columns for clarity
    df.columns = ['label', 'message']
    
    # Convert labels: ham = 0, spam = 1
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Apply text preprocessing
    df['message_clean'] = df['message'].apply(preprocess_text)
    
    return df

if __name__ == "__main__":
    df = load_dataset()
    print(df.head())  # Show first few rows
