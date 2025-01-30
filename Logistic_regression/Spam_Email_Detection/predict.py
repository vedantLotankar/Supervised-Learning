import re
import os
import nltk
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Load the trained model and vectorizer
model_path = os.path.join(script_dir, 'spam_detector.pkl')
vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Tokenization
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Join words back into a sentence
    return ' '.join(words)


def predict_spam(email_text):
    email_tfidf = vectorizer.transform([email_text])  # Convert text to TF-IDF features
    prediction = model.predict(email_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
email = "Congratulations! You have been selected as a winner. Reply to this email to claim your prize."
processed_email = preprocess_text(email)
print(predict_spam(processed_email))
