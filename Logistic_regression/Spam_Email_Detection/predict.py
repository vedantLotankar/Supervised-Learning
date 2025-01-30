import joblib
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Load the trained model and vectorizer
model_path = os.path.join(script_dir, 'spam_detector.pkl')
vectorizer_path = os.path.join(script_dir, 'vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_spam(email_text):
    email_tfidf = vectorizer.transform([email_text])  # Convert text to TF-IDF features
    prediction = model.predict(email_tfidf)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Example usage
email = "Congratulations! You have been selected as a winner. Reply to this email to claim your prize."
print(predict_spam(email))
