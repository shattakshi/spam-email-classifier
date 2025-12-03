import joblib
import re
import string

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

model = joblib.load("models/spam_mid_advanced_model.joblib")
tfidf = joblib.load("models/tfidf_mid_advanced.joblib")

print("\n📩 Spam Classifier Ready!")

while True:
    msg = input("\nEnter message (or 'q' to quit): ")
    if msg.lower() == "q":
        break
    clean = clean_text(msg)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"Prediction: {label}")
