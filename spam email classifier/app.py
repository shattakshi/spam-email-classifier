import streamlit as st
import joblib
import re
import string

from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

model = joblib.load("models/spam_mid_advanced_model.joblib")
tfidf = joblib.load("models/tfidf_mid_advanced.joblib")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.title("📩 Spam Email Classifier ")

message = st.text_area("Enter message:")
if st.button("Predict"):
    clean = clean_text(message)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    label = "SPAM" if pred == 1 else "NOT SPAM"
    st.write("### Prediction:", label)
