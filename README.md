<h1 align="center">📩 Spam Email Classifier</h1> <p align="center"> A clean, structured, and interview-ready NLP + Machine Learning project that classifies SMS/Emails as <b>Spam</b> or <b>Not Spam</b>. </p>
🌟 1. Overview

This project is a mid-advanced, beginner-friendly Machine Learning solution built to detect spam messages using:

Natural Language Processing (NLP)

TF-IDF Vectorization

Logistic Regression & Naive Bayes

Streamlit Web App

The goal was to create a project that is:
✔ Professional
✔ Explainable in interviews
✔ Neat & structured
✔ Believable for a fresher
✔ Strong enough to add to a portfolio

📁 2. Project Structure
spam-email-classifier/
│
├── data/
│   └── spam.csv
│
├── models/
│   ├── spam_mid_advanced_model.joblib
│   └── tfidf_mid_advanced.joblib
│
├── src/
│   ├── train_mid_advanced.py
│   ├── predict_mid_advanced.py
│   └── __init__.py
│
├── app.py
└── README.md


✔ Clean
✔ Logical
✔ Industry-style project layout

🧠 3. Features
🔹 NLP Preprocessing

Lowercasing

Removing URLs

Removing digits

Removing punctuation

Stopword removal

🔹 Vectorization

TF–IDF

1–2 gram features

5000 vocabulary size

min_df=2

🔹 Models Trained

Multinomial Naive Bayes

Logistic Regression (Winner)

🔹 Additional Highlights

Model comparison (F1-score)

Saved model + vectorizer

Real-time prediction script

Web interface using Streamlit

📊 4. Model Performance
Metric	Score
Accuracy	96.86%
Precision	97.50%
Recall	78.52%
F1 Score	86.99%
Best Model	Logistic Regression

Balanced performance with strong precision.

⚙️ 5. Installation
Install requirements:
pip install -r requirements.txt

Download NLTK stopwords:
python
>>> import nltk
>>> nltk.download("stopwords")
>>> exit()

🏋️ 6. Train the Model

Run:

python src/train_mid_advanced.py


This script will:

Clean text

Vectorize data with TF-IDF

Train 2 ML models

Compare F1-scores

Save the best model + vectorizer

🔍 7. Make Predictions

Run:

python src/predict_mid_advanced.py


Example:

Enter message: You won a free prize!!!
Prediction: SPAM

🌐 8. Streamlit App (UI)

Launch the app:

streamlit run app.py


Provides a simple, user-friendly interface for testing messages.

🛠 9. Tech Stack

Python

Pandas

Scikit-learn

NLTK

TF–IDF

Logistic Regression

Naive Bayes

Streamlit

Joblib

🎯 10. What I Learned

How preprocessing impacts ML performance

Why Logistic Regression performs strongly in text classification

Best practices in structuring ML projects

Saving & loading ML pipelines

Building minimal ML web apps

Model evaluation (Precision/Recall/F1)

🪪 11. License

This project is licensed under the MIT License.

✨ 12. Contact

Shatakshi Tiwari
📩 Open for AI/ML internship & beginner roles
🔗 Connect with me on LinkedIn
