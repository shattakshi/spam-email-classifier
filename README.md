<h1 align="center">📩 Spam Email Classifier — NLP + Machine Learning</h1> <p align="center"> A mid-advanced, beginner-friendly Machine Learning project that classifies emails/messages as <b>Spam</b> or <b>Not Spam</b> using classical NLP techniques and TF-IDF based text vectorization. </p>
🚀 Project Overview

This project is an end-to-end implementation of a Spam Email Classification system using Python, NLP, and Machine Learning.

It covers:

Text cleaning & preprocessing

Feature extraction using TF-IDF

Model training & comparison

Evaluation using Precision, Recall, F1-Score

Saving/loading ML models

A simple Streamlit UI for real-time predictions

The project is intentionally built to be advanced enough to impress, yet realistic for a fresher—so you can explain every part confidently in an interview.

📂 Folder Structure
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

🧠 Features
📝 Natural Language Processing

Lowercasing

Regex-based cleaning

Stopword removal (NLTK)

Punctuation removal

🔡 Feature Engineering

TF-IDF vectorization

1–2 gram features

max_features=5000

min_df=2

🤖 Machine Learning Models

Trained & compared:

Model	Why It's Used
Multinomial Naive Bayes	Strong baseline for text problems
Logistic Regression	High precision & strong binary classifier

➡ Logistic Regression achieved the best F1-Score.

📊 Model Performance
Metric	Score
Accuracy	96.86%
Precision	97.50%
Recall	78.52%
F1 Score	86.99%
Selected Model	Logistic Regression

The model strikes a balance between precision (avoiding false positives) and recall.

⚙️ Installation & Setup
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Download NLTK stopwords
python
>>> import nltk
>>> nltk.download("stopwords")
>>> exit()

🏋️ Train the Model

Run:

python src/train_mid_advanced.py


This will:

Clean the dataset

Train 2 ML models

Compare results

Pick the best model

Save model + TF-IDF vectorizer

Outputs are stored in:

/models/

🔍 Run Predictions
python src/predict_mid_advanced.py


Example:

Enter message: You have won a free prize!!
Prediction: SPAM

🌐 Streamlit Web App

Start the UI:

streamlit run app.py


A simple web interface opens where users can:

Enter a message

View classification result

See whether it is Spam or Not Spam

🛠 Tech Stack

Python

Pandas

scikit-learn

NLTK

TF-IDF

Logistic Regression

Streamlit

Joblib

Regex

📘 Key Learnings

Through this project, I learned:

How NLP-based preprocessing improves ML accuracy

Difference between Naive Bayes & Logistic Regression

Why TF-IDF is effective for text classification

How to structure ML projects professionally

How to save and reuse ML pipelines

Basics of building ML-powered web apps

🪪 License

This project is open-source and available under the MIT License.
You are free to use, modify, and distribute it.

✨ Contact

Shatakshi Tiwari
📧 Open to AI/ML projects & collaborations
🔗 Connect on LinkedIn!
