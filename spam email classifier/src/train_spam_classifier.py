import pandas as pd
import re
import string
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =====================================
# 1. LOAD DATA
# =====================================
DATA_PATH = "data/spam.csv"

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df = df.rename(columns={"v1": "label", "v2": "text"})
df = df[["label", "text"]]


# =====================================
# 2. MID-ADVANCED TEXT CLEANING
# =====================================
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

df["clean_text"] = df["text"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})


# =====================================
# 3. TRAIN-TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_num"],
    test_size=0.2, random_state=42,
    stratify=df["label_num"]
)


# =====================================
# 4. TF-IDF (MID-ADVANCED)
# =====================================
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)


# =====================================
# 5. TRAIN TWO MODELS
# =====================================

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=300)
}

scores = {}

for name, mdl in models.items():
    print(f"\nTraining {name}...")
    mdl.fit(X_train_vec, y_train)
    preds = mdl.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    scores[name] = f1

    print(f"{name} Scores:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

# pick best model
best_model_name = max(scores, key=scores.get)
best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")


# =====================================
# 6. SAVE MODEL + TF-IDF
# =====================================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/spam_mid_advanced_model.joblib")
joblib.dump(tfidf, "models/tfidf_mid_advanced.joblib")

print("\nModel saved successfully!")
