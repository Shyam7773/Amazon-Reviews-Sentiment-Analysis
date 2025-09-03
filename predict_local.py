# predict_local.py
import sys, joblib

vec = joblib.load("models/tfidf/vectorizer.joblib")
clf = joblib.load("models/tfidf/clf.joblib")

text = " ".join(sys.argv[1:]) or "This is great."
X = vec.transform([text])
proba = clf.predict_proba(X)[0]
pred = "positive" if proba[1] >= 0.5 else "negative"
print({"text": text, "prediction": pred, "scores": {"neg": float(proba[0]), "pos": float(proba[1])}})
