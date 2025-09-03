# tests/test_predict.py
import joblib

def test_basic_prediction():
    vec = joblib.load("models/tfidf/vectorizer.joblib")
    clf = joblib.load("models/tfidf/clf.joblib")

    X = vec.transform(["i love this", "this is awful"])
    preds = clf.predict(X)

    assert preds.shape[0] == 2
    assert all(p in [0, 1] for p in preds)
