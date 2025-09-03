# tests/test_artifacts.py
import os
import joblib

def test_artifacts_exist_and_load():
    vec_path = "models/tfidf/vectorizer.joblib"
    clf_path = "models/tfidf/clf.joblib"

    assert os.path.exists(vec_path), "Vectorizer is missing"
    assert os.path.exists(clf_path), "Classifier is missing"

    vec = joblib.load(vec_path)
    clf = joblib.load(clf_path)

    assert vec is not None
    assert clf is not None
