import os, json, joblib, numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

OUT_DIR = "models/tfidf"
os.makedirs(OUT_DIR, exist_ok=True)

def load_amazon_dataset(n_tr = 20000, n_te = 5000, seed = 24):
    ds = load_dataset("amazon_polarity")
    tr = ds["train"].shuffle(seed=seed).select(range(n_tr))
    te = ds["test"].shuffle(seed=seed).select(range(n_te))
    def join_text(batch):
        return{"text":[(t or "") + "" + (c or "") for t,c in zip(batch["title"], batch["content"])], "label":batch["label"]}
    tr = tr.map(join_text, batched=True, remove_columns =ds["train"].column_names)
    te = te.map(join_text, batched = True, remove_columns = ds["test"].column_names)
    return tr["text"], np.array(tr["label"]), te["text"], np.array(te["label"])

def main():
    Xtr,ytr,Xte,yte = load_amazon_dataset()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
    ])
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    p,r,f1,_ = precision_recall_fscore_support(yte, yhat, average = "binary", pos_label = 1)

    joblib.dump(pipe.named_steps["tfidf"], f"{OUT_DIR}/vectorizer.joblib")
    joblib.dump(pipe.named_steps["clf"], f"{OUT_DIR}/clf.joblib")

    metrics = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
    with open("metrics.json","w") as f: json.dump(metrics, f, indent=2)
    
    print("Saved TF-IDF model to:", OUT_DIR)
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()