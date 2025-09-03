# training/train_tfidf_logreg.py
# aim:
# simplicity for 2–3 min training on laptop
# saving config + git hash

import os, json, joblib, numpy as np, subprocess, argparse, time
from datasets import load_dataset, DownloadConfig
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

def git_short_sha_or_na():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "na"

def load_amazon_subset(n_train, n_test, seed):
    # making downloads more resilient
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    splits = {"train": f"train[:{n_train}]", "test": f"test[:{n_test}]"}
    ds = load_dataset(
        "amazon_polarity",
        split=splits,
        download_config=DownloadConfig(max_retries=5, resume_download=True)  # removed timeout
    )
    tr, te = ds["train"].shuffle(seed=seed), ds["test"].shuffle(seed=seed)

    def join_text(batch):
        return {"text": [(t or "") + " " + (c or "") for t, c in zip(batch["title"], batch["content"])],
                "label": batch["label"]}

    tr = tr.map(join_text, batched=True, remove_columns=tr.column_names)
    te = te.map(join_text, batched=True, remove_columns=te.column_names)
    return tr["text"], np.array(tr["label"]), te["text"], np.array(te["label"])

def main():
    ap = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression on Amazon Polarity (small subset).")
    ap.add_argument("--n-train", type=int, default=int(os.getenv("N_TRAIN", "20000")))
    ap.add_argument("--n-test", type=int, default=int(os.getenv("N_TEST", "5000")))
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    ap.add_argument("--max-features", type=int, default=int(os.getenv("MAX_FEATURES", "50000")))
    ap.add_argument("--ngram-min", type=int, default=int(os.getenv("NGRAM_MIN", "1")))
    ap.add_argument("--ngram-max", type=int, default=int(os.getenv("NGRAM_MAX", "2")))
    ap.add_argument("--min-df", type=int, default=int(os.getenv("MIN_DF", "2")))
    ap.add_argument("--max-iter", type=int, default=int(os.getenv("MAX_ITER", "2000")))
    ap.add_argument("--artifact-dir", type=str, default=os.getenv("ARTIFACT_DIR", "models/tfidf"))
    ap.add_argument("--metrics-path", type=str, default=os.getenv("METRICS_PATH", "metrics.json"))
    args = ap.parse_args()

    os.makedirs(args.artifact_dir, exist_ok=True)

    print(f"[info] loading data n_train={args.n_train} n_test={args.n_test} seed={args.seed}")
    Xtr, ytr, Xte, yte = load_amazon_subset(args.n_train, args.n_test, args.seed)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(args.ngram_min, args.ngram_max),
            min_df=args.min_df
        )),
        ("clf", LogisticRegression(max_iter=args.max_iter, n_jobs=-1))
    ])

    t0 = time.time()
    print("[info] training…")
    pipe.fit(Xtr, ytr)
    train_secs = time.time() - t0

    print("[info] evaluating…")
    yhat = pipe.predict(Xte)
    acc = accuracy_score(yte, yhat)
    p, r, f1, _ = precision_recall_fscore_support(yte, yhat, average="binary", pos_label=1)

    print("[report]")
    print(classification_report(yte, yhat, digits=3))
    print("confusion_matrix:\n", confusion_matrix(yte, yhat))

    vec_path = os.path.join(args.artifact_dir, "vectorizer.joblib")
    clf_path = os.path.join(args.artifact_dir, "clf.joblib")
    joblib.dump(pipe.named_steps["tfidf"], vec_path, compress=3)
    joblib.dump(pipe.named_steps["clf"],   clf_path, compress=3)

    config = {
        "n_train": args.n_train, "n_test": args.n_test, "seed": args.seed,
        "max_features": args.max_features, "ngram_range": [args.ngram_min, args.ngram_max],
        "min_df": args.min_df, "max_iter": args.max_iter
    }
    with open(os.path.join(args.artifact_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    metrics = {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1),
               "train_secs": round(train_secs, 2)}
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.artifact_dir, "model_version.txt"), "w") as f:
        f.write(git_short_sha_or_na() + "\n")

    print("[done]", {"metrics": metrics, "artifacts": [vec_path, clf_path]})

if __name__ == "__main__":
    main()
