# 📦 Amazon Reviews Sentiment (MLOps Student Project)

![CI](https://github.com/Shyam7773/amazon-sentiment-mlops/actions/workflows/ci.yml/badge.svg)

This repository demonstrates an **end-to-end Machine Learning workflow** built around sentiment analysis on Amazon reviews.  
It was created as a **student project** to practice **MLOps concepts**: training, saving artifacts, serving predictions with an API, building a simple frontend, containerizing with Docker, and setting up CI with GitHub Actions.

---

## ✨ Highlights
- **Model Training**: TF-IDF + Logistic Regression on the Amazon Polarity dataset  
- **Artifact Handling**: Save vectorizer & classifier with joblib  
- **Serving**: Predictions via FastAPI (`/predict` and `/health`)  
- **Frontend**: Minimal dark-themed GUI (HTML/CSS/JS)  
- **Containerization**: Dockerfile to run the app anywhere  
- **Testing**: Pytest suite for artifacts & endpoints  
- **CI Pipeline**: GitHub Actions to train on a small slice & run tests automatically  

---

## 📂 Project Structure

```text
amazon-sentiment-mlops/
├── training/             # Training script
│   └── train_tfidf_logreg.py
├── serving/              # FastAPI app + GUI
│   ├── app.py
│   └── static/
│       ├── styles.css
│       └── app.js
├── models/               # Saved model artifacts
│   ├── vectorizer.joblib
│   ├── clf.joblib
│   ├── config.json
│   └── model_version.txt
├── tests/                # Pytest test files
│   ├── test_artifacts.py
│   ├── test_predict.py
│   └── test_integration.py
├── .github/workflows/    # CI workflow
│   └── ci.yml
├── Dockerfile
├── requirements.txt
├── metrics.json          # Sample metrics from training
└── README.md
```


## 🏋️ Training
Dataset: [Amazon Polarity dataset](https://huggingface.co/datasets/amazon_polarity)  
- Fields: `title`, `content`, and `label` (0 = negative, 1 = positive)

Run training (subset for speed):
```bash
python training/train_tfidf_logreg.py --n-train 20000 --n-test 5000
```
Artifacts and metrics are saved to models/ and metrics.json.

## 🚀 Serving the API
```bash
uvicorn serving.app:app --reload --port 8000
```
Endpoints:

/ → Minimal dark GUI

/health → Service status

/predict → JSON input → sentiment output

Example (curl):
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely loved this product!"}'
```

## 🎨 Web UI

Built with vanilla HTML/CSS/JS

Dark-themed, minimal, easy to use

Shows prediction + positive/negative probabilities

Health panel to check model status



## 🐳 Docker

Build and run:
```bash
docker build -t reviews-api .
docker run -p 8000:8000 reviews-api
```

##CI (Continuous Integration)

Workflow: ci.yml

What it does:

Installs dependencies

Trains a small model (2000/500 split)

Runs tests in tests/

Badge at the top shows the status!

## Future Work

Try transformer models (DistilBERT, etc.)

Deploy to cloud (Render, Railway, Cloud Run)

Add MLflow or experiment tracking

Improve GUI with charts

## About Me

I’m a student learning Data Science & MLOps.
This repo is my way of practicing the full ML pipeline and showing what I can do.
Thanks for checking it out! 🙂
shyam.rathore01@outlook.com
