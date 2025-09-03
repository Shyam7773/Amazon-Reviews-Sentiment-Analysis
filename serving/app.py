# serving/app.py


import os
import joblib
import time
from fastapi import FastAPI, Request
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# loading the trained stuff
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "models/tfidf")
VEC_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.joblib")
CLF_PATH = os.path.join(ARTIFACT_DIR, "clf.joblib")
VER_PATH = os.path.join(ARTIFACT_DIR, "model_version.txt")

print("loading vectorizer + classifier...")
vectorizer = joblib.load(VEC_PATH)
clf = joblib.load(CLF_PATH)
if os.path.exists(VER_PATH):
    model_version = open(VER_PATH).read().strip()
else:
    model_version = "na"
print("loaded! (version:", model_version, ")")

# fastapi app
app = FastAPI(
    title="Amazon Reviews Sentiment API (student build)",
    description="Send in a review → get Positive/Negative prediction back 🙂",
    version="0.1"
)

# allow calls from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# just keeping schemas minimal
class Payload(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: str
    scores: dict = {}
    version: str = "na"

# static files (css + js)
here = Path(__file__).resolve().parent               
static_dir = here / "static"                         
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/static", StaticFiles(directory="serving/static"), name="static")

# my own small homepage
INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Amazon Reviews Sentiment Demo</title>
  <link rel="stylesheet" href="/static/styles.css"/>
</head>
<body>
  <main class="container">
    <h1>Amazon Reviews Sentiment Analysis</h1>
    <p>Type a review below to see if it's positive or negative. 👍👎</p>

    <section class="card">
      <h2>Try it out</h2>
      <form id="predict-form">
        <textarea id="input-text" rows="4">I absolutely loved this product!</textarea>
        <button type="submit">Predict</button>
      </form>
      <div id="result" hidden>
        <p><b>Prediction:</b> <span id="prediction"></span></p>
        <p><b>Scores:</b> <span id="scores"></span></p>
      </div>
    </section>

    <section class="card">
      <h2>Health</h2>
      <button id="health-btn">Check Health</button>
      <div id="health"></div>
    </section>
  </main>
  <script src="/static/app.js"></script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return INDEX_HTML

# health check
@app.get("/health")
def health():
    return {"status": "ok", "model": "tfidf-logreg", "version": model_version}

# predicting the endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(p: Payload):
    X = vectorizer.transform([p.text])
    prob = clf.predict_proba(X)[0].tolist()
    pred = "positive" if prob[1] >= 0.5 else "negative"
    return {
        "prediction": pred,
        "scores": {"negative": prob[0], "positive": prob[1]},
        "version": model_version
    }
