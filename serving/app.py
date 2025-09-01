# serving/app.py

import os                                 # read env vars, paths
import logging                            # basic logs
from typing import Dict                   # type hint for response
import joblib                             # load sklearn artifacts
from fastapi import FastAPI, Request      # web framework primitives
from fastapi.middleware.cors import CORSMiddleware  # allow local testing (CORS)
from pydantic import BaseModel            # request validation

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
log = logging.getLogger("api")

# Read env with defaults
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "models/tfidf")
VEC_PATH = os.path.join(ARTIFACT_DIR, "vectorizer.joblib")
CLF_PATH = os.path.join(ARTIFACT_DIR, "clf.joblib")

# Create FastAPI app with a title (shows in docs at /docs)
app = FastAPI(title="Amazon Reviews Sentiment API (TF-IDF)")

# Allow local frontend or tools to call the API (adjust origins in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # demo-friendly; restrict in real deployments
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts once at startup (fast requests thereafter)
log.info(f"Loading artifacts: {VEC_PATH}, {CLF_PATH}")
vectorizer = joblib.load(VEC_PATH)
clf = joblib.load(CLF_PATH)
log.info("Artifacts loaded.")

# Validate incoming request body
class Payload(BaseModel):
    text: str  # single review text to classify

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Basic access log for each request (path + method)."""
    log.info(f">>> {request.method} {request.url.path}")
    response = await call_next(request)
    log.info(f"<<< {request.method} {request.url.path} {response.status_code}")
    return response

@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness/readiness probe endpoint."""
    return {"status": "ok", "model": "tfidf-logreg"}

@app.post("/predict")
def predict(p: Payload) -> Dict:
    """Predict sentiment for a single review text."""
    # Transform text using the same TF-IDF vectorizer used at train time
    X = vectorizer.transform([p.text])
    # Predict class probabilities: [neg_prob, pos_prob]
    prob = clf.predict_proba(X)[0].tolist()
    # Convert to label for convenience
    pred = "positive" if prob[1] >= 0.5 else "negative"
    # Return both the label and the raw scores
    return {"prediction": pred, "scores": {"negative": prob[0], "positive": prob[1]}}
