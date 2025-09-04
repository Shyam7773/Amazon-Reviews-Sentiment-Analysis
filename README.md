Amazon Reviews Sentiment (Student Project)

This project is an end‑to‑end machine‑learning deployment pipeline built as a student exercise in MLOps. It demonstrates how to train a text classification model on real‐world data, save and version the resulting artifacts, serve predictions through a web API, add a simple dark‑themed GUI, containerise the service with Docker and run automated tests via GitHub Actions. The codebase is deliberately kept clear and annotated so that other learners can follow along.

Project overview
Stage	Description	Key tools/files
Data & training	We use the public Amazon Polarity dataset (about 4 million Amazon product reviews). Each example has a short title, a long review and a label indicating whether the sentiment is positive or negative. We train a simple model using a TF‑IDF vectoriser plus a logistic regression classifier. The training script (training/train_tfidf_logreg.py) downloads a slice of the dataset, joins the title and content, trains the model, evaluates its accuracy and persists the artefacts into the models/ directory.	[training/train_tfidf_logreg.py], Hugging Face datasets library
huggingface.co

Model artefacts	After training, the script stores vectorizer.joblib and clf.joblib in models/tfidf/. It also writes a small model_version.txt so the API can report which version of the model it is serving. You can re‑train with different parameters to produce fresh artefacts.	models/
Serving API	A FastAPI application (serving/app.py) loads the saved artefacts and exposes three routes: GET / returns a minimal dark‑themed web page for interactive testing, POST /predict accepts a JSON body with a review and returns the predicted label and probabilities, and GET /health reports the status and model version.	serving/app.py
Frontend GUI	The app serves static HTML/CSS/JS stored under serving/static/. The page is intentionally dark, with a textarea to type reviews, a Predict button that calls the API, and a health check panel below. This allows anyone to try the model without writing code.	serving/static/
Dockerisation	A Dockerfile builds a slim Python image, copies over the app and model artefacts, installs dependencies and exposes port 8000. Running docker build and docker run launches the API with the GUI.	Dockerfile
Tests & CI	The tests/ directory contains simple Pytest tests that load the artefacts and hit the API. GitHub Actions (.github/workflows/ci.yml) installs dependencies, trains the model on a small subset of data, runs the tests and reports status via the badge above.	tests/, .github/workflows/ci.yml
Quickstart
1. Clone and install
# clone the repo
git clone https://github.com/your‑username/amazon‑sentiment‑mlops.git
cd amazon‑sentiment‑mlops

# create a virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

2. Train the model

Run the training script to build the TF‑IDF vectoriser and logistic regression classifier. By default it trains on 20 000 reviews and tests on 5 000; adjust with --n-train and --n-test flags. Results and metrics will be written to models/ and metrics.json.

python training/train_tfidf_logreg.py --n-train 20000 --n-test 5000

3. Start the API

Use Uvicorn to serve the FastAPI app locally:

uvicorn serving.app:app --reload --port 8000


Open http://127.0.0.1:8000
 in your browser to access the GUI. You can also call the API programmatically:

curl -X POST -H "Content-Type: application/json" \
     -d '{"text": "The battery life on this phone is amazing"}' \
     http://127.0.0.1:8000/predict

4. Run inside Docker

Build and run a containerised version:

# build the image
docker build -t reviews-api .

# run it
docker run -p 8000:8000 reviews-api


The service will be available at http://localhost:8000/.

5. Run tests

Pytest ensures the artefacts exist and the API returns valid responses. Execute the tests with:

pytest -q

CI pipeline (GitHub Actions)

Every push to the main branch triggers an automated workflow defined in .github/workflows/ci.yml. The workflow sets up Python 3.11, installs dependencies, trains a small model on 2 000 training examples and 500 test examples and runs the Pytests. The status badge at the top of this README reflects the outcome. View the pipeline logs in the Actions tab on GitHub.

Project purpose and learnings

This repository started as a student exercise in MLOps. My goals were to practise:

Working with real datasets; the Amazon Polarity dataset from Hugging Face includes millions of reviews with fields for title, review text and a sentiment label
huggingface.co
.

Building a classic text classification model using TF‑IDF and logistic regression.

Saving and versioning model artefacts for repeatable serving.

Developing a lightweight API with FastAPI that can return not only the predicted label but also the probability scores.

Designing a minimal UI so non‑technical users can explore the model output.

Packaging the project in a Docker container for easy deployment.

Implementing a simple CI pipeline with Pytest to give confidence that the code and model artefacts work on each commit.

While the model is simple, the pipeline demonstrates the key steps of a production ML workflow. In the future I would like to explore more sophisticated models (e.g. transformers), track experiments with MLflow and deploy the service to a cloud platform.

License

This project is distributed under the MIT License. Feel free to fork and experiment.
