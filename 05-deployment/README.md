# Module 5: Model Deployment (Machine Learning Zoomcamp 2026)

This repository contains the end-to-end model deployment pipeline for Module 5 of the DataTalksClub Machine Learning Zoomcamp (2026 Cohort). It covers serializing model artifacts, building a RESTful inference API using FastAPI, managing isolated reproducible environments with `uv`, and containerizing the microservice using Docker.

---

## 📋 Homework 5 Verified Answers

| Question | Topic | Submission Value | Raw / Verification Details |
|---|---|---|---|
| **Question 1** | Local `uv` check | *(Ungraded setup)* | `uv 0.13.5`[cite: 1] |
| **Question 2** | Pinned Scikit-Learn version | **`1.7.2`** | `scikit-learn==1.7.2` declared in `uv.lock` & `pyproject.toml`[cite: 1] |
| **Question 3** | Lead 1 conversion probability | **`0.533`** | `0.532994` (standalone inference)[cite: 1] |
| **Question 4** | Lead 2 API probability | **`0.770`** | `0.769799` (via local Uvicorn `/predict` endpoint)[cite: 1] |
| **Question 5** | Docker base image tag | **`python:3.11.15-slim-bookworm`** | Target base image in `Dockerfile`[cite: 1] |
| **Question 6** | Lead 2 Container probability | **`0.770`** | `0.769799` (via containerized `/predict` endpoint)[cite: 1] |

---

## 📂 Project Structure

```text
05-deployment/
├── .python-version          # Pinned Python version (3.11.15)[cite: 2]
├── Dockerfile               # Multi-stage container recipe based on python:3.11.15-slim-bookworm[cite: 1, 2]
├── pyproject.toml           # Project dependencies managed via uv[cite: 2]
├── uv.lock                  # Pinned dependency lockfile (scikit-learn 1.7.2)[cite: 1, 2]
├── pipeline.bin             # Serialized DictVectorizer + LogisticRegression pipeline[cite: 1, 2]
├── feature_defaults.json    # Imputation medians and fallback defaults for missing fields[cite: 1, 2]
├── model_metadata.json      # Pipeline metadata, artifact SHA-256 hash, and training schema[cite: 2]
├── model.py                 # Normalization layer handling missing numerical/categorical values[cite: 2]
├── predict.py               # FastAPI application defining /health and /predict endpoints[cite: 2]
├── smoke_test.py            # Local environment startup smoke test[cite: 2]
├── q6_test.py               # Test client for scoring Lead 2 against port 9696[cite: 2]
├── course_lead_scoring_2026.csv # Cohort training dataset[cite: 1, 2]
└── homework5.ipynb          # Exploratory scoring, checksum check, and endpoint testing notebook[cite: 1, 2]
```

---

## 🛡️ Model Integrity & Defensive Normalization

### 1. Checksum Verification
The production model artifact `pipeline.bin` is cryptographically verified to prevent tampering and corruption[cite: 1]:
* **SHA-256 Hash:** `1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`[cite: 1]

### 2. Defensive Imputation (`model.py`)
To prevent inference crashes from partial input dictionaries[cite: 1]:
* **Missing Categoricals:** Automatically assigned to `"NA"` to safely map into unseen/unobserved category bins[cite: 1].
* **Missing Numericals:** Imputed using training-set medians loaded from `feature_defaults.json`[cite: 1]:
  * `annual_income`: `55051.0`[cite: 1]
  * `number_of_courses_viewed`: `2.0`[cite: 1]
  * `interaction_count`: `4.0`[cite: 1]
  * `lead_score`: `0.49`[cite: 1]

---

## 🚀 Reproduction & Setup Guide

### 1. Local Environment Setup with `uv`

Ensure `uv` is installed, then synchronize the pinned virtual environment[cite: 1]:

```bash
# Verify uv installation
uv --version

# Install exact locked dependencies without dev packages
uv sync --locked --no-dev
```

### 2. Running the FastAPI Server Locally

Start the microservice locally using Uvicorn on port `9696`:

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

### 3. Containerizing with Docker

Build and run the containerized service:

```bash
# Build the Docker image
docker build -t zoomcamp-model:2026-hw5 .

# Run the container mapping port 9696
docker run --rm -p 9696:9696 zoomcamp-model:2026-hw5
```

---

## 🧪 Testing & Validation

### Health Check Endpoint (`GET /health`)
Verify model loading and SHA-256 integrity[cite: 1]:

```bash
curl -s http://localhost:9696/health
```

**Response:**
```json
{
  "status": "ok",
  "model_sha256": "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1",
  "feature_defaults_sha256": "1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"
}
```
[cite: 1]

### Lead 2 Inference Endpoint (`POST /predict`)
Send the reference evaluation payload via Python client or curl[cite: 1]:

```bash
python q6_test.py
```

Or via direct HTTP request[cite: 1]:
```bash
curl -s -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74
  }'
```

**Response:**
```json
{
  "conversion_probability": 0.769799,
  "conversion": true
}
```
[cite: 1]
*(Rounds to **`0.770`** for Questions 4 and 6)*[cite: 1]