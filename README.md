# ATS Resume Analyzer — Backend

A minimal FastAPI server that wraps your notebook's ATS pipeline behind one route.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

First startup downloads `BAAI/bge-base-en-v1.5` (~440MB) and computes role
embeddings once — this takes a bit, then the server is ready.

## API

### `POST /analyze`

Upload a PDF resume, get back the ATS analysis as JSON.

**Request:** `multipart/form-data` with a `file` field (the PDF).
Optional query param `include_feedback=true` also generates written LLM
feedback for the top-matched role (loads `microsoft/Phi-3.5-mini-instruct`
on first use — a few GB, slow on CPU, so it's off by default).

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@resume.pdf"

# with LLM feedback included
curl -X POST "http://localhost:8000/analyze?include_feedback=true" \
  -F "file=@resume.pdf"
```

**Response:**

```json
{
  "resume_skills_detected": ["python", "machine learning", "sql", "..."],
  "results": [
    {
      "Role": "Machine Learning Engineer",
      "ATS Score": 69,
      "Matched Skills": ["python", "sql", "machine learning", "..."],
      "Missing Skills": ["docker"]
    },
    { "Role": "Data Scientist", "ATS Score": 58, "...": "..." }
  ],
  "top_match": {
    "Role": "Machine Learning Engineer",
    "ATS Score": 69,
    "Matched Skills": ["..."],
    "Missing Skills": ["docker"]
  },
  "feedback": "Only present if include_feedback=true"
}
```

### `GET /health`

Simple liveness check, returns `{"status": "ok"}`.

## Using it from your major project

Point your frontend/other service at `POST http://<server>:8000/analyze`
with the PDF as form-data field `file`. CORS is open (`*`) for convenience —
tighten `allow_origins` in `main.py` before deploying publicly.

## Notes

- The role list, required skills, and skill-alias dictionary are the same
  fixed dataset from the notebook (5 roles: ML Engineer, Data Scientist,
  AI Engineer, Data Analyst, Backend Developer). Extend `ROLE_DEFINITIONS` /
  `ROLE_REQUIRED_SKILLS` / `SKILL_ALIASES` in `main.py` to add more.
- Models are loaded once at startup (embedder) or lazily on first request
  (feedback LLM), not per-request — keeps things fast after warmup.
- A GPU speeds up both models significantly but isn't required.
