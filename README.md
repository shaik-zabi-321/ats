# ATS Resume Analyzer — Backend

A minimal FastAPI server that wraps your notebook's ATS pipeline behind one route.
This version is **lightweight and free-tier friendly**: no `torch`, no
`transformers`, no `sentence-transformers` — just `scikit-learn` TF-IDF for
semantic matching and a rule-based (non-LLM) feedback generator. Boots in
seconds and comfortably fits ~512MB RAM hosts like Render's free tier.

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

No model downloads — starts up almost instantly.

## API

### `POST /analyze`

Upload a PDF resume, get back the ATS analysis as JSON.

**Request:** `multipart/form-data` with a `file` field (the PDF).
Optional query param `include_feedback=true` also includes a written,
rule-based review for the top-matched role.

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@resume.pdf"

# with written feedback included
curl -X POST "http://localhost:8000/analyze?include_feedback=true" \
  -F "file=@resume.pdf"
```

**Response:**

```json
{
  "resume_skills_detected": ["python", "sql", "git", "..."],
  "results": [
    {
      "Role": "Backend Developer",
      "ATS Score": 62,
      "Matched Skills": ["sql", "git", "python"],
      "Missing Skills": ["django", "fastapi"]
    },
    { "Role": "Data Analyst", "ATS Score": 35, "...": "..." }
  ],
  "top_match": {
    "Role": "Backend Developer",
    "ATS Score": 62,
    "Matched Skills": ["..."],
    "Missing Skills": ["django", "fastapi"]
  },
  "feedback": "Only present if include_feedback=true"
}
```

### `GET /health`

Simple liveness check, returns `{"status": "ok"}`.

## Deploying (e.g. Render free tier)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

Because there's no heavy ML runtime, this fits comfortably in free-tier RAM
limits and starts up in seconds instead of minutes.

## Using it from your website

Point your frontend at `POST https://<your-server>/analyze` with the PDF as
form-data field `file`. CORS is open (`*`) for convenience — tighten
`allow_origins` in `main.py` before a real production launch.

## Notes / how this differs from the original notebook

- **Semantic matching:** the notebook used `sentence-transformers`
  (`BAAI/bge-base-en-v1.5`) embeddings + cosine similarity. This version uses
  TF-IDF + cosine similarity instead — same idea (compare resume text to role
  descriptions), much lighter weight, no GPU/large-model dependency.
- **Feedback:** the notebook used `microsoft/Phi-3.5-mini-instruct` (a ~3.8B
  parameter LLM) to write the review. No free hosting tier realistically runs
  that, so this version generates the review from a template driven by the
  actual computed score/matched/missing skills — still grounded, still useful,
  just not LLM-generated prose.
- **Skill extraction bug fix:** the original alias matching used plain
  substring checks (`"ai" in text`), which false-matched inside unrelated
  words like "tr**ai**ning" or "han**dl**e". This version uses word-boundary
  regex matching so only real, standalone mentions of a skill count.
- The role list, required skills, and skill-alias dictionary are still the
  same fixed dataset (5 roles: ML Engineer, Data Scientist, AI Engineer, Data
  Analyst, Backend Developer). Extend `ROLE_DEFINITIONS` / `ROLE_REQUIRED_SKILLS`
  / `SKILL_ALIASES` in `main.py` to add more.

## Want the LLM feedback back?

If you later move to a paid tier (or run this locally for demos), you can
swap `generate_feedback()` back to an LLM-based version — either a locally
loaded Hugging Face pipeline (as in the notebook) or a call to a hosted API
(e.g. Anthropic's API) so you don't need to load model weights yourself at all.
Ask if you'd like that version.