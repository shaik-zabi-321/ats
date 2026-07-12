"""
ATS Resume Analyzer - Backend API (lightweight / free-tier friendly)
----------------------------------------------------------------------
Single endpoint: POST /analyze  (upload a PDF resume, get JSON back)

Pipeline:
  1. Extract text from the uploaded PDF (PyMuPDF)
  2. Clean the text
  3. TF-IDF vectorize resume + a fixed set of role descriptions
  4. Cosine-similarity resume -> each role = semantic score
  5. Keyword/alias-based skill extraction (word-boundary matched) -> skill score per role
  6. Weighted ATS score (0.6 * semantic + 0.4 * skill) per role, sorted best-first
  7. (optional) Rule-based written feedback for the top role - no LLM, so it
     runs comfortably within free-tier hosting memory limits (e.g. Render free plan).

This version intentionally avoids torch / sentence-transformers / transformers.
Those pull in a GB+ of dependencies and need 1GB+ RAM to load, which reliably
gets OOM-killed on free hosting tiers (~512MB). TF-IDF + scikit-learn gets you
the same "does this resume look like this role" comparison in a few MB.

Run:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Then POST a PDF to:  http://localhost:8000/analyze
"""

import re
from contextlib import asynccontextmanager

from pathlib import Path

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STATIC_DIR = Path(__file__).parent / "static"

# ----------------------------------------------------------------------
# Static data: roles, role skill requirements, skill alias dictionary
# (same dataset used in the notebook)
# ----------------------------------------------------------------------

ROLE_DEFINITIONS = [
    {
        "role": "Machine Learning Engineer",
        "description": """
        Build machine learning models using Python,
        TensorFlow, PyTorch, deep learning, neural networks,
        data preprocessing, model deployment and MLOps.
        """,
    },
    {
        "role": "Data Scientist",
        "description": """
        Analyze data, create predictive models,
        statistics, machine learning, Python, SQL,
        data visualization and experimentation.
        """,
    },
    {
        "role": "AI Engineer",
        "description": """
        Develop artificial intelligence systems,
        deep learning, NLP, computer vision,
        generative AI, LLMs and AI applications.
        """,
    },
    {
        "role": "Data Analyst",
        "description": """
        Analyze business data using SQL,
        Python, Excel, statistics, dashboards,
        Power BI and data visualization.
        """,
    },
    {
        "role": "Backend Developer",
        "description": """
        Build backend applications using Python,
        APIs, databases, server architecture,
        Django, FastAPI and software development.
        """,
    },
]

ROLE_REQUIRED_SKILLS = {
    "Machine Learning Engineer": [
        "python", "machine learning", "deep learning",
        "tensorflow", "pytorch", "sql", "docker",
    ],
    "Data Scientist": [
        "python", "machine learning", "statistics", "sql", "data analysis",
    ],
    "AI Engineer": [
        "artificial intelligence", "python", "deep learning",
        "natural language processing", "large language models", "generative ai",
    ],
    "Data Analyst": [
        "python", "sql", "statistics", "data analysis",
    ],
    "Backend Developer": [
        "python", "fastapi", "django", "sql", "git",
    ],
}

SKILL_ALIASES = {
    "machine learning": ["ml", "machine learning", "machine-learning"],
    "artificial intelligence": ["ai", "artificial intelligence"],
    "deep learning": ["dl", "deep learning"],
    "natural language processing": ["nlp", "natural language processing"],
    "large language models": ["llm", "llms", "large language model"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "python": ["python"],
    "sql": ["sql"],
    "docker": ["docker"],
    "aws": ["aws"],
    "fastapi": ["fastapi"],
    "django": ["django"],
    "git": ["git"],
    "github": ["github"],
}

# ----------------------------------------------------------------------
# Model container - built once at startup, reused across requests.
# No heavyweight ML runtime here: just a TF-IDF vectorizer fit on the
# role descriptions, which is a few KB and loads instantly.
# ----------------------------------------------------------------------


class Models:
    vectorizer: TfidfVectorizer | None = None
    role_vectors = None  # sparse matrix, one row per role


models = Models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Fitting TF-IDF vectorizer on role descriptions...")
    descriptions = [r["description"] for r in ROLE_DEFINITIONS]
    models.vectorizer = TfidfVectorizer(stop_words="english")
    models.role_vectors = models.vectorizer.fit_transform(descriptions)
    print("Startup complete. Ready to accept requests.")
    yield
    print("Shutting down.")


app = FastAPI(title="ATS Resume Analyzer", lifespan=lifespan)

# Allow the frontend of your major project to call this API from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Helpers (ported from the notebook)
# ----------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"(?<=\w) (?=\w)", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u2022", " ")
    text = text.replace("\xa0", " ")
    return text.strip()


def extract_skills(text: str, skill_aliases: dict) -> list:
    text = text.lower()
    detected = []
    for main_skill, aliases in skill_aliases.items():
        for alias in aliases:
            # Word-boundary match so short aliases like "ai", "ml", "dl", "tf"
            # don't false-positive inside unrelated words (e.g. "training", "handle").
            pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
            if re.search(pattern, text):
                detected.append(main_skill)
                break
    return detected


def calculate_skill_score(user_skills, required_skills):
    matched = set(user_skills) & set(required_skills)
    missing = set(required_skills) - set(user_skills)
    score = (len(matched) / len(required_skills)) * 100
    return round(score, 2), list(matched), list(missing)


def generate_feedback(ats_result: dict) -> str:
    """
    Rule-based written feedback - no LLM involved. Deliberately templated
    (not invented) so it stays 100% grounded in the scores/skills we actually
    computed, and costs ~0 extra memory/latency to generate.
    """
    role = ats_result["Role"]
    score = ats_result["ATS Score"]
    matched = ats_result["Matched Skills"]
    missing = ats_result["Missing Skills"]

    if score >= 75:
        tier = "a strong match"
    elif score >= 55:
        tier = "a solid, moderate match"
    else:
        tier = "a partial match"

    lines = [
        f"Overall Assessment: Your resume is {tier} for the {role} role, "
        f"with an ATS score of {score}%.",
        "",
    ]

    if matched:
        lines.append(
            "Strengths: Your resume clearly demonstrates "
            + ", ".join(matched) +
            ", which are directly relevant to this role."
        )
    else:
        lines.append(
            "Strengths: No directly matching keywords for this role were detected "
            "in the resume text - consider making relevant skills more explicit."
        )

    lines.append("")

    if missing:
        lines.append(
            "Missing Skill Impact: The role typically also looks for "
            + ", ".join(missing) + ". Their absence may lower how strongly "
            "ATS systems and recruiters match your profile to this role."
        )
        lines.append("")
        lines.append(
            "Improvement Suggestions: Consider adding concrete, truthful experience "
            "with " + ", ".join(missing) + " to your resume - e.g. a project, "
            "coursework, or certification that used them. Keep skill mentions specific "
            "(tools, versions, outcomes) rather than just listing keywords."
        )
    else:
        lines.append(
            "Missing Skill Impact: No major required skills for this role appear to be missing."
        )
        lines.append("")
        lines.append(
            "Improvement Suggestions: Focus on quantifying your impact for the skills "
            "you already have (metrics, scale, outcomes) to strengthen the resume further."
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# API route
# ----------------------------------------------------------------------

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    include_feedback: bool = Query(
        False,
        description="If true, also include a written (rule-based) review for the top role.",
    ),
):
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Please upload a PDF file.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1-2. Extract + clean text
    try:
        raw_text = extract_text_from_pdf(file_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not read PDF: {exc}")

    clean_resume = clean_text(raw_text)
    if not clean_resume:
        raise HTTPException(
            status_code=422, detail="No extractable text found in PDF.")

    # 3-4. TF-IDF vectorize resume + cosine similarity to each role
    resume_vector = models.vectorizer.transform([clean_resume])
    similarities = cosine_similarity(resume_vector, models.role_vectors)[0]

    role_results = [
        {"role": ROLE_DEFINITIONS[i]["role"], "score": round(
            float(similarities[i]) * 100, 2)}
        for i in range(len(ROLE_DEFINITIONS))
    ]

    role_results.sort(key=lambda x: x["score"], reverse=True)

    # 5. Skill extraction
    resume_skills = extract_skills(clean_resume, SKILL_ALIASES)

    # 6. Weighted ATS score per role
    ats_results = []
    for result in role_results:
        role = result["role"]
        semantic_score = min(result["score"] * 1.2, 100)
        skill_score, matched, missing = calculate_skill_score(
            resume_skills, ROLE_REQUIRED_SKILLS[role]
        )
        final_score = (0.6 * semantic_score) + (0.4 * skill_score)
        ats_results.append({
            "Role": role,
            "ATS Score": int(round(final_score, 2)),
            "Matched Skills": matched,
            "Missing Skills": missing,
        })

    ats_results.sort(key=lambda x: x["ATS Score"], reverse=True)

    response = {
        "resume_skills_detected": resume_skills,
        "results": ats_results,
        "top_match": ats_results[0],
    }

    # 7. Optional written feedback for the top match (rule-based, not an LLM)
    if include_feedback:
        response["feedback"] = generate_feedback(ats_results[0])

    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the tester page so visiting the deployed link shows a real UI,
    not a bare JSON 404."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>ATS Resume Analyzer API</h1><p>See /health and /analyze.</p>"
