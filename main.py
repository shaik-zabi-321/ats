"""
ATS Resume Analyzer - Backend API
----------------------------------
Single endpoint: POST /analyze  (upload a PDF resume, get JSON back)

Pipeline (ported straight from the notebook):
  1. Extract text from the uploaded PDF (PyMuPDF)
  2. Clean the text
  3. Embed resume + a fixed set of role descriptions (SentenceTransformer bge-base-en-v1.5)
  4. Cosine-similarity resume -> each role = semantic score
  5. Keyword/alias-based skill extraction -> skill score per role
  6. Weighted ATS score (0.6 * semantic + 0.4 * skill) per role, sorted best-first
  7. (optional) LLM-generated written feedback for the top role (Phi-3.5-mini-instruct)

Run:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Then POST a PDF to:  http://localhost:8000/analyze
"""

import io
import re
from contextlib import asynccontextmanager

import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
# Model container - loaded once at startup, reused across requests
# ----------------------------------------------------------------------


class Models:
    embedder: SentenceTransformer | None = None
    role_embeddings: list | None = None
    feedback_pipeline = None  # loaded lazily, only if feedback is requested


models = Models()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading sentence embedding model (BAAI/bge-base-en-v1.5)...")
    models.embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    print("Pre-computing role description embeddings...")
    models.role_embeddings = [
        {"role": r["role"], "embedding": models.embedder.encode(
            r["description"])}
        for r in ROLE_DEFINITIONS
    ]
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


def get_feedback_pipeline():
    """Lazily load the (large) feedback LLM only on first use."""
    if models.feedback_pipeline is None:
        from transformers import pipeline
        print("Loading feedback LLM (microsoft/Phi-3.5-mini-instruct)... this may take a while.")
        models.feedback_pipeline = pipeline(
            "text-generation",
            model="microsoft/Phi-3.5-mini-instruct",
            max_new_tokens=300,
            device_map="auto",
        )
    return models.feedback_pipeline


def generate_feedback(ats_result: dict) -> str:
    prompt = f"""
You are an expert ATS resume reviewer.

Analyze this candidate result.

Recommended Role:
{ats_result['Role']}

ATS Score:
{ats_result['ATS Score']}%

Matched Skills:
{', '.join(ats_result['Matched Skills'])}

Missing Skills:
{', '.join(ats_result['Missing Skills'])}

Generate a professional resume review with:

1. Overall assessment
2. Candidate strengths
3. Missing skill impact
4. Improvement suggestions

Rules:
- Do not invent skills or experience.
- Only use information provided.
- Keep the response under 250 words.
- Give actionable suggestions.

Keep it concise and useful for a job seeker.
"""
    pipe = get_feedback_pipeline()
    response = pipe(prompt, max_new_tokens=300, return_full_text=False)
    return response[0]["generated_text"].strip()


# ----------------------------------------------------------------------
# API route
# ----------------------------------------------------------------------

@app.post("/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    include_feedback: bool = Query(
        False,
        description="If true, also generate written LLM feedback for the top role "
                    "(slower - loads Phi-3.5-mini-instruct on first call).",
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

    # 3-4. Resume embedding + semantic similarity to each role
    resume_embedding = models.embedder.encode(clean_resume)

    role_results = []
    for role in models.role_embeddings:
        similarity = cosine_similarity(
            [resume_embedding], [role["embedding"]])[0][0]
        role_results.append(
            {"role": role["role"], "score": round(similarity * 100, 2)})

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

    # 7. Optional LLM-written feedback for the top match
    if include_feedback:
        response["feedback"] = generate_feedback(ats_results[0])

    return response


@app.get("/health")
async def health():
    return {"status": "ok"}
