from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import json
import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None

try:
    import PyPDF2
except Exception:  # pragma: no cover
    PyPDF2 = None

try:
    import docx
except Exception:  # pragma: no cover
    docx = None

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
METRICS_PATH = "model/metrics.json"

DEFAULT_JOB_DESC = (
    "Backend Developer with Java, Spring Boot, REST APIs, and SQL. "
    "Experience with Git and basic system design preferred."
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

embedder = None
if SentenceTransformer is not None:
    try:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception:
        embedder = None

nlp = None
if spacy is not None:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

ALLOWED_EXTS = {".txt", ".pdf", ".docx"}

SKILL_LIST = [
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "flask",
    "django",
    "fastapi",
    "react",
    "node",
    "spring",
    "spring boot",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "nlp",
    "computer vision",
    "spark",
    "hadoop",
    "git",
    "linux",
    "rest",
    "rest api",
    "graphql",
    "html",
    "css",
    ".net",
    "devops",
    "selenium",
    "jira",
    "power bi",
    "tableau",
    "excel",
    "agile",
]

_SKILL_PATTERNS = []
for skill in SKILL_LIST:
    escaped = re.escape(skill)
    if re.search(r"\w", skill):
        pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
    else:
        pattern = re.compile(escaped, re.IGNORECASE)
    _SKILL_PATTERNS.append((skill, pattern))

ROLE_PROFILES = {
    "Sales Executive": {
        "triggers": [
            "sales executive",
            "sales",
            "business development",
            "account executive",
            "inside sales",
        ],
        "keywords": [
            "sales",
            "business development",
            "lead generation",
            "prospecting",
            "crm",
            "client relationship",
            "negotiation",
            "closing",
            "revenue",
            "pipeline",
            "cold calling",
            "b2b",
            "b2c",
        ],
    },
    "Data Scientist": {
        "triggers": ["data scientist", "data science", "machine learning", "ml", "ai"],
        "keywords": [
            "python",
            "machine learning",
            "deep learning",
            "tensorflow",
            "pytorch",
            "nlp",
            "statistics",
            "sql",
            "modeling",
            "data analysis",
        ],
    },
    "Software Engineer": {
        "triggers": ["software engineer", "developer", "backend", "frontend", "full stack"],
        "keywords": [
            "java",
            "python",
            "javascript",
            "typescript",
            "api",
            "microservices",
            "react",
            "node",
            "spring",
            "git",
        ],
    },
    "Cybersecurity Analyst": {
        "triggers": ["cybersecurity", "security analyst", "soc", "ethical hacking"],
        "keywords": [
            "cybersecurity",
            "ethical hacking",
            "siem",
            "incident response",
            "network security",
            "linux",
            "vulnerability",
            "threat",
            "soc",
            "pen testing",
        ],
    },
}

ROLE_FALLBACK_STOPWORDS = {
    "and", "or", "the", "a", "an", "to", "for", "with", "of", "in", "on", "at",
    "is", "are", "as", "by", "be", "this", "that", "from", "will", "must", "should",
    "experience", "preferred", "required", "years", "year", "role", "job",
}

SEMANTIC_STRONG_THRESHOLD = 0.45
SEMANTIC_SOFT_THRESHOLD = 0.30


def extract_text(file_storage):
    filename = file_storage.filename or ""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTS:
        return None, f"Unsupported file type: {ext}"

    if ext == ".txt":
        data = file_storage.read()
        try:
            return data.decode("utf-8", errors="ignore"), None
        except Exception:
            return data.decode("latin-1", errors="ignore"), None

    if ext == ".pdf":
        if PyPDF2 is None:
            return None, "PyPDF2 not installed"
        try:
            reader = PyPDF2.PdfReader(file_storage)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, None
        except Exception as e:
            return None, f"PDF read error: {e}"

    if ext == ".docx":
        if docx is None:
            return None, "python-docx not installed"
        try:
            document = docx.Document(file_storage)
            text = "\n".join(p.text for p in document.paragraphs)
            return text, None
        except Exception as e:
            return None, f"DOCX read error: {e}"

    return None, "Unsupported file"


def extract_skills(text):
    text = text or ""
    if nlp is not None:
        try:
            doc = nlp(text)
            ents = {ent.text.strip().lower() for ent in doc.ents}
            matched = []
            for skill, pattern in _SKILL_PATTERNS:
                if pattern.search(text) or skill.lower() in ents:
                    matched.append(skill)
            return sorted(set(matched))[:12]
        except Exception:
            pass
    matched = [skill for skill, pattern in _SKILL_PATTERNS if pattern.search(text)]
    return sorted(set(matched))[:12]


def infer_role_profile(job_description):
    text = (job_description or "").lower()
    for role_name, profile in ROLE_PROFILES.items():
        if any(trigger in text for trigger in profile["triggers"]):
            return role_name, profile["keywords"]

    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{2,}", text)
    fallback_keywords = []
    seen = set()
    for token in tokens:
        t = token.lower()
        if t in ROLE_FALLBACK_STOPWORDS:
            continue
        if t in seen:
            continue
        seen.add(t)
        fallback_keywords.append(t)
        if len(fallback_keywords) >= 15:
            break
    return "General Role", fallback_keywords


def semantic_similarity(job_desc, resume_text):
    if embedder is None:
        return None
    try:
        embeddings = embedder.encode([job_desc, resume_text], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))
    except Exception:
        return None


def role_keyword_hits(resume_text, keywords):
    text = (resume_text or "").lower()
    matches = []
    for kw in keywords:
        key = kw.lower().strip()
        if key and key in text:
            matches.append(kw)
    return matches


def compute_role_match(semantic_score, keyword_hits, total_keywords):
    sem = semantic_score if isinstance(semantic_score, (int, float)) else 0.0
    if sem >= SEMANTIC_STRONG_THRESHOLD:
        return True
    if total_keywords <= 0:
        return sem >= SEMANTIC_SOFT_THRESHOLD
    coverage = len(keyword_hits) / total_keywords
    return sem >= SEMANTIC_SOFT_THRESHOLD and (len(keyword_hits) >= 2 or coverage >= 0.2)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/metrics", methods=["GET"])
def metrics():
    if not os.path.exists(METRICS_PATH):
        return jsonify({"error": "Metrics not found. Run training first."}), 404
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def predict():
    files = request.files.getlist("resumes")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    if len(files) > 100:
        return jsonify({"error": "Too many files. Max 100."}), 400

    job_description = request.form.get("job_description", "") or DEFAULT_JOB_DESC
    target_role, role_keywords = infer_role_profile(job_description)
    results = []
    class_labels = list(label_encoder.classes_)
    lower_to_index = {str(lbl).lower(): i for i, lbl in enumerate(class_labels)}
    hire_index = lower_to_index.get("hire")
    reject_index = lower_to_index.get("reject")

    for f in files:
        text, err = extract_text(f)
        if err:
            results.append({
                "filename": f.filename,
                "error": err,
            })
            continue

        vec = vectorizer.transform([text])
        proba = model.predict_proba(vec)[0]
        pred = int(model.predict(vec)[0])
        predicted_category = str(label_encoder.inverse_transform([pred])[0])
        confidence = float(proba[pred])
        prediction_lower = predicted_category.lower()
        sim = semantic_similarity(job_description, text)
        matched_role_keywords = role_keyword_hits(text, role_keywords)
        is_role_match = compute_role_match(sim, matched_role_keywords, len(role_keywords))
        is_hire = prediction_lower == "hire"
        label = "Relevant" if (is_hire and is_role_match) else "Not Relevant"
        skills = extract_skills(text)
        hire_probability = float(proba[hire_index]) if hire_index is not None else None
        reject_probability = float(proba[reject_index]) if reject_index is not None else None

        results.append({
            "filename": f.filename,
            "predicted_category": predicted_category,
            "prediction": label,
            "confidence": confidence,
            "hire_reject_prediction": predicted_category,
            "role_target": target_role,
            "role_match": is_role_match,
            "role_keyword_hits": matched_role_keywords,
            "role_keywords_used": role_keywords,
            "hire_probability": hire_probability,
            "reject_probability": reject_probability,
            "semantic_score": sim,
            "matched_skills": skills,
        })

    return jsonify({
        "job_description": job_description,
        "target_category": target_role,
        "count": len(results),
        "semantic_enabled": embedder is not None,
        "results": results,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
