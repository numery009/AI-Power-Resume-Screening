# ==========================================
# AI-Powered Resume Screening — JUCS Revision (Final, single file)
# (Includes SHAP dtype fix + unified Explainer API)
# ==========================================

import os
import re
import json
import time
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import pdfplumber
import docx
import spacy
import nltk
import matplotlib.pyplot as plt
import shap  # SHAP for explanations

from io import BytesIO
from nltk.corpus import names, stopwords
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from xgboost import XGBClassifier

# ---- Build / cache salt ----
BUILD_TAG  = "JUCS_shapfix_v3"
CACHE_SALT = BUILD_TAG

# ---- OpenAI (LLM mode) ----
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False
    OpenAI = None

# ---- Quiet NLTK downloads ----
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("names", quiet=True)
except Exception:
    pass

# ---- Reproducibility ----
RANDOM_SEED = 42
TEST_SIZE   = 0.20
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"                 # explicit for paper + reproducibility
LLM_MODEL_NAME   = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
np.random.seed(RANDOM_SEED)

# ---- Helper to enforce numeric float32 arrays (prevents SHAP casting errors) ----
def _f32(a):
    """Return a contiguous float32 NumPy array."""
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))

# ---- spaCy model (resilient load) ----
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    try:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")  # minimal fallback

# ---- SBERT (cached) ----
@st.cache_resource
def load_sbert_model():
    # reference CACHE_SALT so a new build can invalidate the resource cache if needed
    _ = CACHE_SALT
    return SentenceTransformer(SBERT_MODEL_NAME)

sbert_model = load_sbert_model()

# ---------- Token filters for XAI ----------
name_list = set(name.lower() for name in names.words())
stop_words = set(stopwords.words('english'))
irrelevant_tokens = {
    "name","address","street","city","state","road","main","email","phone","resume","curriculum","vitae",
    "zip","pin","number","contact","mobile","tel","fax","home","house","location","area","block",
    "india","usa","uk","canada","gmail","yahoo","hotmail","123",
    "linkedin","github","portfolio","profile","url","website"
}
BLACKLIST_PATTERNS = (r"^(http|https)://", r"^(www\.)", r".+\.(com|net|org|io)$")
BLACKLIST_SUBSTRINGS = {"linkedin","github","bitbucket","twitter","gmail","hotmail","outlook","portfolio","website","phone","email"}

def _is_good_token(clean_word: str) -> bool:
    if not clean_word or len(clean_word) <= 2:
        return False
    if clean_word in name_list or clean_word in irrelevant_tokens or clean_word in stop_words:
        return False
    if re.match(r"^[\d\-\(\)\s]+$", clean_word):
        return False
    if re.match(r"^\S+@\S+\.\S+$", clean_word):
        return False
    if any(re.search(pat, clean_word) for pat in BLACKLIST_PATTERNS):
        return False
    if any(s in clean_word for s in BLACKLIST_SUBSTRINGS):
        return False
    return True

# ---------- Data loading ----------
@st.cache_data
def load_dataset(_salt: str = CACHE_SALT):
    dataset_path = "resume_dataset.csv"
    df = pd.read_csv(dataset_path)
    if "Resume" not in df.columns or "Category" not in df.columns:
        raise ValueError("Dataset must contain 'Resume' and 'Category' columns.")
    X = df["Resume"].astype(str)
    y = df["Category"].astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return df, X, y_encoded, y.values, label_encoder

# ---------- File text extraction ----------
def extract_text(file_path: str) -> str:
    if file_path.endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        except Exception:
            return ""
    elif file_path.endswith(".docx"):
        try:
            document = docx.Document(file_path)
            return "\n".join([para.text for para in document.paragraphs if para.text.strip()])
        except Exception:
            return ""
    return ""

# ---------- Resume structure (Experience + Education) ----------
def extract_resume_entities(text: str) -> dict:
    doc = nlp(text)

    # Experience (sum of mentions)
    total_years = 0
    seen_phrases = set()
    try:
        for ent in doc.ents:
            if ent.label_ in ["DATE","QUANTITY"]:
                phrase = ent.text.strip().lower()
                if phrase not in seen_phrases:
                    seen_phrases.add(phrase)
                    m = re.search(r"(?:over\s+|more\s+than\s+)?(\d+)\+?\s*(year|years)", phrase)
                    if m: total_years += int(m.group(1))
    except Exception:
        pass

    # Education
    degree_keywords = ["bachelor","master","mba","phd","b.sc","m.sc","msc","btech","mtech","diploma","bfa"]
    education_lines = set()
    for line in text.split('\n'):
        if any(kw in line.lower() for kw in degree_keywords):
            cleaned = line.strip()
            if len(cleaned) > 5 and not re.fullmatch(r"(?i)(bachelor|master|phd|mba|bfa|b\.sc|m\.sc|msc|btech|mtech|diploma)", cleaned):
                education_lines.add(cleaned)
    regex_matches = re.findall(r"(?i)(Bachelor|Master|PhD|MBA|BFA|B\.Sc|M\.Sc|Diploma).*", text)
    for line in regex_matches:
        line = line.strip() if isinstance(line,str) else str(line)
        if len(line) > 5 and not re.fullmatch(r"(?i)(bachelor|master|phd|mba|bfa|b\.sc|m\.sc|msc|btech|mtech|diploma)", line):
            education_lines.add(line)

    return {
        "Experience": f"{total_years} years" if total_years > 0 else "Not Found",
        "Education": "\n".join(sorted(education_lines)) if education_lines else "Not Found"
    }

# ---------- Cache SBERT embeddings ----------
@st.cache_data
def get_cached_embeddings(text: str, _salt: str = CACHE_SALT) -> np.ndarray:
    vec = sbert_model.encode(text, convert_to_numpy=True)
    return _f32(vec)

# ---------- Train XGBoost on SBERT embeddings ----------

def train_xgb_model():
    df, X, y, y_raw, label_encoder = load_dataset()
    X_embeddings = _f32(np.vstack(X.apply(get_cached_embeddings).values))
    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train = _f32(X_train)
    X_test  = _f32(X_test)

    model = XGBClassifier(
        n_estimators=50, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', verbosity=1, n_jobs=-1, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model, label_encoder, X_train, X_test, y_test

# ---------- Evaluation & artifacts (CLASSICAL ONLY) ----------

def evaluate_and_report(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("📊 Classification Report")
    st.dataframe(report_df, use_container_width=True)
    report_df.to_csv("classification_report.csv", index=True)
    st.caption("Saved: classification_report.csv")

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(18, 14), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', aspect='auto', cmap='viridis')
    ax.set_title('Normalized Confusion Matrix', fontsize=20, pad=20)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=14)
    classes = list(label_encoder.classes_)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=35, ha='right', fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    ax.set_ylabel('True Label', fontsize=16, labelpad=12)
    ax.set_xlabel('Predicted Label', fontsize=16, labelpad=12)
    plt.tight_layout()
    st.subheader("🧩 Confusion Matrix (Normalized)")
    st.pyplot(fig, use_container_width=True)
    fig.savefig("confusion_matrix.png", bbox_inches='tight', dpi=300)
    st.caption("Saved: confusion_matrix.png")
    return report_df

# ---------- XAI: TF-IDF keyword highlights ----------

def interpret_keywords(resume_text: str, job_desc: str, top_k: int = 8) -> dict:
    raw_tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+_.-]{2,}", resume_text.lower())
    tokens = [t for t in raw_tokens if _is_good_token(t)]
    if not tokens:
        return {}
    vectorizer = TfidfVectorizer(max_features=300)
    tfidf_vec = vectorizer.fit_transform([" ".join(tokens)])
    scores = tfidf_vec.toarray()[0]
    words = vectorizer.get_feature_names_out()
    jd = job_desc.lower() if job_desc else ""
    boosted = []
    for w, s in zip(words, scores):
        b = s * (1.25 if w in jd else 1.0)
        boosted.append((w, b))
    boosted.sort(key=lambda x: x[1], reverse=True)
    top = boosted[:top_k]
    return {w: round(float(score), 4) for w, score in top}

# ---------- OpenAI helpers (LLM evaluation) ----------

def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not (_OPENAI_AVAILABLE and api_key):
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def llm_predict_label_openai(client, text: str, candidate_labels: List[str], model: str = "gpt-4o-mini") -> str:
    """
    Deterministic single-label classification via JSON. Falls back to first label on parse error.
    """
    if client is None:
        return candidate_labels[0]

    system_msg = (
        "You are a strict classifier. "
        "Only answer with a single JSON object: {\"label\": \"<one_of_the_labels>\"}. "
        "Choose the best matching label from the provided list. Do not add commentary."
    )
    user_msg = (
        "Classify the following resume text into one of the labels.\n\n"
        f"Labels: {candidate_labels}\n\n"
        f"Text:\n{text[:6000]}"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=20,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        obj = json.loads(raw)
        label = obj.get("label", "").strip()
        if label in candidate_labels:
            return label
    except Exception:
        pass
    # Salvage bare string if it exactly matches
    if raw in candidate_labels:
        return raw
    return candidate_labels[0]

@st.cache_data(show_spinner=False)
def evaluate_llm_on_dataset_sample(sample_size: int, model_name: str = "gpt-4o-mini", _salt: str = CACHE_SALT):
    """
    Runs the LLM as a classifier on a balanced sample of the dataset.
    Produces:
      - llm_classification_report.csv
      - llm_confusion_matrix.png
    Returns (report_df, cm, labels).
    """
    client = _get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is not set or OpenAI package is unavailable.")

    df, X, y_enc, y_raw, label_encoder = load_dataset()
    labels = list(label_encoder.classes_)

    # Balanced sample across classes as best as possible
    df_balanced = []
    per_class = max(1, sample_size // len(labels))
    for lbl in labels:
        df_lbl = df[df["Category"] == lbl]
        if len(df_lbl) == 0:
            continue
        take = min(len(df_lbl), per_class)
        df_balanced.append(resample(df_lbl, replace=False, n_samples=take, random_state=RANDOM_SEED))
    if not df_balanced:
        raise RuntimeError("No data available for balanced sampling.")
    df_eval = pd.concat(df_balanced, ignore_index=True)

    y_true = df_eval["Category"].tolist()
    y_pred = []

    progress = st.progress(0.0, text="Classifying with LLM…")
    for i, row in enumerate(df_eval.itertuples(index=False)):
        resume_text = str(getattr(row, "Resume", ""))
        pred = llm_predict_label_openai(client, resume_text, labels, model=model_name)
        y_pred.append(pred)
        if (i + 1) % 5 == 0 or i == len(df_eval) - 1:
            progress.progress((i + 1) / len(df_eval))

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("llm_classification_report.csv", index=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

    # Plot & save
    fig, ax = plt.subplots(figsize=(18, 14), dpi=150)
    im = ax.imshow(cm, interpolation='nearest', aspect='auto', cmap='viridis')
    ax.set_title('LLM Normalized Confusion Matrix', fontsize=20, pad=20)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=14)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_ylabel('True Label', fontsize=16, labelpad=12)
    ax.set_xlabel('Predicted Label', fontsize=16, labelpad=12)
    plt.tight_layout()
    fig.savefig("llm_confusion_matrix.png", bbox_inches='tight', dpi=300)

    return report_df, cm, labels

# ---------- LLM baseline (per-resume match) ----------

def llm_match_score(resume_text: str, job_desc: str):
    """
    Returns (score_0_to_1, predicted_category, latency_seconds).
    If OPENAI_API_KEY missing or openai lib unavailable → (None, None, 0.0).
    """
    if not job_desc.strip():
        return 0.0, "N/A", 0.0
    api_key = os.getenv("OPENAI_API_KEY")
    if not (_OPENAI_AVAILABLE and api_key):
        return None, None, 0.0

    client = OpenAI(api_key=api_key)
    start = time.time()
    prompt = f"""
You are assisting with resume screening. Read the JOB DESCRIPTION and RESUME below.
Return:
- match_score: a number between 0 and 1 (0 = no match, 1 = perfect)
- category: one of these coarse roles if obvious (else "Unknown"):
  Advocate, Arts, Automation Testing, Blockchain, Business Analyst, Civil Engineer, Data Science, Database,
  DevOps Engineer, DotNet Developer, ETL Developer, Electrical Engineering, HR, Hadoop, Health and Fitness,
  Java Developer, Mechanical Engineer, Network Security Engineer, Operations Manager, PMO, Python Developer,
  SAP Developer, Sales, Testing, Web Designing

JOB DESCRIPTION:
{job_desc}

RESUME:
{resume_text}

Respond as compact JSON: {{"match_score": <float>, "category": "<str>"}}
"""
    try:
        msg = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
        )
        txt = msg.choices[0].message.content.strip()
        m = re.search(r'"match_score"\s*:\s*([0-9]*\.?[0-9]+)', txt)
        c = re.search(r'"category"\s*:\s*"([^"]+)"', txt)
        score = float(m.group(1)) if m else 0.0
        cat   = c.group(1) if c else "Unknown"
        latency = time.time() - start
        score = max(0.0, min(1.0, score))
        return float(score), cat, float(latency)
    except Exception:
        latency = time.time() - start
        return None, None, float(latency)

# ---------- SBERT cosine similarity for match score ----------

def match_resume_with_job(resume_text: str, job_desc: str) -> float:
    if not job_desc.strip():
        return 0.0
    resume_embedding = get_cached_embeddings(resume_text)
    job_embedding    = get_cached_embeddings(job_desc)
    return float(cosine_similarity(resume_embedding.reshape(1,-1),
                                  job_embedding.reshape(1,-1))[0][0])

# ---------- Dataset Overview helpers ----------
@st.cache_data
def compute_class_distribution(_salt: str = CACHE_SALT):
    df, X, y_enc, y_raw, label_encoder = load_dataset()
    counts = pd.Series(y_raw).value_counts().sort_index()
    summary = pd.DataFrame({"Category": counts.index, "Count": counts.values})
    total_rows = len(df); n_classes = len(label_encoder.classes_)
    return df, summary, total_rows, n_classes, label_encoder


def render_class_distribution(summary: pd.DataFrame):
    summary.to_csv("class_distribution.csv", index=False)
    st.caption("Saved: class_distribution.csv")
    fig, ax = plt.subplots(figsize=(18, 6), dpi=150)
    ax.bar(summary["Category"], summary["Count"])
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution", fontsize=16, pad=10)
    plt.xticks(rotation=35, ha='right', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    fig.savefig("class_distribution.png", bbox_inches='tight', dpi=300)
    st.caption("Saved: class_distribution.png")

# ---------- UI helpers ----------
ENGINE_ABOUT = {
    "Classical ML: SBERT + XGBoost (primary)": """
- **How it works:** SBERT embeddings → XGBoost classifier  
- **Pros:** Reproducible, **explainable (SHAP over embeddings + TF-IDF tokens)**, evaluation artifacts (report & confusion matrix)  
- **Outputs:** Category + probability, match score, classification report, confusion matrix, **SHAP global & local plots**  
""",
    "LLM Baseline (GPT-4o) – optional": """
- **How it works:** Prompt an LLM with resume + JD for a 0–1 match score and category  
- **Pros:** Flexible reading comprehension  
- **Limitations:** Non-deterministic, API cost/latency, weaker reproducibility  
- **Outputs:** Match score (+ optional category) and **TF-IDF key terms** for transparency.  
  **Evaluation (optional):** You can run the LLM as a classifier over a labeled sample to produce a classification report & confusion matrix.
"""
}


def badges_row(deterministic: bool, explainable: bool, offline: bool):
    flags = []
    flags.append("🧪 Reproducible" if deterministic else "⚠️ Non-deterministic")
    flags.append("🪪 Explainable" if explainable else "🪫 Limited explainability")
    flags.append("🔒 Local/Offline" if offline else "☁️ Uses external API")
    st.markdown("**Mode summary:** " + " • ".join(flags))


def ethics_callout(run_llm_baseline: bool):
    st.subheader("⚖️ Ethics & Limitations")
    notes = [
        "This tool is an **assistive** system; final decisions should remain **human-in-the-loop**.",
        "Models may reflect **dataset bias**. Avoid feeding protected attributes.",
        "PII filters are applied to explanations, but PII can still appear in résumés.",
        "LLM mode is **non-deterministic**; classical mode is **seeded** and reproducible.",
        "Offline evaluation may not reflect **deployment drift**; re-evaluate periodically.",
        "Use explanations to **audit** decisions, not as sole grounds."
    ]
    if run_llm_baseline:
        notes.append("LLM evaluation can incur **token costs** and depends on external API availability/policies.")
    st.info("• " + "\n• ".join(notes))

# ---------- Streamlit App ----------

def resume_screening_dashboard():
    st.title("📄 AI-Powered Resume Screening")
    st.caption(f"Build: {BUILD_TAG}")

    # Sidebar: Engine selection + explainer
    with st.sidebar:
        st.header("Model Options")
        if st.button("🔄 Clear caches (data & resources)"):
            try:
                st.cache_data.clear(); st.cache_resource.clear()
                st.success("Caches cleared. Click Rerun.")
            except Exception as _e:
                st.warning(f"Could not clear caches: {_e}")
            st.stop()

        model_choice = st.selectbox(
            "Engine",
            ["Classical ML: SBERT + XGBoost (primary)", "LLM Baseline (GPT-4o) – optional"]
        )
        run_llm_baseline = model_choice.startswith("LLM")

    st.markdown("### ℹ️ About this engine")
    st.markdown(ENGINE_ABOUT[model_choice])

    # Badges
    if run_llm_baseline:
        badges_row(deterministic=False, explainable=False, offline=False)
    else:
        badges_row(deterministic=True, explainable=True, offline=True)

    # Dataset Overview
    st.markdown("### 🗃️ Dataset Overview")
    try:
        df, class_summary, total_rows, n_classes, label_encoder = compute_class_distribution()
        c1, c2 = st.columns(2)
        with c1: st.metric("Total samples", f"{total_rows}")
        with c2: st.metric("Number of classes", f"{n_classes}")
        st.dataframe(class_summary, use_container_width=True, hide_index=True)
        render_class_distribution(class_summary)
    except Exception as e:
        st.warning(f"Dataset overview unavailable: {e}")

    # Inputs
    job_desc = st.text_area("Enter Job Description", "Enter job requirements...")
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf","docx"])

    if uploaded_file is not None:
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract resume text + structure
        t0 = time.time()
        resume_text = extract_text(file_path)
        structured_data = extract_resume_entities(resume_text)
        extract_latency = time.time() - t0

        # SBERT match score
        if not job_desc.strip():
            st.warning("⚠️ Please enter a job description to calculate match score.")
            match_score = 0.0; match_latency = 0.0
        else:
            t1 = time.time()
            match_score = match_resume_with_job(resume_text, job_desc)
            match_latency = time.time() - t1

        st.subheader("🔍 Resume Match Score:")
        st.write(f"**{match_score * 100:.2f}% Match**")
        st.caption(f"Extraction latency: {extract_latency:.2f}s • Match calc latency: {match_latency:.2f}s")

        if match_score >= 0.75:
            st.success("✅ High match: Resume strongly aligns with the job description.")
        elif match_score >= 0.5:
            st.info("ℹ️ Moderate match: Resume partially aligns with the job description.")
        else:
            st.warning("⚠️ Low match: Resume has limited alignment with the job description.")

        # Education / Experience
        education_lines = [
            line.strip().lstrip("•").strip()
            for line in structured_data["Education"].split('\n')
            if line.strip() and not re.match(r"(?i)^education\s*$", line.strip())
        ]
        st.subheader("🎓 Education")
        for line in education_lines:
            st.markdown(f"- {line}")
        st.subheader("🧭 Experience")
        st.markdown(structured_data["Experience"])

        # Branches
        if run_llm_baseline:
            st.subheader("🤖 GPT-4o Screening")
            llm_score, llm_cat, llm_latency = llm_match_score(resume_text, job_desc)
            if llm_score is None:
                st.info("Set OPENAI_API_KEY to enable LLM mode.")
            else:
                st.write(f"**LLM Match Score:** {llm_score*100:.2f}%")
                st.write(f"**LLM Predicted Category:** {llm_cat}")
                st.caption(f"LLM call latency: {llm_latency:.2f}s")

            # XAI terms
            st.subheader("🔍 Human-Readable Terms (TF-IDF Heuristic — LLM Baseline)")
            keyw = interpret_keywords(resume_text, job_desc, top_k=8)
            if not keyw:
                st.write("🔹 No influential words found.")
            else:
                st.write("🔹 " + ", ".join([f"{w} (score={v:.4f})" for w, v in keyw.items()]))

            # --- Optional: LLM evaluation over dataset sample ---
            st.markdown("---")
            st.subheader("🧪 Evaluate LLM on the dataset (optional)")
            st.caption("Runs the LLM as a classifier over a labeled sample to produce a real classification report and confusion matrix. Uses your OpenAI API key and may incur token cost.")
            sample_size = st.number_input("Sample size", min_value=25, max_value=500, step=25, value=100)
            llm_model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)

            if st.button("Run LLM evaluation"):
                try:
                    with st.spinner("Evaluating LLM on dataset…"):
                        rep_df, cm_llm, class_labels = evaluate_llm_on_dataset_sample(sample_size, llm_model_name)
                    st.subheader("📊 LLM Classification Report (dataset sample)")
                    st.dataframe(rep_df, use_container_width=True)
                    st.caption("Saved: llm_classification_report.csv")

                    st.subheader("🧩 LLM Confusion Matrix (Normalized)")
                    st.image("llm_confusion_matrix.png", use_column_width=True, caption="Saved: llm_confusion_matrix.png")
                except RuntimeError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error(f"LLM evaluation failed: {e}")

        else:
            with st.spinner("Training classifier (SBERT + XGBoost)..."):
                ttrain = time.time()
                model, label_encoder2, X_train, X_test, y_test = train_xgb_model()
                train_latency = time.time() - ttrain
            st.success("✅ Model Trained Successfully!")
            st.caption(f"Training latency: {train_latency:.2f}s")

            # Evaluation (CLASSICAL ONLY)
            evaluate_and_report(model, X_test, y_test, label_encoder2)

            # ---- SHAP EXPLAINABILITY (GLOBAL over test set) ----
            st.subheader("🧠 SHAP Global Explanation (Classical Model)")
            st.caption(f"[debug] X_train dtype={X_train.dtype}, X_test dtype={X_test.dtype}")
            try:
                booster = model.get_booster()
                explainer = shap.TreeExplainer(booster)
                shap_values = explainer.shap_values(_f32(X_test))  # list per class

                # Aggregate mean |SHAP| across classes
                if isinstance(shap_values, list):
                    mean_abs = sum(np.abs(v).mean(axis=0) for v in shap_values) / len(shap_values)
                else:
                    mean_abs = np.abs(shap_values).mean(axis=0)

                order = np.argsort(mean_abs)[::-1][:20]
                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                ax.bar(range(len(order)), mean_abs[order])
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels([f"emb_{i}" for i in order], rotation=45, ha='right', fontsize=9)
                ax.set_ylabel("mean |SHAP value|")
                ax.set_title("Top embedding dimensions by mean |SHAP|")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                fig.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
                st.caption("Saved: shap_summary.png")
            except Exception as e:
                st.warning(f"SHAP global explanation unavailable: {e}")

            # Per-resume prediction
            tpred = time.time()
            pred_vec = _f32(get_cached_embeddings(resume_text).reshape(1,-1))
            predicted_probs = model.predict_proba(pred_vec)[0]
            infer_latency = time.time() - tpred

            sorted_preds = sorted(
                zip(label_encoder2.classes_, predicted_probs),
                key=lambda x: x[1], reverse=True
            )[:5]
            st.subheader("🏆 Predicted Job Categories (Pipeline):")
            for category, confidence in sorted_preds:
                st.write(f"🔹 **{category}: {confidence:.2%} confidence**")
            st.caption(f"Inference latency: {infer_latency:.2f}s")

            # ---- SHAP LOCAL (for this resume, predicted class) ----
            st.subheader("🧩 SHAP Local Explanation for This Resume (Predicted Class)")
            try:
                booster = model.get_booster()
                explainer_local = shap.TreeExplainer(booster)
                shap_local = explainer_local.shap_values(pred_vec)

                top_idx = int(np.argmax(predicted_probs))

                if isinstance(shap_local, list):  # multiclass
                    contrib = shap_local[top_idx][0]
                else:  # binary
                    contrib = shap_local[0]

                k = 15
                order = np.argsort(np.abs(contrib))[::-1][:k]
                fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
                colors = ["#4CAF50" if v > 0 else "#F44336" for v in contrib[order]]
                ax.bar(range(k), contrib[order], color=colors)
                ax.set_xticks(range(k))
                ax.set_xticklabels([f"emb_{i}" for i in order], rotation=45, ha='right', fontsize=9)
                ax.set_ylabel("SHAP value (this resume)")
                ax.set_title(f"Top embedding-dimension contributions → {label_encoder2.classes_[top_idx]}")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                fig.savefig("shap_top_features.png", bbox_inches='tight', dpi=300)
                st.caption("Saved: shap_top_features.png")

            except Exception as e:
                st.warning(f"SHAP local explanation unavailable: {e}")


            # XAI (TF-IDF keywords)
            st.subheader("🔍 Human-Readable Feature Insights (XAI — Classical Model)")
            keyw = interpret_keywords(resume_text, job_desc, top_k=8)
            if not keyw:
                st.write("🔹 No influential words found.")
            else:
                st.write("🔹 " + ", ".join([f"{w} (score={v:.4f})" for w, v in keyw.items()]))

        # Minimal run log
        results_row = {
            "resume_file": uploaded_file.name,
            "engine": "LLM Baseline" if run_llm_baseline else "SBERT+XGBoost",
            "pipeline_match_score": round(float(match_score), 4),
        }
        if not run_llm_baseline and locals().get("sorted_preds"):
            results_row.update({
                "pipeline_top_category": sorted_preds[0][0],
                "pipeline_top_confidence": round(float(sorted_preds[0][1]), 4),
            })
        log_path = "screening_runs.csv"
        pd.DataFrame([results_row]).to_csv(
            log_path, index=False, mode='a', header=not os.path.exists(log_path)
        )
        st.caption(f"Appended run to: {log_path}")

    # Ethics block
    st.markdown("---")
    ethics_callout(run_llm_baseline)


if __name__ == '__main__':
    resume_screening_dashboard()
