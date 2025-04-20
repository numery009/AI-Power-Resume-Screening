# ✅ Import necessary libraries 
import os
import pandas as pd
import numpy as np
import re
import docx
import pdfplumber
import spacy
import nltk
import streamlit as st
import shap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from nltk.corpus import names, stopwords

# ✅ Load NLP resources
nltk.download('stopwords')
nltk.download('names')
nlp = spacy.load("en_core_web_sm")

# ✅ Load Pre-trained SBERT Model (cached for performance)
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sbert_model = load_sbert_model()
name_list = set(name.lower() for name in names.words())
stop_words = set(stopwords.words('english'))
irrelevant_tokens = {
    "name", "address", "street", "city", "state", "road", "main", "email", "phone", "resume", "curriculum", "vitae",
    "zip", "pin", "number", "contact", "mobile", "tel", "fax", "home", "house", "location", "area", "block",
    "india", "usa", "uk", "canada", "gmail", "yahoo", "hotmail", "123",
    "linkedin", "github", "portfolio", "profile", "url", "website"
}

# ✅ Load Dataset with Class Mapping
def load_dataset():
    dataset_path = "resume_dataset.csv"
    df = pd.read_csv(dataset_path)

    if "Resume" not in df.columns or "Category" not in df.columns:
        raise ValueError("Dataset must contain 'Resume' and 'Category' columns.")

    X = df["Resume"]
    y = df["Category"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

# ✅ Extract Text from PDF/DOCX
def extract_text(file_path):
    if file_path.endswith(".pdf"):
        try:
            with pdfplumber.open(file_path) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        except Exception:
            return ""
    elif file_path.endswith(".docx"):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        except Exception:
            return ""
    else:
        return ""

# ✅ Named Entity Recognition (NER) for Experience, Education
def extract_resume_entities(text):
    doc = nlp(text)

    # --- Extract Experience ---
    experience_entities = []
    for ent in doc.ents:
        if ent.label_ in ["DATE", "QUANTITY"] and "year" in ent.text.lower():
            experience_entities.append(ent.text.strip())

    # --- Extract Education ---
    degree_keywords = ["bachelor", "master", "mba", "phd", "b.sc", "m.sc", "msc", "btech", "mtech", "diploma"]
    education_lines = set()

    for line in text.split('\n'):
        if any(kw in line.lower() for kw in degree_keywords):
            cleaned = line.strip()
            if len(cleaned) > 5 and not re.fullmatch(r"(?i)(bachelor|master|phd|mba|b\.sc|m\.sc|msc|btech|mtech|diploma)", cleaned):
                education_lines.add(cleaned)

    # --- Backup: Regex-based degree pattern matching ---
    regex_matches = re.findall(r"(?i)(Bachelor|Master|PhD|MBA|B\.Sc|M\.Sc|Diploma).*", text)
    for match in regex_matches:
        line = match.strip()
        if len(line) > 5 and not re.fullmatch(r"(?i)(bachelor|master|phd|mba|b\.sc|m\.sc|msc|btech|mtech|diploma)", line):
            education_lines.add(line)

    return {
        "Experience": ", ".join(set(experience_entities)) if experience_entities else "Not Found",
        "Education": "\n".join(sorted(education_lines)) if education_lines else "Not Found"
    }

# ✅ Cache SBERT Embeddings
@st.cache_data
def get_cached_embeddings(text):
    return sbert_model.encode(text)

# ✅ Train XGBoost Model with evaluation
def train_xgb_model():
    X, y, label_encoder = load_dataset() 
    X_embeddings = np.vstack(X.apply(get_cached_embeddings).values) 
    X_train, X_test, y_train, y_test = train_test_split(X_embeddings, y, test_size=0.2, random_state=42) 

    model = XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          tree_method='hist', verbosity=1, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("\nModel Evaluation Report:\n", report)

    return model, label_encoder, X_train

# ✅ SHAP Explanation with Filtering
def interpret_feature_importance(model, resume_text):
    tokens = resume_text.split()
    embedding = get_cached_embeddings(resume_text).reshape(1, -1)
    explainer = shap.Explainer(model)
    shap_values = explainer(embedding)
    importance = np.abs(shap_values.values[0])

    top_indices = np.argsort(importance)[-15:][::-1]
    top_words = {}

    for idx in top_indices:
        i = int(idx[0]) if isinstance(idx, (np.ndarray, list)) else int(idx)
        if i < len(tokens):
            raw_word = tokens[i]
            clean_word = re.sub(r"[^a-zA-Z0-9@.+-]", "", raw_word).lower()

            if (
                not clean_word or
                len(clean_word) <= 1 or
                clean_word in name_list or
                clean_word in irrelevant_tokens or
                clean_word in stop_words or
                re.match(r"^[\d\-\(\)\s]+$", clean_word) or
                re.match(r"^\S+@\S+\.\S+$", clean_word) or
                re.match(r"^(http|www)\.", clean_word) or
                re.search(r"linkedin\.com|github\.com|\.io|\.me|\.net", clean_word)
            ):
                continue

            top_words[clean_word] = max(float(importance[i]), top_words.get(clean_word, 0.0))

        if len(top_words) >= 5:
            break

    if len(top_words) < 5:
        fallback = {}
        for w in tokens:
            clean_fallback = re.sub(r"[^a-zA-Z0-9@.+-]", "", w.lower())
            if (
                len(clean_fallback) > 2 and
                clean_fallback not in name_list and
                clean_fallback not in irrelevant_tokens and
                clean_fallback not in stop_words and
                not re.match(r"^[\d\-\(\)\s]+$", clean_fallback) and
                not re.match(r"^\S+@\S+\.\S+$", clean_fallback) and
                not re.match(r"^(http|www)\.", clean_fallback) and
                not re.search(r"linkedin\.com|github\.com|\.io|\.me|\.net", clean_fallback)
            ):
                fallback[clean_fallback] = 0.0
                if len(fallback) >= 5:
                    break
        for k, v in fallback.items():
            if k not in top_words:
                top_words[k] = v

    return top_words

# ✅ Resume Matching using SBERT + Cosine Similarity
def match_resume_with_job(resume_text, job_desc):
    if not job_desc.strip():
        return 0.0

    resume_embedding = get_cached_embeddings(resume_text)
    job_embedding = get_cached_embeddings(job_desc)
    return cosine_similarity(resume_embedding.reshape(1, -1), job_embedding.reshape(1, -1))[0][0]

# ✅ Streamlit Web Dashboard 
def resume_screening_dashboard():
    st.title("📄 AI-Powered Resume Screening")

    job_desc = st.text_area("Enter Job Description", "Enter job requirements...")
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])

    if uploaded_file is not None:
        file_path = f"./temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        resume_text = extract_text(file_path)
        structured_data = extract_resume_entities(resume_text)

        if not job_desc.strip():
            st.warning("⚠️ Please enter a job description to calculate match score.")
            match_score = 0.0
        else:
            match_score = match_resume_with_job(resume_text, job_desc)

        st.subheader("🔍 Resume Match Score:")
        st.write(f"**{match_score * 100:.2f}% Match**")

        if match_score >= 0.75:
            st.success("✅ High match: Resume strongly aligns with the job description.")
        elif match_score >= 0.5:
            st.info("ℹ️ Moderate match: Resume partially aligns with the job description.")
        else:
            st.warning("⚠️ Low match: Resume has limited alignment with the job description.")

        education_lines = [
            line.strip().lstrip("•").strip()
            for line in structured_data["Education"].split('\n')
            if line.strip() and not re.match(r"(?i)^education\s*$", line.strip())
        ]

        st.markdown("**Education:**")
        for line in education_lines:
            st.markdown(f"- {line}")

        st.markdown("**Experience:** " + structured_data["Experience"])

        model, label_encoder, X_train = train_xgb_model()
        st.success("✅ Model Trained Successfully!")

        predicted_probs = model.predict_proba(get_cached_embeddings(resume_text).reshape(1, -1))[0]
        sorted_preds = sorted(zip(label_encoder.classes_, predicted_probs), key=lambda x: x[1], reverse=True)[:5]

        st.subheader("🏆 Predicted Job Categories:")
        for category, confidence in sorted_preds:
            st.write(f"🔹 **{category}: {confidence:.2%} confidence**")

        top_words = interpret_feature_importance(model, resume_text)
        seen = set()
        unique_words = []
        for word in top_words.keys():
            key = word.strip().lower()
            if key not in seen:
                unique_words.append(word)
                seen.add(key)

        st.subheader("🔍 Top Influential Words in Resume:")
        if not unique_words:
            st.write("🔹 No influential words found.")
        else:
            st.write(f"🔹 Words Influencing Prediction: {', '.join(unique_words)}")

if __name__ == '__main__':
    resume_screening_dashboard()
