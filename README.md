
# For project_v6.py

<img width="818" height="428" alt="image" src="https://github.com/user-attachments/assets/59f5f9a8-8ac5-45ba-86a1-8a5d433affcf" />

##	Implementation

The AI-powered resume screening system is implemented using Python (v3.11) as the primary programming language. The system integrates multiple technologies, including Natural Language Processing (NLP), Machine Learning (ML), semantic similarity (SBERT), and Explainable AI (XAI), into a unified web-based application. The entire project was developed in a Visual Studio IDE on a Windows 11 machine with an Intel i7 processor and 16GB of RAM, using the pip package manager for dependency installation. The final application was deployed using streamlit, enabling real-time recruiter interaction through a lightweight browser-based interface. The core functionalities include resume parsing, feature extraction, embedding generation using Sentence-BERT (SBERT), multi-class classification using XGBoost, cosine similarity-based matching, and SHAP-based interpretability.

The dataset used was a CSV file containing synthetically labeled resumes across predefined job categories such as "Software Engineer", "Data Scientist", "Cybersecurity Analyst", and more. The dataset was created for academic research and adheres to ethical guidelines. Resume texts were first parsed and cleaned, then transformed into 768-dimensional embeddings using the pre-trained all-MiniLM-L6-v2 SBERT model. These embeddings were fed into an XGBoost classifier trained using an 80/20 train-test split, with key parameters such as max_depth = 5, n_estimators = 50, and tree_method = 'hist' for efficient boosting. The Streamlit interface enables users to upload resumes, input job descriptions, and instantly view resume match scores (via cosine similarity), predicted job categories with confidence scores, extracted structured information like education and experience, and top influential words from SHAP explanations [29]. This end-to-end methodology ensures transparency, modularity, and real-time user feedback, providing a practical and explainable solution for AI-powered resume screening.

## Install
pip install pandas numpy scikit-learn xgboost spacy nltk streamlit shap pdfplumber python-docx sentence-transformers

## Run
streamlit run project_v6.py

# For project_v7.py

<img width="834" height="485" alt="image" src="https://github.com/user-attachments/assets/40ccd7db-4e53-4b24-b4ef-8ac48a6ddd6e" />

## Implementation
This version (project_v7.py) of the AI-Powered Resume Screening System, integrating dual explainable engines — a Classical Explainable Engine (SBERT + XGBoost + SHAP) and an LLM Baseline Engine (GPT-4o / GPT-4o-mini + TF-IDF transparency).

### 🔑 Key Features

- **Resume Parsing** from PDF/DOCX using [`pdfplumber`](https://pypi.org/project/pdfplumber/) and [`python-docx`](https://pypi.org/project/python-docx/)
- **Semantic Embeddings** via [`Sentence-BERT`](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Explainable ML Pipeline** using [`XGBoost`](https://xgboost.readthedocs.io/) and [`SHAP`](https://shap.readthedocs.io/)
- **LLM Integration** using `GPT-4o` / `GPT-4o-mini` for contextual matching
- **Streamlit Dashboard** with dual-engine mode selection and interpretability layers
- **XAI Transparency** with TF-IDF keywords + SHAP (local & global)
- **Ethics & Governance** guidance to support human-in-the-loop decision-making

## ⚙️ Environment Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/numery009/AI-Power-Resume-Screening.git
cd AI-Power-Resume-Screening
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3️⃣ Install Required Dependencies
```bash
pip install -r requirements.txt
```
If you don’t have a requirements.txt, install manually:
```bash
pip install streamlit numpy pandas scikit-learn xgboost shap spacy nltk pdfplumber python-docx sentence-transformers matplotlib openai
```

### 4️⃣ Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```

###  📂 Dataset Requirements
The application expects a CSV file named:
```bash
resume_dataset.csv
```

