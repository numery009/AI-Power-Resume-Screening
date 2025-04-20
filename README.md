

#	Implementation
The AI-powered resume screening system is implemented using Python (v3.11) as the primary programming language. The system integrates multiple technologies, including Natural Language Processing (NLP), Machine Learning (ML), semantic similarity (SBERT), and Explainable AI (XAI), into a unified web-based application. The entire project was developed in a Visual Studio IDE on a Windows 11 machine with an Intel i7 processor and 16GB of RAM, using the pip package manager for dependency installation. The final application was deployed using streamlit [28], enabling real-time recruiter interaction through a lightweight browser-based interface. The core functionalities include resume parsing, feature extraction, embedding generation using Sentence-BERT (SBERT), multi-class classification using XGBoost, cosine similarity-based matching, and SHAP-based interpretability.

# Install
pip install pandas numpy scikit-learn xgboost spacy nltk streamlit shap pdfplumber python-docx sentence-transformers

# Run
streamlit run project_v6.py
