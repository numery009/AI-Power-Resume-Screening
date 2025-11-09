

#	Implementation
## For project_v6.py

The AI-powered resume screening system is implemented using Python (v3.11) as the primary programming language. The system integrates multiple technologies, including Natural Language Processing (NLP), Machine Learning (ML), semantic similarity (SBERT), and Explainable AI (XAI), into a unified web-based application. The entire project was developed in a Visual Studio IDE on a Windows 11 machine with an Intel i7 processor and 16GB of RAM, using the pip package manager for dependency installation. The final application was deployed using streamlit, enabling real-time recruiter interaction through a lightweight browser-based interface. The core functionalities include resume parsing, feature extraction, embedding generation using Sentence-BERT (SBERT), multi-class classification using XGBoost, cosine similarity-based matching, and SHAP-based interpretability.

The dataset used was a CSV file containing synthetically labeled resumes across predefined job categories such as "Software Engineer", "Data Scientist", "Cybersecurity Analyst", and more. The dataset was created for academic research and adheres to ethical guidelines. Resume texts were first parsed and cleaned, then transformed into 768-dimensional embeddings using the pre-trained all-MiniLM-L6-v2 SBERT model. These embeddings were fed into an XGBoost classifier trained using an 80/20 train-test split, with key parameters such as max_depth = 5, n_estimators = 50, and tree_method = 'hist' for efficient boosting. The Streamlit interface enables users to upload resumes, input job descriptions, and instantly view resume match scores (via cosine similarity), predicted job categories with confidence scores, extracted structured information like education and experience, and top influential words from SHAP explanations [29]. This end-to-end methodology ensures transparency, modularity, and real-time user feedback, providing a practical and explainable solution for AI-powered resume screening.

# Install
pip install pandas numpy scikit-learn xgboost spacy nltk streamlit shap pdfplumber python-docx sentence-transformers

# Run
streamlit run project_v6.py
