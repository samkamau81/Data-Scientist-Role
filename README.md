# Data-Scientist-Role
Data Scientist - AI Acceleration (AIA) Technical Assessment

Repository Structure
Part1_Structured_Data_ML_Challenge/  
├── Structured_Data_ML_Challenge.ipynb # Jupyter notebook with EDA, modeling, and segmentation  
├── report_part1.pdf                  # Summary of approach and findings  
└── visuals/                          # Generated plots/figures  
│  
├── Part2_LLM_Vector_DB_Challenge/  
│   ├── review_processing_pipeline.py     # Document processing & embeddings  
│   ├── llm_application.py                # Summaries, Q&A, and issue detection  
│   ├── sentiment_analysis.py             # Classifier and evaluation  
│   ├── requirements.txt                  # Python dependencies (see below)  
│   └── report_part2.pdf  
│  
├── Part3_MLOps_Design_Exercise/  
│   ├── mlops_design_document.pdf         # Detailed architecture and strategies  
│   └── monitoring_dashboard_mockup.png   # Example dashboard within the report
│  
└── README.md  

Setup Instructions
Part 1: Structured Data & ML Challenge
Requirements: Python 3.8+, pandas, scikit-learn, matplotlib, seaborn

Run: Execute Structured_Data_ML_Challenge.ipynb directly in Jupyter.

Part 2: LLM & Vector Database Challenge
Virtual Environment (Recommended):

bash
python -m venv venv
venv\Scripts\activate (Windows)
pip install -r requirements.txt

Key Dependencies:
1. transformers (Hugging Face)
2. sentence-transformers (for embeddings)
3. openai (if using OpenAI API)
4. faiss or chromadb (vector DB)

See full list in requirements.txt.

Part 3: MLOps Design Exercise
No code execution needed. Review PDF and diagrams directly.
