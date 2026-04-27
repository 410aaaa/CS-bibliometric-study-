# CS-bibliometric-study-

CNS Journals Computer Science Paper Analysis (2016–2025)

A pipeline for detecting, classifying, and modeling computer science (CS) publications in Cell, Nature, and Science (CNS) journals.

This project integrates:

document embeddings (SPECTER2)
keyword-based filtering
multi-disciplinary classification
topic modeling (BERTopic)

to identify CS-related papers and analyze their research themes.

🚀 Installation
pip install -r requirements.txt

Or install manually:

pip install torch transformers adapters sentence-transformers bertopic
pip install xgboost scikit-learn pandas numpy openpyxl
pip install gensim umap-learn hdbscan matplotlib seaborn tqdm
📂 Data

Input files should include:

Article Title
Abstract
Document Type
Publication Year
Training data (Q1 only)
is_computer (1 = CS, 0 = non-CS)
Example structure
data/
├── train_data.xlsx
└── CNS_data/
    ├── 2016/
    ├── 2017/
    └── ...

Replace data/ with your local path.

⚙️ Usage
Step 1 — CS Detection
python q1_cs_detection_threshold_confusion.py
Step 2 — Discipline Classification
python q2_discipline_classification_threshold_sensitivity.py
Step 3 — Discipline-Specific Topic Modeling
python q3_bertopic_include_only_discipline.py

Modify inside script:

TARGET_DISCIPLINE = "Clinical Medicine"

Outputs:

topic distribution per discipline
coherence & stability metrics
include vs only comparison
Step 4 — Topic Modeling (All Papers)
python q3_bertopic_all_papers_overlap_semantic.py

Outputs:

global topic structure
topic similarity matrix
topic-word overlap analysis
evaluation metrics
📦 Outputs

The pipeline generates:

CS detection predictions (yearly)
discipline-level annotations
discipline-specific topic models
global topic modeling results
evaluation metrics and analysis files
📚 Citation

A manuscript describing this work is currently under review.
Citation details will be updated upon publication.

📄 License

This project is licensed under the MIT License.

💬 Notes
Update file paths according to your environment
Results may vary depending on dataset and parameter settings
