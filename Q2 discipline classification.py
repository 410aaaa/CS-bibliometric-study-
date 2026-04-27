# =========================================================
# 0. Basic Environment + Random Seed Setup
# =========================================================
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Random seed setup
# --------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Ensure cudnn determinism
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------
# Device selection
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =========================================================
# 1. Load Data (Merge by Year)
# =========================================================
years = range(2016, 2026)
all_dfs = []

for year in years:
    path = f"D:/Desktop/code/CNS data/{year}/{year}_predict_article.xlsx"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    df = pd.read_excel(path)
    if "Article Title" not in df.columns or "Abstract" not in df.columns:
        raise ValueError(f"Missing required columns in {path}")

    df["Publication Year"] = year
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Loaded {len(df_all)} papers")


# =========================================================
# 2. Load SPECTER2
# =========================================================
print("Loading SPECTER2...")

tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

model.to(DEVICE)
model.eval()


# =========================================================
# 3. Encoding Function (mean pooling)
# =========================================================
def encode_specter2(texts, batch_size=8, max_length=512):
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(DEVICE)

            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


# =========================================================
# 4. Encode Papers
# =========================================================
texts = (
    df_all["Article Title"].fillna("") + ". " +
    df_all["Abstract"].fillna("")
).tolist()

doc_embeddings = encode_specter2(texts)
print("Document embeddings shape:", doc_embeddings.shape)


# =========================================================
# 5. Discipline Anchors
# =========================================================
DISCIPLINE_ANCHORS = {
    "Agriculture, Biology & Environmental Sciences": [
        "Agriculture/Agronomy", "Agricultural Chemistry", "Animal Sciences",
        "Aquatic Sciences", "Biology", "Biotechnology & Applied Microbiology",
        "Entomology/Pest Control", "Environment/Ecology",
        "Food Science/Nutrition", "Multidisciplinary",
        "Plant Sciences", "Veterinary Medicine/Animal Health"
    ],
    "Arts & Humanities": [
        "Archaeology", "Art & Architecture", "Classical Studies",
        "General", "History", "Language & Linguistics",
        "Literature", "Performing Arts", "Philosophy",
        "Religion & Theology"
    ],
    "Business & Economics": [
        "Accounting & Finance", "Business & Economics",
        "Business Law & Reviews", "Computer Technology & Information Systems",
        "Employee Relations & Human Resources",
        "Management & Organization",
        "Marketing & Business Communication",
        "Political Science",
        "Public Administration & Development"
    ],
    "Clinical Medicine": [
        "Anesthesia & Intensive Care",
        "Cardiovascular & Respiratory Systems",
        "Clinical Immunology & Infectious Disease",
        "Clinical Psychology & Psychiatry",
        "Dentistry/Oral Surgery & Medicine",
        "Dermatology",
        "Endocrinology, Metabolism & Nutrition",
        "Environmental Medicine & Public Health",
        "Gastroenterology & Hepatology",
        "General & Internal Medicine",
        "Health Care Sciences & Services",
        "Hematology", "Neurology", "Nursing", "Oncology",
        "Ophthalmology",
        "Orthopedics, Rehabilitation & Sports Medicine",
        "Otolaryngology", "Pediatrics",
        "Pharmacology/Toxicology",
        "Radiology, Nuclear Medicine & Imaging",
        "Reproductive Medicine",
        "Research/Laboratory Medicine & Medical Technology",
        "Rheumatology",
        "Surgery, Urology & Nephrology"
    ],
    "Electronics & Telecommunications": [
        "Applied Physics", "Computer Science",
        "Electronics & Electrical Engineering",
        "Optics & Laser Research",
        "Semiconductors & Solid State Materials",
        "Signal Processing/Circuits & Systems",
        "Telecommunications Technology",
        "Technology R&D/Management"
    ],
    "Engineering, Computing & Technology": [
        "Aerospace Engineering", "Artificial Intelligence",
        "Deep Learning", "Machine Learning",
        "Natural Language Processing",
        "Robotics & Automatic Control",
        "Chemical Engineering", "Civil Engineering",
        "Computer Science & Engineering",
        "Electrical & Electronics Engineering",
        "Engineering Management/General",
        "Engineering Mathematics",
        "Environmental Engineering & Energy",
        "Geological, Petroleum, & Mining Engineering",
        "Information Technology & Communication Systems",
        "Instrumentation & Measurement",
        "Materials Science & Engineering",
        "Mechanical Engineering", "Metallurgy",
        "Nuclear Engineering", "Optics & Acoustics"
    ],
    "Life Sciences": [
        "Animal & Plant Sciences", "Biochemistry & Biophysics",
        "Cardiovascular & Hematology Research",
        "Cell & Developmental Biology",
        "Chemistry & Analysis",
        "Endocrinology, Nutrition & Metabolism",
        "Experimental Biology", "Immunology",
        "Medical Research, Diagnosis & Treatment",
        "Medical Research, General Topics",
        "Medical Research, Organs & Systems",
        "Microbiology, Molecular Biology & Genetics",
        "Neurosciences & Behavior",
        "Oncogenesis & Cancer Research",
        "Pharmacology & Toxicology", "Physiology"
    ],
    "Physical, Chemical & Earth Sciences": [
        "Applied Physics/Condensed Matter/Materials Science",
        "Chemistry", "Earth Sciences",
        "Inorganic & Nuclear Chemistry",
        "Mathematics, Multidisciplinary",
        "Organic Chemistry/Polymer Science",
        "Physical Chemistry/Chemical Physics",
        "Physics, Space Science",
        "Spectroscopy/Instrumentation/Analytical Sciences"
    ],
    "Social & Behavioral Sciences": [
        "Anthropology", "Communication", "Economics",
        "Education", "Environmental Studies",
        "Geography & Development", "Law",
        "Library & Information Sciences", "Management",
        "Political Science & Public Administration",
        "Psychiatry", "Psychology",
        "Public Health & Health Care Science",
        "Rehabilitation, Social Work & Social Policy",
        "Sociology & Social Sciences"
    ]
}

# =========================================================
# Anchor Text
# =========================================================
anchor_names = list(DISCIPLINE_ANCHORS.keys())
anchor_texts = [", ".join(sublist) for sublist in DISCIPLINE_ANCHORS.values()]

anchor_embeddings = encode_specter2(anchor_texts)
anchor_emb_dict = dict(zip(anchor_names, anchor_embeddings))


# =========================================================
# 6. Compute Similarity
# =========================================================
def compute_scores(doc_emb):
    return {
        d: float(cosine_similarity(doc_emb.reshape(1, -1), emb.reshape(1, -1))[0][0])
        for d, emb in anchor_emb_dict.items()
    }

df_all["Discipline_Scores"] = [compute_scores(e) for e in doc_embeddings]


# =========================================================
# 7. Discipline Selection Algorithm
# =========================================================
def select_disciplines_paper_level(score_dict,
                                   second_ratio=0.95,
                                   second_min=0.80,
                                   third_ratio=0.95,
                                   third_min=0.80):

    if not score_dict:
        return []

    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    disciplines, sims = zip(*sorted_items)

    selected = [disciplines[0]]

    if len(sims) > 1 and sims[1] >= second_ratio * sims[0] and sims[1] >= second_min:
        selected.append(disciplines[1])

    if len(sims) > 2 and len(selected) == 2 and sims[2] >= third_ratio * sims[1] and sims[2] >= third_min:
        selected.append(disciplines[2])

    return selected


df_all["Present_Disciplines"] = df_all["Discipline_Scores"].apply(select_disciplines_paper_level)
df_all["Presence_Breadth"] = df_all["Present_Disciplines"].apply(len)

breadth_counts = df_all["Presence_Breadth"].value_counts().sort_index()
print("\nPresence_Breadth Statistics")
for n in [1, 2, 3]:
    print(f"{n} disciplines: {breadth_counts.get(n, 0)} papers")

breadth_ratio = df_all["Presence_Breadth"].value_counts(normalize=True).sort_index()
print("\nPresence_Breadth Ratio")
for n in [1, 2, 3]:
    print(f"{n} disciplines: {breadth_ratio.get(n, 0)*100:.2f}%")


# =========================================================
# 8. Threshold Sensitivity Analysis
# =========================================================
print("\nRunning threshold grid search...")

df_grid = df_all.copy()

ratio_list = np.arange(0.85, 0.99, 0.01)
min_list   = np.arange(0.70, 0.91, 0.02)

results = []
scores_list = df_grid["Discipline_Scores"].tolist()

for r in ratio_list:
    for m in min_list:

        present_disciplines = [
            select_disciplines_paper_level(scores, r, m, r, m)
            for scores in scores_list
        ]

        presence_breadth = [len(p) for p in present_disciplines]

        multi_rate = np.mean([b >= 2 for b in presence_breadth])
        single_rate = np.mean([b == 1 for b in presence_breadth])

        top1_sim = np.mean([max(scores.values()) for scores in scores_list])

        selected_sim = np.mean([
            np.mean([scores[d] for d in selected])
            for scores, selected in zip(scores_list, present_disciplines)
            if len(selected) > 0
        ])

        all_disciplines = [d for p in present_disciplines for d in p]
        if len(all_disciplines) > 0:
            counts = Counter(all_disciplines)
            p = np.array(list(counts.values()))
            p = p / p.sum()
            entropy_score = -np.sum(p * np.log(p + 1e-10))
        else:
            entropy_score = 0

        results.append({
            "ratio": r,
            "min": m,
            "multi_rate": multi_rate,
            "single_rate": single_rate,
            "top1_sim": top1_sim,
            "selected_sim": selected_sim,
            "entropy": entropy_score
        })

results_df = pd.DataFrame(results)

print("Grid search finished")

OUT_DIR = "D:/Desktop/code/threshold/3.25"
os.makedirs(OUT_DIR, exist_ok=True)

results_df.to_excel(
    os.path.join(OUT_DIR, "3.26_threshold_grid_search.xlsx"),
    index=False
)

print("\nGrid search results saved")