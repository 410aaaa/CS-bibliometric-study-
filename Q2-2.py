# =========================================================
# 0. Environment and seed setup
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
# Set random seeds
# --------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------
# Device
# --------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =========================================================
# 1. Load data (single pre‑processed file)
# =========================================================
all_dfs = []
path = "C:/Users/LLT/Desktop/data/out1/20162025_all_predict_article_Hybrid(1).xlsx"
df = pd.read_excel(path)
if "Article Title" not in df.columns or "Abstract" not in df.columns:
    raise ValueError(f"Missing required columns in {path}")

all_dfs.append(df)
df_all = pd.concat(all_dfs, ignore_index=True)
print(f"Loaded {len(df_all)} papers")

# =========================================================
# 2. Load SPECTER2 (no manual adapter activation)
# =========================================================
print("Loading SPECTER2...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.to(DEVICE)
model.eval()

# =========================================================
# 3. Encoding function (mean pooling)
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
# 4. Encode papers
# =========================================================
texts = (
    df_all["Article Title"].fillna("") + ". " +
    df_all["Abstract"].fillna("")
).tolist()
doc_embeddings = encode_specter2(texts)
print("Document embeddings shape:", doc_embeddings.shape)

# =========================================================
# 5. Discipline anchors
# =========================================================
DISCIPLINE_ANCHORS = {
    "Agriculture, Biology & Environmental Sciences": [
        "Agriculture/Agronomy",
        "Agricultural Chemistry",
        "Animal Sciences",
        "Aquatic Sciences",
        "Biology",
        "Biotechnology & Applied Microbiology",
        "Entomology/Pest Control",
        "Environment/Ecology",
        "Food Science/Nutrition",
        "Multidisciplinary",
        "Plant Sciences",
        "Veterinary Medicine/Animal Health"
    ],
    "Arts & Humanities": [
        "Archaeology",
        "Art & Architecture",
        "Classical Studies",
        "General",
        "History",
        "Language & Linguistics",
        "Literature",
        "Performing Arts",
        "Philosophy",
        "Religion & Theology"
    ],
    "Business & Economics": [
        "Accounting & Finance",
        "Business & Economics",
        "Business Law & Reviews",
        "Computer Technology & Information Systems",
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
        "Hematology",
        "Neurology",
        "Nursing",
        "Oncology",
        "Ophthalmology",
        "Orthopedics, Rehabilitation & Sports Medicine",
        "Otolaryngology",
        "Pediatrics",
        "Pharmacology/Toxicology",
        "Radiology, Nuclear Medicine & Imaging",
        "Reproductive Medicine",
        "Research/Laboratory Medicine & Medical Technology",
        "Rheumatology",
        "Surgery, Urology & Nephrology"
    ],
    "Electronics & Telecommunications": [
        "Applied Physics",
        "Computer Science",
        "Electronics & Electrical Engineering",
        "Optics & Laser Research",
        "Semiconductors & Solid State Materials",
        "Signal Processing/Circuits & Systems",
        "Telecommunications Technology",
        "Technology R&D/Management"
    ],
    "Engineering, Computing & Technology": [
        "Aerospace Engineering",
        "Artificial Intelligence",
        "Deep Learning",
        "Machine Learning",
        "Natural Language Processing",
        "Robotics & Automatic Control",
        "Chemical Engineering",
        "Civil Engineering",
        "Computer Science & Engineering",
        "Electrical & Electronics Engineering",
        "Engineering Management/General",
        "Engineering Mathematics",
        "Environmental Engineering & Energy",
        "Geological, Petroleum, & Mining Engineering",
        "Information Technology & Communication Systems",
        "Instrumentation & Measurement",
        "Materials Science & Engineering",
        "Mechanical Engineering",
        "Metallurgy",
        "Nuclear Engineering",
        "Optics & Acoustics"
    ],
    "Life Sciences": [
        "Animal & Plant Sciences",
        "Biochemistry & Biophysics",
        "Cardiovascular & Hematology Research",
        "Cell & Developmental Biology",
        "Chemistry & Analysis",
        "Endocrinology, Nutrition & Metabolism",
        "Experimental Biology",
        "Immunology",
        "Medical Research, Diagnosis & Treatment",
        "Medical Research, General Topics",
        "Medical Research, Organs & Systems",
        "Microbiology, Molecular Biology & Genetics",
        "Neurosciences & Behavior",
        "Oncogenesis & Cancer Research",
        "Pharmacology & Toxicology",
        "Physiology"
    ],
    "Physical, Chemical & Earth Sciences": [
        "Applied Physics/Condensed Matter/Materials Science",
        "Chemistry",
        "Earth Sciences",
        "Inorganic & Nuclear Chemistry",
        "Mathematics, Multidisciplinary",
        "Organic Chemistry/Polymer Science",
        "Physical Chemistry/Chemical Physics",
        "Physics, Space Science",
        "Spectroscopy/Instrumentation/Analytical Sciences"
    ],
    "Social & Behavioral Sciences": [
        "Anthropology",
        "Communication",
        "Economics",
        "Education",
        "Environmental Studies",
        "Geography & Development",
        "Law",
        "Library & Information Sciences",
        "Management",
        "Political Science & Public Administration",
        "Psychiatry",
        "Psychology",
        "Public Health & Health Care Science",
        "Rehabilitation, Social Work & Social Policy",
        "Sociology & Social Sciences"
    ]
}

# Build anchor texts
anchor_names = list(DISCIPLINE_ANCHORS.keys())
anchor_texts = [", ".join(sublist) for sublist in DISCIPLINE_ANCHORS.values()]
anchor_embeddings = encode_specter2(anchor_texts)
anchor_emb_dict = dict(zip(anchor_names, anchor_embeddings))

# =========================================================
# 6. Compute similarity scores
# =========================================================
def compute_scores(doc_emb):
    return {
        d: float(
            cosine_similarity(
                doc_emb.reshape(1, -1),
                emb.reshape(1, -1)
            )[0][0]
        )
        for d, emb in anchor_emb_dict.items()
    }

df_all["Discipline_Scores"] = [compute_scores(e) for e in doc_embeddings]

# =========================================================
# 7. Discipline selection (strict + robust)
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

df_all["Present_Disciplines"] = df_all["Discipline_Scores"].apply(
    select_disciplines_paper_level
)
df_all["Presence_Breadth"] = df_all["Present_Disciplines"].apply(len)

# =========================================================
# 8. Statistics and plots
# =========================================================
breadth_counts = df_all["Presence_Breadth"].value_counts().sort_index()
print("\n📊 Presence_Breadth Statistics")
for b, count in breadth_counts.items():
    print(f"{b} disciplines: {count} articles ({count/len(df_all)*100:.2f}%)")

# Count articles per discipline
discipline_counts = {}
for disciplines in df_all["Present_Disciplines"]:
    for d in disciplines:
        discipline_counts[d] = discipline_counts.get(d, 0) + 1
discipline_counts = pd.Series(discipline_counts).sort_values(ascending=False)

# Bar plot
plt.figure(figsize=(12, 6))
sns.barplot(
    x=discipline_counts.index,
    y=discipline_counts.values,
    color="mediumslateblue",
    edgecolor="black"
)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Discipline")
plt.ylabel("Number of Articles")
plt.title("Number of Articles per Discipline")
plt.tight_layout()
plt.show()

# =========================================================
# 9. Sensitivity analysis (threshold sweep)
# =========================================================
thresholds = np.arange(0.75, 0.95, 0.01)
breadth_ratio = []

for t in thresholds:
    df_all["Present_Disciplines"] = df_all["Discipline_Scores"].apply(
        lambda x: select_disciplines_paper_level(x,
                                                 second_ratio=t,
                                                 second_min=t,
                                                 third_ratio=t,
                                                 third_min=t)
    )
    # Recalculate breadth for each threshold
    df_all["Presence_Breadth"] = df_all["Present_Disciplines"].apply(len)
    breadth_ratio.append(df_all["Presence_Breadth"].value_counts(normalize=True).to_dict())

# Plot sensitivity
plt.plot(thresholds, [b.get(3, 0) for b in breadth_ratio], label="3 disciplines")
plt.plot(thresholds, [b.get(2, 0) for b in breadth_ratio], label="2 disciplines")
plt.plot(thresholds, [b.get(1, 0) for b in breadth_ratio], label="1 discipline")
plt.xlabel("Threshold")
plt.ylabel("Proportion of papers")
plt.legend()
plt.show()

# =========================================================
# 10. Save results
# =========================================================
OUT_DIR = "C:/Users/LLT/Desktop/data/out1/"
os.makedirs(OUT_DIR, exist_ok=True)
df_all.to_excel(
    os.path.join(OUT_DIR, "paper_level_discipline_assignment.xlsx"),
    index=False
)
print(f"\n✅ Results saved to: {OUT_DIR}")