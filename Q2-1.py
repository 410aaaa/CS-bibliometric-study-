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

# Ensure deterministic behavior
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
# years = range(2016, 2026)
# all_dfs = []
# for year in years:
#     path = f"D:/桌面/code/CNS data/{year}/{year}_predict_article.xlsx"
#     if not os.path.exists(path):
#         print(f"⚠️ Missing: {path}")
#         continue
#     df = pd.read_excel(path)
#     if "Article Title" not in df.columns or "Abstract" not in df.columns:
#         raise ValueError(f"Missing required columns in {path}")
#     df["Publication Year"] = year
#     all_dfs.append(df)
# df_all = pd.concat(all_dfs, ignore_index=True)

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

# Statistics
breadth_counts = df_all["Presence_Breadth"].value_counts().sort_index()
print("\n📊 Presence_Breadth Statistics")
for n in [1, 2, 3]:
    count = breadth_counts.get(n, 0)
    print(f"{n} disciplines: {count} papers")

breadth_ratio = df_all["Presence_Breadth"].value_counts(normalize=True).sort_index()
print("\n📊 Presence_Breadth Proportion")
for n in [1, 2, 3]:
    ratio = breadth_ratio.get(n, 0)
    print(f"{n} disciplines: {ratio*100:.2f}%")

# =========================================================
# 8. Threshold grid search (fixed version)
# =========================================================
print("\n🔬 Running threshold grid search...")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Work on a copy to avoid modifying original
df_grid = df_all.copy()

# Parameter grid
ratio_list = np.arange(0.85, 0.99, 0.01)
min_list   = np.arange(0.70, 0.91, 0.02)

def select_disciplines_paper_level(score_dict,
                                   second_ratio,
                                   second_min,
                                   third_ratio,
                                   third_min):
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

# Grid search
results = []
scores_list = df_grid["Discipline_Scores"].tolist()

for r in ratio_list:
    for m in min_list:
        present_disciplines = [
            select_disciplines_paper_level(
                scores,
                second_ratio=r,
                second_min=m,
                third_ratio=r,
                third_min=m
            )
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
print("✅ Grid search finished")

# (Optional: uncomment the following blocks to generate heatmaps and Pareto plots)
# pivot_multi = results_df.pivot(index="ratio", columns="min", values="multi_rate")
# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_multi, cmap="viridis")
# plt.title("Multidisciplinary Rate")
# plt.xlabel("Absolute Threshold")
# plt.ylabel("Relative Threshold")
# plt.show()
#
# pivot_sim = results_df.pivot(index="ratio", columns="min", values="selected_sim")
# plt.figure(figsize=(10, 6))
# sns.heatmap(pivot_sim, cmap="magma")
# plt.title("Selected Similarity")
# plt.xlabel("Absolute Threshold")
# plt.ylabel("Relative Threshold")
# plt.show()
#
# plt.figure(figsize=(7, 6))
# plt.scatter(results_df["multi_rate"], results_df["selected_sim"])
# plt.xlabel("Multidisciplinary Rate")
# plt.ylabel("Average Similarity")
# plt.title("Pareto Trade-off")
# target = results_df[(np.isclose(results_df["ratio"], 0.95)) & (np.isclose(results_df["min"], 0.80))]
# if len(target) > 0:
#     x = target["multi_rate"].values[0]
#     y = target["selected_sim"].values[0]
#     plt.scatter(x, y)
#     plt.text(x, y, " (0.95, 0.80)")
# plt.show()

# Current parameter performance
current = results_df[(np.isclose(results_df["ratio"], 0.95)) & (np.isclose(results_df["min"], 0.80))]
print("\n📌 Your Current Setting (0.95, 0.80):")
print(current)

# Save grid search results
OUT_DIR = "C:/Users/LLT/Desktop/data/question2-1/"
os.makedirs(OUT_DIR, exist_ok=True)
results_df.to_excel(
    os.path.join(OUT_DIR, "3.26_threshold_grid_search.xlsx"),
    index=False
)
print("\n✅ Grid search results saved")

# =========================================================
# 9. Discipline trends by year
# =========================================================
print("\n📈 Generating discipline trend files...")
OUT_RESULT_DIR = "C:/Users/LLT/Desktop/data/question2-1/"
os.makedirs(OUT_RESULT_DIR, exist_ok=True)

# Expand paper-discipline relationships
discipline_records = []
for _, row in df_all.iterrows():
    year = row["Publication Year"]
    for discipline in row["Present_Disciplines"]:
        discipline_records.append({
            "Publication Year": year,
            "Discipline": discipline
        })
discipline_long_df = pd.DataFrame(discipline_records)

# 1. Counts by year and discipline
discipline_year_counts = (
    discipline_long_df
    .groupby(["Publication Year", "Discipline"])
    .size()
    .reset_index(name="Paper Count")
)
discipline_year_counts.to_excel(
    os.path.join(OUT_RESULT_DIR, "discipline_trends_by_year_counts.xlsx"),
    index=False
)
print("✅ discipline trends by year saved")

# 2. Percentage distribution
year_total = (
    discipline_year_counts
    .groupby("Publication Year")["Paper Count"]
    .sum()
    .reset_index(name="Total Discipline Assignment")
)
discipline_percentage = discipline_year_counts.merge(year_total, on="Publication Year")
discipline_percentage["Percentage (%)"] = (
    discipline_percentage["Paper Count"] /
    discipline_percentage["Total Discipline Assignment"] * 100
)
discipline_percentage = discipline_percentage.sort_values(
    ["Publication Year", "Percentage (%)"], ascending=[True, False]
)
discipline_percentage.to_excel(
    os.path.join(OUT_RESULT_DIR, "percentage_distribution_of_each_discipline.xlsx"),
    index=False
)
print("✅ discipline percentage distribution saved")

# 3. Single vs Interdisciplinary
single_interdisciplinary = (
    df_all
    .groupby(["Publication Year", "Presence_Breadth"])
    .size()
    .reset_index(name="Paper Count")
)
single_interdisciplinary["Research Type"] = (
    single_interdisciplinary["Presence_Breadth"]
    .apply(lambda x: "Single-disciplinary" if x == 1 else "Interdisciplinary")
)
single_interdisciplinary = single_interdisciplinary.drop(columns=["Presence_Breadth"])
year_paper_total = df_all.groupby("Publication Year").size().reset_index(name="Total Papers")
single_interdisciplinary = single_interdisciplinary.merge(year_paper_total, on="Publication Year")
single_interdisciplinary["Percentage (%)"] = (
    single_interdisciplinary["Paper Count"] /
    single_interdisciplinary["Total Papers"] * 100
)
single_interdisciplinary.to_excel(
    os.path.join(OUT_RESULT_DIR, "single_vs_interdisciplinary.xlsx"),
    index=False
)
print("✅ single vs interdisciplinary saved")

# =========================================================
# 10. Output summary
# =========================================================
print("\n==============================")
print("Generated files:")
print("==============================")
print("1. discipline_trends_by_year_counts.xlsx")
print("2. percentage_distribution_of_each_discipline.xlsx")
print("3. single_vs_interdisciplinary.xlsx")
print("4. threshold_grid_search.xlsx")
print("==============================")