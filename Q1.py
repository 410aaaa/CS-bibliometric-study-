# ====================== Imports ======================
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix
)

# ====================== Device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print(f"🚀 Using device: {device}")
print("=" * 70)

# ====================== Load SPECTER2 ======================
print("\n📥 Loading SPECTER2 Classification model...")
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
model.load_adapter(
    "allenai/specter2_classification",
    source="hf",
    load_as="classification",
    set_active=True
)
model.to(device)
model.eval()

# ====================== SPECTER2 encoding function (CLS + Mean pooling) ======================
def encode_specter2_classification(texts, batch_size=8, max_length=512):
    texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if len(texts) == 0:
        return np.empty((0, model.config.hidden_size * 2))
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding (SPECTER2)"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_token_type_ids=False
            ).to(device)
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            cls_emb = hidden[:, 0, :]          # CLS token
            mean_emb = hidden.mean(dim=1)      # mean pooling
            emb = torch.cat([cls_emb, mean_emb], dim=1)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

# ====================== Keyword system ======================
VERY_STRONG = [
    r'\bLarge Language Model(s)?\b',
    r'\bLLM(s)?\b',
    r'\bTransformer(s)?\b'
]
MEDIUM_STRONG = [
    r'\bDeep Learning\b',
    r'\bNeural Network(s)?\b',
    r'\bMachine Learning\b',
    r'\bNatural Language Processing\b',
    r'\bNLP\b',
    r'\bComputer Vision\b',
    r'\bReinforcement Learning\b',
    r'\bAI\b'
]
very_pattern = re.compile("|".join(VERY_STRONG), re.I)
medium_pattern = re.compile("|".join(MEDIUM_STRONG), re.I)

def keyword_score(text):
    text = str(text)
    if very_pattern.search(text):
        return 0.90
    if medium_pattern.search(text):
        return 0.75
    return 0.0

# ====================== Load training data ======================
print("\n📚 Loading training data...")
train_df = pd.read_excel("C:/Users/LLT/Desktop/data/train_data(only).xls")
train_df["is_computer"] = pd.to_numeric(train_df["is_computer"], errors="coerce")
train_df = train_df[train_df["is_computer"].isin([0, 1])]
train_df["is_computer"] = train_df["is_computer"].astype(int)

# ====================== Build text ======================
train_df["text"] = (
    train_df["Article Title"].fillna("")
    + tokenizer.sep_token
    + train_df["Abstract"].fillna("")
)
train_df = train_df[train_df["text"].str.strip().ne("")]
print(f"Training samples: {len(train_df)}")

# ====================== Labels ======================
y_all = train_df["is_computer"].values

# ====================== Encode all training data (Scheme B) ======================
print("\n🔬 Encoding all training data...")
X_all = encode_specter2_classification(train_df["text"].tolist())
print("\nEmbedding shape:", X_all.shape)
print("\n✅ Part 1 finished")
print("=" * 70)

# ============================================================
# Part 2/4: XGBoost Training + Hybrid Validation
# ============================================================

# ====================== Train/validation split ======================
print("\n📚 Splitting train / validation...")
X_train, X_val, y_train, y_val, text_train, text_val = train_test_split(
    X_all,
    y_all,
    train_df["text"].values,
    test_size=0.2,
    stratify=y_all,
    random_state=42
)
print(f"Training size: {len(y_train)}")
print(f"Validation size: {len(y_val)}")

# ====================== Class weight ======================
scale_weight = ((y_train == 0).sum() / max((y_train == 1).sum(), 1))
print(f"⚖️ scale_pos_weight = {scale_weight:.2f}")

# ====================== XGBoost ======================
print("\n🤖 Training XGBoost...")
clf = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="auc",
    scale_pos_weight=scale_weight,
    random_state=42
)
clf.fit(X_train, y_train)
print("✅ XGBoost training finished")

# ============================================================
# Hybrid Validation
# ============================================================
print("\n==============================")
print("Hybrid Validation Evaluation")
print("==============================")

y_val_proba = clf.predict_proba(X_val)[:, 1]
val_keyword_score = np.array([keyword_score(text) for text in text_val])
val_final_score = np.maximum(y_val_proba, val_keyword_score)

# ====================== Threshold search ======================
thresholds = np.linspace(0.2, 0.8, 61)
best_threshold = 0.5
best_f1 = 0
for t in thresholds:
    y_pred = (val_final_score >= t).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_val_pred = (val_final_score >= best_threshold).astype(int)

print("\n🎯 Hybrid Validation Results")
print(classification_report(y_val, y_val_pred))
hybrid_auc = roc_auc_score(y_val, val_final_score)
hybrid_f1 = f1_score(y_val, y_val_pred)
print(f"Hybrid AUC: {hybrid_auc:.4f}")
print(f"Hybrid F1: {hybrid_f1:.4f}")
print(f"Best Hybrid threshold: {best_threshold:.2f}")

# ============================================================
# Confusion matrix
# ============================================================
cm = confusion_matrix(y_val, y_val_pred)
tn, fp, fn, tp = cm.ravel()
print("\n📊 Confusion Matrix")
print(cm)
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

# ============================================================
# Validation prediction source analysis
# ============================================================
print("\n📊 Validation Prediction Source Analysis")
source_list = []
for pred, model_s, keyword_s in zip(y_val_pred, y_val_proba, val_keyword_score):
    if pred == 0:
        continue
    if keyword_s > model_s:
        source_list.append("keyword_only")
    elif model_s > keyword_s:
        source_list.append("model_only")
    else:
        source_list.append("both")
source_df = pd.DataFrame({"source": source_list})
print(source_df["source"].value_counts())
print("\nPercentage (%)")
print(source_df["source"].value_counts(normalize=True).mul(100).round(2))

# ============================================================
# Validation threshold sensitivity
# ============================================================
print("\n📊 Threshold Sensitivity Analysis")
sensitivity_thresholds = [0.5, 0.6, 0.65, 0.7, 0.8]
val_results = []
for t in sensitivity_thresholds:
    y_pred = (val_final_score >= t).astype(int)
    report = classification_report(y_val, y_pred, output_dict=True)
    val_results.append({
        "threshold": t,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": f1_score(y_val, y_pred)
    })
val_sensitivity_df = pd.DataFrame(val_results)
print(val_sensitivity_df)
print("\n✅ Part 2 finished")

# ============================================================
# Part 3/4: Test Set Prediction + Hybrid Screening
# ============================================================
print("\n📥 Loading test data...")
# test_df = pd.read_excel("C:/Users/LLT/Desktop/data/2016_2025_data_article.xlsx")
test_df = pd.read_excel("C:/Users/LLT/Desktop/data/2016/2016.xls")

# ============================================================
# Metadata Cleaning Pipeline (WoS CNS, 2016-2025)
# ============================================================
print("\n" + "="*80)
print("🧹 Metadata Cleaning Pipeline")
print("="*80)

initial_count = len(test_df)
print(f"\nInitial records: {initial_count:,}")

# Backup original fields
test_df["Original_Document_Type"] = test_df["Document Type"] if "Document Type" in test_df.columns else None
test_df["Original_Source_Title"] = test_df["Source Title"] if "Source Title" in test_df.columns else None

removed_records = []

def save_removed(df, reason):
    if len(df) > 0:
        temp = df.copy()
        temp["Removal_reason"] = reason
        removed_records.append(temp)

# 1. Source Title normalization
print("\n🔧 Step 1: Source Title normalization")
if "Source Title" in test_df.columns:
    test_df["Source Title"] = test_df["Source Title"].astype(str).str.strip().str.upper()
    source_mapping = {
        "SCIENCE (NEW YORK, N.Y.)": "SCIENCE",
        "SCIENCE": "SCIENCE",
        "NATURE": "NATURE",
        "CELL": "CELL"
    }
    test_df["Source Title"] = test_df["Source Title"].replace(source_mapping)
    print("✅ Source Title normalized")

# 2. Remove invalid document types
print("\n🚫 Step 2: Removing invalid document types")
invalid_pattern = (
    "Retracted Publication|Retraction|News|Book Review|"
    "Editorial|Interview|Correction|Meeting Abstract"
)
invalid_mask = test_df["Document Type"].astype(str).str.contains(invalid_pattern, case=False, regex=True, na=False)
removed_invalid = test_df[invalid_mask].copy()
save_removed(removed_invalid, "Invalid document type")
test_df = test_df[~invalid_mask]
print(f"Removed invalid document types: {len(removed_invalid)}")

# 3. Publication Year filtering
print("\n📅 Step 3: Publication year filtering")
test_df["Publication Year"] = pd.to_numeric(test_df["Publication Year"], errors="coerce")
missing_year_mask = test_df["Publication Year"].isna()
removed_missing_year = test_df[missing_year_mask].copy()
save_removed(removed_missing_year, "Missing publication year")
test_df = test_df[~missing_year_mask]
outside_year_mask = (test_df["Publication Year"] < 2016) | (test_df["Publication Year"] > 2025)
removed_year = test_df[outside_year_mask].copy()
save_removed(removed_year, "Publication year outside 2016-2025")
test_df = test_df[~outside_year_mask]
print(f"Removed missing year: {len(removed_missing_year)}")
print(f"Removed outside 2016-2025: {len(removed_year)}")

# 4. Document Type normalization
print("\n📄 Step 4: Document Type normalization")
def normalize_document_type(doc_type):
    if pd.isna(doc_type):
        return None
    doc_type = str(doc_type).lower()
    if "review" in doc_type:
        return "review"
    elif "article" in doc_type or "journal article" in doc_type:
        return "article"
    else:
        return doc_type
test_df["Document Type"] = test_df["Document Type"].apply(normalize_document_type)
print(test_df["Document Type"].value_counts())

# 5. Keep Article and Review only
print("\n📚 Step 5: Keep article and review")
unsupported_mask = ~test_df["Document Type"].isin(["article", "review"])
removed_type = test_df[unsupported_mask].copy()
save_removed(removed_type, "Unsupported document type")
test_df = test_df[~unsupported_mask]
print(f"Removed unsupported types: {len(removed_type)}")

# 6. Missing title and abstract removal
print("\n📝 Step 6: Missing title/abstract filtering")
missing_text_mask = test_df["Article Title"].isna() | test_df["Abstract"].isna()
removed_missing_text = test_df[missing_text_mask].copy()
save_removed(removed_missing_text, "Missing title or abstract")
test_df = test_df[~missing_text_mask]
print(f"Removed missing title/abstract: {len(removed_missing_text)}")

# 7. Duplicate removal
print("\n🔁 Step 7: Duplicate removal")
if "UT" in test_df.columns:
    ut_duplicate_mask = test_df.duplicated(subset=["UT"], keep="first")
    removed_ut = test_df[ut_duplicate_mask].copy()
    save_removed(removed_ut, "Duplicate identical UT")
    test_df = test_df[~ut_duplicate_mask]
    print(f"Removed duplicate UT: {len(removed_ut)}")
test_df["Article Title_clean"] = test_df["Article Title"].astype(str).str.strip().str.lower()
title_year_duplicate_mask = test_df.duplicated(subset=["Article Title_clean", "Publication Year"], keep="first")
removed_title_year = test_df[title_year_duplicate_mask].copy()
save_removed(removed_title_year, "Duplicate normalized title and year")
test_df = test_df[~title_year_duplicate_mask]
print(f"Removed duplicate title-year: {len(removed_title_year)}")

# 8. CNS journal verification
print("\n🧬 Step 8: CNS journal verification")
non_cns_mask = ~test_df["Source Title"].isin(["NATURE", "SCIENCE", "CELL"])
removed_non_cns = test_df[non_cns_mask].copy()
save_removed(removed_non_cns, "Non-CNS source")
test_df = test_df[~non_cns_mask]
print(f"Removed non-CNS records: {len(removed_non_cns)}")

# 9. Save cleaning log
print("\n💾 Saving cleaning log")
if len(removed_records) > 0:
    cleaning_log = pd.concat(removed_records, ignore_index=True)
    cleaning_log.to_excel("data_cleaning_removed_records.xlsx", index=False)
    print("Saved: data_cleaning_removed_records.xlsx")

# 10. Build SPECTER2 input text
print("\n🤖 Building SPECTER2 text")
test_df["text"] = (
    test_df["Article Title"].fillna("")
    + tokenizer.sep_token
    + test_df["Abstract"].fillna("")
)
empty_text_mask = test_df["text"].str.strip().eq("")
test_df = test_df[~empty_text_mask]

print("\n" + "="*80)
print("📊 Final Clean Corpus")
print("="*80)
print(f"Initial records: {initial_count:,}")
print(f"Final records: {len(test_df):,}")
print(f"Total removed: {initial_count-len(test_df):,}")
print("\nClean corpus ready for SPECTER2.")
print("="*80)

# ============================================================
# SPECTER2 encoding
# ============================================================
print("\n🔬 Encoding test documents...")
X_test = encode_specter2_classification(test_df["text"].tolist())

# ============================================================
# XGBoost prediction
# ============================================================
print("\n🤖 Predicting with XGBoost...")
test_df["model_score"] = clf.predict_proba(X_test)[:, 1]

# ============================================================
# Keyword score
# ============================================================
test_df["keyword_score"] = test_df["text"].apply(keyword_score)

# ============================================================
# Hybrid score
# ============================================================
test_df["final_score"] = np.maximum(test_df["model_score"], test_df["keyword_score"])

# ============================================================
# Final screening
# ============================================================
CUSTOM_THRESHOLD = max(0.35, best_threshold)
print("\n🎯 Final screening threshold:", round(CUSTOM_THRESHOLD, 2))
df_final = test_df[test_df["final_score"] >= CUSTOM_THRESHOLD].copy()
df_final["predicted_is_computer"] = 1
df_final = df_final.sort_values("final_score", ascending=False)

# ============================================================
# Final source analysis
# ============================================================
print("\n📊 Final Corpus Source Analysis")
final_sources = []
for _, row in df_final.iterrows():
    if row["keyword_score"] > row["model_score"]:
        final_sources.append("keyword_only")
    elif row["model_score"] > row["keyword_score"]:
        final_sources.append("model_only")
    else:
        final_sources.append("both")
df_final["prediction_source"] = final_sources
print(df_final["prediction_source"].value_counts())
print("\nPercentage (%)")
print(df_final["prediction_source"].value_counts(normalize=True).mul(100).round(2))

# ============================================================
# Test threshold sensitivity
# ============================================================
print("\n📊 Test Threshold Sensitivity")
test_sensitivity_results = []
for t in sensitivity_thresholds:
    temp_df = test_df[test_df["final_score"] >= t]
    count_ai = len(temp_df)
    proportion = count_ai / initial_count
    test_sensitivity_results.append({
        "threshold": t,
        "ai_count": count_ai,
        "proportion": proportion
    })
test_sensitivity_df = pd.DataFrame(test_sensitivity_results)
print(test_sensitivity_df)
print("\n✅ Part 3 finished")

# ============================================================
# Part 4/4: Final Statistics + Save Results
# ============================================================
print("\n" + "=" * 70)
print("📊 FINAL CORPUS STATISTICS")
print("=" * 70)

final_count = len(df_final)
print(f"Original documents: {initial_count:,}")
print(f"Documents after filtering: {len(test_df):,}")
print(f"Final identified computer science papers: {final_count:,}")
print(f"Final proportion: {(final_count / initial_count) * 100:.2f}%")
print("=" * 70)

# ============================================================
# Save paths
# ============================================================
# output_path = "C:/Users/LLT/Desktop/data/out1/20162025_all_predict_article_Hybrid(1).xlsx"
# analysis_output_path = "C:/Users/LLT/Desktop/data/out1/20162025_all_Hybrid_validation_analysis(1).xlsx"
output_path = "C:/Users/LLT/Desktop/data/out1/2016_all_predict_article_Hybrid(1).xlsx"
analysis_output_path = "C:/Users/LLT/Desktop/data/out1/2016_all_Hybrid_validation_analysis(1).xlsx"
# ============================================================
# Save final predictions
# ============================================================
df_final.to_excel(output_path, index=False)
print("\n💾 Final corpus saved:")
print(output_path)

# ============================================================
# Build validation summary
# ============================================================
validation_summary = pd.DataFrame({
    "Metric": ["AUC", "F1", "Best Threshold", "Validation Size"],
    "Value": [hybrid_auc, hybrid_f1, best_threshold, len(y_val)]
})

# ============================================================
# Validation source summary
# ============================================================
if len(source_df) > 0:
    validation_source_summary = source_df["source"].value_counts().reset_index()
    validation_source_summary.columns = ["source", "count"]
else:
    validation_source_summary = pd.DataFrame({"source": [], "count": []})

# ============================================================
# Final source summary
# ============================================================
final_source_summary = df_final["prediction_source"].value_counts().reset_index()
final_source_summary.columns = ["source", "count"]

# ============================================================
# Save analysis results
# ============================================================
with pd.ExcelWriter(analysis_output_path) as writer:
    validation_summary.to_excel(writer, sheet_name="Hybrid_Performance", index=False)
    val_sensitivity_df.to_excel(writer, sheet_name="Validation_Threshold", index=False)
    test_sensitivity_df.to_excel(writer, sheet_name="Test_Threshold", index=False)
    validation_source_summary.to_excel(writer, sheet_name="Validation_Source", index=False)
    final_source_summary.to_excel(writer, sheet_name="Final_Source", index=False)

print("\n💾 Analysis results saved:")
print(analysis_output_path)

# ============================================================
# Final output
# ============================================================
print("\n" + "=" * 70)
print("🎉 Hybrid SPECTER2-XGBoost Pipeline Completed")
print("=" * 70)
print("\nKey outputs:")
print(f"✔ Hybrid AUC: {hybrid_auc:.4f}")
print(f"✔ Hybrid F1: {hybrid_f1:.4f}")
print(f"✔ Threshold: {best_threshold:.2f}")
print(f"✔ Final corpus size: {final_count:,}")
print("=" * 70)