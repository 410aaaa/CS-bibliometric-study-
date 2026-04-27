# ====================== Basic Dependencies ======================
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
    f1_score
)

# ====================== Device ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== Load SPECTER2 ======================
print("Loading SPECTER2 Classification model...")

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

# ====================== SPECTER2 Encoding Function (CLS + MEAN) ======================
def encode_specter2_classification(texts, batch_size=8, max_length=512):
    texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
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

            cls_emb = hidden[:, 0, :]
            mean_emb = hidden.mean(dim=1)

            emb = torch.cat([cls_emb, mean_emb], dim=1)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)

# ====================== Keyword System (Hierarchical) ======================
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

# ====================== Load Training Set ======================
print("Loading training data...")
train_df = pd.read_excel("D:/Desktop/code/train_data.xls")

train_df['is_computer'] = pd.to_numeric(train_df['is_computer'], errors='coerce')
train_df = train_df[train_df['is_computer'].isin([0, 1])]
train_df['is_computer'] = train_df['is_computer'].astype(int)

train_df['text'] = (
    train_df['Article Title'].fillna('') +
    tokenizer.sep_token +
    train_df['Abstract'].fillna('')
)
train_df = train_df[train_df['text'].str.strip().ne('')]

# ====================== Encode Training Set ======================
X_all = encode_specter2_classification(train_df['text'].tolist())
y_all = train_df['is_computer'].values

# ====================== Train / Validation Split ======================
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

scale_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f"scale_pos_weight = {scale_weight:.2f}")

# ====================== Train XGBoost ======================
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

# ====================== Validation Evaluation ======================
y_val_proba = clf.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.2, 0.8, 61)
best_threshold, best_f1 = 0.5, 0

for t in thresholds:
    f1 = f1_score(y_val, (y_val_proba >= t).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print("\nValidation Results")
print(classification_report(y_val, (y_val_proba >= best_threshold).astype(int)))
print(f"AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
print(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")

# ====================== Threshold Sensitivity (Validation) ======================
print("\nThreshold Sensitivity Analysis (Validation Set)")

sensitivity_thresholds = [0.5, 0.6, 0.65, 0.7, 0.8]

val_results = []

for t in sensitivity_thresholds:
    y_pred = (y_val_proba >= t).astype(int)

    precision = classification_report(y_val, y_pred, output_dict=True)['1']['precision']
    recall = classification_report(y_val, y_pred, output_dict=True)['1']['recall']
    f1 = f1_score(y_val, y_pred)

    val_results.append({
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

val_sensitivity_df = pd.DataFrame(val_results)
print(val_sensitivity_df)

from sklearn.metrics import confusion_matrix

y_val_pred = (y_val_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_val, y_val_pred)

tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix (Validation Set)")
print(cm)

print(f"\nTN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")

# ====================== Load Test Set ======================
print("\nLoading test data...")
test_df = pd.read_excel("D:/Desktop/code/CNS data/2025/2025.xls")
original_count = len(test_df)

test_df = test_df.dropna(subset=['Article Title', 'Abstract'])
test_df = test_df[
    test_df['Document Type'].astype(str)
        .str.lower().str.contains('article|review', na=False)
]

test_df['text'] = (
    test_df['Article Title'].fillna('') +
    tokenizer.sep_token +
    test_df['Abstract'].fillna('')
)
test_df = test_df[test_df['text'].str.strip().ne('')]

print(f"Final number of documents used for prediction: {len(test_df):,}")

# ====================== Prediction ======================
X_test = encode_specter2_classification(test_df['text'].tolist())
test_df['model_score'] = clf.predict_proba(X_test)[:, 1]

# ====================== Fusion ======================
test_df['keyword_score'] = test_df['text'].apply(keyword_score)
test_df['final_score'] = np.maximum(test_df['model_score'], test_df['keyword_score'])

CUSTOM_THRESHOLD = max(0.35, best_threshold)
print(f"\nUsing final threshold: {CUSTOM_THRESHOLD:.2f}")

df_final = test_df[test_df['final_score'] >= CUSTOM_THRESHOLD].copy()
df_final['predicted_is_computer'] = 1
df_final = df_final.sort_values('final_score', ascending=False)

# ====================== Statistics ======================
final_count = len(df_final)
print("\nFINAL STATISTICS")
print("-" * 40)
print(f"Total original documents: {original_count:,}")
print(f"Final identified AI papers: {final_count:,}")
print(f"Overall proportion: {(final_count / original_count) * 100:.2f}%")
print("-" * 40)

# ====================== Save ======================
output_path = "D:/Desktop/code/threshold/2025_predict_article.xlsx"
df_final.to_excel(output_path, index=False)

print(f"\nResults saved to: {output_path}")
print("=" * 60)