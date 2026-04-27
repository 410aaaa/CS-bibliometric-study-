# =========================================================
# BERTopic Pipeline (Optimized, Reproducible + Evaluation)
# INCLUDE / ONLY + Excel I/O + Coherence (c_npmi/c_v) + Representative Docs + Topic Stability + Topic Diversity
# =========================================================

import os
import ast
import re
import random
import numpy as np
import pandas as pd
from multiprocessing import freeze_support
from itertools import combinations
import torch

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from umap import UMAP

# =========================================================
# Fix random seeds for reproducibility
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =========================================================
# Tokenizer
# =========================================================
def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()

def tokenize_docs(docs):
    return [simple_tokenize(d) for d in docs]

# =========================================================
# Coherence functions
# =========================================================
def compute_coherence(docs_tokenized, topic_model, top_n=15):
    topics = topic_model.get_topics()
    topic_words = [[w for w,_ in words[:top_n]] for tid, words in topics.items() if tid != -1]
    dictionary = Dictionary(docs_tokenized)
    corpus = [dictionary.doc2bow(doc) for doc in docs_tokenized]

    cm_npmi = CoherenceModel(topics=topic_words, texts=docs_tokenized,
                             corpus=corpus, dictionary=dictionary, coherence='c_npmi')
    cm_cv = CoherenceModel(topics=topic_words, texts=docs_tokenized,
                           dictionary=dictionary, coherence='c_v')
    return cm_npmi.get_coherence(), cm_cv.get_coherence()

# =========================================================
# Topic diversity (1 - avg Jaccard similarity)
# =========================================================
def compute_topic_diversity(topic_model, top_n=15):
    topics = topic_model.get_topics()
    topic_word_sets = [set([w for w,_ in words[:top_n]]) for tid, words in topics.items() if tid != -1]
    n = len(topic_word_sets)
    jaccards = []
    for i,j in combinations(range(n),2):
        inter = len(topic_word_sets[i] & topic_word_sets[j])
        union = len(topic_word_sets[i] | topic_word_sets[j])
        jaccards.append(inter/union)
    avg_similarity = sum(jaccards)/len(jaccards) if jaccards else 0
    return 1 - avg_similarity

# =========================================================
# Manual stability (Jaccard)
# =========================================================
def compute_topic_stability_manual(topic_model, docs, nr_runs=3, top_n=15):
    first_run_topics = topic_model.get_topics()
    first_run_words = {tid: set([w for w,_ in words[:top_n]])
                       for tid, words in first_run_topics.items() if tid != -1}

    stability_scores = {tid: [] for tid in first_run_words.keys()}

    for run in range(nr_runs):
        run_seed = SEED + run

        hdbscan_model = HDBSCAN(
            min_cluster_size=topic_model.hdbscan_model.min_cluster_size,
            min_samples=topic_model.hdbscan_model.min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=run_seed
        )

        temp_model = BERTopic(
            embedding_model=topic_model.embedding_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=topic_model.vectorizer_model,
            representation_model=KeyBERTInspired(),
            top_n_words=top_n,
            umap_model=umap_model,
            verbose=False
        )

        topics, _ = temp_model.fit_transform(docs)

        run_topic_words = [
            set([w for w,_ in words[:top_n]])
            for tid, words in temp_model.get_topics().items() if tid != -1
        ]

        for tid, words_set in first_run_words.items():
            max_j = max([len(words_set & w)/len(words_set | w) for w in run_topic_words])
            stability_scores[tid].append(max_j)

    return {tid: np.mean(scores) for tid,scores in stability_scores.items()}

# =========================================================
# Main Pipeline
# =========================================================
def run_bertopic_pipeline():
    INPUT_EXCEL = r"D:/Desktop/code/threshold/paper_level_discipline_assignment(v1.12).xlsx"
    OUTPUT_DIR = r"D:/Desktop/code/BERTopic_results_method1_final"

    TEXT_COLUMN = "text"
    DISCIPLINE_COLUMN = "Present_Disciplines"
    TARGET_DISCIPLINE = "Clinical Medicine"

    MODES = ["include","only"]

    MIN_CLUSTER_SIZE = 5
    MIN_SAMPLES = 2
    TOP_N_WORDS = 15
    N_STABILITY_RUNS = 3

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_excel(INPUT_EXCEL)

    df["disciplines_clean"] = df[DISCIPLINE_COLUMN].apply(
        lambda x: [i.strip().lower() for i in ast.literal_eval(x)] if pd.notna(x) else []
    )

    target = TARGET_DISCIPLINE.lower()

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    for mode in MODES:

        if mode=="include":
            df_sel = df[df["disciplines_clean"].apply(lambda x: target in x)]
        else:
            df_sel = df[df["disciplines_clean"].apply(lambda x: len(x)==1 and x[0]==target)]

        print(f"\n{mode.upper()} mode: {len(df_sel)} papers selected")

        docs = df_sel[TEXT_COLUMN].astype(str).str.strip()
        docs = docs[docs.str.len()>50].tolist()

        if len(docs)<20:
            print("Too few documents, skipped.")
            continue

        print(f"{len(docs)} cleaned documents ready")

        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=1,
            max_df=0.95,
            max_features=10000
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            min_samples=MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )

        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=SEED
        )

        topic_model = BERTopic(
            embedding_model=embedding_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=KeyBERTInspired(),
            top_n_words=TOP_N_WORDS,
            umap_model=umap_model,
            verbose=True
        )

        print("Running BERTopic...")
        topics, probs = topic_model.fit_transform(docs)

        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        print(f"Number of topics: {n_topics}")

        tokenized_docs = tokenize_docs(docs)

        pd.DataFrame({"Document":docs, "Topic":topics}).to_excel(
            os.path.join(OUTPUT_DIR,f"{TARGET_DISCIPLINE.replace(',','')}_{mode}_doc_topics.xlsx"),
            index=False
        )

        topic_model.get_topic_info().to_excel(
            os.path.join(OUTPUT_DIR,f"{TARGET_DISCIPLINE.replace(',','')}_{mode}_topic_info.xlsx"),
            index=False
        )

        rep_docs_list = []
        for tid in set(topics):
            if tid==-1: continue
            rep_docs = topic_model.get_representative_docs(tid)
            rep_docs_list.append({"Topic":tid,"Representative_Docs":rep_docs[:3]})

        pd.DataFrame(rep_docs_list).to_excel(
            os.path.join(OUTPUT_DIR,f"{TARGET_DISCIPLINE.replace(',','')}_{mode}_rep_docs.xlsx"),
            index=False
        )

        print("Calculating coherence scores...")
        c_npmi, c_v = compute_coherence(tokenized_docs, topic_model, TOP_N_WORDS)
        print(f"c_npmi: {c_npmi:.4f}, c_v: {c_v:.4f}")

        print("Calculating topic stability...")
        stability = compute_topic_stability_manual(topic_model, docs, nr_runs=N_STABILITY_RUNS, top_n=TOP_N_WORDS)

        stability_df = pd.DataFrame(list(stability.items()), columns=["Topic","Stability_Jaccard"])
        stability_df.to_excel(
            os.path.join(OUTPUT_DIR,f"{TARGET_DISCIPLINE.replace(',','')}_{mode}_stability.xlsx"),
            index=False
        )

        diversity_score = compute_topic_diversity(topic_model, TOP_N_WORDS)
        print(f"Topic Diversity: {diversity_score:.4f}")

        eval_df = pd.DataFrame({
            "Metric": [
                "Number of Topics",
                "Topic Coherence (c_npmi)",
                "Topic Coherence (c_v)",
                "Topic Diversity"
            ],
            "Value": [n_topics, c_npmi, c_v, diversity_score]
        })

        eval_df.to_excel(
            os.path.join(OUTPUT_DIR,f"{TARGET_DISCIPLINE.replace(',','')}_{mode}_evaluation.xlsx"),
            index=False
        )

        print(f"{mode.upper()} pipeline completed!")

# =========================================================
# Windows Safe Entry
# =========================================================
if __name__=="__main__":
    freeze_support()
    run_bertopic_pipeline()