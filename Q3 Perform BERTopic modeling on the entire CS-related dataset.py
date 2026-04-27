# =====================================================
# Computer Science Literature Topic Modeling System (Stable Research Version)
# BERTopic + Windows + Excel + Academic Cleaning + Post-processing Analysis
# =====================================================

import os
import re
import json
import warnings
import multiprocessing as mp

import pandas as pd
import numpy as np
import gensim
from gensim.models import CoherenceModel

warnings.filterwarnings("ignore")

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# Embedding model
# =====================================================
def load_embedding_model():
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)

# =====================================================
# Academic text cleaning
# =====================================================
def clean_academic_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|doi:\S+", " ", text)
    text = re.sub(r"\[\d+\]", " ", text)
    text = re.sub(r"\bfig\b|\btable\b|\beq\b|\bsection\b", " ", text)
    text = re.sub(r"[^a-z0-9_\s]", " ", text)
    domain_words = ["edu","cn","com","org","stanford","harvard","mit","turing",
                    "microsoft","google","deepmind","ibm"]
    text = re.sub(r"\b(" + "|".join(domain_words) + r")\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =====================================================
# Core Class
# =====================================================
class TopicAnalyzer:

    def __init__(self, base_path, output_path, random_state=42):
        self.base_path = base_path
        self.output_path = output_path
        self.random_state = random_state

        self.data = None
        self.model = None
        self.coherence = {}
        self.diversity = {}

        print("Topic analyzer initialized")

    def _find_col(self, df, keys):
        for col in df.columns:
            for k in keys:
                if k.lower() in col.lower():
                    return col
        return None

    def load_data(self):
        dfs = []
        for year in sorted(os.listdir(self.base_path)):
            if not year.isdigit():
                continue
            file = os.path.join(self.base_path, year, f"{year}_predict_article.xlsx")
            if not os.path.exists(file):
                continue
            print(f"Loading data for year {year}")
            df = pd.read_excel(file)
            text_col = self._find_col(df, ["text"])
            if not text_col:
                print(f"Missing text column in {year}, skipped")
                continue
            text = df[text_col].fillna("").astype(str)
            df["Cleaned_Text"] = text.apply(clean_academic_text)
            df = df[df["Cleaned_Text"].str.len() >= 50]
            dfs.append(df[["Cleaned_Text"]])
        if not dfs:
            raise ValueError("No data files loaded successfully")
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.data)} documents")

    def train(self):
        docs = self.data["Cleaned_Text"].tolist()
        embedding_model = load_embedding_model()

        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=UMAP(n_neighbors=30, n_components=5, min_dist=0.1, metric="cosine",
                            random_state=self.random_state),
            hdbscan_model=HDBSCAN(min_cluster_size=max(15, len(docs)//50), min_samples=5, prediction_data=True),
            vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1,3), min_df=2, max_df=0.9),
            ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
            representation_model={"KeyBERT": KeyBERTInspired(top_n_words=15),
                                  "MMR": MaximalMarginalRelevance(diversity=0.7)},
            top_n_words=15,
            verbose=True
        )

        topics, _ = self.model.fit_transform(docs)
        self.data["Topic"] = topics

        self.model.reduce_topics(docs, nr_topics="auto")
        self._evaluate(docs)

    def _evaluate(self, docs):
        tokens = [d.split() for d in docs if len(d.split())>10]
        dictionary = gensim.corpora.Dictionary(tokens)
        dictionary.filter_extremes(no_below=5, no_above=0.5)

        topic_words = [[w for w,_ in self.model.get_topic(t)[:10]]
                       for t in self.model.get_topic_info()["Topic"] if t!=-1]

        if len(topic_words)>=2:
            cv = CoherenceModel(topics=topic_words, texts=tokens, dictionary=dictionary,
                                coherence="c_v", processes=1).get_coherence()
            self.coherence["c_v"] = cv
            print(f"Topic coherence C_v = {cv:.4f}")

        sets = [set(t) for t in topic_words]
        sims = []
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                sims.append(len(sets[i]&sets[j])/len(sets[i]|sets[j]))

        if sims:
            self.diversity["jaccard"] = 1-np.mean(sims)
            print(f"Topic diversity = {1-np.mean(sims):.4f}")

    def save(self):
        os.makedirs(self.output_path, exist_ok=True)

        self.data.to_excel(os.path.join(self.output_path, "documents_with_topics.xlsx"), index=False)
        self.model.get_topic_info().to_excel(os.path.join(self.output_path, "topic_info.xlsx"), index=False)

        with open(os.path.join(self.output_path, "evaluation.json"), "w", encoding="utf-8") as f:
            json.dump({"coherence": self.coherence, "diversity": self.diversity}, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {self.output_path}")

    def analyze_topic_overlap(self):
        if self.model is None:
            raise ValueError("Model not trained")

        topic_info = self.model.get_topic_info()
        topic_ids = topic_info["Topic"].tolist()

        topic_words = {t: set([w for w,_ in self.model.get_topic(t)[:10]]) for t in topic_ids}
        overlap_matrix = pd.DataFrame(index=topic_ids, columns=topic_ids, dtype=float)

        for i in topic_ids:
            for j in topic_ids:
                inter = len(topic_words[i] & topic_words[j])
                union = len(topic_words[i] | topic_words[j])
                overlap_matrix.loc[i,j] = inter/union if union else 0

        overlap_matrix.to_excel(os.path.join(self.output_path,"topic_word_overlap.xlsx"))
        print("Word-level overlap matrix saved")

        embeddings = self.model.topic_embeddings_
        sim_matrix = cosine_similarity(embeddings)
        semantic_matrix = pd.DataFrame(sim_matrix, index=topic_ids, columns=topic_ids)

        semantic_matrix.to_excel(os.path.join(self.output_path,"topic_semantic_similarity.xlsx"))
        print("Semantic similarity matrix saved")

        tmp = overlap_matrix.copy(); np.fill_diagonal(tmp.values,0)
        max_word_pair = tmp.stack().idxmax()

        tmp2 = semantic_matrix.copy(); np.fill_diagonal(tmp2.values,0)
        max_sem_pair = tmp2.stack().idxmax()

        result = {
            "max_word_overlap_pair":[int(max_word_pair[0]),int(max_word_pair[1])],
            "max_semantic_similarity_pair":[int(max_sem_pair[0]),int(max_sem_pair[1])]
        }

        with open(os.path.join(self.output_path,"overlap_analysis.json"),"w") as f:
            json.dump(result,f,indent=2)

        print("Overlap analysis completed")

# =====================================================
# Main
# =====================================================
def set_global_seed(seed=42):
    import random, numpy as np, torch, os
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def main():
    set_global_seed(42)

    base_path = "D:/Desktop/code/CNS data"
    output_path = "D:/Desktop/code/BERTopic_final"

    analyzer = TopicAnalyzer(base_path, output_path)
    analyzer.load_data()
    analyzer.train()
    analyzer.save()
    analyzer.analyze_topic_overlap()

    print("Pipeline completed")

if __name__ == "__main__":
    mp.freeze_support()
    main()