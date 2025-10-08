# scripts/cluster_kmeans_semantic.py
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scripts.utils import load_seed_map, match_seed_labels

# ---------------------------------------------------------
# Load cases
# ---------------------------------------------------------
def load_cases(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

# ---------------------------------------------------------
# Get top TF-IDF terms per cluster (for interpretation)
# ---------------------------------------------------------
def top_terms_by_cluster(docs, labels, top_n=10):
    df = pd.DataFrame({"doc": docs, "cluster": labels})
    terms = {}
    for c in sorted(set(labels)):
        texts = df[df.cluster == c].doc.values
        if len(texts) == 0:
            terms[c] = []
            continue
        vect = TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words='english')
        X = vect.fit_transform(texts)
        sums = np.array(X.sum(axis=0)).ravel()
        idxs = sums.argsort()[-top_n:][::-1]
        feature_names = vect.get_feature_names_out()
        terms[c] = [feature_names[i] for i in idxs]
    return terms

# ---------------------------------------------------------
# Assign seed-based label for each cluster (majority voting)
# ---------------------------------------------------------
def assign_cluster_labels(df, seed_map):
    cluster_labels = {}
    for cluster_id in df["Cluster_ID"].unique():
        subset = df[df["Cluster_ID"] == cluster_id]
        labels = []
        for _, row in subset.iterrows():
            case_labels = match_seed_labels(
                row.get("summary", ""),
                row.get("ipc_sections", []),
                seed_map
            )
            labels.extend(list(case_labels.keys()))
        if labels:
            common = Counter(labels).most_common(1)[0][0]
            cluster_labels[cluster_id] = common
        else:
            cluster_labels[cluster_id] = "unlabeled"
    df["Cluster_Label"] = df["Cluster_ID"].map(cluster_labels)
    return df

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main(args):
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    df = load_cases(args.input)
    texts = df["summary"].fillna("").tolist()

    # --- Embeddings ---
    print("🧠 Generating sentence embeddings...")
    model = SentenceTransformer(args.model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # --- PCA ---
    print(f"📉 Reducing dimensions to {args.pca_components}...")
    pca = PCA(n_components=args.pca_components, random_state=42)
    X_red = pca.fit_transform(embeddings)
    print(f"✅ Reduced shape: {X_red.shape}")

    # --- KMeans ---
    print(f"🎯 Running KMeans with {args.k} clusters...")
    kmeans = KMeans(n_clusters=args.k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_red)
    df["Cluster_ID"] = labels

    # --- Save raw results ---
    raw_out = Path(args.output_folder) / f"kmeans_semantic_k{args.k}_raw.xlsx"
    df.to_excel(raw_out, index=False)
    print(f"✅ Raw clustering saved to {raw_out}")

    # --- Seed-based labeling ---
    seed_map = load_seed_map(args.seed)
    df = assign_cluster_labels(df, seed_map)

    labeled_out = Path(args.output_folder) / f"kmeans_semantic_k{args.k}_with_seed.xlsx"
    df.to_excel(labeled_out, index=False)
    print(f"✅ Seed-integrated clustering saved to {labeled_out}")

    # --- Top terms per cluster ---
    terms = top_terms_by_cluster(df["summary"].fillna("").tolist(), labels, top_n=args.top_n_terms)
    with open(Path(args.output_folder) / f"kmeans_semantic_k{args.k}_top_terms.json", "w", encoding="utf-8") as f:
        # Convert np.int32 keys to normal int for JSON
        terms_str_keys = {int(k): v for k, v in terms.items()}
        json.dump(terms_str_keys, f, ensure_ascii=False, indent=2)
        print(f"🗂️  Top terms saved to kmeans_semantic_k{args.k}_top_terms.json")

    # --- Print cluster info ---
    sizes = pd.Series(labels).value_counts().sort_index()
    print("\n📊 Cluster sizes:\n", sizes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="cases.jsonl file")
    parser.add_argument("--output_folder", default="outputs", help="Output directory")
    parser.add_argument("--seed", default="data/seed_map.json", help="Seed map file")
    parser.add_argument("--k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--pca_components", type=int, default=50, help="PCA dimensionality")
    parser.add_argument("--model_name", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--top_n_terms", type=int, default=10)
    args = parser.parse_args()
    main(args)
