# scripts/tune_kmeans.py
# this algorithm will decide the number of clusters for KMeans clustering
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def load_cases(jsonl_file):
    texts = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            case = json.loads(line)
            # prefer summary if available, else raw text path
            if "summary" in case and case["summary"]:
                texts.append(case["summary"])
            elif "raw_text_path" in case and os.path.exists(case["raw_text_path"]):
                with open(case["raw_text_path"], "r", encoding="utf-8") as rf:
                    texts.append(rf.read())
    return texts

def main(args):
    print(f"Loading cases from {args.jsonl_file}...")
    texts = load_cases(args.jsonl_file)

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(texts)

    inertias = []
    silhouettes = []
    k_values = range(2, args.max_k + 1)

    print("Running KMeans for different k values...")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 5))

    # Plot elbow (inertia)
    plt.subplot(1, 2, 1)
    plt.plot(list(k_values), inertias, marker="o")
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")

    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(list(k_values), silhouettes, marker="o")
    plt.title("Silhouette Score")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Score")

    output_path = "outputs/kmeans_tuning.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Tuning results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_file", type=str, default="outputs/cases.jsonl",
                        help="Path to cases.jsonl")
    parser.add_argument("--max_k", type=int, default=15,
                        help="Maximum number of clusters to test")
    args = parser.parse_args()
    main(args)
