# scripts/tune_gmm.py
import argparse
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def load_cases(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def main(args):
    df = load_cases(args.input)
    texts = df["summary"].fillna("").tolist()

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts).toarray()  # GMM needs dense array

    bics, aics = [], []
    K_range = range(2, args.max_k+1)

    for k in K_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(K_range, bics, marker='o', label="BIC")
    plt.plot(K_range, aics, marker='o', label="AIC")
    plt.xlabel("Number of clusters (n_components)")
    plt.ylabel("Score (lower is better)")
    plt.title("GMM Cluster Tuning")
    plt.legend()
    plt.savefig("outputs/gmm_tuning.png")
    plt.show()

    print("✅ Plot saved to outputs/gmm_tuning.png")
    print("👉 Pick the k where BIC/AIC is lowest.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--max_k", type=int, default=15)
    args = parser.parse_args()
    main(args)
