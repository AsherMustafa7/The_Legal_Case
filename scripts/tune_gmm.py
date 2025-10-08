# scripts/tune_gmm.py
"""
Tune optimal number of Gaussian Mixture clusters (semantic version)
using SentenceTransformer embeddings + PCA dimension reduction.

Output: BIC/AIC curves saved to outputs/gmm_tuning_semantic.png
"""

import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from tqdm import tqdm

# ✅ Semantic embeddings
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------
# Load cases
# --------------------------------------------------------------------
def load_cases(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main(args):
    print(f"\n📘 Loading cases from: {args.input}")
    df = load_cases(args.input)
    texts = df["summary"].fillna("").tolist()

    if not texts:
        print("❌ No texts found in the JSONL.")
        return

    # ----------------------------------------------------------------
    # 1️⃣ Sentence Embeddings
    # ----------------------------------------------------------------
    print("🧠 Generating sentence embeddings (using all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print(f"✅ Embeddings shape: {embeddings.shape}")

    # ----------------------------------------------------------------
    # 2️⃣ PCA Dimensionality Reduction
    # ----------------------------------------------------------------
    n_components = min(embeddings.shape[1], args.pca_components)
    print(f"📉 Reducing dimensions to {n_components} using PCA...")
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(embeddings)

    print(f"✅ Reduced shape: {X_reduced.shape}")

    # ----------------------------------------------------------------
    # 3️⃣ Fit GMMs for multiple K and compute BIC/AIC
    # ----------------------------------------------------------------
    bics, aics = [], []
    K_range = range(2, args.max_k + 1)

    print(f"\n🎲 Fitting GMM models for K = 2 to {args.max_k}...")
    for k in tqdm(K_range):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
            reg_covar=1e-6
        )
        gmm.fit(X_reduced)
        bics.append(gmm.bic(X_reduced))
        aics.append(gmm.aic(X_reduced))

    # ----------------------------------------------------------------
    # 4️⃣ Plot results
    # ----------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, bics, marker="o", label="BIC")
    plt.plot(K_range, aics, marker="o", label="AIC")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Score (lower = better)")
    plt.title("Semantic GMM Cluster Tuning (Embeddings + PCA)")
    plt.legend()
    plt.grid(True)

    output_path = "outputs/gmm_tuning_semantic.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    # ----------------------------------------------------------------
    # 5️⃣ Print summary
    # ----------------------------------------------------------------
    print("\n✅ Plot saved to:", output_path)
    best_k_bic = K_range[np.argmin(bics)]
    best_k_aic = K_range[np.argmin(aics)]
    print(f"📊 Best K by BIC: {best_k_bic}")
    print(f"📊 Best K by AIC: {best_k_aic}")
    print("\n👉 Choose a value near these for your main GMM clustering.")


# --------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune GMM with semantic embeddings")
    parser.add_argument("--input", required=True, help="Path to cases.jsonl")
    parser.add_argument("--max_k", type=int, default=25, help="Maximum number of clusters to test")
    parser.add_argument("--pca_components", type=int, default=50, help="Number of PCA components")
    args = parser.parse_args()
    main(args)
