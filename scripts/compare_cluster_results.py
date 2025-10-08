# scripts/compare_cluster_results.py
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# -----------------------------------------
# Helper: evaluate one clustering file
# -----------------------------------------
def evaluate_clustering(filepath, text_column="summary"):
    print(f"🔍 Evaluating {filepath.name}")
    df = pd.read_excel(filepath)
    df = df.dropna(subset=[text_column])

    # --- Detect which label column to use ---
    if "Cluster_ID" in df.columns:
        label_column = "Cluster_ID"
    elif "Topic_ID" in df.columns:
        label_column = "Topic_ID"
    else:
        print(f"⚠️ No cluster/topic column found in {filepath.name}. Skipping.")
        return None

    # Ensure numeric cluster IDs
    try:
        df[label_column] = df[label_column].astype(int)
    except:
        df[label_column] = df[label_column].astype(str)

    unique_clusters = df[label_column].nunique()
    if unique_clusters <= 1:
        print(f"⚠️ Only one cluster found in {filepath.name}, skipping.")
        return None

    # --- TF-IDF Vectorization for metrics ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(df[text_column])
    labels = df[label_column].values

    # --- Compute metrics ---
    try:
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X.toarray(), labels)
        davies = davies_bouldin_score(X.toarray(), labels)
    except Exception as e:
        print(f"⚠️ Error calculating metrics for {filepath.name}: {e}")
        silhouette, calinski, davies = np.nan, np.nan, np.nan

    # --- Label coverage (if seed labels exist) ---
    label_coverage = None
    if "Cluster_Label" in df.columns:
        labeled = df[df["Cluster_Label"].notna() & (df["Cluster_Label"] != "unlabeled")]
        label_coverage = len(labeled) / len(df) * 100

    algo_name = filepath.stem.replace("_with_seed", "").replace("_raw", "")
    return {
        "Algorithm": algo_name,
        "Num_Clusters": unique_clusters,
        "Silhouette": silhouette,
        "Calinski-Harabasz": calinski,
        "Davies-Bouldin": davies,
        "Label_Coverage(%)": label_coverage,
    }

# -----------------------------------------
# Main
# -----------------------------------------
def main(args):
    files = list(Path(args.folder).glob("*with_seed.xlsx"))
    if not files:
        print("⚠️ No *_with_seed.xlsx files found in", args.folder)
        return

    results = []
    for f in files:
        r = evaluate_clustering(f)
        if r:
            results.append(r)

    df_results = pd.DataFrame(results).sort_values(by="Silhouette", ascending=False)
    out_path = Path(args.folder) / "cluster_comparison_summary.xlsx"
    df_results.to_excel(out_path, index=False)
    print(f"✅ Results saved to {out_path}")

    # --- Plot comparison ---
    metrics = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
    plt.figure(figsize=(10, 6))
    width = 0.25
    x = np.arange(len(df_results))

    # Bar chart for all metrics
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df_results[metric], width, label=metric)

    plt.xticks(x + width, df_results["Algorithm"], rotation=25)
    plt.title("Clustering Performance Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()

    plt.savefig(Path(args.folder) / "cluster_comparison_plot.png", dpi=300)
    plt.show()
    print("📊 Graph saved to cluster_comparison_plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="outputs", help="Folder containing *_with_seed.xlsx files")
    args = parser.parse_args()
    main(args)
