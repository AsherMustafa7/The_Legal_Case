# scripts/compare_visualize_results.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def normalize(series, invert=False):
    """Normalize metric to [0,1]. Invert=True means lower is better."""
    s = (series - series.min()) / (series.max() - series.min() + 1e-9)
    return 1 - s if invert else s

def main():
    df = pd.read_excel("outputs/cluster_comparison_summary.xlsx")
    df["Algorithm"] = df["Algorithm"].str.replace("output_", "").str.upper()

    # Normalize metrics for radar chart
    df["Silhouette_norm"] = normalize(df["Silhouette"])
    df["Calinski_norm"] = normalize(df["Calinski-Harabasz"])
    df["Davies_norm"] = normalize(df["Davies-Bouldin"], invert=True)
    df["Coverage_norm"] = normalize(df["Label_Coverage(%)"])

    # ---- BAR CHART ----
    metrics = ["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"]
    plt.figure(figsize=(10, 6))
    width = 0.25
    x = np.arange(len(df))

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df[metric], width, label=metric)

    plt.xticks(x + width, df["Algorithm"], rotation=25)
    plt.title("Clustering Performance Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/cluster_comparison_barchart.png", dpi=300)
    print("✅ Saved: outputs/cluster_comparison_barchart.png")

    # ---- RADAR (SPIDER) CHART ----
    categories = ["Silhouette_norm", "Calinski_norm", "Davies_norm", "Coverage_norm"]
    labels = ["Silhouette", "Calinski–Harabasz", "Davies–Bouldin", "Label Coverage"]

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, row in df.iterrows():
        values = row[categories].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Algorithm"])
        ax.fill(angles, values, alpha=0.1)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Normalized Clustering Quality Comparison", size=14)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.savefig("outputs/cluster_comparison_radar.png", dpi=300)
    print("✅ Saved: outputs/cluster_comparison_radar.png")
    plt.show()

if __name__ == "__main__":
    main()
