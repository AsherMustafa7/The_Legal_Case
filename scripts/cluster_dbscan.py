# scripts/cluster_dbscan.py
import argparse
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
from scripts.utils import load_seed_map, match_seed_labels

def load_cases(jsonl_path):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

def assign_cluster_labels(df, seed_map):
    cluster_labels = {}
    for cluster_id in df["Cluster_ID"].unique():
        if cluster_id == -1:
            cluster_labels[cluster_id] = "outlier"
            continue
        subset = df[df["Cluster_ID"] == cluster_id]
        labels = []
        for _, row in subset.iterrows():
            case_labels = match_seed_labels(row.get("summary", ""), row.get("ipc_sections", []), seed_map)
            labels.extend(list(case_labels.keys()))
        if labels:
            common = Counter(labels).most_common(1)[0][0]
            cluster_labels[cluster_id] = common
        else:
            cluster_labels[cluster_id] = "unlabeled"
    df["Cluster_Label"] = df["Cluster_ID"].map(cluster_labels)
    return df

def main(args):
    df = load_cases(args.input)
    texts = df["summary"].fillna("").tolist()

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(texts)

    dbscan = DBSCAN(eps=0.7, min_samples=3, metric="cosine")
    df["Cluster_ID"] = dbscan.fit_predict(X)

    # Raw
    raw_out = args.output.replace(".xlsx", "_raw.xlsx")
    df.to_excel(raw_out, index=False)
    print(f"✅ DBSCAN raw clustering saved to {raw_out}")

    # With seed
    seed_map = load_seed_map(args.seed)
    df = assign_cluster_labels(df, seed_map)
    labeled_out = args.output.replace(".xlsx", "_with_seed.xlsx")
    df.to_excel(labeled_out, index=False)
    print(f"✅ DBSCAN with-seed clustering saved to {labeled_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", default="data/seed_map.json")
    args = parser.parse_args()
    main(args)
