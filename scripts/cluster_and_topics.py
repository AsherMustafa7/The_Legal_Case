# scripts/cluster_and_topics.py
import json
import argparse
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np

def load_cases(jsonl_path):
    docs = []
    objs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            objs.append(o)
            # Use summary first, fallback to raw text file
            if o.get("summary"):
                docs.append(o["summary"])
            elif o.get("raw_text_path"):
                try:
                    docs.append(open(o["raw_text_path"], "r", encoding="utf-8").read())
                except:
                    docs.append("")
            else:
                docs.append("")
    return objs, docs

def top_terms_by_cluster(docs, labels, top_n=10):
    df = pd.DataFrame({"doc": docs, "cluster": labels})
    terms = {}
    for c in sorted(set(labels)):
        if c == -1:
            continue
        texts = df[df.cluster == c].doc.values
        vect = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
        X = vect.fit_transform(texts)
        if X.shape[0] == 0:
            terms[c] = []
            continue
        # sum tfidf per term
        sums = np.array(X.sum(axis=0)).ravel()
        idxs = sums.argsort()[-top_n:][::-1]
        feature_names = vect.get_feature_names_out()
        terms[c] = [feature_names[i] for i in idxs]
    return terms

def main(input_jsonl, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    objs, docs = load_cases(input_jsonl)
    if not docs:
        print("No documents found.")
        return

    print("Encoding documents with SentenceTransformer (this may take a while first run)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    print("Reducing dimensions with UMAP...")
    reducer = umap.UMAP(n_neighbors=15, n_components=20, metric='cosine', random_state=42)
    emb_reduced = reducer.fit_transform(embeddings)

    print("Clustering with HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom')
    labels = clusterer.fit_predict(emb_reduced)

    print("Computing top terms per cluster (TF-IDF based)...")
    terms = top_terms_by_cluster(docs, labels, top_n=10)

    # attach clusters to objects and save CSV
    rows = []
    for obj, lab in zip(objs, labels):
        obj["cluster_id"] = int(lab)
        rows.append({
            "file_name": obj.get("file_name"),
            "case_title": obj.get("case_title"),
            "cluster_id": int(lab),
            "crime_labels": ",".join(obj.get("crime_labels", []))
        })

    df = pd.DataFrame(rows)
    csv_path = Path(output_folder) / "clusters_summary.csv"
    df.to_csv(csv_path, index=False)
    print("Clusters saved to", csv_path)

    # Save top terms
    with open(Path(output_folder) / "cluster_top_terms.json", "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)

    # Optionally overwrite JSONL with cluster ids
    out_jsonl = Path(output_folder) / "cases_with_clusters.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as outf:
        for obj in objs:
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("Done. top terms saved and cases_with_clusters.jsonl created at", output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", default="outputs/cases.jsonl")
    parser.add_argument("--output_folder", default="outputs")
    args = parser.parse_args()
    main(args.input_jsonl, args.output_folder)
