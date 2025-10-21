# scripts/create_consensus_labels.py
import pandas as pd
import json
from pathlib import Path
from collections import Counter, defaultdict

def load_label_df(path):
    df = pd.read_excel(path)
    # ensure columns exist
    if "file_name" not in df.columns:
        # try raw_text_path or fallback to index
        df["file_name"] = df.get("file_name", df.index.astype(str))
    # normalize crime_labels column to list
    def to_list(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            # excel often stores as string representation
            parsed = json.loads(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        # fallback split by comma
        if isinstance(x, str):
            # strip brackets
            s = x.strip()
            for ch in ("[", "]"):
                s = s.strip(ch)
            parts = [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
            return parts
        return []
    df["crime_labels_parsed"] = df["crime_labels"].apply(to_list) if "crime_labels" in df.columns else [[] for _ in range(len(df))]
    return df[["file_name", "crime_labels_parsed"]].rename(columns={"crime_labels_parsed":"crime_labels"})

def main(inputs, out_path):
    dfs = []
    for p in inputs:
        print("Loading", p)
        dfs.append(load_label_df(p))
    # merge on file_name
    merged = defaultdict(list)
    for df in dfs:
        for _, row in df.iterrows():
            fn = row["file_name"]
            labs = row["crime_labels"] or []
            # pick first label if list non-empty
            if labs:
                merged[fn].append(labs[0])
            else:
                merged[fn].append(None)
    # compute consensus
    out_records = []
    for fn, labels in merged.items():
        # labels is list of label or None for each algorithm
        votes = [l for l in labels if l]
        if votes:
            counter = Counter(votes)
            most_common, count = counter.most_common(1)[0]
            confidence = count / len(labels)  # fraction of algos agreeing (includes None)
            out_records.append({
                "file_name": fn,
                "consensus_label": most_common,
                "consensus_confidence": confidence,
                "votes": dict(counter),
                "num_algorithms": len(labels)
            })
        else:
            out_records.append({
                "file_name": fn,
                "consensus_label": None,
                "consensus_confidence": 0.0,
                "votes": {},
                "num_algorithms": len(labels)
            })
    out_df = pd.DataFrame(out_records)
    out_df.to_excel(out_path, index=False)
    print("Wrote consensus to", out_path)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="List of *_with_seed.xlsx files (e.g. output_kmeans_with_seed.xlsx ...)")
    p.add_argument("--out", default="outputs/consensus_labels.xlsx")
    args = p.parse_args()
    main(args.inputs, args.out)
