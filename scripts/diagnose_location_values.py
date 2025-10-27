# scripts/diagnose_location_values.py
import json, pandas as pd, unicodedata, re
from collections import Counter
from pathlib import Path

def norm(s):
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_candidates(x):
    if isinstance(x, dict):
        for k in ("state","State","district","District","city","City","raw"):
            v = x.get(k)
            if v:
                yield norm(v)
    elif isinstance(x, str):
        yield norm(x)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    df = pd.read_excel(args.input)
    vals = []
    if "location" not in df.columns:
        print("No 'location' column found.")
        raise SystemExit(1)
    for v in df["location"].dropna():
        for c in extract_candidates(v):
            if c:
                vals.append(c)
    cnt = Counter(vals)
    print("Total rows:", len(df))
    print("Unique candidate tokens:", len(cnt))
    print("\nTop 50 candidates:")
    for k,n in cnt.most_common(50):
        print(f"{n:5d}  {k}")
