#!/usr/bin/env python3
"""
scripts/count_cases_by_state.py

Counts how many crime cases belong to each Indian state based on the 'location' column.
Works with output_kmeans_with_seed.xlsx or cases.jsonl files produced by extract_and_parse.py.

Usage:
    python scripts/count_cases_by_state.py --input outputs/output_kmeans_with_seed.xlsx
"""

import pandas as pd
import json
import argparse
import ast
from collections import Counter

def try_parse_location(val):
    """Parse the location field safely from JSON/dict/string."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        v = val.strip()
        if not v:
            return {}
        # Try Python literal
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        # Try JSON
        try:
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Excel (.xlsx) or JSONL file containing extracted cases")
    args = p.parse_args()
    path = args.input

    # Load file (Excel or JSONL)
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Input file must be .xlsx or .jsonl")

    if "location" not in df.columns:
        raise ValueError("No 'location' column found in the dataset!")

    # Parse location column
    df["parsed_location"] = df["location"].apply(try_parse_location)

    # Extract state names
    df["state_name"] = df["parsed_location"].apply(lambda x: (x.get("state") or "").strip() if isinstance(x, dict) else "")

    # Count states
    counts = Counter(df["state_name"].fillna("").str.title())
    total = sum(counts.values())

    print(f"\n=== Crime Case Counts by State ===")
    print(f"Total cases: {total}\n")
    for state, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        if state:
            print(f"{state:<25} {cnt:>6}")
    print("\n✅ Done.")

if __name__ == "__main__":
    main()
