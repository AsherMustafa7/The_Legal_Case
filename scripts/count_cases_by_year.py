#!/usr/bin/env python3
"""
scripts/count_cases_by_year.py

Counts how many cases belong to each judgment year based on the 'dates' column
or by extracting a year from the file name (fallback).

Usage:
    python scripts/count_cases_by_year.py --input outputs/output_kmeans_with_seed.xlsx
"""

import pandas as pd
import json
import re
import argparse
import ast
from collections import Counter

def try_extract_year(value, fallback=None):
    """Extracts a 4-digit year (>=1900) from text or dict."""
    year = None

    # Case 1: if it's a dict with 'judgment_date'
    if isinstance(value, dict):
        date_str = value.get("judgment_date", "")
        m = re.search(r"(19|20)\d{2}", str(date_str))
        if m:
            year = int(m.group(0))
    elif isinstance(value, str):
        # Case 2: parse from string (like JSON text or file name)
        m = re.search(r"(19|20)\d{2}", value)
        if m:
            year = int(m.group(0))

    # fallback if nothing found
    if not year and fallback:
        m = re.search(r"(19|20)\d{2}", fallback)
        if m:
            year = int(m.group(0))
    return year


def safe_parse_dates(val):
    """Parse the 'dates' field safely from dict/JSON/text."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return {}
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Excel (.xlsx) or JSONL file containing parsed cases")
    args = p.parse_args()
    path = args.input

    # Load file (Excel or JSONL)
    if path.endswith(".xlsx"):
        df = pd.read_excel(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Input file must be .xlsx or .jsonl")

    # Parse date info
    df["parsed_dates"] = df["dates"].apply(safe_parse_dates) if "dates" in df.columns else [{}]

    # Try extracting year
    df["year"] = df.apply(
        lambda row: try_extract_year(
            row.get("parsed_dates", {}),
            fallback=row.get("file_name", "") if "file_name" in row else None,
        ),
        axis=1
    )

    # Count cases by year
    year_counts = Counter(df["year"].dropna().astype(int))
    total = sum(year_counts.values())

    # Print results sorted by year descending
    print("\n=== Case Counts by Year ===")
    print(f"Total cases analyzed: {total}\n")
    for year, count in sorted(year_counts.items(), key=lambda x: x[0], reverse=True):
        print(f"{year:<10} {count:>6} cases")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()
