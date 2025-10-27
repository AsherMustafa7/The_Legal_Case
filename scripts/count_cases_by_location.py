#!/usr/bin/env python3
"""
scripts/count_cases_by_location.py

Counts cases grouped by city+district+state and by district+state, printing
human-readable summaries and writing CSVs.

Usage examples:
    python scripts/count_cases_by_location.py --input outputs/output_kmeans_with_seed.xlsx
    python scripts/count_cases_by_location.py --input cases.jsonl --min_count 5 --out_dir outputs
"""
import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required to run this script. Install via pip install pandas") from e

# --- helpers to parse location field robustly ---
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
        # otherwise return raw string under 'raw'
        return {"raw": v}
    return {}

# Normalizers and text extractors
_norm_space_re = re.compile(r"\s+")
def norm_text(s):
    if s is None:
        return ""
    return _norm_space_re.sub(" ", str(s).strip())

# Common key variants we'll accept for state/district/city
STATE_KEYS = {"state", "State", "STATE", "province", "region", "state_name"}
DISTRICT_KEYS = {"district", "District", "DISTRICT", "dist", "Dist", "taluk", "tehsil", "taluka"}
CITY_KEYS = {"city", "City", "town", "Town", "village", "major_city", "major_cities", "city_name"}

# Patterns to pull "District: X" or "City: Y" from raw text
PATTERNS = {
    "district": re.compile(r"(?:district|dist|districts?)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)", re.I),
    "city": re.compile(r"(?:city|town|village)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)", re.I),
    "state": re.compile(r"(?:state|province|region)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)", re.I),
}

def extract_from_raw_text(raw):
    """Try to pull state/district/city from an unstructured raw string using patterns."""
    if not raw:
        return {}
    out = {}
    s = raw
    for key, pat in PATTERNS.items():
        m = pat.search(s)
        if m:
            out[key] = norm_text(m.group(1))
    # Additional fallback: comma-separated tokens, look for "City, District, State" style
    if "state" not in out or "district" not in out:
        # take last 3 comma-separated chunks (common in addresses)
        parts = [p.strip() for p in re.split(r"[,\n;]+", s) if p.strip()]
        if len(parts) >= 2:
            # try to interpret last as state (if it's short-ish)
            cand_state = parts[-1]
            if len(cand_state) > 1 and len(cand_state) < 40 and "court" not in cand_state.lower():
                out.setdefault("state", norm_text(cand_state))
        if len(parts) >= 3:
            out.setdefault("district", norm_text(parts[-2]))
            out.setdefault("city", norm_text(parts[-3]))
    return out

def extract_location_fields(location_field):
    """
    Return tuple (city, district, state) extracted from either:
      - dict-like parsed location (preferred),
      - or raw string heuristics.
    All values are normalized (stripped). Returns None where not found.
    """
    parsed = try_parse_location(location_field)
    city = district = state = None

    # 1) If parsed is dict, check common keys first
    if isinstance(parsed, dict) and parsed:
        for k in CITY_KEYS:
            if k in parsed and parsed[k]:
                city = norm_text(parsed[k]) if not isinstance(parsed[k], (list, dict)) else None
                break
        for k in DISTRICT_KEYS:
            if k in parsed and parsed[k]:
                district = norm_text(parsed[k]) if not isinstance(parsed[k], (list, dict)) else None
                break
        for k in STATE_KEYS:
            if k in parsed and parsed[k]:
                state = norm_text(parsed[k]) if not isinstance(parsed[k], (list, dict)) else None
                break
        # nested shapes or alternate field names (e.g., {"location":"X, Y, Z"})
        if not (city or district or state):
            # try flattening any string values in parsed
            for v in parsed.values():
                if isinstance(v, str) and ("," in v or ":" in v):
                    heur = extract_from_raw_text(v)
                    city = city or heur.get("city")
                    district = district or heur.get("district")
                    state = state or heur.get("state")
    else:
        # parsed may be {'raw': '...'} or string -> try raw heuristics
        raw = parsed.get("raw") if isinstance(parsed, dict) else ""
        heur = extract_from_raw_text(raw)
        city = heur.get("city")
        district = heur.get("district")
        state = heur.get("state")

    # Final fallback: if still empty and original was a plain string, try global heuristics
    if not (city or district or state) and isinstance(location_field, str):
        heur = extract_from_raw_text(location_field)
        city = city or heur.get("city")
        district = district or heur.get("district")
        state = state or heur.get("state")

    # normalize empty -> None
    city = city if city and city.strip() else None
    district = district if district and district.strip() else None
    state = state if state and state.strip() else None
    return city, district, state

# --- main CLI logic ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Excel (.xlsx) or JSONL (.jsonl) file containing extracted cases")
    p.add_argument("--out_dir", default=".", help="Directory to write CSV outputs")
    p.add_argument("--min_count", type=int, default=1, help="Minimum count threshold to print (default 1)")
    p.add_argument("--top", type=int, default=200, help="Max number of lines printed for city-level (default 200)")
    args = p.parse_args()

    infile = args.input
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if infile.endswith(".xlsx"):
        df = pd.read_excel(infile)
    elif infile.endswith(".jsonl"):
        df = pd.read_json(infile, lines=True)
    else:
        raise ValueError("Input must be .xlsx or .jsonl")

    if "location" not in df.columns:
        raise ValueError("No 'location' column found in dataset!")

    # parse and extract fields
    extracted = df["location"].apply(lambda v: extract_location_fields(v))
    # create columns
    df["_city"], df["_district"], df["_state"] = zip(*extracted)

    # if there are explicit columns already (sometimes present), use them as fallbacks
    if "city" in df.columns:
        df["_city"] = df["_city"].fillna(df["city"].apply(lambda x: norm_text(x) if pd.notna(x) else None))
    if "district" in df.columns:
        df["_district"] = df["_district"].fillna(df["district"].apply(lambda x: norm_text(x) if pd.notna(x) else None))
    if "state" in df.columns:
        df["_state"] = df["_state"].fillna(df["state"].apply(lambda x: norm_text(x) if pd.notna(x) else None))

    # Build readable strings and counters
    city_district_state_counter = Counter()
    district_state_counter = Counter()
    rows_with_any_location = 0

    for idx, row in df.iterrows():
        city = row.get("_city") or None
        district = row.get("_district") or None
        state = row.get("_state") or None

        if city or district or state:
            rows_with_any_location += 1

        # canonical keys (use empty strings to keep groups consistent)
        key_cds = (city or "").strip(), (district or "").strip(), (state or "").strip()
        key_ds = (district or "").strip(), (state or "").strip()

        city_district_state_counter[key_cds] += 1
        district_state_counter[key_ds] += 1

    # --- print human readable city+district+state results ---
    print(f"\nTotal rows with some location info: {rows_with_any_location} / {len(df)}\n")
    print("=== Top city,district,state groups ===\n")
    printed = 0
    for (city, district, state), cnt in city_district_state_counter.most_common():
        if cnt < args.min_count:
            continue
        pretty = []
        if city:
            pretty.append(f"city {city}")
        if district:
            pretty.append(f"district {district}")
        if state:
            pretty.append(f"state {state}")
        label = ", ".join(pretty) if pretty else "(unspecified)"
        print(f"{cnt:5d} cases found in {label}")
        printed += 1
        if printed >= args.top:
            break

    # --- print district,state only ---
    print("\n=== District + State groups ===\n")
    for (district, state), cnt in district_state_counter.most_common():
        if cnt < args.min_count:
            continue
        if district or state:
            if district and state:
                print(f"{cnt:5d} cases found in district {district} of state {state}")
            elif district and not state:
                print(f"{cnt:5d} cases found in district {district} (state unspecified)")
            elif state and not district:
                print(f"{cnt:5d} cases found in state {state} (district unspecified)")

    # --- write CSV outputs ---
    cds_rows = []
    for (city, district, state), cnt in city_district_state_counter.items():
        cds_rows.append({"city": city or "", "district": district or "", "state": state or "", "count": cnt})
    ds_rows = []
    for (district, state), cnt in district_state_counter.items():
        ds_rows.append({"district": district or "", "state": state or "", "count": cnt})

    df_cds = pd.DataFrame(cds_rows).sort_values(["count"], ascending=False)
    df_ds = pd.DataFrame(ds_rows).sort_values(["count"], ascending=False)

    out1 = out_dir / "city_district_state_counts.csv"
    out2 = out_dir / "district_state_counts.csv"
    df_cds.to_csv(out1, index=False)
    df_ds.to_csv(out2, index=False)

    print(f"\nCSV outputs written:\n  {out1}\n  {out2}\n")
    print("✅ Done.")

if __name__ == "__main__":
    main()
