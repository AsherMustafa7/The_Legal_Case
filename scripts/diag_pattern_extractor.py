#!/usr/bin/env python3
"""
scripts/diag_pattern_extractor.py

Scan raw text files and produce diagnostics helping to improve detect_location().
Outputs:
 - outputs/location_diagnostics.csv
 - outputs/location_diagnostics.jsonl

Usage:
 python scripts/diag_pattern_extractor.py --raw_texts raw_texts --places data/india_5_states_sample.json --out_dir outputs
"""
import argparse
import json
import os
from pathlib import Path
import re
import csv
from collections import Counter, defaultdict
import difflib
import math

# ---------------- helpers ----------------
def _norm(s):
    if s is None: return ""
    s = str(s)
    s = re.sub(r"[\r\n]+", " ", s)
    s = s.strip()
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def read_places(path):
    with open(path, "r", encoding="utf-8") as f:
        places = json.load(f)
    # Build normalized lookups
    states = {}
    districts = {}   # norm -> (state, original)
    cities = {}      # norm -> (state, district_if_any, original)
    for st, info in places.items():
        st_norm = _norm(st)
        states[st_norm] = st
        d_obj = info.get("districts") or {}
        # districts may be dict or list
        if isinstance(d_obj, dict):
            for dname, dinfo in d_obj.items():
                if not dname: continue
                dnorm = _norm(dname)
                districts.setdefault(dnorm, []).append((st, dname))
                # major cities under district
                mcs = dinfo.get("major_cities") or dinfo.get("cities") or {}
                if isinstance(mcs, dict):
                    for cname in mcs.keys():
                        cnorm = _norm(cname)
                        cities.setdefault(cnorm, []).append((st, dname, cname))
                elif isinstance(mcs, list):
                    for cname in mcs:
                        cnorm = _norm(cname)
                        cities.setdefault(cnorm, []).append((st, dname, cname))
        elif isinstance(d_obj, list):
            for dname in d_obj:
                if not dname: continue
                dnorm = _norm(dname)
                districts.setdefault(dnorm, []).append((st, dname))
        # top-level major cities
        top_cities = info.get("major_cities") or info.get("cities") or {}
        if isinstance(top_cities, dict):
            for cname in top_cities.keys():
                cnorm = _norm(cname)
                cities.setdefault(cnorm, []).append((st, None, cname))
        elif isinstance(top_cities, list):
            for cname in top_cities:
                cnorm = _norm(cname)
                cities.setdefault(cnorm, []).append((st, None, cname))
    return {"raw": places, "states": states, "districts": districts, "cities": cities}

# explicit patterns to find (capturing groups)
EXPLICIT_PATTERNS = [
    r"(?:district|dist|districts?)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)",
    r"(?:tehsil|taluk|taluka)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)",
    r"(?:police station|p\.s\.|ps)\s*[:\-\–]?\s*([A-Z][A-Za-z0-9\-\s]+)",
    r"(?:city|town)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)",
    r"district\s*[-\n\r\s]+([A-Z][A-Za-z0-9\-\s]+)",   # "District-\nKanpur Nagar"
    r"(?:sitting at|sitting in|court at|court of)\s+([A-Z][A-Za-z0-9\-\s]+)"
]

CAPITALIZED_PHRASE_RE = re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}")

# blacklist short/frequent false tokens to exclude from fuzzy attempts
COMMON_FALSE = {"judicature","judgment","judgement","the","mon","tue","wed","thu","fri","sat","sun",
                "developers","election","petition","order","case","court","appellant","respondent"}

# ---------------- main per-file analysis ----------------
def analyze_text(text, places_idx, fuzzy_threshold=0.88, header_pref_len=800):
    """Return a dict of diagnostics for a single document text."""
    out = {
        "explicit_matches": [],
        "exact_state_matches": [],
        "exact_district_matches": [],
        "exact_city_matches": [],
        "fuzzy_matches": [],
        "capitalized_phrases": [],
        "header_snippet": text.strip()[:1200].replace("\n", " "),
    }
    text_flat = re.sub(r"[\r\n]+", " ", text)
    text_norm = _norm(text)

    # 1) explicit patterns
    for pat in EXPLICIT_PATTERNS:
        for m in re.finditer(pat, text, re.I):
            candidate = m.group(1).strip()
            out["explicit_matches"].append({"pattern": pat, "candidate": candidate, "span": (m.start(), m.end()), "snippet": text[max(0,m.start()-80):m.end()+80].replace("\n"," ")})
    # 2) exact dictionary matches via word-boundary on normalized text
    # states
    for st_norm, st_orig in places_idx["states"].items():
        if re.search(r"\b" + re.escape(st_norm) + r"\b", text_norm):
            out["exact_state_matches"].append({"state_norm": st_norm, "state": st_orig, "count": len(re.findall(r"\b"+re.escape(st_norm)+r"\b", text_norm))})
    # districts
    for dnorm, dlist in places_idx["districts"].items():
        if re.search(r"\b" + re.escape(dnorm) + r"\b", text_norm):
            out["exact_district_matches"].append({"district_norm": dnorm, "candidates": dlist, "count": len(re.findall(r"\b"+re.escape(dnorm)+r"\b", text_norm))})
    # cities
    for cnorm, clist in places_idx["cities"].items():
        if re.search(r"\b" + re.escape(cnorm) + r"\b", text_norm):
            out["exact_city_matches"].append({"city_norm": cnorm, "candidates": clist, "count": len(re.findall(r"\b"+re.escape(cnorm)+r"\b", text_norm))})

    # 3) capitalized phrases and fuzzy match them to district/city lists (only if we don't have strong exact district)
    caps = CAPITALIZED_PHRASE_RE.findall(text)
    caps = list(dict.fromkeys(caps))[:250]  # unique preserve order, limit
    out["capitalized_phrases"] = caps

    # Build lists for fuzzy comparison
    district_names = []
    for dnorm, tuples in places_idx["districts"].items():
        # get the original form(s)
        for st, dname in tuples:
            if dname and dname not in district_names:
                district_names.append(dname)
    city_names = []
    for cnorm, tuples in places_idx["cities"].items():
        for st, dname, cname in tuples:
            if cname and cname not in city_names:
                city_names.append(cname)

    # fuzzy compare capitalized phrases to district and city names
    for tok in caps:
        tnorm = _norm(tok)
        if not tnorm or tnorm in COMMON_FALSE or len(tnorm) < 3:
            continue
        # fuzzy to districts
        best = difflib.get_close_matches(tok, district_names, n=1, cutoff=fuzzy_threshold)
        if best:
            out["fuzzy_matches"].append({"token": tok, "type": "district", "match": best[0], "ratio_est": None})
            continue
        bestc = difflib.get_close_matches(tok, city_names, n=1, cutoff=fuzzy_threshold)
        if bestc:
            out["fuzzy_matches"].append({"token": tok, "type": "city", "match": bestc[0], "ratio_est": None})
            continue

    # 4) assemble a "best guess" using simple heuristics:
    # priority: explicit district -> exact district -> exact city -> state in header -> top freq state -> fuzzy
    best_guess = {"state": None, "district": None, "city": None, "reason": None, "snippet": None}

    if out["explicit_matches"]:
        # try to resolve explicit to known district/city/state
        em = out["explicit_matches"][0]
        cand = _norm(em["candidate"])
        # check exact district
        if cand in places_idx["districts"]:
            dlist = places_idx["districts"][cand]
            st = dlist[0][0]
            dname = dlist[0][1]
            best_guess.update(state=st, district=dname, city=None, reason="explicit_district", snippet=em["snippet"])
        else:
            # fuzzy against districts
            fm = difflib.get_close_matches(em["candidate"], district_names, n=1, cutoff=fuzzy_threshold)
            if fm:
                # find state
                match = fm[0]
                # find state via index
                for dnorm, tuples in places_idx["districts"].items():
                    for st, d in tuples:
                        if d == match:
                            best_guess.update(state=st, district=match, city=None, reason="explicit_district_fuzzy", snippet=em["snippet"])
                            break
                    if best_guess["district"]: break
            else:
                best_guess.update(reason="explicit_but_unresolved", snippet=em["snippet"])

    elif out["exact_district_matches"]:
        d = out["exact_district_matches"][0]
        dnorm = d["district_norm"]
        st, dname = places_idx["districts"][dnorm][0]
        best_guess.update(state=st, district=dname, reason="exact_district", snippet=None)

    elif out["exact_city_matches"]:
        c = out["exact_city_matches"][0]
        cnorm = c["city_norm"]
        st, dname, cname = places_idx["cities"][cnorm][0]
        best_guess.update(state=st, district=dname, city=cname, reason="exact_city")

    else:
        # try header-level state detection
        header = _norm(text[:header_pref_len])
        for st_norm, st_orig in places_idx["states"].items():
            if re.search(r"\b" + re.escape(st_norm) + r"\b", header):
                best_guess.update(state=st_orig, reason="state_in_header", snippet=text[:min(300,len(text))].replace("\n"," "))
                break

        if not best_guess["state"]:
            # fallback top frequency exact state in document
            if out["exact_state_matches"]:
                best_guess.update(state=out["exact_state_matches"][0]["state"], reason="state_freq")

    # also include top few candidate tokens for manual inspection
    top_candidates = {
        "explicit": [e["candidate"] for e in out["explicit_matches"][:3]],
        "exact_districts": [d["district_norm"] for d in out["exact_district_matches"][:3]],
        "exact_cities": [c["city_norm"] for c in out["exact_city_matches"][:3]],
        "fuzzy": [f"{f['token']}->{f['match']}" for f in out["fuzzy_matches"][:5]]
    }
    return {"diagnostics": out, "best_guess": best_guess, "top_candidates": top_candidates}

# ---------------- CLI ----------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_texts", required=True, help="Folder containing raw .txt files")
    p.add_argument("--places", required=True, help="india_5_states_sample.json (places JSON)")
    p.add_argument("--out_dir", default="outputs", help="output directory")
    p.add_argument("--fuzzy_threshold", type=float, default=0.88, help="difflib cutoff for fuzzy matching")
    p.add_argument("--header_pref_len", type=int, default=800, help="how many chars of header get priority")
    args = p.parse_args()

    places_idx = read_places(args.places)
    raw_dir = Path(args.raw_texts)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "location_diagnostics.csv"
    jsonl_path = out_dir / "location_diagnostics.jsonl"

    file_paths = sorted(list(raw_dir.glob("*.txt")))
    if not file_paths:
        print("No .txt files found in", raw_dir)
        return

    rows = []
    token_counter = Counter()
    for fp in file_paths:
        text = fp.read_text(encoding="utf-8", errors="ignore")
        res = analyze_text(text, places_idx, fuzzy_threshold=args.fuzzy_threshold, header_pref_len=args.header_pref_len)
        best = res["best_guess"]
        diag = res["diagnostics"]
        row = {
            "file": str(fp.name),
            "best_state": best.get("state"),
            "best_district": best.get("district"),
            "best_city": best.get("city"),
            "reason": best.get("reason"),
            "snippet": (best.get("snippet") or "")[:500],
            "explicit": ";".join(res["top_candidates"]["explicit"]) or "",
            "exact_districts": ";".join(res["top_candidates"]["exact_districts"]) or "",
            "exact_cities": ";".join(res["top_candidates"]["exact_cities"]) or "",
            "fuzzy": ";".join(res["top_candidates"]["fuzzy"]) or ""
        }
        rows.append(row)
        # token counters for summary
        for k in ("explicit_matches","exact_district_matches","exact_city_matches"):
            for e in diag.get(k, [])[:5]:
                token = e.get("candidate") or e.get("district_norm") or e.get("city_norm")
                if token:
                    token_counter[token] += 1

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        fieldnames = ["file","best_state","best_district","best_city","reason","snippet","explicit","exact_districts","exact_cities","fuzzy"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # write JSONL full diagnostics (re-run per file to include verbose)
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for fp in file_paths:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            res = analyze_text(text, places_idx, fuzzy_threshold=args.fuzzy_threshold, header_pref_len=args.header_pref_len)
            outobj = {"file": str(fp.name), "diagnostic": res}
            jf.write(json.dumps(outobj, ensure_ascii=False) + "\n")

    # print summary
    print(f"Wrote CSV -> {csv_path}")
    print(f"Wrote JSONL -> {jsonl_path}")
    print("\nTop matched candidate tokens (sample 40):")
    for tok, c in token_counter.most_common(40):
        print(f"{c:4d}  {tok}")

if __name__ == "__main__":
    import argparse
    main()
