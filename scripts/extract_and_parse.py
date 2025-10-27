# scripts/extract_and_parse.py
import os
import json
import argparse
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from pathlib import Path
from data import regex_patterns as rp
import spacy
import re
import math
import difflib
from collections import defaultdict
import unicodedata
nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------------------------
# Load place data
# -------------------------------------------------------------------
def load_places(path="data/indian_places.json"):
    """
    Loads place lookup JSON. Prefer 'india_5_states_sample.json' in the same directory
    as `path` if it exists; otherwise fall back to the provided path.
    """
    base_dir = os.path.dirname(path) or "."
    richer = os.path.join(base_dir, "india_5_states_sample.json")

    if os.path.exists(richer):
        chosen = richer
    elif os.path.exists(path):
        chosen = path
    else:
        raise FileNotFoundError(f"Neither '{richer}' nor '{path}' found. Place a JSON file in the data folder.")

    with open(chosen, "r", encoding="utf-8") as f:
        places = json.load(f)
    return places

# -------------------------------------------------------------------
# Detect state/district/city using both regex + JSON
# -------------------------------------------------------------------
# --- Replace your existing detect_location with the following ---
# ---- Replace detect_location with improved scoring-based version ----

# ---------- Improved detect_location (drop-in) ----------

def _norm(s):
    if not s: return ""
    s = str(s)
    s = re.sub(r"[\r\n]+", " ", s)         # collapse newlines
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# optional alias map: add common alternatives you know (extend as needed)
ALIASES = {
    _norm("gautam buddha nagar"): ["gautam budh nagar", "gautham budh nagar", "ghaziabad-north"],  # examples
    _norm("kanpur nagar"): ["kanpur", "kanpur nagar district"]
    # add more as you discover mismatches
}

def _closest_match(token, candidates, min_ratio=0.86):
    # returns candidate,ratio or (None,0)
    if not token or not candidates: return None, 0.0
    best = None
    best_ratio = 0.0
    for c in candidates:
        r = difflib.SequenceMatcher(None, token, _norm(c)).ratio()
        if r > best_ratio:
            best_ratio = r; best = c
    if best_ratio >= min_ratio:
        return best, best_ratio
    return None, best_ratio













def detect_location(text, places, nlp_obj=None):
    """
    Strict High Court-first location detector (drop-in).
    - If a High Court is detected (except Supreme Court), only scan that court's
      jurisdiction state(s) first and accept any match found in those state(s).
    - If nothing found in the court-priority states, fall back to explicit regex check
      and then to global collection + fuzzy scoring (same as before).
    Returns: dict with keys {raw,state,district,city,confidence,match_context,score_breakdown}
    """
    if not text:
        return {"raw": None, "state": None, "district": None, "city": None,
                "confidence": 0.0, "match_context": None, "score_breakdown": {}}

    # helpers
    def snippet_from_match(m):
        if not m:
            return None
        s, e = m.start(), m.end()
        start = max(0, s - 60)
        end = min(len(text), e + 60)
        return text[start:end].replace("\n", " ")

    text_flat = re.sub(r"[\r\n]+", " ", text)
    text_n = _norm(text)

    # ---------------- 0) Extract court name and build prioritized state list ---------------
    court_state_priorities = None
    court_name_snippet = None
    court_match = None
    try:
        court_match = rp.court_re.search(text)
    except Exception:
        court_match = None

    if court_match:
        court_name = court_match.group(0).strip()
        court_name_snippet = court_name
        uname = court_name.upper()

        # map substrings -> prioritized list of place JSON keys
        # priority lists (ordered) — use this to try multiple jurisdictions in order
        COURT_TO_STATE_PRIORITIES = {
            "ALLAHABAD": ["Uttar Pradesh"],
            "LUCKNOW": ["Uttar Pradesh"],
            "BOMBAY": ["Maharashtra", "Goa", "Dadra and Nagar Haveli and Daman and Diu"],
            "MUMBAI": ["Maharashtra", "Goa", "Dadra and Nagar Haveli and Daman and Diu"],
            "BENGALURU": ["Karnataka"],
            "BANGALORE": ["Karnataka"],
            "MADRAS": ["Tamil Nadu", "Puducherry"],
            "CHENNAI": ["Tamil Nadu", "Puducherry"],
            "CALCUTTA": ["West Bengal", "Andaman and Nicobar Islands"],
            "KOLKATA": ["West Bengal", "Andaman and Nicobar Islands"],
            "KERALA": ["Kerala", "Lakshadweep"],
            "PUNJAB": ["Punjab"],
            "HARYANA": ["Haryana"],
            "CHANDIGARH": ["Punjab", "Haryana", "Chandigarh"],   # Chandigarh shares HC arrangements
            "DELHI": ["Delhi"],
            "GUJARAT": ["Gujarat"],
            "RAJASTHAN": ["Rajasthan"],
            "MADHYA PRADESH": ["Madhya Pradesh"],
            "UTTAR PRADESH": ["Uttar Pradesh"],
            "TELANGANA": ["Telangana"],
            "ANDHRA PRADESH": ["Andhra Pradesh"],
            "HIMACHAL": ["Himachal Pradesh"],
            "JAMMU": ["Jammu and Kashmir", "Ladakh"],
            "SRINAGAR": ["Jammu and Kashmir", "Ladakh"],
            "KASHMIR": ["Jammu and Kashmir", "Ladakh"],
            "ASSAM": ["Assam"],
            "GAUHATI": ["Assam", "Arunachal Pradesh", "Nagaland", "Manipur", "Meghalaya", "Tripura", "Mizoram"],  # Gauhati historically covers NE; include all NE states it serves
            "ARUNACHAL": ["Arunachal Pradesh"],
            "MIZORAM": ["Mizoram"],
            "NAGALAND": ["Nagaland"],
            "MEGHALAYA": ["Meghalaya"],
            "MANIPUR": ["Manipur"],
            "TRIPURA": ["Tripura"],
            "GOA": ["Goa"],
            "ODISHA": ["Odisha"],
            "JHARKHAND": ["Jharkhand"],
            "CHHATTISGARH": ["Chhattisgarh"],
            "SIKKIM": ["Sikkim"],
            "ANDAMAN": ["Andaman and Nicobar Islands", "West Bengal"],  # Calcutta HC has jurisdictional linkage historically
            "DADRA": ["Dadra and Nagar Haveli and Daman and Diu", "Gujarat", "Maharashtra"], # DNH/DD often grouped with Bombay HC in practice
            "LAKSHADWEEP": ["Lakshadweep", "Kerala"],  # often serviced administratively via Kerala
            "LADAKH": ["Ladakh", "Jammu and Kashmir"],
            "PUDUCHERRY": ["Puducherry", "Tamil Nadu"],
            "BIHAR": ["Bihar"],
            "PATNA": ["Bihar"],
            "UTTARAKHAND": ["Uttarakhand"],
            "NAINITAL": ["Uttarakhand"]
        }



        # SUPREME COURT: special — do NOT restrict to a single state
        if "SUPREME COURT" not in uname:
            # find first keyed priority whose token appears in the court name
            for key, st_list in COURT_TO_STATE_PRIORITIES.items():
                if key in uname:
                    court_state_priorities = st_list
                    break

        # fallback: if "HIGH COURT OF <STATE>" present, try to extract that phrase and reconcile
        if court_state_priorities is None:
            m2 = re.search(r"HIGH COURT OF\s+([A-Z\s\.&]+)", court_name.upper())
            if m2:
                cand = m2.group(1).strip().title()
                matched = []
                for p in places.keys():
                    if _norm(cand) in _norm(p) or _norm(p) in _norm(cand):
                        matched.append(p)
                if matched:
                    # keep matched order (could be multiple variants)
                    court_state_priorities = matched

    # ---------------- helper: strict scan within one state ----------------
    def scan_one_state(state_key):
        """Return candidate dict or None — strict search within that state's districts and cities."""
        st_info = places.get(state_key)
        if not st_info or not isinstance(st_info, dict):
            return None

        context_re = re.compile(r"(district|dist|tehsil|taluk|taluka|city|village|at|in|court|bench|police station)", re.I)
        best_local = None
        best_score = -1.0
        norm_text = _norm(text)

        # districts first (with city check)
        districts_obj = st_info.get("districts", {}) or {}
        if isinstance(districts_obj, dict):
            for dname, dinfo in districts_obj.items():
                if not dname:
                    continue
                pat = r"\b" + re.escape(_norm(dname)) + r"\b"
                m = re.search(pat, norm_text)
                if m:
                    pos = m.start()
                    score = 1.0
                    if pos < 800:
                        score += 1.5
                    ctx_snip = text[max(0, pos-60): pos+60]
                    if context_re.search(ctx_snip):
                        score += 1.5
                    # look for city under district
                    city_found = None
                    mcs = dinfo.get("major_cities") or dinfo.get("cities") or {}
                    if isinstance(mcs, dict):
                        for cname in mcs.keys():
                            cm = re.search(r"\b" + re.escape(_norm(cname)) + r"\b", norm_text)
                            if cm:
                                city_found = cname
                                score += 0.9
                                break
                    if score > best_score:
                        best_score = score
                        best_local = {
                            "raw": city_found or dname,
                            "state": state_key,
                            "district": dname,
                            "city": city_found,
                            "confidence": round(min(0.995, 1 - (1.0 / (1.0 + score))), 3),
                            "match_context": ctx_snip,
                            "score_breakdown": {"local_score": score, "method": "state_prioritized_district_match"}
                        }

        # if no district match, try top-level state cities
        if not best_local:
            state_cities = st_info.get("major_cities") or st_info.get("cities") or {}
            if isinstance(state_cities, dict):
                for cname in state_cities.keys():
                    cm = re.search(r"\b" + re.escape(_norm(cname)) + r"\b", _norm(text))
                    if cm:
                        pos = cm.start()
                        score = 1.2
                        if pos < 800:
                            score += 1.0
                        ctx_snip = text[max(0, pos-60): pos+60]
                        if context_re.search(ctx_snip):
                            score += 1.2
                        best_local = {
                            "raw": cname,
                            "state": state_key,
                            "district": None,
                            "city": cname,
                            "confidence": round(min(0.995, 1 - (1.0 / (1.0 + score))), 3),
                            "match_context": ctx_snip,
                            "score_breakdown": {"local_score": score, "method": "state_prioritized_city_match"}
                        }
                        break

        return best_local

    # --------------- 1) If court priorities exist, scan them STRICTLY and accept first hit --------------
    if court_state_priorities:
        # court_state_priorities may be list of state names; try each in order
        for st_hint in court_state_priorities:
            # normalize to an actual places key
            st_key = None
            for p in places.keys():
                if p.lower() == st_hint.lower() or _norm(p) == _norm(st_hint):
                    st_key = p
                    break
            if not st_key:
                # try fuzzy-ish containment
                for p in places.keys():
                    if _norm(st_hint) in _norm(p) or _norm(p) in _norm(st_hint):
                        st_key = p
                        break
            if not st_key:
                continue
            found = scan_one_state(st_key)
            if found:
                # strict HC-first behavior: accept any match within the HC's state(s)
                found["note"] = "strict_match_from_court_state"
                found["court_name_snippet"] = court_name_snippet
                return found
        # if no match found in any prioritized state, continue to explicit/global logic below

    # ---------------- 2) explicit-pattern extraction (District: X / Tehsil: X) across all states ----------------
    explicit_patterns = [
        r"(?:district|dist|districts?)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)",
        r"(?:tehsil|taluk|taluka)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)",
        r"(?:city|town)\s*[:\-\–]\s*([A-Z][A-Za-z0-9\-\s]+)",
        r"district\s*[-\n\r\s]+([A-Z][A-Za-z0-9\-\s]+)"
    ]
    for pat in explicit_patterns:
        m = re.search(pat, text, re.I)
        if m:
            candidate = m.group(1).strip()
            cand_norm = _norm(candidate)
            for state_name, info in places.items():
                districts = info.get("districts") or {}
                if isinstance(districts, dict):
                    for dname in districts.keys():
                        if _norm(dname) == cand_norm:
                            match_snip = text[max(0, m.start()-80): m.end()+80].replace("\n", " ")
                            return {
                                "raw": dname, "state": state_name, "district": dname, "city": None,
                                "confidence": 0.99, "match_context": match_snip,
                                "score_breakdown": {"explicit_regex": 1.0, "court_name_snippet": court_name_snippet}
                            }
            # if not exact, fall through to fuzzy/global below

    # ---------------- 3) Global candidate collection (word-boundary exact matches) ----------------
    candidates = []
    for state_name, info in places.items():
        if re.search(r"\b" + re.escape(_norm(state_name)) + r"\b", text_n):
            candidates.append({"state": state_name, "district": None, "city": None, "pos": text_n.find(_norm(state_name)), "match_text": state_name})
        districts_obj = info.get("districts", {}) or {}
        if isinstance(districts_obj, dict):
            for dname in districts_obj.keys():
                if not dname:
                    continue
                dnorm = _norm(dname)
                m = re.search(r"\b" + re.escape(dnorm) + r"\b", text_n)
                if m:
                    pos = m.start()
                    candidates.append({"state": state_name, "district": dname, "city": None, "pos": pos, "match_text": dname})
                    dinfo = districts_obj.get(dname, {}) or {}
                    mcs = dinfo.get("major_cities") or dinfo.get("cities") or {}
                    if isinstance(mcs, dict):
                        for cname in mcs.keys():
                            cm = re.search(r"\b" + re.escape(_norm(cname)) + r"\b", text_n)
                            if cm:
                                candidates.append({"state": state_name, "district": dname, "city": cname, "pos": cm.start(), "match_text": cname})
        else:
            if isinstance(districts_obj, list):
                for dname in districts_obj:
                    if not dname:
                        continue
                    m = re.search(r"\b" + re.escape(_norm(dname)) + r"\b", text_n)
                    if m:
                        candidates.append({"state": state_name, "district": dname, "city": None, "pos": m.start(), "match_text": dname})

        state_cities = info.get("major_cities") or info.get("cities") or {}
        if isinstance(state_cities, dict):
            for cname in state_cities.keys():
                cm = re.search(r"\b" + re.escape(_norm(cname)) + r"\b", text_n)
                if cm:
                    candidates.append({"state": state_name, "district": None, "city": cname, "pos": cm.start(), "match_text": cname})

    # ------------- 4) fuzzy fallback if no exact candidates ----------------
    if not candidates:
        tokens = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}", text)
        uniq = list(dict.fromkeys(tokens))[:300]
        district_names = []
        city_names = []
        for st, info in places.items():
            dlist = info.get("districts") or {}
            if isinstance(dlist, dict):
                district_names.extend(list(dlist.keys()))
                for d, dinfo in dlist.items():
                    mcs = dinfo.get("major_cities") or dinfo.get("cities") or {}
                    if isinstance(mcs, dict):
                        city_names.extend(list(mcs.keys()))
            state_cities = info.get("major_cities") or info.get("cities") or {}
            if isinstance(state_cities, dict):
                city_names.extend(list(state_cities.keys()))
            elif isinstance(state_cities, list):
                city_names.extend(state_cities)
        district_names = list(dict.fromkeys(district_names))
        city_names = list(dict.fromkeys(city_names))

        for tok in uniq:
            tnorm = _norm(tok)
            if len(tnorm) < 3:
                continue
            dmatch, dr = _closest_match(tnorm, district_names, min_ratio=0.88)
            if dmatch:
                mpos = text_n.find(_norm(dmatch))
                candidates.append({"state": None, "district": dmatch, "city": None, "pos": mpos if mpos >= 0 else 999999, "match_text": dmatch, "fuzzy": dr})
                continue
            cmatch, cr = _closest_match(tnorm, city_names, min_ratio=0.88)
            if cmatch:
                mpos = text_n.find(_norm(cmatch))
                candidates.append({"state": None, "district": None, "city": cmatch, "pos": mpos if mpos >= 0 else 999999, "match_text": cmatch, "fuzzy": cr})
                continue
            for canonical, altlist in ALIASES.items():
                for alt in altlist:
                    if _norm(alt) == tnorm:
                        for stn, info in places.items():
                            if isinstance(info.get("districts"), dict) and any(_norm(d) == canonical for d in info.get("districts").keys()):
                                dname = next(d for d in info.get("districts").keys() if _norm(d) == canonical)
                                candidates.append({"state": stn, "district": dname, "city": None, "pos": text_n.find(tnorm), "match_text": dname})
                                break

    if not candidates:
        return {"raw": None, "state": None, "district": None, "city": None,
                "confidence": 0.0, "match_context": None, "score_breakdown": {}}

    # ---------------- 5) Score candidates and pick best ----------------
    best = None
    best_score = -1e9
    best_breakdown = None
    gpe_set = set()
    if nlp_obj is not None:
        try:
            doc_small = nlp_obj(text[:20000])
            gpe_set = {ent.text.lower() for ent in doc_small.ents if ent.label_ == "GPE"}
        except Exception:
            gpe_set = set()

    for cand in candidates:
        st = cand.get("state")
        d = cand.get("district")
        c = cand.get("city")
        pos = cand.get("pos", 999999)
        token = cand.get("match_text") or (c or d or st)
        token_norm = _norm(token) if token else ""
        breakdown = defaultdict(float)
        score = 0.0

        # base exact match weight
        score += 1.0; breakdown["base"] = 1.0

        # header / early position bonus
        if pos < 800:
            score += 2.5; breakdown["header_bonus"] = 2.5

        # context proximity
        ctx_segment = text_flat[max(0, pos-60): pos+60]
        if re.search(r"(district|tehsil|at|in|bench|sitting at|court of|court at|police station)", ctx_segment, re.I):
            score += 2.0; breakdown["ctx"] = 2.0

        # spaCy GPE confirmation
        if token_norm and token_norm in gpe_set:
            score += 1.6; breakdown["gpe"] = 1.6

        # fuzzy boost
        if cand.get("fuzzy"):
            score += 0.5; breakdown["fuzzy_boost"] = 0.5

        # length bonus
        length_bonus = min(0.8, len(token_norm) / 40.0)
        score += length_bonus; breakdown["len"] = round(length_bonus, 3)

        # citation penalty
        snippet = text[max(0, pos-60): pos+60]
        if re.search(r"\b(v|vs|v\.)\b.*\bstate of\b", snippet, re.I) or re.search(r"\bstate of\s+" + re.escape(token_norm), snippet, re.I):
            score -= 2.0; breakdown["citation_penalty"] = -2.0

        # blacklist tokens penalty
        if token_norm in {"judicature", "judgment", "judgement", "the","mon","tue","wednesday","monday","developers","election"}:
            score -= 5.0; breakdown["blacklist"] = -5.0

        # state-count bonus
        if st:
            occ = len(re.findall(r"\b" + re.escape(_norm(st)) + r"\b", text_n))
            if occ > 0:
                b = min(1.0, occ * 0.15)
                score += b; breakdown["state_occ"] = round(b, 3)

        # position penalty for deep occurrences
        pos_pen = max(0.0, (pos / max(1, len(text_flat))))
        score -= pos_pen * 0.9; breakdown["pos_penalty"] = -round(pos_pen * 0.9, 4)

        if score > best_score:
            best_score = score
            best = dict(st=st, d=d, c=c, token=token, pos=pos, snippet=snippet)
            best_breakdown = dict(breakdown)

    conf = 1.0 - (1.0 / (1.0 + max(0.0, best_score)))
    raw_choice = best.get("c") or best.get("d") or best.get("st")
    out = {
        "raw": raw_choice,
        "state": best.get("st"),
        "district": best.get("d"),
        "city": best.get("c"),
        "confidence": float(round(conf, 3)),
        "match_context": best.get("snippet"),
        "score_breakdown": best_breakdown
    }
    # attach court debug info if present
    if court_name_snippet:
        out["court_name_snippet"] = court_name_snippet
    return out










# -------------------------------------------------------------------
# OCR and PDF text extraction
# -------------------------------------------------------------------
def ocr_pdf(pdf_path):
    text = ""
    try:
        images = convert_from_path(pdf_path, dpi=200)
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        print(f"OCR conversion error for {pdf_path}: {e}")
    return text

def extract_text(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text()
                if ptext:
                    text += ptext + "\n"
    except Exception as e:
        print(f"pdfplumber error for {pdf_path}: {e}")
    if not text.strip():
        print(f"No text extracted via pdfplumber — using OCR for {pdf_path}")
        text = ocr_pdf(pdf_path)
    return text

# -------------------------------------------------------------------
# Case parsing logic
# -------------------------------------------------------------------
def parse_case(text, filename, seed_map, places):
    out = {
        "file_name": filename,
        "case_title": None,
        "petitioner": None,
        "respondent": None,
        "court_name": None,
        "location": {"raw": None, "state": None, "district": None, "city": None, "confidence": 0.0},
        "dates": {},
        "ipc_sections": [],
        "acts": [],
        "crime_labels": [],
        "keywords": [],
        "entities": {},
        "summary": None,
        "raw_text_path": None
    }

    # --- Case title ---
    m = rp.case_title_re.search(text)
    if m:
        out["petitioner"] = m.group("p1").strip()
        out["respondent"] = m.group("p2").strip()
        out["case_title"] = f"{out['petitioner']} vs {out['respondent']}"

    # --- Court name ---
    cm = rp.court_re.search(text)
    if cm:
        out["court_name"] = cm.group(1).strip()

    # --- IPC Sections ---
    secs = rp.section_re.findall(text)
    if secs:
        cleaned = [s for s in secs if s]
        out["ipc_sections"] = list({s.upper() for s in cleaned})

    # --- Acts ---
    acts_found = []
    for a_re in rp.act_patterns:
        if a_re.search(text):
            acts_found.append(a_re.pattern)
    out["acts"] = list(dict.fromkeys(acts_found))

    # --- DATE EXTRACTION (merged improved version) ---
    found_dates = []
    for regex in [rp.date_re_textual, rp.date_re_slash, rp.date_re_iso, rp.date_re_dot]:
        for m in regex.finditer(text):
            found_dates.append(m.group(1))

    if found_dates:
        # Choose earliest appearance (closest to judgment header)
        out["dates"]["judgment_date"] = found_dates[0]
        out["dates"]["confidence"] = 1.0 if len(found_dates) == 1 else 0.8
    else:
        out["dates"]["judgment_date"] = None
        out["dates"]["confidence"] = 0.0

    # --- Entities ---
    doc = nlp(text[:20000])
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    out["entities"]["persons"] = list(dict.fromkeys(persons))
    out["entities"]["gpes"] = list(dict.fromkeys(gpes))

    # --- Seed Labeling (fixed for nested seed_map) ---
    lower = text.lower()
    labels, keywords = set(), set()

    for label, entry in seed_map.items():
        # each entry is a dict with keys: triggers, ipc_sections, acts
        trigger_list = entry.get("triggers", [])
        ipc_list = entry.get("ipc_sections", [])
        act_list = entry.get("acts", [])

        # 1️⃣ Match trigger words
        for t in trigger_list:
            if t.lower() in lower:
                labels.add(label)
                keywords.add(t)
                break

        # 2️⃣ Match IPC sections if text includes them (e.g. "Section 420")
        for sec in ipc_list:
            if re.search(rf"\b{sec}\b", lower):
                labels.add(label)
                keywords.add(f"section {sec}")
                break

        # 3️⃣ Match Acts by name
        for act in act_list:
            if act.lower() in lower:
                labels.add(label)
                keywords.add(act)
                break

    out["crime_labels"] = sorted(list(labels))
    out["keywords"] = sorted(list(keywords))


    # --- Location Detection (dual-layer) ---
    loc = detect_location(text, places)
    out["location"] = loc

    # --- Summary ---
    judg = re.search(r"(JUDGMENT|JUDGMENT:|JUDGEMENT|JUDGMENT OF THE COURT)(.*?)(\n\n|\Z)", text, re.I | re.S)
    if judg:
        s = judg.group(2).strip()
        out["summary"] = s[:1000]
    else:
        out["summary"] = text[:1000]

    return out

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main(pdfs_folder, raw_texts_folder, output_jsonl, seed_map_path, places_path):
    os.makedirs(raw_texts_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(seed_map_path, "r", encoding="utf-8") as f:
        seed_map = json.load(f)
    places = load_places(places_path)

    pdf_paths = sorted(Path(pdfs_folder).glob("*.pdf"))
    if not pdf_paths:
        print("No PDFs found in", pdfs_folder)
        return

    with open(output_jsonl, "w", encoding="utf-8") as outf:
        for pdf_path in pdf_paths:
            print("Processing:", pdf_path.name)
            text = extract_text(str(pdf_path))

            txt_name = Path(raw_texts_folder) / (pdf_path.stem + ".txt")
            with open(txt_name, "w", encoding="utf-8") as tf:
                tf.write(text)

            parsed = parse_case(text, pdf_path.name, seed_map, places)
            parsed["raw_text_path"] = str(txt_name)
            outf.write(json.dumps(parsed, ensure_ascii=False) + "\n")

    print("✅ Done. Output:", output_jsonl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs_folder", default="pdfs")
    parser.add_argument("--raw_texts", default="raw_texts")
    parser.add_argument("--output_jsonl", default="outputs/cases.jsonl")
    parser.add_argument("--seed_map", default="data/seed_map.json")
    parser.add_argument("--places", default="data/indian_places.json")
    args = parser.parse_args()
    main(args.pdfs_folder, args.raw_texts, args.output_jsonl, args.seed_map, args.places)
