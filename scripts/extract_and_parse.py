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

nlp = spacy.load("en_core_web_sm")

# -------------------------------------------------------------------
# Load place data
# -------------------------------------------------------------------
def load_places(path="data/indian_places.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------------------------------------------------
# Detect state/district/city using both regex + JSON
# -------------------------------------------------------------------
def detect_location(text, places):
    text_lower = text.lower()
    found_state, found_district, found_city = None, None, None
    confidence = 0.0

    # --- 1️⃣ Regex-based quick detection ---
    # Explicit state mentions
    state_match = rp.state_re.search(text)
    if state_match:
        found_state = state_match.group(1).title()
        confidence = max(confidence, 0.8)

    # Generic city/district hints (e.g., “Court at Jaipur”)
    hint_match = rp.place_hint_re.search(text)
    if hint_match:
        found_city = hint_match.group(2).title()
        confidence = max(confidence, 0.6)

    # --- 2️⃣ JSON lookup confirmation ---
    for state, info in places.items():
        if state.lower() in text_lower:
            found_state = state
            confidence = max(confidence, 0.7)

        for district in info.get("districts", []):
            if district.lower() in text_lower:
                found_district = district
                found_state = state
                confidence = max(confidence, 0.9)

        for city in info.get("major_cities", []):
            if city.lower() in text_lower:
                found_city = city
                found_state = state
                confidence = max(confidence, 0.8)

    return {
        "raw": found_city or found_district or found_state,
        "state": found_state,
        "district": found_district,
        "city": found_city,
        "confidence": confidence
    }

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
