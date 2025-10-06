# scripts/utils.py
import json, re

def load_seed_map(path="data/seed_map.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_text(text):
    return text.lower()

def match_seed_labels(case_text, case_sections, seed_map):
    """
    case_text: full summary/raw text
    case_sections: list of detected section numbers (strings like "302","420")
    seed_map: loaded JSON
    returns: dict of matched labels with provenance/confidence
    """
    text = normalize_text(case_text)
    labels = {}
    # 1) Section-based exact matches (highest priority)
    for label, meta in seed_map.items():
        for s in meta.get("ipc_sections", []):
            if s and any(s == sec.strip().lstrip("0") or s in ("s."+sec, "section "+sec) for sec in case_sections):
                labels[label] = {"source":"section","match":s,"confidence":1.0}
                break

    # 2) Act-based matches (high)
    for label, meta in seed_map.items():
        if label in labels: continue
        for act in meta.get("acts", []):
            if act and act.lower() in text:
                labels[label] = {"source":"act","match":act,"confidence":0.9}
                break

    # 3) Trigger phrase matches (lower)
    for label, meta in seed_map.items():
        if label in labels: continue
        for trig in meta.get("triggers", []):
            if not trig: continue
            if trig.lower() in text:
                labels.setdefault(label, {"source":"trigger","matches":[],"confidence":0.7})
                labels[label]["matches"].append(trig)
    return labels




# 1) What will utils.py do now?

# utils.py is your helper functions file. Right now, it will contain the code I gave you for:

# Loading the seed_map.json.

# Normalizing case text.

# Matching labels (crime categories) against case text and section numbers.

# So in short: utils.py = the logic for applying your seed_map to each case.

# Instead of duplicating this code in every script, you’ll just from utils import match_seed_labels and use it wherever you want to attach legal labels to cases.