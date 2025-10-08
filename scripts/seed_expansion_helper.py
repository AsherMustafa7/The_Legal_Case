# scripts/seed_expansion_helper.py
import argparse
import json
import pandas as pd
import ast
import os
import re
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

# Optional KeyBERT import
try:
    from keybert import KeyBERT
except ImportError:
    KeyBERT = None

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
LEGAL_STOPWORDS = {
    "state", "applicant", "respondent", "counsel", "learned", "court", "case",
    "petition", "petitioner", "order", "application", "party", "vs", "vs state",
    "supreme", "high", "scc", "dated", "2024", "2025", "advocate", "section",
    "judge", "justice", "law", "legal", "said", "stated", "india", "manoj",
    "kumar", "respondent no", "state of", "appellant", "defendant", "plaintiff",
    "submitted", "hon’ble", "lordship", "learned counsel", "opposite party",
    "opposite", "bench", "neutral", "citation", "record", "heard", "impugned"
}

GENERIC_ACTS = {
    "indian penal code|ipc".lower(),
    "code of criminal procedure|crpc".lower(),
    "constitution of india|constitution".lower()
}

nlp = spacy.load("en_core_web_sm")

# ----------------------------------------------
# HELPERS
# ----------------------------------------------
def safe_parse_list(x):
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return []
    x = x.strip()
    if not x or x.lower() in ("nan", "none"):
        return []
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

def remove_named_entities(text):
    doc = nlp(text)
    cleaned = text
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"}:
            cleaned = cleaned.replace(ent.text, "")
    return cleaned

def extract_meaningful_tokens(text):
    doc = nlp(text)
    tokens = []
    for token in doc:
        if (
            token.is_alpha
            and not token.is_stop
            and token.pos_ in {"NOUN", "PROPN", "ADJ"}
            and token.lemma_.lower() not in LEGAL_STOPWORDS
            and len(token.text) > 3
        ):
            tokens.append(token.lemma_.lower())
    return " ".join(tokens)

def top_terms_tfidf(texts, top_n=15):
    vect = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), stop_words="english")
    X = vect.fit_transform(texts)
    sums = X.sum(axis=0).A1
    features = vect.get_feature_names_out()
    idxs = sums.argsort()[::-1][:top_n]
    return [features[i] for i in idxs]

def top_terms_keybert(texts, top_n=15):
    if KeyBERT is None:
        raise ImportError("⚠️ KeyBERT not installed. Run: pip install keybert sentence-transformers")
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    joined = " ".join(texts)
    keywords = kw_model.extract_keywords(
        joined,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.7,
        top_n=top_n
    )
    return [k for k, _ in keywords]

def normalize_act(a):
    return re.sub(r"[\\\s]+", " ", a.lower()).strip()

def flatten_seed(seed_map):
    known = set()
    for _, entry in seed_map.items():
        for t in entry.get("triggers", []):
            known.add(t.lower())
        for s in entry.get("ipc_sections", []):
            known.add(str(s).lower())
        for a in entry.get("acts", []):
            known.add(a.lower())
    return known

# ----------------------------------------------
# MAIN
# ----------------------------------------------
def main(args):
    df = pd.read_excel(args.input)
    for col in ["ipc_sections", "acts", "crime_labels"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # dynamic stopwords from case parties
    for col in ["petitioner", "respondent", "case_title"]:
        if col in df.columns:
            for val in df[col].dropna().unique():
                if isinstance(val, str):
                    LEGAL_STOPWORDS.add(val.lower())

    seed_path = Path(args.seed)
    if seed_path.exists():
        with open(seed_path, "r", encoding="utf-8") as f:
            seed_map = json.load(f)
    else:
        print("⚠️ No seed_map.json found; starting fresh.")
        seed_map = {}

    known_terms = flatten_seed(seed_map)
    unlabeled_df = df[df["Cluster_Label"].isin(["unlabeled", "miscellaneous"])]

    suggestions = {}
    print("\n🧩 Analyzing unlabeled clusters...")

    for cid in tqdm(unlabeled_df["Cluster_ID"].unique(), desc="Processing Clusters"):
        subset = unlabeled_df[unlabeled_df["Cluster_ID"] == cid]
        texts = subset["summary"].dropna().tolist()
        if not texts:
            continue

        cleaned_texts = [extract_meaningful_tokens(remove_named_entities(t)) for t in texts]

        # --- Choose extraction method ---
        if args.method == "keybert":
            try:
                keywords = top_terms_keybert(cleaned_texts)
            except Exception as e:
                print(f"⚠️ KeyBERT failed for cluster {cid}: {e}. Falling back to TF-IDF.")
                keywords = top_terms_tfidf(cleaned_texts)
        else:
            keywords = top_terms_tfidf(cleaned_texts)

        filtered = [kw for kw in keywords if kw.lower() not in LEGAL_STOPWORDS and len(kw) > 3]

        all_ipc = [sec for lst in subset["ipc_sections"] for sec in lst]
        all_acts = [act for lst in subset["acts"] for act in lst]
        ipc_common = [i for i, _ in Counter(all_ipc).most_common(5)]
        act_common = [i for i, _ in Counter(all_acts).most_common(5)]

        new_keywords = [kw for kw in filtered if kw.lower() not in known_terms]
        existing = [kw for kw in filtered if kw.lower() in known_terms]

        # Detect procedural clusters
        acts_clean = [normalize_act(a) for a in all_acts]
        acts_generic = all(a in GENERIC_ACTS for a in acts_clean if a)
        if acts_generic and len(new_keywords) < 3:
            print(f"\n⚠️ Cluster {cid} looks procedural — skip? (y/n): ", end="")
            resp = input().strip().lower()
            if resp == "y":
                continue

        suggestions[f"cluster_{cid}"] = {
            "suggested_keywords": filtered,
            "new_keywords": new_keywords,
            "already_in_seed": existing,
            "common_ipc_sections": ipc_common,
            "common_acts": act_common,
            "comment": f"Auto-generated for cluster {cid}"
        }

    # --- Interactive Review ---
    print("\n🧠 Interactive review: Approve or skip new categories.")
    approved = {}
    for cid, data in suggestions.items():
        if not data["new_keywords"]:
            continue
        if data["common_ipc_sections"]:
            cat_name = f"ipc_{data['common_ipc_sections'][0]}_related"
        elif data["common_acts"]:
            act_clean = data["common_acts"][0].split("|")[0].replace(" ", "_").lower()
            cat_name = f"{act_clean}_cases"
        else:
            cat_name = f"cluster_{cid}"

        print("\n----------------------------------------")
        print(f"Cluster {cid} → Proposed Category: {cat_name}")
        print("Common IPC:", data["common_ipc_sections"])
        print("Common Acts:", data["common_acts"])
        print("New Keywords:", ", ".join(data["new_keywords"][:10]))
        resp = input("Add this category to seed_map.json? (y/n): ").strip().lower()
        if resp == "y":
            approved[cat_name] = {
                "triggers": data["new_keywords"][:10],
                "ipc_sections": data["common_ipc_sections"],
                "acts": data["common_acts"]
            }

    if approved:
        backup_path = seed_path.with_suffix(".backup.json")
        if seed_path.exists():
            shutil.copy(seed_path, backup_path)
            print(f"\n🗄️ Backup created: {backup_path}")
        seed_map.update(approved)
        with open(seed_path, "w", encoding="utf-8") as f:
            json.dump(seed_map, f, indent=2, ensure_ascii=False)
        print(f"✅ Added {len(approved)} new categories to {seed_path}")
    else:
        print("\n⚠️ No new categories approved. Seed map unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Excel file with seed-labeled clusters")
    parser.add_argument("--seed", default="data/seed_map.json", help="Path to seed map JSON")
    parser.add_argument("--method", default="tfidf", choices=["tfidf", "keybert"], help="Keyword extraction method")
    args = parser.parse_args()
    main(args)
