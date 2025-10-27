# scripts/court_state_helpers.py
import json, re, unicodedata
from pathlib import Path

def _norm(s):
    if s is None: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_court_map(path="data/court_to_state.json"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found. Create a court->state JSON.")
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # create normalized lookup keys for fuzzy matching
    norm_map = {}
    for k, states in raw.items():
        norm_map[_norm(k)] = states
    return norm_map

def map_court_to_states(court_name, norm_map):
    """
    Returns list of matching states (strings) or None.
    Performs normalized substring/fuzzy matching on court_name.
    """
    if not court_name:
        return None
    ck = _norm(court_name)
    # direct exact normalized match
    if ck in norm_map:
        return norm_map[ck]
    # try substring match: which map key is substring of court text
    for key in norm_map:
        if key in ck or ck in key:
            return norm_map[key]
    # token-wise match: keys words appear in court text
    for key in norm_map:
        key_tokens = set(key.split())
        if key_tokens & set(ck.split()):
            return norm_map[key]
    return None
