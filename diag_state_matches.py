# diag_state_matches.py
import pandas as pd, json, unicodedata, re
from pathlib import Path

def norm_key(s):
    if not s: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"\b(state|the|of|dist|district|province|ut|union territory|and|city|town)\b"," ", s)
    s = re.sub(r"[^a-z0-9\s]"," ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def load_places(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

df = pd.read_excel("outputs/output_kmeans_with_seed.xlsx")
places = load_places("data/india_5_states_sample.json")

# build normalized set from places keys
place_keys = {norm_key(k): k for k in places.keys()}

# parse 'location' column similarly to your script (support stringified dicts)
import ast, json
def try_parse_location_value(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        v = val.strip()
        if not v: return {}
        if v.startswith("{") and v.endswith("}"):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, dict): return parsed
            except Exception:
                pass
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict): return parsed
            except Exception:
                pass
        return {"raw": v}
    return {}

# collect normalized tokens seen in df
state_tokens = {}
for i, loc in enumerate(df['location'].fillna('').tolist()):
    locp = try_parse_location_value(loc) if not isinstance(loc, dict) else loc
    state = locp.get('state') or locp.get('State') or ''
    n = norm_key(state)
    if n:
        state_tokens.setdefault(n, 0)
        state_tokens[n] += 1

# report
print("States found in dataframe (normalized) with counts:")
for k, cnt in sorted(state_tokens.items(), key=lambda x: -x[1])[:200]:
    print(f"{k:40}  {cnt:5}  -> places.json match: {'YES' if k in place_keys else 'NO'}")

print("\nMissing normalized states (present in df but not in places.json):")
for k, cnt in sorted(state_tokens.items(), key=lambda x: -x[1]):
    if k not in place_keys:
        print(f"{k:40}  {cnt}")
