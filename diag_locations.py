import pandas as pd, json, ast, re, unicodedata

def try_parse_location_value(val):
    import ast, json
    if isinstance(val, dict): return val
    if isinstance(val, str):
        v = val.strip()
        if not v: return {}
        if v.startswith("{") and v.endswith("}"):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, dict): return parsed
            except Exception: pass
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict): return parsed
            except Exception: pass
        return {"raw": v}
    return {}

df = pd.read_excel("outputs/output_kmeans_with_seed.xlsx")
df['parsed_loc'] = df['location'].apply(lambda v: try_parse_location_value(v) if not isinstance(v, dict) else v)

missing = df[df['parsed_loc'].apply(lambda x: not bool(x.get('state') or x.get('district') or x.get('city')))]
print("=== Rows with no state/district/city (sample 30) ===")
print(missing['location'].astype(str).value_counts().head(30).to_string())

print("\n=== First 20 non-null location examples ===")
no_lat = df[df['location'].notna()].head(20)
print(no_lat['location'].astype(str).head(20).to_string())

# Also print counts of top normalized state tokens (helpful)
def norm_key(s):
    if not s: return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"\b(state|the|of|dist|district|province|ut|union territory|and|city|town)\b"," ", s)
    s = re.sub(r"[^a-z0-9\s]"," ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

states = df['parsed_loc'].apply(lambda d: d.get('state') if isinstance(d, dict) else None).dropna().map(norm_key)
print("\n=== Top normalized state tokens (sample 60) ===")
print(states.value_counts().head(60).to_string())
