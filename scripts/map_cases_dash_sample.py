#!/usr/bin/env python3
"""
scripts/map_cases_dash_sample.py

Interactive Dash map: plot each case individually, colored by Cluster_Label.
Coordinate matching priority:
  1) city coordinates (places[state]['districts'][district]['major_cities'][city])
  2) district coordinates (places[state]['districts'][district]['coordinates'])
  3) state centroid  (places[state]['centroid'])
If nothing matches, row is dropped.

This version robustly handles stringified location dicts in the Excel,
adds a fallback extracting state from case_title/summary, and jitters duplicate points.
"""
import argparse
import json
import re
import unicodedata
import ast
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# ---------------- Helpers ----------------
def norm_key(s):
    """Normalize a string key for fuzzy matching"""
    if not s:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    s = re.sub(r"\b(state|the|of|dist|district|province|ut|union territory|and|city|town)\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_places(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_indices_from_places(places):
    """
    Build useful lookup indexes:
      - state_centroids: { state_norm: (lat, lon) }
      - district_to_state: { district_norm: state_norm }
      - city_to_state: { city_norm: state_norm }
      - district_coords: { district_norm: (lat,lon) }   (optional)
      - city_coords: { city_norm: (lat,lon) }           (optional)
    """
    state_centroids = {}
    district_to_state = {}
    city_to_state = {}
    district_coords = {}
    city_coords = {}

    for state_name, info in places.items():
        st_norm = norm_key(state_name)
        centroid = None
        if isinstance(info, dict):
            for key in ("centroid", "center", "latlon", "center_latlon"):
                c = info.get(key)
                if isinstance(c, (list, tuple)) and len(c) == 2:
                    try:
                        lat = float(c[0]); lon = float(c[1])
                        centroid = (lat, lon)
                        break
                    except Exception:
                        centroid = None
        if centroid:
            state_centroids[st_norm] = centroid

        districts_obj = info.get("districts") if isinstance(info, dict) else None
        if isinstance(districts_obj, dict):
            for dname, dinfo in districts_obj.items():
                dnorm = norm_key(dname)
                district_to_state[dnorm] = st_norm
                if isinstance(dinfo, dict):
                    dcoords = dinfo.get("coordinates")
                    if isinstance(dcoords, (list, tuple)) and len(dcoords) == 2:
                        try:
                            district_coords[dnorm] = (float(dcoords[0]), float(dcoords[1]))
                        except Exception:
                            pass
                    mcs = dinfo.get("major_cities") or dinfo.get("cities") or {}
                    if isinstance(mcs, dict):
                        for cname, ccoords in mcs.items():
                            cnorm = norm_key(cname)
                            city_to_state[cnorm] = st_norm
                            if isinstance(ccoords, (list, tuple)) and len(ccoords) == 2:
                                try:
                                    city_coords[cnorm] = (float(ccoords[0]), float(ccoords[1]))
                                except Exception:
                                    pass
                    elif isinstance(mcs, list):
                        for cname in mcs:
                            cnorm = norm_key(cname)
                            city_to_state[cnorm] = st_norm
        else:
            if isinstance(districts_obj, list):
                for dname in districts_obj:
                    dnorm = norm_key(dname)
                    district_to_state[dnorm] = st_norm
            state_cities = info.get("major_cities") if isinstance(info, dict) else None
            if isinstance(state_cities, list):
                for cname in state_cities:
                    cnorm = norm_key(cname)
                    city_to_state[cnorm] = st_norm
            elif isinstance(state_cities, dict):
                for cname, ccoords in state_cities.items():
                    cnorm = norm_key(cname)
                    city_to_state[cnorm] = st_norm
                    if isinstance(ccoords, (list, tuple)) and len(ccoords) == 2:
                        try:
                            city_coords[cnorm] = (float(ccoords[0]), float(ccoords[1]))
                        except Exception:
                            pass

    return {
        "state_centroids": state_centroids,
        "district_to_state": district_to_state,
        "city_to_state": city_to_state,
        "district_coords": district_coords,
        "city_coords": city_coords
    }

def try_parse_location_value(val):
    """
    If val is a stringified dict (e.g. "{'state': 'Karnataka', ...}"), try to parse it into a real dict.
    Accepts Python-literal dict strings (ast.literal_eval), JSON strings, or passes through original.
    """
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        v = val.strip()
        if not v:
            return {}
        if v.startswith("{") and v.endswith("}"):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            try:
                parsed = json.loads(v)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {"raw": v}
    return {}

def pick_candidates_from_location(loc_obj):
    """
    Return list of normalized keys to try matching (ordered preference).
    Tries city coords -> district coords -> state name -> raw tokens.
    """
    candidates = []
    loc = try_parse_location_value(loc_obj)

    # district -> city -> state -> raw tokens
    for k in ("district", "District", "DISTRICT"):
        v = loc.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(norm_key(v))
    for k in ("city", "City", "town", "major_city"):
        v = loc.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(norm_key(v))
    for k in ("state", "State", "province", "region"):
        v = loc.get(k)
        if isinstance(v, str) and v.strip():
            candidates.append(norm_key(v))

    raw = loc.get("raw") or loc.get("location") or None
    if isinstance(raw, str) and raw.strip():
        candidates.append(norm_key(raw))
        for token in re.split(r"[,\n;/\|\-]+|\s{2,}", raw):
            token = token.strip()
            if token and len(token) > 2:
                candidates.append(norm_key(token))

    out = []
    seen = set()
    for c in candidates:
        if not c:
            continue
        for variant in (c, c.replace(" ", "")):
            if variant and variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out

def extract_state_from_text(text, places):
    """Try to find a state name mentioned in text by matching normalized place keys."""
    if not text or not isinstance(text, str):
        return None
    txt = text.lower()
    # quick direct match
    for st in places.keys():
        if st.lower() in txt:
            return st
    # normalized match
    ntxt = norm_key(text)
    for st in places.keys():
        if norm_key(st) in ntxt:
            return st
    return None

def jitter_duplicate_points(coords_df, max_radius_deg=0.05):
    """
    coords_df: DataFrame with 'lat','lon' columns (float), in original order.
    For groups with identical coords, apply a tiny circular jitter so stacked points become visible.
    Returns arrays (jlats, jlons) aligned with coords_df.index order.
    """
    lats = coords_df['lat'].to_numpy(copy=True)
    lons = coords_df['lon'].to_numpy(copy=True)
    pairs = list(zip(lats, lons))
    groups = defaultdict(list)
    for i, p in enumerate(pairs):
        if np.isnan(p[0]) or np.isnan(p[1]):
            continue
        # rounding for grouping to avoid tiny fp differences
        key = (round(float(p[0]), 6), round(float(p[1]), 6))
        groups[key].append(i)

    for key, idxs in groups.items():
        n = len(idxs)
        if n <= 1:
            continue
        # scale radius inversely with sqrt(n) so many points cluster modestly
        radius = max_radius_deg * (1.0 / max(1.0, np.sqrt(n) / 2.0))
        for k, i in enumerate(idxs):
            angle = (2 * np.pi * k) / n
            lats[i] += radius * np.sin(angle)
            lons[i] += radius * np.cos(angle)
    return lats, lons

def assign_coords_for_df(df, places, max_jitter_deg=0.05):
    """
    Assign lat/lon for each row using places indices. Returns new df with lat/lon columns.
    Also applies small jitter to duplicate coordinates so stacked points are visible.
    """
    idx = build_indices_from_places(places)
    state_centroids = idx["state_centroids"]
    district_to_state = idx["district_to_state"]
    city_to_state = idx["city_to_state"]
    district_coords = idx["district_coords"]
    city_coords = idx["city_coords"]

    lats, lons = [], []
    total = len(df)
    matched = 0

    for _, row in df.iterrows():
        loc_raw = row.get("location")
        candidates = pick_candidates_from_location(loc_raw)

        lat = lon = np.nan
        found = False

        # 1) try city coords
        for cand in candidates:
            if cand in city_coords:
                lat, lon = city_coords[cand]
                found = True
                break
        # 2) district coords
        if not found:
            for cand in candidates:
                if cand in district_coords:
                    lat, lon = district_coords[cand]
                    found = True
                    break
        # 3) city -> state -> centroid
        if not found:
            for cand in candidates:
                if cand in city_to_state:
                    st = city_to_state[cand]
                    if st in state_centroids:
                        lat, lon = state_centroids[st]
                        found = True
                        break
        # 4) district -> state -> centroid
        if not found:
            for cand in candidates:
                if cand in district_to_state:
                    st = district_to_state[cand]
                    if st in state_centroids:
                        lat, lon = state_centroids[st]
                        found = True
                        break
        # 5) direct state match
        if not found:
            for cand in candidates:
                if cand in state_centroids:
                    lat, lon = state_centroids[cand]
                    found = True
                    break
        # 6) try raw state field in parsed loc dict
        if not found and isinstance(loc_raw, dict):
            state_raw = loc_raw.get("state") or loc_raw.get("State") or None
            if isinstance(state_raw, str):
                stn = norm_key(state_raw)
                if stn in state_centroids:
                    lat, lon = state_centroids[stn]
                    found = True

        # 7) fallback: try to extract state from case_title/summary text
        if not found:
            text_search = ""
            if isinstance(row.get("case_title"), str):
                text_search += " " + row.get("case_title")
            if isinstance(row.get("summary"), str):
                text_search += " " + row.get("summary")
            st_from_text = extract_state_from_text(text_search, places)
            if st_from_text:
                stn = norm_key(st_from_text)
                if stn in state_centroids:
                    lat, lon = state_centroids[stn]
                    found = True

        if found and not (np.isnan(lat) or np.isnan(lon)):
            matched += 1
        lats.append(lat); lons.append(lon)

    df2 = df.copy()
    df2["lat"] = lats
    df2["lon"] = lons

    print(f"Total rows: {total}, with assigned coords (before jitter): {matched}, dropped: {total - matched}")

    # jitter duplicates so stacked points are visible
    coords_mask = (~df2["lat"].isna()) & (~df2["lon"].isna())
    coords_df = df2.loc[coords_mask, ["lat", "lon"]].reset_index(drop=True)
    if not coords_df.empty:
        jlats, jlons = jitter_duplicate_points(coords_df, max_radius_deg=max_jitter_deg)
        # write jittered back into df2 in same order as coords_df rows
        idxs = df2.index[coords_mask].tolist()
        for i, orig_idx in enumerate(idxs):
            df2.at[orig_idx, "lat"] = float(jlats[i])
            df2.at[orig_idx, "lon"] = float(jlons[i])

    with_coords = df2[~df2["lat"].isna() & ~df2["lon"].isna()].shape[0]
    print(f"After jitter: assigned coords: {with_coords}, dropped: {total - with_coords}")
    if with_coords == 0:
        print("⚠️ No coordinates assigned. Check that your places JSON includes state centroids or matching keys.")
    return df2

def build_hover(row):
    """
    Build concise HTML hover text for a case with:
      - Title (bold)
      - State | District | City
      - Date (judgment_date if available)
      - Petitioner / Respondent
      - Cluster_Label (visualization label)
      - Seed crime labels (crime_labels)
      - IPC sections
    Summary is intentionally omitted.
    """

    # parse/normalize location (works if location is dict or string)
    loc_raw = row.get("location") or {}
    try:
        if callable(try_parse_location_value):
            loc = try_parse_location_value(loc_raw) if not isinstance(loc_raw, dict) else loc_raw
        else:
            loc = loc_raw if isinstance(loc_raw, dict) else {}
    except Exception:
        loc = loc_raw if isinstance(loc_raw, dict) else {}

    # Safe extraction of location parts
    state = (loc.get("state") or loc.get("State") or "") if isinstance(loc, dict) else ""
    district = (loc.get("district") or loc.get("District") or "") if isinstance(loc, dict) else ""
    city = (loc.get("city") or loc.get("City") or "") if isinstance(loc, dict) else ""

    # Date extraction (supports row['dates'] or flat columns)
    date_val = ""
    dates_obj = row.get("dates") or {}
    if isinstance(dates_obj, dict):
        date_val = dates_obj.get("judgment_date") or dates_obj.get("date") or ""
    else:
        date_val = row.get("judgment_date") or row.get("date") or ""

    # Parties and title
    title = row.get("case_title") or ""
    petitioner = row.get("petitioner") or ""
    respondent = row.get("respondent") or ""

    # Visualization cluster label (KMeans label)
    cluster_label = row.get("Cluster_Label") or row.get("cluster_label") or ""

    # Seed-based crime labels (list) — join into readable string
    seed_labels = row.get("crime_labels") or []
    if isinstance(seed_labels, list):
        seed_labels_str = ", ".join([str(x) for x in seed_labels if x])
    else:
        seed_labels_str = str(seed_labels) if seed_labels else ""

    # IPC sections
    ipc = row.get("ipc_sections") or []
    if isinstance(ipc, list):
        ipc_str = ", ".join([str(x) for x in ipc if x])
    else:
        ipc_str = str(ipc) if ipc else ""

    # Build parts (omit empty)
    parts = []

    if title:
        parts.append(f"<b>{title}</b>")

    # Single-line Location (City, District, State) if present
    loc_line = ", ".join([p for p in (city, district, state) if p])
    if loc_line:
        parts.append(f"<b>Location:</b> {loc_line}")

    if date_val:
        parts.append(f"<b>Date:</b> {date_val}")

    if petitioner:
        parts.append(f"<b>Petitioner:</b> {petitioner}")
    if respondent:
        parts.append(f"<b>Respondent:</b> {respondent}")

    if cluster_label:
        parts.append(f"<b>Cluster:</b> {cluster_label}")

    if seed_labels_str:
        parts.append(f"<b>Crime label(s):</b> {seed_labels_str}")

    if ipc_str:
        parts.append(f"<b>IPC:</b> {ipc_str}")

    # join with HTML line breaks and return
    hover_html = "<br>".join(parts)
    return hover_html



    # join with HTML line breaks
    hover_html = "<br>".join(parts)
    return hover_html


# deterministic color assignment for labels
def label_color_map(labels):
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3 + px.colors.qualitative.Bold
    palette_len = len(palette)
    mapping = {}
    for i, lab in enumerate(sorted(labels)):
        mapping[lab] = palette[i % palette_len]
    return mapping

# ---------------- App builder ----------------
def build_app(df, places_path, states_geojson_path=None):
    places = load_places(places_path)
    df2 = df.copy()

    # ensure Cluster_Label column
    if "Cluster_Label" not in df2.columns and "cluster_label" in df2.columns:
        df2.rename(columns={"cluster_label": "Cluster_Label"}, inplace=True)
    if "Cluster_Label" not in df2.columns:
        df2["Cluster_Label"] = "unlabeled"

    # parse list-like columns if needed
    for col in ("ipc_sections", "acts"):
        if col in df2.columns:
            df2[col] = df2[col].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

    # attempt to convert any stringified 'location' to dict using try_parse_location_value
    if "location" in df2.columns:
        df2["location"] = df2["location"].apply(lambda v: try_parse_location_value(v) if not isinstance(v, dict) else v)
    else:
        df2["location"] = [{} for _ in range(len(df2))]

    # assign coords (with fallback and jitter)
    df2 = assign_coords_for_df(df2, places, max_jitter_deg=0.05)

    # drop rows without coords
    df2 = df2.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # hover text + id
    df2["hover_text"] = df2.apply(build_hover, axis=1)
    df2["id_row"] = df2.index.astype(str)

    # unique labels and colors
    unique_labels = sorted(df2["Cluster_Label"].fillna("unlabeled").unique())
    if not unique_labels:
        unique_labels = ["unlabeled"]
    colors = label_color_map(unique_labels)

    # load states geojson for clear boundaries (optional)
    india_geo = None
    chosen_prop = None
    if states_geojson_path and Path(states_geojson_path).exists():
        try:
            with open(states_geojson_path, "r", encoding="utf-8") as f:
                india_geo = json.load(f)
            sample_props = india_geo.get("features", [{}])[0].get("properties", {})
            candidates = ["st_nm", "ST_NM", "STATE_NAME", "NAME_1", "NAME", "name", "state"]
            chosen_prop = next((c for c in candidates if c in sample_props), None)
        except Exception as e:
            print("⚠️ Failed to load geojson:", e)
            india_geo = None
            chosen_prop = None

    # build base figure with choropleth borders (visible) + scatter points grouped by label
    fig = go.Figure()
    if india_geo and chosen_prop:
        locations = [feat["properties"].get(chosen_prop) for feat in india_geo["features"]]
        z = [0] * len(locations)
        fig.add_trace(
            go.Choropleth(
                geojson=india_geo,
                locations=locations,
                z=z,
                featureidkey=f"properties.{chosen_prop}",
                showscale=False,
                marker_line_color="black",
                marker_line_width=0.6,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                hoverinfo="none",
                name="states"
            )
        )

    # Add points grouped by label (legend) — each group's points plotted with label color
    for label, group in df2.groupby("Cluster_Label"):
        fig.add_trace(
            go.Scattergeo(
                lon=group["lon"],
                lat=group["lat"],
                text=group["hover_text"],
                mode="markers",
                name=str(label),
                marker=dict(size=8, opacity=0.85, color=colors.get(label, "#636EFA")),
                hovertemplate="%{text}<extra></extra>"
            )
        )

    fig.update_geos(
        projection_type="mercator",
        center={"lat": 22.5937, "lon": 78.9629},
        lataxis_range=[6, 37],
        lonaxis_range=[68, 98],
        showcountries=True,
        countrycolor="black",
        showland=True,
        landcolor="rgb(245,245,245)",
    )
    fig.update_layout(
        title="Indian Legal Cases — Points colored by Cluster_Label",
        height=720,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend=dict(x=0.92, y=0.95)
    )

    # Dash app
    app = Dash(__name__)
    server = app.server

    cluster_options = [{"label": c, "value": c} for c in unique_labels]

    app.layout = html.Div([
        html.H3("🗺️ Indian Legal Cases — Points colored by Cluster_Label"),
        html.Div([
            html.Label("Filter by Cluster_Label:"),
            dcc.Dropdown(id="label-filter", options=cluster_options, multi=True, placeholder="Select labels"),
            html.Br(),
            html.Label("Date substring filter (summary or case_title):"),
            dcc.Input(id="date-from", type="text", placeholder="From e.g. 2020"),
            dcc.Input(id="date-to", type="text", placeholder="To e.g. 2024"),
            html.Button("Reset", id="reset-btn"),
            html.Div(id="stats", style={"marginTop": "8px", "fontSize": "13px"})
        ], style={"width": "25%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),
        html.Div([dcc.Graph(id="map-graph", figure=fig, config={"displayModeBar": True})], style={"width": "72%", "display": "inline-block"}),
        html.Div(id="clicked-detail", style={"padding": "15px", "borderTop": "1px solid #ccc", "background": "#f8f8f8"})
    ])

    @app.callback(
        Output("map-graph", "figure"),
        Output("stats", "children"),
        Input("label-filter", "value"),
        Input("date-from", "value"),
        Input("date-to", "value"),
        Input("reset-btn", "n_clicks"),
    )
    def update_map(labels, date_from, date_to, reset_clicks):
        dff = df2.copy()
        if labels:
            dff = dff[dff["Cluster_Label"].isin(labels)]
        if date_from:
            s = str(date_from).strip()
            if s:
                dff = dff[dff["summary"].str.contains(s, case=False, na=False) | dff["case_title"].str.contains(s, case=False, na=False)]
        if date_to:
            s2 = str(date_to).strip()
            if s2:
                dff = dff[dff["summary"].str.contains(s2, case=False, na=False) | dff["case_title"].str.contains(s2, case=False, na=False)]

        fig2 = go.Figure()
        if india_geo and chosen_prop:
            locations = [feat["properties"].get(chosen_prop) for feat in india_geo["features"]]
            z = [0] * len(locations)
            fig2.add_trace(
                go.Choropleth(geojson=india_geo, locations=locations, z=z,
                              featureidkey=f"properties.{chosen_prop}", showscale=False,
                              marker_line_color="black", marker_line_width=0.6,
                              colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                              hoverinfo="none", name="states")
            )

        for label, group in dff.groupby("Cluster_Label"):
            fig2.add_trace(
                go.Scattergeo(
                    lon=group["lon"],
                    lat=group["lat"],
                    text=group["hover_text"],
                    mode="markers",
                    name=str(label),
                    marker=dict(size=8, opacity=0.85, color=colors.get(label, "#636EFA")),
                    hovertemplate="%{text}<extra></extra>"
                )
            )

        fig2.update_geos(projection_type="mercator", center={"lat": 22.5937, "lon": 78.9629},
                         lataxis_range=[6, 37], lonaxis_range=[68, 98], showcountries=True, countrycolor="black")
        fig2.update_layout(title="Indian Legal Cases — Points colored by Cluster_Label", height=720, margin={"r": 0, "t": 40, "l": 0, "b": 0}, legend=dict(x=0.92, y=0.95))

        return fig2, f"Showing {len(dff)} cases | Labels visible: {len(dff['Cluster_Label'].unique())}"

    @app.callback(
        Output("clicked-detail", "children"),
        Input("map-graph", "clickData")
    )
    def show_details(clickData):
        if not clickData:
            return html.Div("Click a point for case details.")
        pt = clickData["points"][0]
        lat = pt.get("lat"); lon = pt.get("lon")
        candidates = df2[(np.isclose(df2["lat"], lat)) & (np.isclose(df2["lon"], lon))]
        if candidates.empty:
            return html.Pre(json.dumps(clickData, indent=2))
        row = candidates.iloc[0]
        return html.Div([
            html.H4(row.get("case_title") or "Case"),
            html.P(f"Petitioner: {row.get('petitioner', '-')}", style={"marginBottom": "5px"}),
            html.P(f"Respondent: {row.get('respondent', '-')}", style={"marginBottom": "5px"}),
            html.P(f"Cluster Label: {row.get('Cluster_Label', '-')}", style={"marginBottom": "5px"}),
            html.P(f"IPC Sections: {row.get('ipc_sections', [])}", style={"marginBottom": "5px"}),
            html.P(f"Acts: {row.get('acts', [])}", style={"marginBottom": "5px"}),
            html.H5("Summary:"), html.Div(row.get('summary', "No summary available"), style={"whiteSpace": "pre-wrap"})
        ], style={"maxWidth": "800px"})

    return app, server

# ----------------- CLI / Main -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Excel with cluster results")
    parser.add_argument("--places", required=True, help="places JSON (e.g. india_5_states_sample.json)")
    parser.add_argument("--states_geojson", default=None, help="optional India states geojson to draw boundaries")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    app, server = build_app(df, args.places, args.states_geojson)
    app.run(port=args.port, debug=True)

if __name__ == "__main__":
    main()
