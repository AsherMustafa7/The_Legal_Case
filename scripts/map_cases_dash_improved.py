# scripts/map_cases_dash_improved.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import html as html_lib

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ----------------- Helpers -----------------
def safe_parse_list(x):
    """Try to coerce column values that may be stringified lists into lists."""
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        try:
            return json.loads(x)
        except Exception:
            # fallback: split common separators
            if x.startswith("[") and x.endswith("]"):
                inner = x[1:-1]
                parts = [p.strip().strip("'\"") for p in inner.split(",") if p.strip()]
                return parts
            return [x]
    return [x]

def load_places(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def ensure_centroid_for_state(places):
    """Return map of normalized state name -> (lat, lon) using 'centroid' field if present."""
    centroids = {}
    for st, info in places.items():
        # normalize state name
        key = st.strip().lower()
        # check possible centroid fields
        c = info.get("centroid") or info.get("center") or info.get("latlon") or None
        if c and isinstance(c, (list, tuple)) and len(c) == 2:
            # unify to (lat, lon)
            try:
                lat = float(c[0])
                lon = float(c[1])
                centroids[key] = (lat, lon)
            except Exception:
                pass
    return centroids

def pick_state_from_location(loc_obj):
    """Given a location object (may be dict or string), return normalized state name or None."""
    if isinstance(loc_obj, dict):
        state = loc_obj.get("state") or loc_obj.get("State") or loc_obj.get("province")
        if isinstance(state, str) and state.strip():
            return state.strip().lower()
        # sometimes 'raw' contains state or city; attempt fallback
        raw = loc_obj.get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw.strip().lower()
        return None
    elif isinstance(loc_obj, str):
        return loc_obj.strip().lower()
    return None

def tidy_text(s, max_len=None):
    if s is None:
        return ""
    s = str(s).strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")
    if max_len and len(s) > max_len:
        return s[:max_len].rstrip() + "..."
    return s

def build_hover_text(row):
    """Return single HTML string for hover text (Plotly will render it)."""
    parts = []
    title = row.get("case_title") or row.get("case_title", "")
    if title:
        parts.append(f"<b>{html_lib.escape(tidy_text(title, 200))}</b>")
    pet = row.get("petitioner") or row.get("petitioner", "")
    if pet:
        parts.append(f"P: {html_lib.escape(tidy_text(pet, 120))}")
    resp = row.get("respondent") or row.get("respondent", "")
    if resp:
        parts.append(f"R: {html_lib.escape(tidy_text(resp, 120))}")
    label = row.get("Cluster_Label") or row.get("cluster_label") or ""
    if label:
        parts.append(f"Label: {html_lib.escape(tidy_text(label))}")
    # show a short summary (first 180 chars)
    summary = row.get("summary") or ""
    if summary:
        parts.append(html_lib.escape(tidy_text(summary, 180)))
    return "<br>".join(parts) if parts else html_lib.escape(tidy_text(title or pet or resp or label))

# ----------------- Build Dash App -----------------
def build_app(df, places_json_path, states_geojson_path=None):
    # normalize dataframe copy
    df = df.copy()

    # ensure expected columns exist and parse list-like columns
    for col in ("ipc_sections", "acts"):
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)
        else:
            df[col] = [[] for _ in range(len(df))]

    # prefer Cluster_Label normalization
    if "Cluster_Label" not in df.columns and "cluster_label" in df.columns:
        df.rename(columns={"cluster_label": "Cluster_Label"}, inplace=True)
    if "Cluster_Label" not in df.columns:
        df["Cluster_Label"] = "unlabeled"

    # load places and centroids
    places = load_places(places_json_path)
    centroids = ensure_centroid_for_state(places)

    # attach lat/lon using centroid lookup (case-insensitive)
    lats, lons = [], []
    for _, row in df.iterrows():
        loc = row.get("location") or {}
        st_norm = pick_state_from_location(loc)
        lat = lon = np.nan
        if st_norm:
            # try exact match
            if st_norm in centroids:
                lat, lon = centroids[st_norm]
            else:
                # try fuzzy simple normalization: remove "state of", punctuation
                candidate = st_norm.replace("state of", "").replace(".", "").strip()
                if candidate in centroids:
                    lat, lon = centroids[candidate]
                else:
                    # try title-casing lookup in places keys
                    for k in centroids.keys():
                        if candidate == k or candidate in k or k in candidate:
                            lat, lon = centroids[k]
                            break
        lats.append(lat)
        lons.append(lon)

    df["lat"] = lats
    df["lon"] = lons

    # drop rows without coords
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    # build hover_text column (single string per row)
    df["hover_text"] = df.apply(build_hover_text, axis=1)
    df["id_row"] = df.index.astype(str)  # stable id used for click matching

    # dropdown options for cluster labels
    unique_labels = sorted(df["Cluster_Label"].dropna().unique())
    cluster_options = [{"label": label, "value": label} for label in unique_labels]

    # ---- Create base map figure ----
    fig = go.Figure()

    # Add India state boundaries (GeoJSON)
    chosen_prop = None
    india_geo = None
    if states_geojson_path and Path(states_geojson_path).exists():
        try:
            with open(states_geojson_path, "r", encoding="utf-8") as f:
                india_geo = json.load(f)
            sample_props = india_geo.get("features", [{}])[0].get("properties", {})
            candidate_names = ["st_nm", "ST_NM", "STATE_NAME", "NAME_1", "name", "NAME", "state"]
            chosen_prop = next((c for c in candidate_names if c in sample_props), None)
            if chosen_prop:
                fig.add_trace(
                    go.Choropleth(
                        geojson=india_geo,
                        locations=[f["properties"][chosen_prop] for f in india_geo["features"]],
                        z=[0] * len(india_geo["features"]),
                        featureidkey=f"properties.{chosen_prop}",
                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                        showscale=False,
                        marker_line_color="black",
                        marker_line_width=0.6,
                        hoverinfo="none"
                    )
                )
        except Exception as e:
            print("⚠️ Could not load states_geojson:", e)
            india_geo = None
            chosen_prop = None

    # Add your case points (one trace per label)
    for label, group in df.groupby("Cluster_Label"):
        fig.add_trace(
            go.Scattergeo(
                lon=group["lon"],
                lat=group["lat"],
                text=group["hover_text"],
                mode="markers",
                name=label,
                marker=dict(size=8, opacity=0.85),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Center & zoom over India
    fig.update_geos(
        scope="asia",
        projection_type="mercator",
        center={"lat": 22.5937, "lon": 78.9629},  # India's center
        lataxis_range=[6, 37],
        lonaxis_range=[68, 98],
        showcountries=True,
        countrycolor="black",
        showsubunits=True,
        subunitcolor="gray",
        showland=True,
        landcolor="rgb(240,240,240)",
        visible=True,   # ensure it's visible initially
    )

    fig.update_layout(
        title="🗺️ Indian Legal Cases — Interactive Map",
        height=700,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        legend=dict(x=1.05, y=1, bordercolor="Black", borderwidth=0.5),
    )

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        legend={"orientation": "v", "x": 0.88, "y": 0.95},
    )

    # Build Dash app and layout
    app = Dash(__name__)
    server = app.server

    app.layout = html.Div([
        html.H3("🗺️ Indian Legal Cases — Interactive Map"),
        html.Div([
            html.Label("Filter by Crime Label:"),
            dcc.Dropdown(
                id="label-filter",
                options=cluster_options,
                multi=True,
                placeholder="Select one or more labels"
            ),
            html.Br(),
            html.Label("Date Filter (substring match on summary or case_title):"),
            dcc.Input(id="date-from", type="text", placeholder="From (e.g. 2020)"),
            dcc.Input(id="date-to", type="text", placeholder="To (e.g. 2024)"),
            html.Br(),
            html.Button("Reset filters", id="reset-btn"),
        ], style={"width": "25%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

        html.Div([
            dcc.Graph(id="map-graph", figure=fig, config={"displayModeBar": True}),
        ], style={"width": "70%", "display": "inline-block"}),

        html.Div(id="clicked-detail", style={"padding": "15px", "borderTop": "1px solid #ccc", "background": "#f8f8f8"})
    ])

    # ---------------- Callbacks ----------------
    @app.callback(
        Output("map-graph", "figure"),
        Input("label-filter", "value"),
        Input("date-from", "value"),
        Input("date-to", "value"),
        Input("reset-btn", "n_clicks"),
    )
    def update_map(labels, date_from, date_to, reset_clicks):
        dff = df.copy()
        # label filter
        if labels:
            dff = dff[dff["Cluster_Label"].isin(labels)]
        # date substring filters (search in summary and case_title)
        if date_from:
            s = str(date_from).strip()
            dff = dff[dff["summary"].str.contains(s, case=False, na=False) | dff["case_title"].str.contains(s, case=False, na=False)]
        if date_to:
            s2 = str(date_to).strip()
            dff = dff[dff["summary"].str.contains(s2, case=False, na=False) | dff["case_title"].str.contains(s2, case=False, na=False)]

        # Build updated figure (preserve same geo centre & ranges so map stays visible)
        fig2 = go.Figure()

        # re-add choropleth borders if we had them
        if india_geo and chosen_prop:
            try:
                fig2.add_trace(
                    go.Choropleth(
                        geojson=india_geo,
                        locations=[f["properties"][chosen_prop] for f in india_geo["features"]],
                        z=[0] * len(india_geo["features"]),
                        featureidkey=f"properties.{chosen_prop}",
                        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                        showscale=False,
                        marker_line_color="black",
                        marker_line_width=0.6,
                        hoverinfo="none"
                    )
                )
            except Exception:
                pass

        # add points grouped by label
        for label, group in dff.groupby("Cluster_Label"):
            fig2.add_trace(
                go.Scattergeo(
                    lon=group["lon"],
                    lat=group["lat"],
                    text=group["hover_text"],
                    mode="markers",
                    name=label,
                    marker=dict(size=8, opacity=0.85),
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        # Important: keep geo visible and locked over India (don't set visible=False)
        fig2.update_geos(
            scope="asia",
            projection_type="mercator",
            center={"lat": 22.5937, "lon": 78.9629},
            lataxis_range=[6, 37],
            lonaxis_range=[68, 98],
            showcountries=True,
            countrycolor="black",
            showland=True,
            landcolor="rgb(240,240,240)",
            visible=True,
        )

        fig2.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, legend={"orientation": "v", "x": 0.88, "y": 0.95}, height=700)
        return fig2

    @app.callback(
        Output("clicked-detail", "children"),
        Input("map-graph", "clickData")
    )
    def show_details(clickData):
        if not clickData:
            return html.Div("Click on a point to view full case details here.")
        pt = clickData["points"][0]
        lat = pt.get("lat")
        lon = pt.get("lon")
        # find candidates using isclose because of float representation
        candidates = df[(np.isclose(df["lat"], lat)) & (np.isclose(df["lon"], lon))]
        if candidates.empty:
            # fallback: return raw clickData pretty-printed
            return html.Pre(json.dumps(clickData, indent=2))
        row = candidates.iloc[0]
        return html.Div([
            html.H4(tidy_text(row.get("case_title") or "Case")),
            html.P(f"Petitioner: {tidy_text(row.get('petitioner', '-'))}", style={"marginBottom": "5px"}),
            html.P(f"Respondent: {tidy_text(row.get('respondent', '-'))}", style={"marginBottom": "5px"}),
            html.P(f"Cluster Label: {tidy_text(row.get('Cluster_Label', '-'))}", style={"marginBottom": "5px"}),
            html.P(f"IPC Sections: {row.get('ipc_sections', [])}", style={"marginBottom": "5px"}),
            html.P(f"Acts: {row.get('acts', [])}", style={"marginBottom": "5px"}),
            html.H5("Summary:"),
            html.Div(row.get("summary", "No summary available"), style={"whiteSpace": "pre-wrap"})
        ], style={"maxWidth": "700px"})

    return app, server

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Excel file with seed-labeled clusters")
    parser.add_argument("--places", required=True, help="indian_places.json (with centroids)")
    parser.add_argument("--states_geojson", default=None, help="India states geojson (optional)")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    df = pd.read_excel(args.input)
    # handle alternate column name
    if "Cluster_Label" not in df.columns and "cluster_label" in df.columns:
        df.rename(columns={"cluster_label": "Cluster_Label"}, inplace=True)

    app, server = build_app(df, args.places, args.states_geojson)
    # Dash v2/3: use app.run
    app.run(port=args.port, debug=True)

if __name__ == "__main__":
    main()
