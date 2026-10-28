"""
Route Optimization page.
Takes the victims collected on the Triage page (with lat/lon) and a rescue
team starting point, then computes a priority-weighted rescue route using
priority_routing.py. Visualized on a free OpenStreetMap/Folium map —
no Google Maps API key or billing account required.
"""

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import sys
import os

DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(DASHBOARD_DIR)
sys.path.append(DASHBOARD_DIR)  # for utils.py
sys.path.append(REPO_ROOT)      # for module2_triage/...
from utils import apply_theme, rescue_banner
from module2_triage.route_optimization.priority_routing import optimize_route, URGENCY_WEIGHT
from module2_triage.route_optimization.osrm_routing import get_road_route

st.set_page_config(page_title="Route Optimization", page_icon="🗺️", layout="wide")
apply_theme()

rescue_banner(
    "🗺️ Route Optimization",
    "Plans a priority-weighted rescue route across the triaged victim log."
)

victims_log = st.session_state.get("victims_log", [])

if not victims_log:
    st.warning(
        "No victims in the log yet. Go to **Triage & Prioritization**, run a "
        "detection, and click **Add to SITREP log** (now with a location) for "
        "each victim."
    )
    st.stop()

missing_location = [v["victim_id"] for v in victims_log if "lat" not in v or "lon" not in v]
if missing_location:
    st.warning(
        f"These victims were logged before location tracking was added, so "
        f"they're missing coordinates and will be skipped: {', '.join(missing_location)}. "
        f"Clear the log on the Triage page and re-add them to include location."
    )
victims_with_location = [v for v in victims_log if "lat" in v and "lon" in v]

if not victims_with_location:
    st.stop()

st.markdown("#### Rescue team starting point")
st.caption("Where the rescue team begins — the route is planned from here.")
bc1, bc2 = st.columns(2)
with bc1:
    start_lat = st.number_input("Start latitude", value=26.2183, format="%.5f")
with bc2:
    start_lon = st.number_input("Start longitude", value=78.1828, format="%.5f")

st.markdown("---")

route = optimize_route(start_lat, start_lon, victims_with_location)

total_km = route[-1]["cumulative_km"] if route else 0.0
m1, m2, m3 = st.columns(3)
m1.metric("Victims on route", len(route))
m2.metric("Total route distance", f"{total_km} km")
m3.metric("First stop", route[0]["victim_id"] if route else "—")

st.markdown("#### Optimized visit order")
st.caption(
    "Each stop is chosen by distance ÷ urgency weight — closer AND more "
    "urgent victims are visited sooner. A straight-line (haversine) "
    "distance heuristic, not a live road-routing API."
)
route_df = pd.DataFrame([{
    "Stop #": r["stop_number"],
    "Victim": r["victim_id"],
    "Urgency": r["urgency"].upper(),
    "Dist. from prev (km)": r["distance_from_prev_km"],
    "Cumulative (km)": r["cumulative_km"],
} for r in route])
st.dataframe(route_df, use_container_width=True)

st.markdown("---")
st.markdown("#### Map")

URGENCY_COLOR = {
    "critical": "red",
    "high": "orange",
    "medium": "beige",
    "low": "green",
}

center_lat = sum([start_lat] + [r["lat"] for r in route]) / (len(route) + 1)
center_lon = sum([start_lon] + [r["lon"] for r in route]) / (len(route) + 1)

fmap = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

folium.Marker(
    [start_lat, start_lon],
    tooltip="Rescue team start",
    icon=folium.Icon(color="blue", icon="home"),
).add_to(fmap)

path_points = [[start_lat, start_lon]]
for r in route:
    path_points.append([r["lat"], r["lon"]])
    folium.Marker(
        [r["lat"], r["lon"]],
        tooltip=f"Stop {r['stop_number']}: {r['victim_id']} ({r['urgency'].upper()})",
        icon=folium.Icon(
            color=URGENCY_COLOR.get(r["urgency"], "gray"),
            icon="info-sign",
        ),
    ).add_to(fmap)

road = get_road_route([tuple(p) for p in path_points])
folium.PolyLine(road["path"], color="#ff6b35", weight=4, opacity=0.85).add_to(fmap)
if road["source"] == "osrm":
    st.success(f"🛣️ Road route via OSRM: {road['distance_km']} km, about {road['duration_min']} min by road.")
else:
    st.warning("OSRM server not reachable, so straight lines are shown instead. Check your internet and refresh.")

st_folium(fmap, width=None, height=500, returned_objects=[])

st.caption(
    "🔧 Uses free OpenStreetMap tiles via Folium — no API key needed. "
    "Road paths come from the free OSRM demo server."
)