"""
Multimodal Disaster Victim Detection System - single-flow dashboard.

One scrolling page, five connected stages:
  1 Pipeline overview  ->  2 Detection (YOLO + audio)  ->  3 Triage score
  ->  4 Situation report (LLM)  ->  5 Route optimization (OSRM roads)

Run from the dashboard folder:   python -m streamlit run app.py
"""

import base64
import hashlib
import html
import io
import os
import sys

import folium
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit_folium import st_folium

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DASHBOARD_DIR)
sys.path.append(DASHBOARD_DIR)   # ui.py, utils.py
sys.path.append(REPO_ROOT)       # module2_triage/...

from ui import (  # noqa: E402
    URGENCY_STYLE, detections_table_html, rescue_order_html, speech_html, victim_color, scored_table_html, empty_card, flowchart_html, gauge_html, icon, inject_css,
    metric_card, route_table_html, stage_head_html, topbar, tracker_html,
    victims_table_html,
)
from utils import (  # noqa: E402
    compute_priority_score, rescue_order_score, load_yolo_model, run_audio_distress_detection,
    run_occlusion_detection,
)
from module2_triage.route_optimization.osrm_routing import get_road_route  # noqa: E402
from module2_triage.route_optimization.priority_routing import optimize_route  # noqa: E402

WEIGHTS = os.path.join(DASHBOARD_DIR, "weights", "best.pt")   # absolute: works locally and on the cloud
DEFAULT_START = (26.2183, 78.1828)

st.set_page_config(
    page_title="Disaster Victim Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

S = st.session_state
S.setdefault("victims_log", [])

model = load_yolo_model(WEIGHTS)
inject_css()
topbar(model is not None)
tracker_ph = st.empty()


def md(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def image_to_data_uri(img: Image.Image, max_w: int = 960) -> str:
    img = img.copy()
    if img.width > max_w:
        img.thumbnail((max_w, max_w * 4))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def road_route_cached(points):
    """Ask OSRM once per set of points. Failures are not cached, so a retry works."""
    cache = S.setdefault("road_cache", {})
    key = tuple((round(a, 5), round(b, 5)) for a, b in points)
    if key in cache:
        return cache[key]
    result = get_road_route(list(points))
    if result["source"] == "osrm":
        cache[key] = result
    return result


def groq_key():
    try:
        saved = st.secrets["GROQ_API_KEY"]
    except Exception:
        saved = None
    return saved or os.environ.get("GROQ_API_KEY") or S.get("groq_api_key")


# ===========================================================================
with st.container(key="pipeline"):

    # ------------------------------------------------------------------ 1
    with st.container(key="stage1"):
        md(stage_head_html(1, "Pipeline Architecture", True))
        md(flowchart_html())

    # ------------------------------------------------------------------ 2
    with st.container(key="stage2"):
        head2 = st.empty()
        left, right = st.columns([5, 7], gap="large")

        image, img_bytes, audio_bytes = None, None, None
        with left:
            mode = st.radio("Scene source", ["Upload image", "Use camera"], horizontal=True, key="image_mode", label_visibility="collapsed")
            if mode == "Upload image":
                f = st.file_uploader("Scene image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            else:
                f = st.camera_input("Capture a scene", label_visibility="collapsed")
            if f is not None:
                img_bytes = f.getvalue()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            conf_min = st.slider("Minimum confidence (lower finds more people)", 0.10, 0.60, 0.25, 0.05, key="conf_min")
            hi_detail = st.checkbox("High detail (slower, finds smaller people)", key="hi_detail")

            amode = st.radio("Audio (optional)", ["None", "Upload clip", "Microphone"], horizontal=True, key="audio_mode")
            if amode == "Upload clip":
                af = st.file_uploader("Audio clip", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
                if af is not None:
                    audio_bytes = af.getvalue()
            elif amode == "Microphone":
                rec = st.audio_input("Record ambient sound or a call for help", label_visibility="collapsed")
                if rec is not None:
                    audio_bytes = rec.getvalue()

        # run each model once per new input (not on every widget click)
        imgsz = 960 if hi_detail else 640
        visible = []
        if image is not None:
            sig = f"{hashlib.md5(img_bytes).hexdigest()}-{imgsz}"
            if S.get("det_sig") != sig:
                S["det_all"] = run_occlusion_detection(image, model, conf=0.10, imgsz=imgsz)
                S["det_sig"] = sig
            # the slider only filters; it never re-runs the model
            kept = [d for d in S["det_all"]["detections"] if d["confidence"] >= conf_min]
            kept.sort(key=lambda d: d["victim_score"], reverse=True)           # most victim-like first
            visible = [dict(d, id=i) for i, d in enumerate(kept, start=1)]
        else:
            S["det_all"], S["det_sig"] = None, None

        if audio_bytes is not None:
            asig = hashlib.md5(audio_bytes).hexdigest()
            if S.get("aud_sig") != asig:
                S["aud"] = run_audio_distress_detection(audio_bytes)
                S["aud_sig"] = asig
        else:
            S["aud"], S["aud_sig"] = None, None

        aud = S.get("aud")
        has_det = len(visible) > 0
        selected_ids = []

        with right:
            img_ph = st.empty()          # filled last, so boxes can show who is selected
            if image is None:
                img_ph.markdown(
                    f'<div class="vempty">{icon("image_search")}<div>Add an image to run detection</div></div>',
                    unsafe_allow_html=True,
                )
            elif not has_det:
                md('<div class="tiny" style="margin-top:8px">No people found. Lower the minimum confidence or turn on High detail.</div>')
            else:
                likely = [d["id"] for d in visible if d["victim_score"] >= 45]
                md(
                    '<div class="mrow">'
                    + metric_card("People found", len(visible), "numbered on the photo", "#0284C7")
                    + metric_card("Likely victims", len(likely), "hidden or lying down", "#DC2626")
                    + metric_card("Upright / visible", len(visible) - len(likely), "probably not victims", "#16A34A")
                    + "</div>"
                )
                selected_ids = st.multiselect(
                    "Who needs rescue? Likely victims are pre-ticked. Confirm or change",
                    options=[d["id"] for d in visible],
                    default=likely,
                    format_func=lambda i: f"#{i}",
                    key=f"sel_{S['det_sig']}_{conf_min}",
                )
                with st.expander("All detections"):
                    md(detections_table_html(visible, selected_ids))
                if S["det_all"].get("is_placeholder"):
                    md('<div class="tiny">Simulated result: model weights not loaded.</div>')

            if image is not None:
                W, H = image.size
                boxes = ""
                for d in visible:
                    x1, y1, x2, y2 = d["bbox"]
                    l, t = max(0, x1 / W * 100), max(0, y1 / H * 100)
                    w, h = min(100 - l, (x2 - x1) / W * 100), min(100 - t, (y2 - y1) / H * 100)
                    sel = " sel" if d["id"] in selected_ids else ""
                    boxes += (
                        f'<div class="vbox{sel}" style="--c:{victim_color(d["victim_score"])};'
                        f'left:{l:.1f}%;top:{t:.1f}%;width:{w:.1f}%;height:{h:.1f}%">'
                        f'<span class="vtag">#{d["id"]} · {d["victim_label"]}</span></div>'
                    )
                img_ph.markdown(
                    f'<div class="viewport"><img src="{image_to_data_uri(image)}" alt="scene">{boxes}</div>',
                    unsafe_allow_html=True,
                )

            if aud:
                md(
                    f'<div class="achip">{icon("record_voice_over", 18)}'
                    f'<span>{html.escape(aud["class"].capitalize())} detected · distress score <b>{aud["distress_score"]:.2f}</b></span></div>'
                )
                if aud.get("is_placeholder"):
                    md('<div class="tiny" style="margin-top:4px">Audio result is simulated until the trained classifier is wired in.</div>')

    # ------------------------------------------------------------------ 3
    result = None
    with st.container(key="stage3"):
        head3 = st.empty()
        chosen = [d for d in visible if d["id"] in selected_ids]
        if not has_det:
            md(empty_card("hourglass_top", "Waiting for a detection"))
        elif not chosen:
            md(empty_card("touch_app", "Tick at least one person above"))
        else:
            cl, cr = st.columns([5, 7], gap="large")
            n = len(S["victims_log"])
            with cr:
                audio_score = st.slider(
                    "Audio distress score",
                    0.0, 1.0,
                    float(aud["distress_score"]) if aud else 0.5,
                    0.05,
                    key=f"distress_{S.get('aud_sig') or 'none'}",
                )
                md('<div class="tiny">Set time buried and how easy each person is to reach.</div>')
                edited = st.data_editor(
                    [{"Person": f"#{d['id']}", "Time buried (min)": 30, "Access": "moderate"} for d in chosen],
                    key=f"edit_{S['det_sig']}_{'-'.join(str(d['id']) for d in chosen)}",
                    hide_index=True, use_container_width=True, disabled=["Person"],
                    column_config={
                        "Time buried (min)": st.column_config.NumberColumn(min_value=0, step=5),
                        "Access": st.column_config.SelectboxColumn(options=["easy", "moderate", "difficult"]),
                    },
                )
                extra = {d["id"]: (int(r["Time buried (min)"]), r["Access"])
                         for d, r in zip(chosen, (edited if isinstance(edited, list) else edited.to_dict("records")))}
                c3, c4 = st.columns(2)
                v_lat = c3.number_input("Scene latitude", value=DEFAULT_START[0] + n * 0.004, format="%.5f", key=f"lat_{n}")
                v_lon = c4.number_input("Scene longitude", value=DEFAULT_START[1] + n * 0.004, format="%.5f", key=f"lon_{n}")
                add = st.button(f"Add {len(chosen)} to rescue log", key="btn_add")

            scored = [
                (d, compute_priority_score(
                    occlusion_confidence=d["confidence"],
                    occlusion_pct=d["occlusion_pct"],
                    audio_distress=audio_score,
                ))
                for d in chosen
            ]
            for d, res in scored:
                t, ac = extra[d["id"]]
                res["order_score"] = rescue_order_score(res["priority_score"], t, ac)
            scored.sort(key=lambda r: r[1]["order_score"], reverse=True)
            top_d, result = scored[0]
            S["last_score"] = result["priority_score"]
            with cl:
                md(gauge_html(result["priority_score"], result["urgency"]))
                md(f'<div class="tiny" style="text-align:center;margin-bottom:10px">Rescue first: person #{top_d["id"]}</div>')
                md(scored_table_html(scored))

            if add:
                for i, (d, res) in enumerate(scored):
                    S["victims_log"].append({
                        "victim_id": f"V{n + i + 1}",
                        "final_confidence": round(res["priority_score"] / 100, 2),
                        "order_score": res["order_score"],
                        "time_buried_minutes": extra[d["id"]][0],
                        "accessibility": extra[d["id"]][1],
                        "urgency": res["urgency"],
                        "lat": v_lat + i * 0.0001,     # same scene, tiny offset so pins do not overlap
                        "lon": v_lon + i * 0.0001,
                    })
                with cr:
                    st.success(f"{len(scored)} people added to the log.")

    # ------------------------------------------------------------------ 4
    log = S["victims_log"]
    with st.container(key="stage4"):
        head4 = st.empty()
        if not log:
            md(empty_card("playlist_add", "No victims in the log yet"))
        else:
            ranked = sorted(log, key=lambda v: v.get("order_score", v["final_confidence"] * 100), reverse=True)
            for i, v in enumerate(ranked, start=1):
                v["rule_based_rank"] = i

            with st.container(key="rescue_order"):
                md(f'<div class="caps" style="margin:10px 0 4px">{icon("campaign", 18)} Rescue order: save {len(ranked)} '
                   f'{"person" if len(ranked) == 1 else "people"} in this sequence</div>')
                md(rescue_order_html(ranked))
                components.html(speech_html(ranked), height=70)

            tl, tr = st.columns([7, 5], gap="large")
            with tl:
                md(victims_table_html(ranked))
                key = groq_key()
                if not key:
                    entered = st.text_input("Free Groq API key (console.groq.com/keys)", type="password")
                    if entered:
                        S["groq_api_key"] = entered
                        key = entered
                b1, b2 = st.columns(2)
                gen = b1.button("Generate SITREP", key="btn_sitrep", disabled=not key)
                clear = b2.button("Clear log", key="btn_clear")

                if gen:
                    try:
                        from module2_triage.sitrep.generate_sitrep_groq import generate_sitrep
                        with st.spinner("Writing the report..."):
                            S["sitrep_text"] = generate_sitrep(ranked, rank_key="rule_based_rank", api_key=key)
                    except ImportError:
                        st.error("The `groq` package is missing. Run: python -m pip install groq")
                    except Exception as e:  # network, quota, bad key...
                        st.error(f"Report generation failed: {e}")
                if clear:
                    S["victims_log"] = []
                    S.pop("sitrep_text", None)
                    st.rerun()

            with tr:
                with st.container(key="sitrep_card"):
                    md(f'<div class="caps" style="margin-bottom:8px">{icon("psychology", 18)} Field SITREP</div>')
                    if S.get("sitrep_text"):
                        st.markdown(S["sitrep_text"])
                    else:
                        md('<div class="tiny">The report appears here after you click Generate.</div>')

    # ------------------------------------------------------------------ 5
    located = [v for v in log if "lat" in v and "lon" in v]
    route, road = [], None
    with st.container(key="stage5"):
        head5 = st.empty()
        if not located:
            md(empty_card("alt_route", "Add at least one victim with a location"))
        else:
            sc1, sc2, _sp = st.columns([1, 1, 2])
            start_lat = sc1.number_input("Start latitude", value=DEFAULT_START[0], format="%.5f", key="start_lat")
            start_lon = sc2.number_input("Start longitude", value=DEFAULT_START[1], format="%.5f", key="start_lon")

            route = optimize_route(start_lat, start_lon, located)
            points = [(start_lat, start_lon)] + [(r["lat"], r["lon"]) for r in route]
            road = road_route_cached(points)

            fmap = folium.Map(location=points[0], zoom_start=14, tiles="OpenStreetMap", control_scale=True)

            def pin(label, color):
                return folium.DivIcon(
                    icon_size=(30, 30), icon_anchor=(15, 15),
                    html=(
                        f'<div style="width:30px;height:30px;border-radius:50%;background:{color};color:#fff;'
                        f'display:flex;align-items:center;justify-content:center;font:600 12px Inter,sans-serif;'
                        f'border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.35)">{label}</div>'
                    ),
                )

            folium.Marker(points[0], tooltip="Rescue team start", icon=pin("HQ", "#0B2545")).add_to(fmap)
            for r in route:
                solid = URGENCY_STYLE.get(r["urgency"], URGENCY_STYLE["medium"])[2]
                folium.Marker(
                    (r["lat"], r["lon"]),
                    tooltip=f"Stop {r['stop_number']}: {r['victim_id']} ({r['urgency'].upper()})",
                    icon=pin(r["stop_number"], solid),
                ).add_to(fmap)
            folium.PolyLine(road["path"], color="#0B2545", weight=6, opacity=0.9).add_to(fmap)
            folium.PolyLine(road["path"], color="#0284C7", weight=3, opacity=1, dash_array="10 8").add_to(fmap)
            fmap.fit_bounds(points, padding=(40, 40), max_zoom=16)

            st_folium(fmap, height=400, use_container_width=True, returned_objects=[], key="route_map")

            by_road = road["source"] == "osrm"
            dist = road["distance_km"] if by_road else route[-1]["cumulative_km"]
            md(
                '<div class="mrow">'
                + metric_card("Stops", len(route), "in visit order", "#0B2545")
                + metric_card("Distance", f"{dist} km", "by road" if by_road else "straight line (OSRM offline)", "#14532D")
                + metric_card("Travel time", f"{road['duration_min']:.0f} min" if by_road else "n/a", "estimated", "#0284C7")
                + "</div>"
            )
            md(route_table_html(route))

# ---------------------------------------------------------------------------
# fill the placeholders now that every stage has run
# ---------------------------------------------------------------------------
sitrep_done = bool(S.get("sitrep_text"))
head2.markdown(stage_head_html(2, "Multimodal Detection", has_det), unsafe_allow_html=True)
head3.markdown(stage_head_html(3, "Triage & Priority Score", result is not None), unsafe_allow_html=True)
head4.markdown(stage_head_html(4, "Situation Report", sitrep_done), unsafe_allow_html=True)
head5.markdown(stage_head_html(5, "Route Optimization", bool(route)), unsafe_allow_html=True)

tracker_ph.markdown(
    tracker_html([
        ("Pipeline", "READY", True),
        ("Detection", "DETECTED" if has_det else "AWAITING INPUT", has_det),
        ("Triage score", f"SCORED: {result['priority_score']:.0f}" if result else "WAITING", result is not None),
        ("SITREP", "GENERATED" if sitrep_done else (f"LOG: {len(log)}" if log else "WAITING"), sitrep_done),
        ("Routing", "ROUTE READY" if route else "WAITING", bool(route)),
    ]),
    unsafe_allow_html=True,
)

md(
    '<div class="foot"><span>Multimodal Disaster Victim Detection System</span>'
    '<span>Detection → Triage → Situation report → Routing</span></div>'
)