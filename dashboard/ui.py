"""
ui.py
-----
Visual layer for the single-flow dashboard (light theme: white + navy + dark green).
Everything that is "how it looks" lives here, so app.py only has the logic.

- inject_css():  fonts, colours, animated background, stage cards, pipeline spine
- topbar(), tracker(), stage_head(), flowchart(), metric_card(), chip(), gauge()
"""

import html
from urllib.parse import quote

import streamlit as st

NAVY = "#0B2545"
GREEN = "#14532D"
CYAN = "#0284C7"

URGENCY_STYLE = {
    "critical": ("#FEE2E2", "#991B1B", "#DC2626"),   # bg, text, solid
    "high":     ("#FEF3C7", "#92400E", "#F59E0B"),
    "medium":   ("#FFEDD5", "#9A3412", "#FB923C"),
    "low":      ("#DCFCE7", "#15803D", "#16A34A"),
}


def _h(markup: str) -> str:
    """Strip indentation + blank lines so Streamlit's markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def _svg_uri(svg: str) -> str:
    return 'url("data:image/svg+xml;utf8,' + quote(svg) + '")'


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
_RADAR_TILE = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='320' height='320'>"
    "<g fill='none' stroke='#0B2545' stroke-opacity='0.07'>"
    "<circle cx='160' cy='160' r='40'/><circle cx='160' cy='160' r='90' stroke-dasharray='3 3'/>"
    "<circle cx='160' cy='160' r='140'/>"
    "<line x1='0' y1='160' x2='320' y2='160'/><line x1='160' y1='0' x2='160' y2='320'/>"
    "</g></svg>"
)
_TOPO_TILE = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='600' height='600'>"
    "<g fill='none' stroke='#14532D' stroke-opacity='0.09' stroke-width='1'>"
    "<path d='M0,150 Q150,90 300,160 T600,150'/>"
    "<path d='M0,300 Q180,240 320,320 T600,300' stroke-dasharray='4 3'/>"
    "<path d='M0,450 Q140,510 310,430 T600,450'/>"
    "<path d='M0,560 Q200,520 330,575 T600,560' stroke-dasharray='2 4'/>"
    "</g></svg>"
)

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

:root{
  --navy:#0B2545; --navy-2:#132F4C; --green:#14532D; --cyan:#0284C7;
  --canvas:#F7F9FC; --card:#FFFFFF; --hair:#E2E8F0; --text:#0F172A; --muted:#475569;
  --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}

/* ---------- page + animated background ---------- */
.stApp{
  background-color:var(--canvas);
  background-image:__RADAR__,__TOPO__;
  background-size:320px 320px,600px 600px;
  animation:bgdrift 140s linear infinite;
  font-family:'Inter',system-ui,sans-serif;
  color:var(--text);
}
@keyframes bgdrift{
  from{background-position:0 0,0 0}
  to{background-position:640px 0,-1200px 600px}
}
/* slow radar rings + sweep, far right */
.stApp::before{
  content:"";position:fixed;right:-18vmax;top:12vh;width:64vmax;height:64vmax;border-radius:50%;
  background:repeating-radial-gradient(circle,transparent 0 70px,rgba(11,37,69,.045) 71px 72px);
  animation:ringpulse 9s ease-in-out infinite;pointer-events:none;z-index:0;
}
.stApp::after{
  content:"";position:fixed;right:-18vmax;top:12vh;width:64vmax;height:64vmax;border-radius:50%;
  background:conic-gradient(from 0deg,rgba(20,83,45,.06),rgba(20,83,45,0) 22%);
  animation:sweep 14s linear infinite;pointer-events:none;z-index:0;
}
@keyframes ringpulse{0%,100%{transform:scale(.96);opacity:.9}50%{transform:scale(1.04);opacity:.5}}
@keyframes sweep{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){
  .stApp,.stApp::before,.stApp::after,.flow-dot,.spine-pulse{animation:none!important}
}

[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],[data-testid="stDecoration"],#MainMenu,footer{display:none!important}
[data-testid="stMain"],[data-testid="stAppViewContainer"]>section{background:transparent!important}
.block-container{max-width:1440px;padding:5.5rem 2rem 3rem 2rem!important;position:relative;z-index:1}
.material-symbols-outlined{font-family:'Material Symbols Outlined';font-weight:normal;font-style:normal;
  font-size:20px;line-height:1;letter-spacing:normal;text-transform:none;display:inline-block;
  white-space:nowrap;-webkit-font-feature-settings:'liga';font-feature-settings:'liga';vertical-align:middle;
  width:1em;overflow:hidden}
.stApp a.fnode,.stApp a.tr-step,.stApp .tb-nav a{text-decoration:none!important}
.stApp a.fnode{color:inherit!important}
.stApp .fnode.big.dark .fk{color:#9FB3D1!important}
.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
.caps{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}

/* ---------- top bar ---------- */
.topbar{position:fixed;top:0;left:0;right:0;height:64px;z-index:1000;background:rgba(11,37,69,.96);
  backdrop-filter:blur(8px);box-shadow:0 1px 8px rgba(0,0,0,.08);display:flex;align-items:center;
  justify-content:space-between;gap:16px;padding:0 2rem;font-family:'Inter',sans-serif}
.tb-left{display:flex;align-items:center;gap:12px;min-width:0}
.tb-logo{width:34px;height:34px;border-radius:9px;background:rgba(255,255,255,.1);display:flex;
  align-items:center;justify-content:center;color:#B1F2BE;flex:none}
.tb-title{color:#fff;font-weight:600;font-size:15px;letter-spacing:-.005em;white-space:nowrap}
.tb-sub{color:#778DB2;font-size:11px;letter-spacing:.08em;text-transform:uppercase;white-space:nowrap}
.tb-nav{display:flex;gap:4px;padding:4px;border-radius:12px;background:rgba(255,255,255,.06)}
.tb-nav a{color:#9FB3D1;text-decoration:none;font-size:13px;font-weight:500;padding:6px 12px;border-radius:8px;
  display:flex;gap:6px;align-items:center;transition:.15s}
.tb-nav a span{font-family:var(--mono);opacity:.6;font-size:11px}
.tb-nav a:hover{background:rgba(255,255,255,.12);color:#fff}
.tb-status{display:flex;align-items:center;gap:8px;padding:5px 12px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.tb-status.ok{background:rgba(177,242,190,.14);color:#B1F2BE}
.tb-status.demo{background:rgba(245,158,11,.16);color:#FCD34D}
.tb-status i{width:8px;height:8px;border-radius:50%;background:currentColor;display:inline-block;animation:blink 1.8s infinite}
@keyframes blink{50%{opacity:.3}}
@media (max-width:1100px){.tb-nav{display:none}}
@media (max-width:640px){.topbar{padding:0 1rem}.tb-sub{display:none}.tb-status{display:none}}

/* ---------- tracker rail ---------- */
.tracker{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:14px;margin-bottom:2rem;
  box-shadow:0 4px 20px -2px rgba(11,37,69,.05);display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.tr-step{display:flex;gap:10px;align-items:center;padding:10px;border-radius:10px;background:#F2F4F7;
  text-decoration:none;transition:.15s;min-width:0}
.tr-step:hover{background:#ECEEF1}
.tr-num{width:30px;height:30px;border-radius:50%;background:var(--navy);color:#fff;display:flex;align-items:center;
  justify-content:center;font-family:var(--mono);font-size:12px;flex:none}
.tr-name{display:block;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tr-state{font-family:var(--mono);font-size:12px;display:flex;align-items:center;gap:6px;color:#94A3B8;white-space:nowrap}
.tr-state i{width:7px;height:7px;border-radius:50%;background:#CBD5E1;display:inline-block}
.tr-step.done .tr-state{color:var(--green)}
.tr-step.done .tr-state i{background:#16A34A}
@media (max-width:900px){.tracker{grid-template-columns:repeat(2,1fr)}}

/* ---------- pipeline spine + stage cards ---------- */
.st-key-pipeline{position:relative;padding-left:3.4rem;gap:2rem!important}
.st-key-pipeline::before{content:"";position:absolute;left:calc(2rem - 1px);top:1rem;bottom:1rem;width:2px;background:#E0E3E6}
.st-key-pipeline::after{content:"";position:absolute;left:calc(2rem - 1px);top:0;width:2px;height:90px;
  background:linear-gradient(to bottom,transparent,#2E6A41,transparent);animation:travel 6s linear infinite}
@keyframes travel{from{top:0}to{top:calc(100% - 90px)}}

[class*="st-key-stage"]{background:var(--card);border:1px solid var(--hair);border-radius:14px;
  padding:1.25rem 1.5rem 1.5rem 1.5rem!important;box-shadow:0 4px 20px -2px rgba(11,37,69,.05),0 2px 6px -1px rgba(11,37,69,.03);
  position:relative;gap:1rem!important}
[class*="st-key-stage"]::before{position:absolute;left:-2.4rem;top:1.15rem;width:2rem;height:2rem;border-radius:50%;
  background:#001026;color:#fff;display:flex;align-items:center;justify-content:center;font-family:var(--mono);
  font-size:13px;box-shadow:0 2px 8px rgba(11,37,69,.25);z-index:2}
.st-key-stage1::before{content:"1"}.st-key-stage2::before{content:"2"}.st-key-stage3::before{content:"3"}
.st-key-stage4::before{content:"4"}.st-key-stage5::before{content:"5"}
@media (max-width:700px){
  .block-container{padding:5rem 1rem 2rem 1rem!important}
  .st-key-pipeline{padding-left:2.2rem}
  .st-key-pipeline::before,.st-key-pipeline::after{left:.8rem}
  [class*="st-key-stage"]::before{left:-2.0rem;width:1.6rem;height:1.6rem;font-size:11px}
  [class*="st-key-stage"]{padding:1rem!important}
}

.stage-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;scroll-margin-top:90px}
.stage-head .l{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.stage-head .no{font-family:var(--mono);font-weight:600;font-size:13px;color:var(--navy)}
.stage-head .sl{color:#94A3B8}
.stage-head h2{margin:0;padding:0;font-size:18px;font-weight:600;letter-spacing:-.01em;color:var(--text);line-height:24px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:3px 11px;border-radius:999px;font-size:11px;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;line-height:14px;white-space:nowrap}
.chip.done{background:#DCFCE7;color:#15803D;border:1px solid #86EFAC}
.chip.wait{background:#F1F5F9;color:#64748B;border:1px solid #E2E8F0}
.chip .dot{width:6px;height:6px;border-radius:50%;background:currentColor;display:inline-block}
.chip.critical .dot{animation:blink 1.2s infinite}

/* ---------- flowchart ---------- */
.flow{overflow-x:auto;padding:4px 2px 10px}
.flow-row{min-width:1060px;display:flex;align-items:stretch;gap:6px}
.fcol{display:flex;flex-direction:column;gap:10px;flex:none}
.fnode{display:flex;align-items:center;gap:12px;padding:12px 14px;border-radius:12px;background:#F2F4F7;
  border:1px solid transparent;text-decoration:none;transition:.18s;color:inherit}
.fnode:hover{transform:translateY(-1px);box-shadow:0 8px 24px -4px rgba(11,37,69,.12);border-color:var(--hair);background:#fff}
.fico{width:38px;height:38px;border-radius:10px;background:var(--navy);color:#fff;display:flex;align-items:center;justify-content:center;flex:none}
.fico.soft{background:#E0E3E6;color:var(--navy)}
.fk{display:block;font-size:10.5px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.fv{display:block;font-size:14.5px;font-weight:600;color:var(--text);line-height:19px}
.fnode.big{flex-direction:column;align-items:flex-start;justify-content:space-between;min-height:100%;gap:8px;padding:14px 16px}
.fnode.big.dark{background:var(--navy);color:#fff}
.fnode.big.dark .fk{color:#9FB3D1}.fnode.big.dark .fv{color:#fff}
.fnode.big.mid{background:#E6E8EB}
.fsub{font-size:12px;color:var(--muted);line-height:16px}
.fnode.big.dark .fsub{color:#9FB3D1}
.farrow{display:flex;align-items:center;flex:none;width:34px;position:relative}
.farrow::before{content:"";height:2px;width:100%;background:#CBD5E1;border-radius:2px}
.farrow::after{content:"";position:absolute;right:0;width:8px;height:8px;border-top:2px solid #94A3B8;border-right:2px solid #94A3B8;transform:rotate(45deg)}
.flow-dot{position:absolute;left:0;top:50%;margin-top:-3px;width:6px;height:6px;border-radius:50%;background:#16A34A;animation:dotmove 2.2s linear infinite}
@keyframes dotmove{from{left:0;opacity:0}15%{opacity:1}85%{opacity:1}to{left:calc(100% - 10px);opacity:0}}

/* ---------- metric cards ---------- */
.mcard{background:#fff;border:1px solid var(--hair);border-radius:14px;padding:14px 16px 14px 18px;position:relative;overflow:hidden;
  box-shadow:0 4px 20px -2px rgba(11,37,69,.05);height:100%}
.mcard::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--accent,#0B2545)}
.mcard .v{font-size:30px;font-weight:700;letter-spacing:-.03em;line-height:36px;color:var(--text);font-variant-numeric:tabular-nums}
.mcard .s{font-size:12px;color:var(--muted);margin-top:2px}
.mrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px}

/* ---------- detection viewport ---------- */
.viewport{position:relative;border-radius:12px;overflow:hidden;background:#0B2545;line-height:0}
.viewport img{width:100%;height:auto;display:block}
.vbox{position:absolute;border:2px solid var(--c,#B1F2BE);border-radius:4px;background:transparent}
.vbox.sel{border-width:3px;box-shadow:0 0 0 2px #fff,0 0 14px var(--c)}
.vtag{position:absolute;left:-2px;top:-16px;background:var(--c,#0B2545);color:#fff;font-family:var(--mono);font-size:10px;
  padding:0 5px;border-radius:4px;line-height:15px;white-space:nowrap;font-weight:600}
.vempty{border:1px dashed #C4C6CF;border-radius:12px;min-height:260px;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:8px;color:#94A3B8;background:#F8FAFC;font-size:13px}
.vempty .material-symbols-outlined{font-size:34px}
.achip{display:inline-flex;align-items:center;gap:8px;background:var(--navy);color:#fff;border-radius:10px;padding:7px 12px;font-size:13px;margin-top:10px}
.achip b{font-family:var(--mono);color:#B1F2BE}

/* ---------- tables ---------- */
.ttable{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--hair);border-radius:12px;overflow:hidden;font-size:14px}
.ttable th{background:#F8FAFC;text-align:left;padding:11px 14px;font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);border-bottom:1px solid var(--hair)}
.ttable td{padding:11px 14px;border-bottom:1px solid #F1F5F9;color:var(--text)}
.ttable tr:hover td{background:#F8FAFC}
.ttable td.r,.ttable th.r{text-align:right}
.rank{display:inline-block;padding:2px 8px;border-radius:5px;font-family:var(--mono);font-size:12px;font-weight:600;background:#E0E3E6;color:var(--navy)}
.rank.first{background:var(--navy);color:#fff}
.wrap-x{overflow-x:auto}

/* ---------- streamlit widgets ---------- */
.stButton>button{background:var(--navy);color:#fff;border:0;border-radius:8px;font-weight:600;min-height:40px;padding:.4rem 1.1rem;
  transition:.15s;box-shadow:0 1px 2px rgba(11,37,69,.15)}
.stButton>button:hover{background:var(--navy-2);color:#fff;transform:translateY(-1px);box-shadow:0 8px 24px -4px rgba(11,37,69,.18)}
.stButton>button p,.stButton>button span{color:#fff!important}
.stButton>button:disabled{background:#CBD5E1;box-shadow:none;transform:none}
.st-key-btn_add button,.st-key-btn_sitrep button{background:var(--green)}
.st-key-btn_add button:hover,.st-key-btn_sitrep button:hover{background:#0F3F23}
.st-key-btn_clear button{background:#fff;border:1px solid #CBD5E1;color:var(--text)}
.st-key-btn_clear button p{color:var(--text)!important}
.st-key-btn_clear button:hover{background:#F1F5F9}
[data-testid="stFileUploaderDropzone"]{background:#F2F4F7;border:1px dashed #C4C6CF;border-radius:12px}
[data-testid="stWidgetLabel"] p,.stRadio label p{font-size:13px;font-weight:500;color:var(--text)}
[data-baseweb="input"],[data-baseweb="select"]>div{border-radius:8px!important}
.st-key-sitrep_card{background:#F2F4F7;border-radius:12px;padding:1rem 1.2rem!important;min-height:200px}
.st-key-sitrep_card p,.st-key-sitrep_card li{font-size:14px;line-height:22px}
.tiny{font-size:12px;color:var(--muted)}
.empty-card{border:1px dashed #C4C6CF;border-radius:12px;padding:28px;text-align:center;color:#94A3B8;background:#F8FAFC;font-size:13px}
.foot{margin-top:2.5rem;padding:1rem 0;border-top:1px solid var(--hair);display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
  font-size:12px;color:var(--muted)}
</style>
"""


def inject_css():
    css = _CSS.replace("__RADAR__", _svg_uri(_RADAR_TILE)).replace("__TOPO__", _svg_uri(_TOPO_TILE))
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Components (all return/emit HTML)
# ---------------------------------------------------------------------------
def icon(name: str, size: int = 20) -> str:
    return f'<span class="material-symbols-outlined" style="font-size:{size}px">{name}</span>'


def topbar(model_loaded: bool):
    status = (
        '<div class="tb-status ok"><i></i>Model loaded</div>'
        if model_loaded else
        '<div class="tb-status demo"><i></i>Demo mode</div>'
    )
    st.markdown(_h(f"""
    <div class="topbar">
      <div class="tb-left">
        <div class="tb-logo">{icon('radar')}</div>
        <div><div class="tb-title">Multimodal Disaster Victim Detection</div>
        <div class="tb-sub">Detection · Triage · Rescue routing</div></div>
      </div>
      <nav class="tb-nav">
        <a href="#stage-1"><span>01</span>Pipeline</a><a href="#stage-2"><span>02</span>Detection</a>
        <a href="#stage-3"><span>03</span>Triage</a><a href="#stage-4"><span>04</span>SITREP</a>
        <a href="#stage-5"><span>05</span>Routing</a>
      </nav>
      {status}
    </div>"""), unsafe_allow_html=True)


def tracker_html(states):
    """states: list of 5 (name, text, done)."""
    parts = []
    for i, (name, text, done) in enumerate(states, start=1):
        parts.append(f"""
        <a class="tr-step {'done' if done else ''}" href="#stage-{i}">
          <div class="tr-num">{i}</div>
          <div style="min-width:0"><span class="tr-name">{html.escape(name)}</span>
          <span class="tr-state"><i></i>{html.escape(text)}</span></div>
        </a>""")
    return _h('<div class="tracker">' + "".join(parts) + "</div>")


def stage_head_html(num: int, title: str, done: bool) -> str:
    chip = (
        f'<span class="chip done">{icon("check_circle", 14)}Done</span>' if done
        else '<span class="chip wait"><span class="dot"></span>Waiting</span>'
    )
    return _h(f"""
    <div class="stage-head" id="stage-{num}">
      <div class="l"><span class="no">STAGE 0{num}</span><span class="sl">//</span><h2>{html.escape(title)}</h2></div>
      {chip}
    </div>""")


def flowchart_html() -> str:
    def arrow():
        return '<div class="farrow"><span class="flow-dot"></span></div>'

    return _h(f"""
    <div class="flow"><div class="flow-row">
      <div class="fcol" style="width:220px">
        <a class="fnode" href="#stage-2"><div class="fico">{icon('photo_camera')}</div>
          <div><span class="fk">Input A</span><span class="fv">Image / Camera</span></div></a>
        <a class="fnode" href="#stage-2"><div class="fico">{icon('mic')}</div>
          <div><span class="fk">Input B</span><span class="fv">Audio / Mic</span></div></a>
      </div>
      {arrow()}
      <div class="fcol" style="width:250px">
        <a class="fnode" href="#stage-2"><div class="fico soft">{icon('crop_free')}</div>
          <div><span class="fk">Vision module</span><span class="fv">Occlusion · YOLO</span></div></a>
        <a class="fnode" href="#stage-2"><div class="fico soft">{icon('graphic_eq')}</div>
          <div><span class="fk">Audio module</span><span class="fv">Distress · SVM</span></div></a>
      </div>
      {arrow()}
      <a class="fnode big mid" href="#stage-3" style="width:210px">
        <div style="display:flex;justify-content:space-between;width:100%"><span class="fk">Fusion</span>{icon('merge', 18)}</div>
        <div><span class="fv">Triage &amp; Priority</span><div class="fsub">Weighted score, 0–100</div></div>
      </a>
      {arrow()}
      <a class="fnode big" href="#stage-4" style="width:190px">
        <div style="display:flex;justify-content:space-between;width:100%"><span class="fk">Report</span>{icon('description', 18)}</div>
        <div><span class="fv">Situation Report</span><div class="fsub">LLM summary for field teams</div></div>
      </a>
      {arrow()}
      <a class="fnode big dark" href="#stage-5" style="width:210px">
        <div style="display:flex;justify-content:space-between;width:100%"><span class="fk">Dispatch</span>{icon('alt_route', 18)}</div>
        <div><span class="fv">Route Optimization</span><div class="fsub">Priority-first road route</div></div>
      </a>
    </div></div>""")


def metric_card(label: str, value: str, sub: str = "", accent: str = NAVY) -> str:
    return _h(f"""
    <div class="mcard" style="--accent:{accent}"><div class="caps">{html.escape(label)}</div>
    <div class="v">{html.escape(str(value))}</div><div class="s">{html.escape(sub)}</div></div>""")


def urgency_chip(level: str) -> str:
    bg, fg, solid = URGENCY_STYLE.get(level, URGENCY_STYLE["medium"])
    return (f'<span class="chip {level}" style="background:{bg};color:{fg};border:1px solid {solid}55">'
            f'<span class="dot"></span>{html.escape(level.upper())}</span>')


def gauge_html(score: float, level: str) -> str:
    _, _, solid = URGENCY_STYLE.get(level, URGENCY_STYLE["medium"])
    arc = 251.3
    offset = arc * (1 - max(0.0, min(100.0, score)) / 100)
    return _h(f"""
    <div style="display:flex;flex-direction:column;align-items:center;text-align:center;padding:8px 0">
      <div class="caps">Priority score</div>
      <div style="position:relative;width:240px;height:138px;margin-top:6px">
        <svg viewBox="0 0 200 110" style="width:100%;height:100%">
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#E0E3E6" stroke-width="18" stroke-linecap="round"/>
          <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{solid}" stroke-width="18" stroke-linecap="round"
                stroke-dasharray="{arc}" stroke-dashoffset="{offset:.1f}"/>
        </svg>
        <div style="position:absolute;left:0;right:0;bottom:2px;display:flex;flex-direction:column;align-items:center">
          <span style="font-size:40px;font-weight:700;letter-spacing:-.03em;line-height:42px">{score:.0f}</span>
          <span class="tiny">/ 100</span>
        </div>
      </div>
      <div style="margin-top:10px">{urgency_chip(level)}</div>
    </div>""")


def victims_table_html(rows) -> str:
    body = []
    for v in rows:
        rank = v["rule_based_rank"]
        cls = "rank first" if rank == 1 else "rank"
        body.append(f"""
        <tr><td><span class="mono" style="font-weight:600;color:{NAVY}">{html.escape(v['victim_id'])}</span></td>
        <td>{urgency_chip(v['urgency'])}</td>
        <td class="mono">{v['time_buried_minutes']} min</td>
        <td>{html.escape(str(v['accessibility']).capitalize())}</td>
        <td class="r"><span class="{cls}">RANK #{rank:02d}</span></td></tr>""")
    return _h(f"""
    <div class="wrap-x"><table class="ttable"><thead><tr><th>Victim</th><th>Urgency</th><th>Time buried</th>
    <th>Access</th><th class="r">Priority</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>""")


def route_table_html(route) -> str:
    body = []
    for r in route:
        body.append(f"""
        <tr><td><span class="rank first">{r['stop_number']}</span></td>
        <td><span class="mono" style="font-weight:600">{html.escape(r['victim_id'])}</span></td>
        <td>{urgency_chip(r['urgency'])}</td>
        <td class="mono r">{r['distance_from_prev_km']} km</td></tr>""")
    return _h(f"""
    <div class="wrap-x"><table class="ttable"><thead><tr><th>Stop</th><th>Victim</th><th>Urgency</th>
    <th class="r">From previous</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>""")


def empty_card(icon_name: str, text: str) -> str:
    return _h(f'<div class="empty-card">{icon(icon_name, 30)}<div style="margin-top:6px">{html.escape(text)}</div></div>')


def occlusion_color(occlusion_pct: float) -> str:
    """Box colour: red = mostly hidden, amber = partly hidden, green = mostly visible."""
    if occlusion_pct >= 60:
        return "#DC2626"
    if occlusion_pct >= 30:
        return "#F59E0B"
    return "#16A34A"


def victim_color(score: float) -> str:
    """Box colour by victim-likelihood: red likely, amber possible, green unlikely."""
    return "#DC2626" if score >= 45 else "#F59E0B" if score >= 30 else "#16A34A"


def rescue_order_html(rows) -> str:
    """rows: ranked victim dicts (already sorted). Big readable 'save in this order' list."""
    items = []
    for i, v in enumerate(rows, start=1):
        items.append(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid #E5E9F0">
        <span class="rank first" style="background:{NAVY}">{i}</span>
        <div style="flex:1"><b>Person {html.escape(str(v['victim_id']))}</b>
        <div class="tiny">{v['time_buried_minutes']} min buried · access {html.escape(v['accessibility'])}</div></div>
        <span class="mono" style="font-weight:600">{v['order_score']:.0f}</span>
        {urgency_chip(v['urgency'])}</div>""")
    return _h(f'<div>{"".join(items)}</div>')


def detections_table_html(dets, selected_ids) -> str:
    body = []
    for d in dets:
        col = victim_color(d["victim_score"])
        mark = f'<span class="chip done">{icon("check", 13)}Victim</span>' if d["id"] in selected_ids else ""
        body.append(f"""
        <tr><td><span class="rank first" style="background:{col}">#{d['id']}</span></td>
        <td class="mono">{d['victim_label']} ({d['victim_score']:.0f})</td>
        <td class="mono">{d['occlusion_pct']:.0f}%</td><td class="r">{mark}</td></tr>""")
    return _h(f"""
    <div class="wrap-x" style="max-height:260px;overflow-y:auto"><table class="ttable"><thead><tr><th>Person</th>
    <th>Victim-like</th><th>Hidden (est.)</th><th class="r">Status</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>""")


def scored_table_html(rows) -> str:
    """rows: list of (detection, priority_result)."""
    body = []
    for d, res in sorted(rows, key=lambda r: r[1]["priority_score"], reverse=True):
        body.append(f"""
        <tr><td><span class="mono" style="font-weight:600;color:{NAVY}">#{d['id']}</span></td>
        <td class="mono">{d['occlusion_pct']:.0f}%</td>
        <td class="mono" style="font-weight:600">{res['priority_score']:.0f}</td>
        <td class="r">{urgency_chip(res['urgency'])}</td></tr>""")
    return _h(f"""
    <div class="wrap-x" style="max-height:240px;overflow-y:auto"><table class="ttable"><thead><tr><th>Person</th>
    <th>Hidden</th><th>Score</th><th class="r">Urgency</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>""")


_URG_HI = {"critical": "गंभीर", "high": "उच्च", "medium": "मध्यम", "low": "कम"}
_ACC_HI = {"easy": "आसान", "moderate": "सामान्य", "difficult": "कठिन"}
_ORD_EN = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]


def speech_texts(rows):
    """English and Hindi spoken versions of the rescue order."""
    n = len(rows)
    en = [f"Rescue order. Save {n} {'person' if n == 1 else 'people'}."]
    hi = [f"बचाव क्रम। कुल {n} लोगों को बचाना है।"]
    for i, v in enumerate(rows):
        num = str(v["victim_id"]).lstrip("V")
        o = _ORD_EN[i] if i < len(_ORD_EN) else f"Number {i + 1}"
        en.append(f"{o}: victim {num}, {v['urgency']} urgency, buried {v['time_buried_minutes']} minutes, "
                  f"access {v['accessibility']}.")
        hi.append(f"{i + 1} नंबर: पीड़ित {num}, {_URG_HI.get(v['urgency'], v['urgency'])} स्थिति, "
                  f"{v['time_buried_minutes']} मिनट से दबे, पहुँच {_ACC_HI.get(v['accessibility'], '')}।")
    return " ".join(en), " ".join(hi)


def speech_html(rows) -> str:
    """Self-contained buttons that read the rescue order aloud (browser speech, free)."""
    import json
    en, hi = speech_texts(rows)
    return """
<style>
 body{margin:0;font-family:Inter,system-ui,sans-serif}
 button{border:0;border-radius:10px;padding:12px 18px;font-size:15px;font-weight:600;cursor:pointer;margin-right:8px;color:#fff;background:#0B2545}
 button.s{background:#fff;color:#0B2545;border:1px solid #0B2545}
 #m{font-size:12px;color:#64748B;margin-top:6px}
</style>
<button onclick="say(EN,'en-US')">&#128266; Speak order (English)</button>
<button onclick="say(HI,'hi-IN')">&#128266; बोलें (हिंदी)</button>
<button class="s" onclick="speechSynthesis.cancel()">Stop</button>
<div id="m"></div>
<script>
const EN=%s, HI=%s;
function say(t,l){
  if(!('speechSynthesis' in window)){document.getElementById('m').textContent='Voice is not supported in this browser.';return;}
  speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance(t); u.lang=l; u.rate=0.95;
  const v=speechSynthesis.getVoices().find(x=>x.lang&&x.lang.startsWith(l.slice(0,2)));
  if(v)u.voice=v; else if(l==='hi-IN')document.getElementById('m').textContent='No Hindi voice on this device; install one in system settings.';
  speechSynthesis.speak(u);
}
</script>""" % (json.dumps(en), json.dumps(hi))