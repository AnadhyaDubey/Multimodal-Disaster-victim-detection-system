# Multi-Modal Disaster Victim Detection Suite — Project Plan

**Type:** Personal/portfolio project (not hackathon-scoped) — depth over breadth.
**Team:** Person A, Person B
**Core thread:** Detecting hidden/occluded humans across sensing modalities (visual, acoustic, sonar, radar), plus turning raw detections into an actionable rescue-priority decision layer.

---

## 1. Module Scope (finalized, post-trim)

### Module 1 — RGB Occlusion Detection (flagship module)
- Pipeline: COCO (pipeline sanity check) → CrowdHuman (real-world occlusion) → synthetic rubble-occlusion augmentation (25/50/75%, original contribution — script already built and validated) → YOLO11n/26n curriculum fine-tune (easy→hard occlusion)
- Audio module: mic → spectrogram → 1D-CNN/SVM tap-vs-noise classifier → visual+audio confidence fusion
- **Real-world validation (not training):** HERIDAL used as a held-out test set only, to confirm synthetic-trained model generalizes to genuine aerial SAR footage
- Evaluation: mAP/precision/recall **stratified by actual measured occlusion %** (not blended accuracy)
- Cut: AID-SAR, C2A (redundant with your own synthetic pipeline — adds no new validation angle)

### Module 2 — Underwater + Drowning Detection
- Sonar transfer learning: optical-pretrained YOLO → freeze early layers → fine-tune on **UATD only**
- Drowning sub-module: MediaPipe pose estimation → distress-posture classifier (real footage shootable since you swim regularly)
- Cut: Marine Debris dataset (redundant with UATD), ShuffleNetv2 embedded variant (defer until real edge hardware exists)
- Optional/low-priority: river-current drift calculator (~20-30 lines, build only if time allows)

### Module 3 — Avalanche/Radar Detection
- STFT/wavelet feature extraction → ANOVA/mRMR feature selection → SVM vs 1D-CNN comparison
- Data: simulated radargrams or public bioradar breathing-detection datasets
- Evaluation: ROC/PR curves

### Module 4 — Multi-Victim Triage & Rescue Prioritization (capstone, ties M1–M3 together)
- Synthetic multi-victim scenario generator (occlusion %, audio distress score, time-since-burial, access difficulty)
- Rule-based baseline (START-protocol-inspired scoring)
- Learned ranking model (pairwise ranking) trained against documented ground-truth priority order
- LLM-generated structured incident report (SITREP), reusing RAG project's LLM tooling
- Evaluation: rank-correlation (Kendall's tau / NDCG) vs ground-truth ranking
- Stretch goal only: rescue-routing optimization (Dijkstra/TSP-with-priority-weights) — build after ranking eval is solid, not before

### Cross-cutting infra (applies to all modules)
- Config: unified Hydra configs across all 4 modules
- Experiment tracking: **MLflow only** (not W&B — avoids redundant tooling, reuses RAG project setup)
- Containerization: Docker
- CI: one smoke test (augmentation/pipeline script runs without error + lint pass) — not a full test suite
- Demo: single Streamlit/Gradio dashboard — upload image/sonar/radar signal → get verdict → feeds into Module 4's prioritization view

### Deferred to future scope (do not build now — note as "Future Work" in README only)
- Thermal camera fusion (Module 1)
- USB endoscope/snake camera (Module 1)
- Jetson/Pi edge deployment (Module 1)
- Contact mic / mic array + direction-of-arrival (Module 1)
- Entry-level single-beam sonar hardware (Module 2)
- mmWave radar dev board — Acconeer/Infineon (Module 3)

---

## 2. Task Division

Full ownership per module (efficient — no internal blocking within a module), rotated across both people so everyone gets breadth (learning goal).

| Module | Lead | Support (code review + integration pairing only) |
|---|---|---|
| Module 1 — RGB Occlusion | **Person A** | Person B |
| Module 2 — Underwater + Drowning | **Person B** | Person A |
| Module 3 — Avalanche/Radar | **Person A** | Person B |
| Module 4 — Triage & Prioritization | **Person B** | Person A |
| Cross-cutting infra (Hydra, MLflow, Docker, CI, dashboard) | **Shared** — split by whoever finishes their module lead first | — |
| Final README, methodology write-up, demo video | **Shared** | — |

**Rule:** whoever leads a module writes the code; the support person reviews every PR on that module before merge — this is where the "equal learning" actually happens without doubling coordination overhead.

---

## 3. Build Sequence

Don't parallelize all 4 modules from day one — sequence gives each person a template from the previous module before tackling the next.

```
Phase 1: Module 1        (A leads, B supports)   ─┐
Phase 2: Module 2        (B leads, A supports)    ├─ can overlap once
Phase 3: Module 3        (A leads, B supports)    │  Phase 1 core pipeline
Phase 4: Module 4        (B leads, A supports)   ─┘  is stable
Phase 5: Integration     (shared) — dashboard, unified eval report, README
```

Reasoning: Module 1's confidence-fusion pattern (vision + audio) is reused conceptually in Module 4's multi-signal ranking. Module 3 is the simplest track technically, good as a "third module" once both people are warmed up. Module 4 depends on M1–M3 outputs (even if synthetic placeholders initially), so it comes last.

---

## 4. Repo Structure (proposed)

```
disaster-victim-detection/
├── configs/                  # Hydra configs, one subtree per module
│   ├── module1_rgb/
│   ├── module2_underwater/
│   ├── module3_radar/
│   └── module4_triage/
├── module1_rgb_occlusion/
│   ├── data_pipeline/        # COCO/CrowdHuman loaders, occlusion augmentation
│   ├── audio_fusion/
│   ├── train.py
│   └── eval_stratified.py
├── module2_underwater/
│   ├── sonar_transfer/
│   ├── drowning_detection/
│   └── train.py
├── module3_radar/
│   ├── signal_processing/
│   └── train.py
├── module4_triage/
│   ├── scenario_generator/
│   ├── ranking_model/
│   ├── sitrep_llm/
│   └── eval_rank_correlation.py
├── dashboard/                 # Streamlit/Gradio unified demo
├── docker/
├── .github/workflows/         # single smoke-test CI
├── mlruns/                    # MLflow tracking (gitignored)
├── README.md                  # portfolio framing (see below)
└── PROJECT_PLAN.md            # this file
```

---

## 5. README / Portfolio Framing (for later)

Frame explicitly, not implicitly:
- **Module 1 = primary deliverable** — fully trained, stratified-evaluated, demoed.
- **Modules 2 & 3 = documented extensions** — working pipeline + partial results, not competing flagships.
- **Module 4 = capstone** — shows the "so what": turns 3 independent detectors into one rescue-coordination decision layer.
- **Future Hardware Scope** — one short section, the deferred hardware list above, framed as roadmap not incompleteness.

---

## 6. Next Steps (after this file)

1. Create shared GitHub repo, both clone.
2. Scaffold the folder structure above + push empty configs.
3. Person A starts Module 1 (augmentation script + COCO/CrowdHuman loaders already have a head start from earlier work).
4. Person B reviews Module 1 PRs as they land, in parallel starts Module 2 scaffolding.
5. Continue per Build Sequence above.
