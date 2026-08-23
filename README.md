\# Multimodal Disaster Victim Detection System



A multimodal detection and rescue-prioritization pipeline that fuses \*\*RGB camera vision\*\* and \*\*audio-based distress signal detection\*\* to locate survivors trapped under rubble in earthquake and building-collapse scenarios — combined with a \*\*multi-victim triage module\*\* that ranks rescue priority, generates human-readable situation reports, computes an optimized rescue route, and ties everything together in a unified interactive dashboard.



\---



\## 🎯 Overview



When a disaster strikes (earthquake, building collapse), rescue teams face two hard problems:

1\. \*\*Finding survivors\*\* who may be partially hidden under rubble and unable to call for help.

2\. \*\*Deciding who to rescue first\*\* when multiple potential survivors are detected, with limited time and resources.



This project addresses both:



\- \*\*Module 1 — RGB Occlusion Detection\*\*: Detects partially occluded human victims via camera, fused with audio-based tap/voice distress signal detection (a real technique used by trapped survivors to signal rescuers).

\- \*\*Module 2 — Multi-Victim Triage \& Rescue Prioritization\*\*: Takes detection outputs from multiple victims, ranks them by rescue urgency, generates a field-ready situation report, and computes the most efficient route to reach them.

\- \*\*Unified Dashboard\*\*: A single Streamlit app tying vision+audio fusion, triage ranking, and route optimization into one interactive demo — runs identically on Windows and Mac.



\---



\## 🧩 System Architecture



```

&#x20;               ┌─────────────────┐        ┌─────────────────┐

&#x20;               │   RGB Camera    │        │   Microphone     │

&#x20;               └────────┬────────┘        └────────┬────────┘

&#x20;                        │                          │

&#x20;                ┌───────▼────────┐        ┌────────▼────────┐

&#x20;                │  Vision Model   │        │  Audio Classifier │

&#x20;                │  (YOLO, curriculum-      │  (SVM: tap/voice/  │

&#x20;                │   trained on occlusion)  │   noise, 90% acc)   │

&#x20;                └───────┬────────┘        └────────┬────────┘

&#x20;                        │  visual\_confidence        │  audio\_confidence

&#x20;                        └──────────┬─────────────────┘

&#x20;                                   ▼

&#x20;                         ┌───────────────────┐

&#x20;                         │   Fusion Module     │

&#x20;                         │  (weighted combine)  │

&#x20;                         └─────────┬──────────┘

&#x20;                                   │  final\_confidence per victim

&#x20;                                   ▼

&#x20;                    ┌──────────────────────────────┐

&#x20;                    │   Module 2: Triage Pipeline    │

&#x20;                    │  • Rule-based baseline          │

&#x20;                    │  • Learned pairwise ranker       │

&#x20;                    │  • Kendall's tau / NDCG eval       │

&#x20;                    │  • LLM-generated SITREP            │

&#x20;                    │  • Route optimization (TSP + OSRM,  │

&#x20;                    │    free, no API key)                │

&#x20;                    └──────────────┬───────────────┘

&#x20;                                   ▼

&#x20;                         ┌───────────────────┐

&#x20;                         │  Streamlit Dashboard │

&#x20;                         │  (all modules, one UI)│

&#x20;                         └───────────────────┘

```



\---



\## 📦 Module 1 — RGB Occlusion Detection



\*\*Vision pipeline\*\* (`module1\_rgb\_occlusion/vision/`):

\- Trained on COCO (pipeline sanity check) → CrowdHuman (real-world occlusion) → custom synthetic rubble-occlusion augmentation (25%/50%/75% occlusion levels, original contribution — procedurally generated debris textures, blob-shaped occlusion masks, actual-vs-nominal occlusion logging)

\- YOLO26n curriculum fine-tuning (easy → hard occlusion, each stage initialized from the previous stage's best checkpoint)

\- Stratified evaluation — mAP/precision/recall broken down by \*\*actual measured occlusion %\*\*, not just nominal target, with seaborn visualizations

\- Live webcam / phone-stream demo (via IP Webcam over WiFi), with recorded video output



\*\*Audio pipeline\*\* (`module1\_rgb\_occlusion/audio/`):

\- Mic recording → statistical feature extraction (MFCC, spectral centroid, zero-crossing rate, RMS energy)

\- SVM classifier distinguishing \*\*tap\*\* (intentional knocking signal), \*\*voice\*\* (calling/moaning), and \*\*noise\*\* (background)

\- \*\*90% test accuracy\*\*

\- Real-time live demo with audible text-to-speech alerts



\*\*Fusion\*\* (`module1\_rgb\_occlusion/fusion/`):

\- Combines `visual\_confidence` and `audio\_confidence` into a single survivor-detection score:

&#x20; ```

&#x20; final\_confidence = 0.7 × visual\_confidence + 0.3 × audio\_confidence

&#x20; ```

\- Fully functional end-to-end with real trained weights on both sides (vision + audio)



\---



\## 📦 Module 2 — Multi-Victim Triage \& Rescue Prioritization



\*(folder: `module4\_triage/`)\*



\- \*\*Scenario generator\*\*: synthetic multi-victim scenarios with confidence scores, time-buried, and accessibility

\- \*\*Rule-based baseline\*\*: explainable, START-protocol-inspired priority scoring

\- \*\*Learned ranking model\*\*: pairwise ranking (Logistic Regression) that outperforms the rule-based baseline on rank correlation

\- \*\*Evaluation\*\*: Kendall's tau and NDCG comparison



&#x20; | Model | Avg Kendall's Tau | Avg NDCG |

&#x20; |---|---|---|

&#x20; | Rule-Based | 0.827 | 0.994 |

&#x20; | Pairwise Ranker | \*\*0.893\*\* | 0.990 |



\- \*\*SITREP generation\*\*: Claude API generates a plain-language situation report for field teams from ranked victim data

\- \*\*Route optimization\*\*: exact TSP solver (brute-force optimal for ≤8 locations) using \*\*OSRM\*\* — free, open-source, no API key or credit card required — for real road-network distances and route geometry, rendered on an interactive Folium map



\---



\## 🖥️ Unified Dashboard



\*(folder: `dashboard/`)\*



A single Streamlit app (`app.py`) with three tabs:

\- \*\*Vision + Audio Fusion\*\* — upload a photo + audio clip, runs the real trained models, shows confidence scores and verdict

\- \*\*Route Optimization\*\* — runs the TSP + OSRM solver, displays the interactive map inline (works fully offline in demo mode)

\- \*\*Triage Ranking\*\* — displays priority-ranked victims



Pure Python, so it runs identically on Windows and Mac with the same two commands.



\---



\## 🛠️ Tech Stack



| Component | Tools |

|---|---|

| Vision | YOLO26n (Ultralytics), OpenCV, PyTorch |

| Audio | librosa, scikit-learn, sounddevice |

| Triage / Ranking | scikit-learn, scipy, NetworkX |

| Situation Reports | Anthropic Claude API |

| Route Optimization | OSRM (free, no key), Folium |

| Dashboard | Streamlit |

| Environment | Python venv |



\---



\## 🚀 Setup



```bash

git clone https://github.com/AnadhyaDubey/Multimodal-Disaster-victim-detection-system.git

cd Multimodal-Disaster-victim-detection-system



python3 -m venv venv

source venv/bin/activate   # Windows: venv\\Scripts\\activate



pip install -r module1\_rgb\_occlusion/vision/requirements.txt

pip install -r module1\_rgb\_occlusion/audio/requirements.txt

```



\### Optional API keys

```bash

export ANTHROPIC\_API\_KEY="your-key"      # for SITREP generation only

```

Route optimization needs \*\*no API key\*\* — OSRM's public server is free and open.



\---



\## ▶️ Running the Demos



\*\*Unified dashboard (recommended — all modules in one place):\*\*

```bash

cd dashboard

pip install -r requirements.txt

streamlit run app.py

```



\*\*Audio live demo\*\* (real-time tap/voice detection):

```bash

cd module1\_rgb\_occlusion/audio

python live\_demo.py

```



\*\*Live vision demo (webcam or phone stream):\*\*

```bash

cd module1\_rgb\_occlusion/vision

python live\_webcam\_demo.py --weights best.pt --source 0

```



\*\*Run the triage pipeline end-to-end:\*\*

```bash

cd module4\_triage/eval

python eval\_ranking.py

```



\*\*Generate a route map:\*\*

```bash

cd module4\_triage/route\_optimization

python route\_solver.py --demo   # offline test

python route\_solver.py --victims victims.json   # real coordinates

```



\---



\## 📊 Results Snapshot



\- \*\*Vision (stratified by actual occlusion %)\*\*: mAP50 improved from 0.643 (20-40% occlusion) to 0.707 (60-80% occlusion) across curriculum training stages — performance held up, rather than collapsing, as occlusion got harder

\- Audio classifier (tap/voice/noise): \*\*90% test accuracy\*\*

\- Pairwise triage ranker vs. rule-based baseline: \*\*higher Kendall's tau (0.893 vs 0.827)\*\*

\- Fusion pipeline: fully functional end-to-end with real trained vision + audio weights

\- Route optimization: exact-optimal TSP route with real road-network geometry, free and open-source



\---



\## 🔭 Future Work



\- HERIDAL dataset validation as a held-out real-world test set

\- Full dataset scaling (current vision training used a subset of CrowdHuman)

\- MLflow experiment tracking integration

\- Additional sensing modalities (underwater/sonar, radar-based breathing detection) as potential future extensions

\- Thermal camera fusion, edge deployment (Jetson/Pi)

\- Dockerized deployment



\---



Built as a collaborative two-person project, covering vision, audio, fusion, triage ranking, SITREP generation, route optimization, and a unified dashboard.

