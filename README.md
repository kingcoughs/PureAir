# Delhi-NCR AI-Driven Air Quality Forecasting & Policy Management Engine (Project Meswak)

An end-to-end physics-informed Spatio-Temporal Graph AI backend and REST API engine for hyper-local air quality index (AQI) forecasting, dynamic source attribution, crowdsourced incident management, and counterfactual policy simulation across the Delhi-National Capital Region (NCR).

---

## 1. System Overview & Dual-Model AI Architecture

Project Meswak treats Delhi-NCR not as a flat map of arbitrary numbers, but as a living, interconnected spatiotemporal grid where pollution flows dynamically across terrain shaped by wind advection, thermal inversions, and ground-level point sources.

Physical space is discretized into an **Uber H3 Hexagonal Grid (Resolution 7, ~5.16 km² per node)** covering 80+ spatial zones, including industrial clusters (Mayapuri, Wazirpur, Okhla, Mundka, Bawana), major freight corridors (Ring Road, NH-48, EPE/WPE), smoldering landfills (Ghazipur, Bhalswa, Okhla), and natural green sinks/ridges.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA INGESTION                                                              │
│ CPCB / DPCC Sensors  •  Open-Meteo API  •  NASA FIRMS Stubble  •  Citizen   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ SPATIAL PROCESSING LAYER (Uber H3 Res 7)                                    │
│ Node Feature Matrix X(t) [N, F]  •  Physics Adjacency Matrix A(t) [N, N]    │
└───────────────────┬─────────────────────────────────────┬───────────────────┘
                    │                                     │
                    ▼                                     ▼
┌──────────────────────────────────────┐ ┌────────────────────────────────────┐
│ MODEL 1: LIVE PREDICTOR (LSP)        │ │ MODEL 2: WEEKLY AUDITOR (CTAA)     │
│ • Spatial Advection Graph Conv       │ │ • Integrated Gradients Attribution │
│ • Recurrent GRU Temporal Layer       │ │ • Residual Error Tracking          │
│ • 1-72h Multi-Step Forecast Head     │ │ • Weekly GRAP Policy Briefs        │
│ • Attention-Based Dominant Driver    │ │ • Closed-Loop Model 1 Retraining   │
│ • MC Dropout 90% Confidence Envelope │ │                                    │
└───────────────────┬──────────────────┘ └─────────────────┬──────────────────┘
                    │                                      │
                    ▼                                      ▼
┌──────────────────────────────────────┐ ┌────────────────────────────────────┐
│ CITIZEN REST API & MOBILE/WEB APP    │ │ GOVERNMENT POLICY COMMAND CENTER   │
│ • Hyper-Local Real-Time AQI          │ │ • Ward Causality Matrix            │
│ • 1-72h Trajectory with Envelopes    │ │ • Counterfactual Policy Simulator  │
│ • Clean Air Activity Window Planner  │ │ • DBSCAN Incident Dispatch Queue   │
│ • Geotagged Incident Impulse Reports │ │ • Weekly Graded Action Plan Briefs │
└──────────────────────────────────────┘ └────────────────────────────────────┘
```

---

## 2. Core Mathematical Formulations

### 2.1 Dynamic Wind & Terrain Adjacency ($A_{ij}(t)$)
Models how strongly Hexagon $i$ (source) pollutes Hexagon $j$ (target) driven by wind direction $\phi_w$, wind speed $u_{\text{wind}}$, distance $d_{ij}$, and elevation resistance $\Delta h$:

$$A_{ij}(t) = \mathbb{I}(x_\parallel > 0) \cdot \exp\left(-\frac{x_\perp^2}{2 \sigma_y^2(x_\parallel)}\right) \cdot \exp\left(-\frac{x_\parallel}{u_{\text{wind}} \cdot \tau_{\text{decay}}}\right) \cdot \exp\left(-\frac{\max(0, h_j - h_i)}{\tau_h}\right) \cdot \psi(\text{VI})$$

* **One-Way Downwind Gate** ($\mathbb{I}(x_\parallel > 0)$): Transport only flows downwind; upwind weights are zeroed.
* **Pasquill-Gifford Lateral Dispersion** ($\sigma_y(x_\parallel) = k_y \cdot x_\parallel^{0.89}$): Smoke plume expands laterally into a Gaussian cone.
* **Particulate Settling Decay** ($\exp(-x_\parallel / (u_{\text{wind}} \tau_{\text{decay}}))$): Heavy particles settle out over distance with half-life $\tau_{\text{decay}} = 3600\text{ s}$.
* **Ridge Elevation Resistance** ($\exp(-\max(0, h_j - h_i) / \tau_h)$): Models the physical barrier effect of the Delhi Ridge ($\tau_h = 50\text{ m}$).
* **Thermal Inversion Lid Gate** ($\psi(\text{VI}) = 1 + \frac{1}{1 + (\text{VI} / 6000)^2}$): When the Ventilation Index ($\text{VI} = z_{\text{PBL}} \times u_{\text{wind}}$) collapses below $6000\text{ m}^2/\text{s}$, edge weights amplify to model stagnant ground-level trapping.

### 2.2 Transient Incident Impulse Injection ($\Delta X(t)$)
Crowdsourced citizen reports inject an immediate localized emission impulse that decays exponentially:

$$\Delta X_i^{\text{incident}}(t) = \sum_{k \in \text{Reports}_i} S_k \cdot \mathbf{w}_{\text{type}} \cdot \exp\left(-\frac{t - t_{0,k}}{\lambda_k}\right) \cdot \mathbb{I}(\text{Confidence}_k \ge \theta)$$

### 2.3 Integrated Gradients Source Apportionment (Model 2)
Game-theoretic path-integral attribution decomposing excess AQI against a clean baseline ($\tilde{X}$ with PM2.5 $\le 15 \mu\text{g/m}^3$):

$$\text{Attribution}_i^f = (X_i^f - \tilde{X}_i^f) \times \int_0^1 \frac{\partial \text{AQI}_i}{\partial X_i^f} \left(\tilde{X} + \alpha(X - \tilde{X}), A(t)\right) d\alpha$$

### 2.4 Counterfactual Policy Simulator ($do$-Calculus)
Evaluates hypothetical government actions by forcing constrained feature states:

$$\widehat{Y}_{\text{policy}} = \text{Model}_1\left(\mathbf{X} \text{ with } do(X_{\text{target}} = \kappa \cdot X_{\text{target}}), A(t)\right)$$

### 2.5 Clean Air Window Optimizer
Finds the lowest cumulative particulate inhalation window for outdoor exercise over a 24-hour horizon:

$$t^* = \arg\min_{t \in [T_0, T_0 + 24 - \Delta t]} \int_t^{t + \Delta t} \widehat{\text{AQI}}_i(\tau) \, d\tau$$

---

## 3. Directory Structure

```
project_meswak/
├── app/
│   ├── config.py                 # System constants, Delhi bounds, physics params
│   ├── grid/
│   │   ├── h3_grid.py            # Uber H3 Hexagon discretization (Res 7)
│   │   ├── topography.py         # Ridge elevation, industrial & traffic hotspots
│   │   └── dynamic_graph.py      # Dynamic physics-informed adjacency A(t)
│   ├── data/
│   │   ├── open_meteo.py         # Meteorology & PBL height ingestion
│   │   ├── cpcb_sensors.py       # CPCB/DPCC ground station sensor engine
│   │   ├── stubble_firms.py      # NASA FIRMS satellite biomass plume model
│   │   ├── incidents_store.py    # Citizen incident store, impulse decay & DBSCAN
│   │   └── dataset_builder.py    # Spatiotemporal sequence dataset generator
│   ├── models/
│   │   ├── st_gnn.py             # Model 1: Live Spatiotemporal Predictor (ST-GNN)
│   │   ├── blame_engine.py       # Model 2: Integrated Gradients blame engine
│   │   ├── policy_simulator.py   # Counterfactual Policy Simulator (do-calculus)
│   │   ├── clean_air_window.py   # Clean Air Window Planner
│   │   ├── train_model1.py       # Model 1 training pipeline with backprop
│   │   └── auditor_model2.py     # Model 2 weekly audit & retraining pipeline
│   ├── api/
│   │   ├── schemas.py            # Pydantic typed request & response models
│   │   ├── citizen_routes.py     # /api/aqi/live, /forecast, /optimal-window, /incidents
│   │   ├── gov_routes.py         # /api/gov/causality-matrix, /simulate-policy, /triage
│   │   └── grid_routes.py        # /api/grid/hexagons, /adjacency, /stations
│   ├── static/                   # Interactive Dashboard UI
│   │   ├── index.html            # Single-page interface with Leaflet & Chart.js
│   │   ├── css/styles.css        # Dark glassmorphism styling
│   │   └── js/
│   │       ├── app.js            # Controller & map rendering
│   │       └── charts.js         # Trajectory & source apportionment charts
│   └── main.py                   # FastAPI application entrypoint
├── data/checkpoints/             # Saved model weights (.npz)
├── train.py                      # Standalone CLI training & audit tool
├── run_server.py                 # FastAPI server startup script
├── test_system.py                # Full automated verification suite
└── README.md
```

---

## 4. Quick Start & Execution

### 4.1 Run the Full Test Suite
```bash
python test_system.py
```

### 4.2 Train / Calibrate Model 1 and Run Model 2 Audit
```bash
python train.py --epochs 20 --batch-size 8 --run-audit
```

### 4.3 Launch the FastAPI Server & Interactive Dashboard
```bash
python run_server.py
```
Open your browser and navigate to:
* **Interactive UI Dashboard:** [http://localhost:8000/](http://localhost:8000/)
* **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc API Reference:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 5. REST API Reference

### Citizen Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/aqi/live?lat=28.6468&lon=77.3160` | Real-time AQI, sub-pollutants, dominant stressor, and health advisory |
| `GET` | `/api/aqi/forecast?lat=28.6468&lon=77.3160` | 1-to-72h predicted AQI trajectory with 90% confidence uncertainty envelopes |
| `GET` | `/api/aqi/optimal-window?duration=2` | Recommended lowest inhalation window over next 24 hours |
| `POST` | `/api/incidents/report` | Submit geotagged crowdsourced incident (injects immediate impulse $\Delta X$) |
| `GET` | `/api/incidents/active` | Active incident reports with exponential decay status |
| `GET` | `/api/user/digest?lat=28.6139&lon=77.2090` | Sunday neighborhood air quality digest |

### Government Policy Command Center Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/gov/causality-matrix` | Ward-level vulnerability rankings, top 1st and 2nd contributors, and GRAP actions |
| `POST` | `/api/gov/simulate-policy` | Counterfactual policy simulation ($do$-calculus) for Odd-Even, Freight Bans, etc. |
| `GET` | `/api/gov/incidents/triage` | DBSCAN clustered incident queue cross-validated against sensor spikes |
| `POST` | `/api/gov/incidents/dispatch` | Dispatch municipal enforcement squad to verified hotspot |
| `GET` | `/api/gov/weekly-audit` | Model 2 Integrated Gradients source apportionment & GRAP brief |
| `POST` | `/api/gov/retrain` | Trigger closed-loop Model 1 active retraining using logged residuals |

### Grid & Topology Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/grid/hexagons` | Uber H3 hexagon polygons, centroids, elevations, and emission weights |
| `GET` | `/api/grid/adjacency` | Time-varying directed wind transport edges and weights |
| `GET` | `/api/grid/stations` | Ground CPCB/DPCC monitoring stations with live sensor sub-indices |

