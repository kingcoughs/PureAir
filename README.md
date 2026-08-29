# Delhi-NCR AI-Driven Air Quality Forecasting & Policy Management Engine (Project Meswak)

An end-to-end physics-informed Spatio-Temporal Graph AI backend and REST API engine with two dedicated mobile/web applications (**Citizen Mobile App** and **Government Command Center**) for hyper-local air quality index (AQI) forecasting, dynamic source attribution, crowdsourced incident management, and counterfactual policy simulation across the Delhi-National Capital Region (NCR).

---

## 1. System Architecture & Dual-App Ecosystem

Project Meswak discretizes the entire Delhi-NCR airshed into **281 contiguous Uber H3 Resolution 7 Hexagons (~5.16 km² per node)** covering all wards and districts across Delhi NCT, Noida, Greater Noida, Ghaziabad, Gurugram, Faridabad, Sonipat, and Bahadurgarh.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA & INGESTION LAYER                                                      │
│ CPCB / DPCC Sensors  •  Open-Meteo API  •  NASA FIRMS Stubble  •  Citizen   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPANDED SPATIAL PROCESSING LAYER (Uber H3 Res 7)                           │
│ 281 Contiguous Hexagons (Delhi NCT, Noida, Ghaziabad, Gurugram, Faridabad)  │
│ Dynamic Physics Adjacency Matrix A(t) [281x281]  •  Elevation Profile       │
└───────────────────┬─────────────────────────────────────┬───────────────────┘
                    │                                     │
                    ▼                                     ▼
┌──────────────────────────────────────┐ ┌────────────────────────────────────┐
│ MODEL 1: LIVE PREDICTOR (ST-GNN)     │ │ MODEL 2: WEEKLY AUDITOR (CTAA)     │
│ • Spatial Advection Graph Conv       │ │ • Recalibrated Integrated Grads    │
│ • 12-Hour Recurrent GRU Layer        │ │ • Multi-Factor Source Attribution  │
│ • 1-72h Multi-Step Forecast Head     │ │ • Residual Error Tracking (R²=.93) │
│ • Attention Dominant Stressor        │ │ • Closed-Loop Model 1 Retraining   │
│ • MC Dropout 90% Confidence Bounds   │ │                                    │
└───────────────────┬──────────────────┘ └─────────────────┬──────────────────┘
                    │                                      │
                    ▼                                      ▼
┌──────────────────────────────────────┐ ┌────────────────────────────────────┐
│ CITIZEN MOBILE APPLICATION           │ │ GOVERNMENT COMMAND CENTER          │
│ (/citizen)                           │ │ (/gov)                             │
│ • Bento Grid Modular Widgets         │ │ • Calibrated Causality Matrix      │
│ • Main AQI Card + Primary Source     │ │ • Hexagon-Specific do-Calculus     │
│ • Expandable Pollutant Drill-down    │ │ • DBSCAN Triage with Photo Viewer  │
│ • 3-Day Daytime Clean Air Planner    │ │ • Squad Dispatch Trigger           │
│ • Photo-Enabled Incident Submission  │ │ • AI Model Lab & Active Learning   │
│ • Free Map with Search & Wind Flow   │ │ • Free Airshed Map & Search        │
│ • Theme & Server Switchers           │ │ • Theme & Server Switchers         │
└──────────────────────────────────────┘ └────────────────────────────────────┘
```

---

## 2. Key Features

### 2.1 Citizen Mobile App (`/citizen`)
* **Responsive Bento Grid (Touch-First)**: Modular 2x2, 1x2, 2x1, 2x4 widget layout optimized for mobile aspect ratios.
* **Main AQI Card**: Displays AQI, Category, Dominant Pollutant, and **Primary Pollution Source** (e.g. *Inbound Stubble Smoke (42%) + Inversion Trap*).
* **Expandable Drill-down Bottom Sheet**: Tap to view full breakdown of PM2.5, PM10, NO2, SO2, CO, O3, temperature, humidity, wind, and boundary layer mixing height.
* **3-Day Daytime Clean Air Planner**: Optimizes outdoor cardio/activity schedules during waking hours (06:00 AM - 09:00 PM) across 3 days, showing safest windows vs peak morning rush and percentage of particulate inhalation avoided.
* **Photo-Enabled Incident Reporting**: Crowdsourced reporting with photo upload & preview, severity slider, and instant injection of transient impulse $\Delta X(t)$ into Model 1.
* **Interactive Airshed Map**: Free OpenStreetMap/CartoDB tiles, full 281-hexagon coverage, locality search autocomplete, dynamic wind flow overlay, and legend explaining blue dashed dynamic transport vectors ($A_{ij}(t)$).
* **AQI Educational Guide**: Explaining AQI categories, sensitive group precautions, and pollutant definitions.
* **Theme & Server URL Switcher**: Dark, Light, AMOLED themes, plus instant toggle between Localhost and Remote Cloud Server.

### 2.2 Government Command Center (`/gov`)
* **Calibrated Hotspot Causality Matrix**: Balanced multi-source attribution (Primary, Secondary, and Tertiary percentages) with specific GRAP regulatory directives.
* **Hexagon-Specific Counterfactual Policy Simulator ($do$-calculus)**: Evaluates Odd-Even traffic cuts, truck bans to EPE/WPE, construction halts, industrial curfews, and anti-smog mist cannons specifically on a chosen ward or citywide with projected before-and-after $\Delta$AQI, percent drop, and time-lag.
* **DBSCAN Incident Triage Queue**: Clustered citizen reports with photo evidence inspection and one-click *"🚨 Dispatch Enforcement Squad"* button.
* **AI Model Lab**: Architecture breakdown, live residual error metrics (RMSE, MAE, R²), and one-click execution of closed-loop active learning retraining.

---

## 3. Quick Start & Execution

### 3.1 Run the Automated Test Suite (All 9 tests passing)
```powershell
python test_system.py
```

### 3.2 Train Model 1 & Run Model 2 Weekly Audit
```powershell
python train.py --epochs 15 --batch-size 8 --run-audit
```

### 3.3 Launch the FastAPI Server
```powershell
python run_server.py
```
Open your browser at:
* **Master App Launcher & Portal:** [http://localhost:8000/](http://localhost:8000/)
* **Citizen Mobile App:** [http://localhost:8000/citizen](http://localhost:8000/citizen)
* **Government Command Center:** [http://localhost:8000/gov](http://localhost:8000/gov)
* **Interactive Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 4. Hosting on Phones & Free Cloud

* **Local Phone Testing (No Cloud Server needed):**
  1. Run `python run_server.py`.
  2. Find PC IP via `ipconfig` (e.g. `192.168.1.15`).
  3. Open `http://192.168.1.15:8000/citizen` on your mobile phone browser &rarr; Tap *"Add to Home Screen"*.
* **Free Cloud Hosting:**
  * Step-by-step guides for **Render.com**, **Hugging Face Spaces**, **Railway.app**, and **GitHub Pages** are provided in [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md).
