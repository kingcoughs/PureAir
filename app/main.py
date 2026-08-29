"""
FastAPI Server Entry Point for Delhi-NCR AI AQI Prediction & Policy Management Engine
"""

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings, STATIC_DIR, CHECKPOINTS_DIR
from app.api.citizen_routes import router as citizen_router
from app.api.gov_routes import router as gov_router
from app.api.grid_routes import router as grid_router
from app.models.train_model1 import trainer_model1

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Hyper-local Spatio-Temporal Graph AI Engine for AQI Forecasting, Dynamic Causality Attribution, and Policy Simulation across Delhi-NCR."
)

# Enable CORS for cross-origin frontend dashboard connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Route Modules
app.include_router(citizen_router)
app.include_router(gov_router)
app.include_router(grid_router)

# Mount Static Assets for Dashboard UI
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.on_event("startup")
async def startup_event():
    """Initializes and primes model checkpoints on application startup."""
    print(f"[{settings.APP_NAME}] Initializing AI Models and Atmospheric Grid...")
    checkpoint_file = CHECKPOINTS_DIR / "st_gnn_model1_latest.npz"
    if not checkpoint_file.exists():
        print(f"[{settings.APP_NAME}] Checkpoint not found. Performing initial calibration training...")
        trainer_model1.train(epochs=15, batch_size=8, verbose=True)
    else:
        print(f"[{settings.APP_NAME}] Loaded calibrated checkpoint: {checkpoint_file.name}")
    print(f"[{settings.APP_NAME}] System is LIVE and ready for citizen & government API requests.")

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "mode": "live_operational"
    }

@app.get("/api/info", tags=["System"])
async def api_info():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "architecture": {
            "spatial_discretization": "Uber H3 Resolution 7 Hexagons (~5.16 km² per node)",
            "model_1": "Live Spatiotemporal Predictor (ST-GNN + Pasquill-Gifford Dynamic Advection + GRU)",
            "model_2": "Causal Trend & Attribution Analyzer (Integrated Gradients + Residual Tracker)",
            "physics_inversion_gate": "Ventilation Index (VI = PBL x Wind Speed < 6000 m²/s)",
            "policy_simulator": "do-calculus counterfactual intervention engine",
            "citizen_features": ["Hyper-local AQI", "1-72h Trajectory", "Clean Air Window Optimizer", "Instant Incident Injection"],
            "government_features": ["Hotspot Causality Matrix", "do-calculus Simulator", "DBSCAN Triage Queue", "Weekly GRAP Audit"]
        }
    }

@app.get("/", tags=["Dashboard"])
async def serve_dashboard():
    """Serves the Interactive Verification UI & Dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": f"Welcome to {settings.APP_NAME}. Interactive UI is loading."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

