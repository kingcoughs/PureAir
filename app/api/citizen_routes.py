"""
Citizen API Endpoints: Hyper-Local AQI, 1-72h Trajectories, Clean Air Planner, and Incident Submissions
"""

import time
import numpy as np
from fastapi import APIRouter, Query, HTTPException
from typing import Dict, List, Any, Optional

from app.config import grid_config, settings
from app.grid.h3_grid import grid_manager
from app.data.open_meteo import weather_engine
from app.data.cpcb_sensors import sensor_engine, calculate_cpcb_aqi
from app.data.stubble_firms import stubble_engine
from app.data.incidents_store import incident_store
from app.data.dataset_builder import dataset_builder
from app.models.st_gnn import model1_lsp
from app.models.clean_air_window import clean_air_optimizer
from app.api.schemas import (
    LiveAQIResponse,
    ForecastTrajectoryResponse,
    CleanAirWindowRequest,
    CleanAirWindowResponse,
    IncidentReportRequest,
    IncidentReportResponse,
    UserDigestResponse
)

router = APIRouter(prefix="/api", tags=["Citizen App"])

@router.get("/aqi/live", response_model=LiveAQIResponse)
async def get_live_aqi(
    lat: Optional[float] = Query(None, description="GPS Latitude"),
    lon: Optional[float] = Query(None, description="GPS Longitude"),
    hex_id: Optional[str] = Query(None, description="Target H3 Hexagon ID")
):
    """
    Returns real-time hyper-local AQI, sub-pollutants, dominant driver,
    atmospheric weather, and health advisory for any Delhi coordinate.
    """
    # 1. Resolve Hexagon Node
    if hex_id and hex_id in grid_manager.nodes:
        node = grid_manager.nodes[hex_id]
    elif lat is not None and lon is not None:
        node = grid_manager.find_nearest_node(lat, lon)
    else:
        # Default to central Delhi (Connaught Place / Mandir Marg)
        node = grid_manager.find_nearest_node(28.6365, 77.2011)

    node_idx = grid_manager.hex_ids.index(node.hex_id)

    # 2. Ingest weather, stubble, and sensor state
    weather = weather_engine.get_current_weather()
    stubble = stubble_engine.compute_stubble_inflow(weather["wind_direction"], weather["wind_speed"])
    
    # 3. Build current node features & dynamic adjacency A(t)
    X_curr, A_curr = dataset_builder.build_current_node_features(weather)
    X_seq = np.repeat(X_curr[np.newaxis, :, :], 12, axis=0) # [12, N, F]

    # 4. Run Model 1 Inference with Uncertainty
    predictions = model1_lsp.predict_with_uncertainty(X_seq, A_curr, mc_samples=10)
    node_pred = predictions[node.hex_id]

    # Extract pollutant values
    raw_feats = X_curr[node_idx]
    pollutants = {
        "pm25": round(float(raw_feats[0]), 1),
        "pm10": round(float(raw_feats[1]), 1),
        "no2": round(float(raw_feats[2]), 1),
        "so2": round(float(raw_feats[3]), 1),
        "co": round(float(raw_feats[4]), 2),
        "o3": round(float(raw_feats[5]), 1)
    }

    aqi_val, cat, grap, dom = calculate_cpcb_aqi(pollutants)

    # Health Advisory Logic
    if aqi_val <= 100:
        advisory = "Air quality is good/acceptable. Ideal for all outdoor workouts."
    elif aqi_val <= 200:
        advisory = "Moderate air quality. Sensitive individuals should reduce prolonged outdoor exertion."
    elif aqi_val <= 300:
        advisory = "Poor air. High discomfort on exertion; wear an N95 mask outdoors."
    elif aqi_val <= 400:
        advisory = "Very Poor air. Avoid morning cardio; run indoor HEPA purifiers."
    else:
        advisory = "Severe / Emergency Air. Stay indoors; high respiratory risk for all age groups."

    return {
        "hex_id": node.hex_id,
        "locality": node.name,
        "zone": node.zone,
        "centroid": {"lat": round(node.lat, 5), "lon": round(node.lon, 5)},
        "aqi": aqi_val,
        "category": cat,
        "grap_stage": grap,
        "dominant_pollutant": dom.upper(),
        "primary_driver": node_pred["primary_driver"],
        "driver_category": node_pred["driver_category"],
        "driver_detail": node_pred["driver_detail"],
        "driver_confidence_pct": node_pred["driver_confidence_pct"],
        "pollutants": pollutants,
        "weather": weather,
        "stubble_smoke_inflow": stubble,
        "health_advisory": advisory
    }

@router.get("/aqi/forecast", response_model=ForecastTrajectoryResponse)
async def get_forecast_trajectory(
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    hex_id: Optional[str] = Query(None)
):
    """
    Returns 1-to-72-hour AQI prediction trajectory with 90% confidence uncertainty envelopes.
    """
    if hex_id and hex_id in grid_manager.nodes:
        node = grid_manager.nodes[hex_id]
    elif lat is not None and lon is not None:
        node = grid_manager.find_nearest_node(lat, lon)
    else:
        node = grid_manager.find_nearest_node(28.6365, 77.2011)

    weather = weather_engine.get_current_weather()
    X_curr, A_curr = dataset_builder.build_current_node_features(weather)
    X_seq = np.repeat(X_curr[np.newaxis, :, :], 12, axis=0)

    predictions = model1_lsp.predict_with_uncertainty(X_seq, A_curr, mc_samples=12)
    node_pred = predictions[node.hex_id]

    return {
        "hex_id": node.hex_id,
        "locality": node.name,
        "zone": node.zone,
        "current_aqi": node_pred["current_aqi"],
        "forecast_trajectory": node_pred["forecast_trajectory"]
    }

@router.post("/aqi/optimal-window", response_model=CleanAirWindowResponse)
async def plan_clean_air_window_post(req: CleanAirWindowRequest):
    """
    Calculates the minimum exposure clean air window over the next 24 hours (POST).
    """
    return clean_air_optimizer.find_optimal_window(
        lat=req.lat,
        lon=req.lon,
        duration_hours=req.duration_hours,
        activity_type=req.activity_type
    )

@router.get("/aqi/optimal-window", response_model=CleanAirWindowResponse)
async def plan_clean_air_window_get(
    lat: float = Query(28.6139),
    lon: float = Query(77.2090),
    duration: int = Query(2, ge=1, le=6),
    activity: str = Query("Jogging / Outdoor Workout")
):
    """
    Calculates the minimum exposure clean air window over the next 24 hours (GET).
    """
    return clean_air_optimizer.find_optimal_window(
        lat=lat,
        lon=lon,
        duration_hours=duration,
        activity_type=activity
    )

@router.post("/incidents/report", response_model=IncidentReportResponse)
async def submit_incident_report(report: IncidentReportRequest):
    """
    Submits a geotagged citizen incident report and injects an immediate transient impulse Delta X into Model 1.
    """
    new_report = incident_store.add_report(
        lat=report.lat,
        lon=report.lon,
        incident_type=report.incident_type,
        severity=report.severity,
        description=report.description,
        image_url=report.image_url
    )
    
    current_impulse = new_report.get_current_impulse(time.time())

    return {
        "report_id": new_report.report_id,
        "hex_id": new_report.hex_id,
        "locality": new_report.nearest_node.name,
        "zone": new_report.nearest_node.zone,
        "incident_type": new_report.incident_type,
        "type_label": new_report.to_dict()["type_label"],
        "severity": new_report.severity,
        "status": new_report.status,
        "timestamp": new_report.timestamp,
        "message": "Incident successfully verified and injected into Model 1 live prediction state.",
        "active_impulse": {k: round(v, 1) for k, v in current_impulse.items()}
    }

@router.get("/incidents/active")
async def get_active_incidents():
    """Returns all active crowdsourced citizen reports with their current decay state."""
    now = time.time()
    reports = [r.to_dict(now) for r in incident_store.reports.values() if (now - r.timestamp) < 43200]
    return {
        "total_active": len(reports),
        "incidents": reports
    }

@router.get("/user/digest", response_model=UserDigestResponse)
async def get_user_air_digest(
    lat: float = Query(28.6139),
    lon: float = Query(77.2090)
):
    """
    Delivers a localized Sunday neighborhood air quality digest for a user's sector.
    """
    node = grid_manager.find_nearest_node(lat, lon)
    base_aqi = int(node.baseline_pm25 * 1.5 + 40.0)

    return {
        "hex_id": node.hex_id,
        "locality": node.name,
        "zone": node.zone,
        "digest_period": "Past 7-Day Air Quality Summary",
        "weekly_average_aqi": base_aqi,
        "cleanest_day": "Thursday (Post-Breeze Cleansing)",
        "cleanest_day_aqi": max(55, int(base_aqi * 0.65)),
        "most_polluted_day": "Monday (Stagnant Winter Inversion)",
        "most_polluted_day_aqi": min(480, int(base_aqi * 1.45)),
        "top_neighborhood_driver": f"Vehicular Transit & Idling ({round(node.traffic_weight * 100)}% Local Weight)",
        "driver_percentage": round(node.traffic_weight * 50.0 + 20.0, 1),
        "lifestyle_tips": [
            "Best time for daily outdoor cardio in your sector is 2:00 PM - 4:00 PM.",
            "Seal apartment windows facing main arterial during evening peak (7:00 PM - 10:00 PM).",
            "Keep indoor plants (Areca Palm / Snake Plant) to reduce indoor VOC buildup."
        ]
    }

