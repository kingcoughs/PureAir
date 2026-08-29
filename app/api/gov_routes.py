"""
Government Policy Command Center Endpoints: Causality Matrix, Policy Simulator (do-calculus),
Incident Triage Queue, Integrated Gradients Audit, and Model Retraining.
"""

import time
import numpy as np
from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

from app.grid.h3_grid import grid_manager
from app.data.dataset_builder import dataset_builder
from app.data.incidents_store import incident_store
from app.models.policy_simulator import policy_simulator
from app.models.blame_engine import model2_auditor
from app.models.auditor_model2 import auditor_batch_runner
from app.api.schemas import (
    CausalityMatrixResponse,
    HotspotCausalityItem,
    PolicySimulationRequest,
    PolicySimulationResponse,
    IncidentTriageResponse,
    DispatchActionRequest,
    DispatchActionResponse,
    WeeklyAuditResponse,
    RetrainRequest,
    RetrainResponse
)

router = APIRouter(prefix="/api/gov", tags=["Government Command Center"])

@router.get("/causality-matrix", response_model=CausalityMatrixResponse)
async def get_hotspot_causality_matrix():
    """
    Ranks municipal wards by vulnerability and decomposes raw AQI into
    actionable regulatory categories with specific GRAP enforcement recommendations.
    """
    X_curr, A_curr = dataset_builder.build_current_node_features()
    ig_results = model2_auditor.compute_integrated_gradients(X_curr, A_curr)
    
    node_breakdowns = ig_results["node_breakdowns"]
    city_apportionment = ig_results["citywide_source_apportionment"]

    sorted_nodes = sorted(
        node_breakdowns.values(),
        key=lambda x: x["total_attributed_delta_aqi"],
        reverse=True
    )

    hotspot_items = []
    for rank, item in enumerate(sorted_nodes[:15], start=1):
        hex_id = item["hex_id"]
        node = grid_manager.nodes[hex_id]
        
        curr_aqi = int(round(node.baseline_pm25 * 1.6 + 45.0))
        if curr_aqi <= 200: grap = "Normal"
        elif curr_aqi <= 300: grap = "GRAP-I"
        elif curr_aqi <= 400: grap = "GRAP-II"
        elif curr_aqi <= 450: grap = "GRAP-III"
        else: grap = "GRAP-IV"

        action = model2_auditor._recommend_action(item["primary_blame"], item["name"])

        hotspot_items.append(HotspotCausalityItem(
            rank=rank,
            hex_id=hex_id,
            locality=item["name"],
            zone=item["zone"],
            current_aqi=curr_aqi,
            grap_stage=grap,
            primary_contributor=item["primary_blame"],
            primary_pct=item["primary_blame_pct"],
            secondary_contributor=item["secondary_blame"],
            secondary_pct=item["secondary_blame_pct"],
            tertiary_contributor=item.get("tertiary_blame"),
            tertiary_pct=item.get("tertiary_blame_pct"),
            primary_recommended_action=action
        ))

    max_aqi = max(h.current_aqi for h in hotspot_items) if hotspot_items else 300
    active_regime = "GRAP-IV" if max_aqi > 450 else ("GRAP-III" if max_aqi > 400 else "GRAP-II")

    return {
        "timestamp": time.time(),
        "active_grap_regime": active_regime,
        "top_impact_zones": hotspot_items,
        "citywide_source_apportionment": city_apportionment
    }

@router.post("/simulate-policy", response_model=PolicySimulationResponse)
async def run_counterfactual_policy_simulation(req: PolicySimulationRequest):
    """
    Counterfactual Policy Simulator (do-calculus engine):
    Projects expected delta AQI reduction and time lag either citywide or specifically for a chosen hexagon.
    """
    return policy_simulator.simulate_interventions(
        target_hex_id=req.target_hex_id,
        odd_even_active=req.odd_even_active,
        truck_diversion_active=req.truck_diversion_active,
        construction_halt_active=req.construction_halt_active,
        industrial_curfew_active=req.industrial_curfew_active,
        smog_guns_units=req.smog_guns_units
    )

@router.get("/incidents/triage", response_model=IncidentTriageResponse)
async def get_incident_triage_queue():
    """
    Returns DBSCAN clustered citizen incident reports prioritized for municipal squad dispatch.
    """
    clusters = incident_store.get_clustered_triage_queue()
    return {
        "total_active_clusters": len(clusters),
        "triage_queue": clusters
    }

@router.post("/incidents/dispatch", response_model=DispatchActionResponse)
async def dispatch_incident(req: DispatchActionRequest):
    """
    Dispatches a municipal enforcement squad to the verified incident hotspot.
    """
    success = incident_store.dispatch_cluster(req.cluster_id)
    if not success:
        raise HTTPException(status_code=404, detail="Cluster ID not found in active queue")
    
    return {
        "cluster_id": req.cluster_id,
        "success": True,
        "status": "Dispatched",
        "message": f"Enforcement squad deployed to cluster {req.cluster_id}. Field units dispatched with geo-coordinates."
    }

@router.get("/weekly-audit", response_model=WeeklyAuditResponse)
async def get_weekly_audit():
    """
    Returns Model 2's Integrated Gradients source apportionment audit and GRAP policy recommendations.
    """
    return auditor_batch_runner.run_weekly_audit()

@router.post("/retrain", response_model=RetrainResponse)
async def trigger_model_retraining(req: RetrainRequest):
    """
    Triggers closed-loop active learning fine-tuning of Model 1 using recent residual errors.
    """
    return auditor_batch_runner.trigger_closed_loop_retraining(epochs=req.epochs)
