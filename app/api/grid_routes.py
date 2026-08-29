"""
Grid & Spatial Topology Endpoints: H3 Hexagons, Dynamic Adjacency Vectors, and Sensor Stations
"""

import time
import numpy as np
from fastapi import APIRouter, Query
from typing import Dict, List, Any

from app.grid.h3_grid import grid_manager
from app.grid.dynamic_graph import dynamic_graph_engine
from app.data.open_meteo import weather_engine
from app.data.cpcb_sensors import sensor_engine
from app.data.stubble_firms import stubble_engine

router = APIRouter(prefix="/api/grid", tags=["Spatial Grid & Topology"])

@router.get("/hexagons")
async def get_grid_hexagons():
    """
    Returns all Uber H3 spatial hexagon nodes with polygon boundary vertices,
    elevation, industrial weights, traffic loads, and landfill proximity.
    """
    return {
        "total_hexagons": grid_manager.num_nodes,
        "resolution": grid_manager.resolution,
        "nodes": grid_manager.get_all_nodes_dict()
    }

@router.get("/adjacency")
async def get_dynamic_adjacency(
    wind_speed: float = Query(2.5, description="Wind speed in m/s"),
    wind_direction: float = Query(315.0, description="Wind direction degrees (where wind is coming from)"),
    pbl_height: float = Query(600.0, description="Planetary Boundary Layer mixing height in meters")
):
    """
    Returns non-zero directed transport edge vectors between hexagons driven by wind advection and terrain.
    """
    A = dynamic_graph_engine.compute_adjacency(
        wind_speed_ms=wind_speed,
        wind_direction_deg=wind_direction,
        pbl_height_m=pbl_height
    )
    
    edges = []
    node_ids = grid_manager.hex_ids
    N = len(node_ids)

    for i in range(N):
        src_node = grid_manager.nodes[node_ids[i]]
        for j in range(N):
            if i == j:
                continue
            weight = float(A[i, j])
            if weight > 0.015: # threshold for visual rendering
                tgt_node = grid_manager.nodes[node_ids[j]]
                edges.append({
                    "source_hex": src_node.hex_id,
                    "target_hex": tgt_node.hex_id,
                    "source_coord": [round(src_node.lat, 4), round(src_node.lon, 4)],
                    "target_coord": [round(tgt_node.lat, 4), round(tgt_node.lon, 4)],
                    "weight": round(weight, 4)
                })

    return {
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "pbl_height": pbl_height,
        "ventilation_index": round(pbl_height * wind_speed, 1),
        "total_active_edges": len(edges),
        "edges": edges[:150] # return top edges for map rendering
    }

@router.get("/stations")
async def get_cpcb_stations():
    """
    Returns live monitoring station readings across all Delhi-NCR CPCB/DPCC ground stations.
    """
    weather = weather_engine.get_current_weather()
    stubble = stubble_engine.compute_stubble_inflow(weather["wind_direction"], weather["wind_speed"])
    stations = sensor_engine.get_station_readings(weather, stubble["transboundary_pm25_inflow"])
    
    return {
        "total_stations": len(stations),
        "weather_summary": weather,
        "stubble_inflow": stubble,
        "stations": list(stations.values())
    }

