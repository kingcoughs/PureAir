"""
CPCB / DPCC Ground Monitoring Stations Ingestion and Indian NAQI Calculation
Calculates official Central Pollution Control Board (CPCB) Sub-Indices and Category Regimes.
"""

import math
import random
from typing import Dict, List, Any, Tuple
from app.grid.h3_grid import grid_manager, DELHI_HOTSPOTS

# CPCB Sub-Index Breakpoint Tables: (C_low, C_high, I_low, I_high)
CPCB_BREAKPOINTS = {
    "pm25": [
        (0.0, 30.0, 0, 50),
        (30.1, 60.0, 51, 100),
        (60.1, 90.0, 101, 200),
        (90.1, 120.0, 201, 300),
        (120.1, 250.0, 301, 400),
        (250.1, 500.0, 401, 500),
    ],
    "pm10": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 250.0, 101, 200),
        (250.1, 350.0, 201, 300),
        (350.1, 430.0, 301, 400),
        (430.1, 600.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 180.0, 101, 200),
        (180.1, 280.0, 201, 300),
        (280.1, 400.0, 301, 400),
        (400.1, 800.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (40.1, 80.0, 51, 100),
        (80.1, 380.0, 101, 200),
        (380.1, 800.0, 201, 300),
        (800.1, 1600.0, 301, 400),
        (1600.1, 2400.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.01, 2.0, 51, 100),
        (2.01, 10.0, 101, 200),
        (10.01, 17.0, 201, 300),
        (17.01, 34.0, 301, 400),
        (34.01, 50.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (50.1, 100.0, 51, 100),
        (100.1, 168.0, 101, 200),
        (168.1, 208.0, 201, 300),
        (208.1, 748.0, 301, 400),
        (748.1, 1000.0, 401, 500),
    ]
}

def calculate_sub_index(conc: float, pollutant: str) -> float:
    """Calculates linear interpolated CPCB sub-index for a single pollutant concentration."""
    breakpoints = CPCB_BREAKPOINTS.get(pollutant.lower())
    if not breakpoints or conc <= 0:
        return 0.0
    
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= conc <= c_high:
            return i_low + ((conc - c_low) / (c_high - c_low)) * (i_high - i_low)
    
    # If concentration exceeds highest bracket
    c_low, c_high, i_low, i_high = breakpoints[-1]
    if conc > c_high:
        extra = ((conc - c_high) / (c_high - c_low)) * (i_high - i_low)
        return min(600.0, i_high + extra)
    return 0.0

def calculate_cpcb_aqi(pollutants: Dict[str, float]) -> Tuple[int, str, str, str]:
    """
    Computes overall CPCB AQI as maximum of sub-indices, along with
    quality category, GRAP Stage, and dominant pollutant.
    """
    sub_indices = {}
    for p, val in pollutants.items():
        if p in CPCB_BREAKPOINTS:
            sub_indices[p] = calculate_sub_index(val, p)

    if not sub_indices:
        return (50, "Good", "Normal", "None")

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    aqi_val = int(round(sub_indices[dominant_pollutant]))

    # Categorization
    if aqi_val <= 50:
        category = "Good"
        grap_stage = "Normal"
    elif aqi_val <= 100:
        category = "Satisfactory"
        grap_stage = "Normal"
    elif aqi_val <= 200:
        category = "Moderate"
        grap_stage = "Normal"
    elif aqi_val <= 300:
        category = "Poor"
        grap_stage = "GRAP-I"
    elif aqi_val <= 400:
        category = "Very Poor"
        grap_stage = "GRAP-II"
    elif aqi_val <= 450:
        category = "Severe"
        grap_stage = "GRAP-III"
    else:
        category = "Severe+"
        grap_stage = "GRAP-IV"

    return aqi_val, category, grap_stage, dominant_pollutant

class CPCBSensorEngine:
    """
    Simulates / Ingests real-time ground monitoring station readings
    across Delhi-NCR and interpolates them onto the H3 Hexagon Grid.
    """
    def __init__(self):
        self.station_cache: Dict[str, Dict[str, Any]] = {}

    def get_station_readings(self, weather_data: Dict[str, Any], stubble_inflow: float = 0.0) -> Dict[str, Dict[str, Any]]:
        """
        Generates/fetches ground readings for all 30+ Delhi-NCR CPCB stations.
        """
        vi = weather_data.get("ventilation_index", 5000.0)
        temp = weather_data.get("temperature", 20.0)
        is_trapped = vi < 6000.0
        
        readings = {}
        for station in DELHI_HOTSPOTS:
            s_id = station["id"]
            node = grid_manager.find_nearest_node(station["lat"], station["lon"])
            
            # Base pollutant levels conditioned on industrial/traffic weight & stagnation
            trap_multiplier = 1.0 + max(0.0, (6000.0 - vi) / 4000.0) if is_trapped else 0.85
            
            # Stubble smoke hits northern and west-facing border stations hardest
            stubble_impact = stubble_inflow * (1.4 if "North" in node.zone or "West" in node.zone else 0.8)
            
            pm25 = max(15.0, (node.baseline_pm25 * trap_multiplier) + stubble_impact + random.uniform(-10.0, 15.0))
            pm10 = max(25.0, (node.baseline_pm10 * trap_multiplier) + (stubble_impact * 0.7) + random.uniform(-15.0, 20.0))
            no2 = max(10.0, (20.0 + node.traffic_weight * 85.0) * trap_multiplier + random.uniform(-5.0, 8.0))
            so2 = max(5.0, (8.0 + node.industrial_weight * 70.0) * trap_multiplier + random.uniform(-3.0, 6.0))
            co = max(0.4, (0.8 + node.traffic_weight * 3.5 + node.landfill_proximity * 2.0) * (trap_multiplier * 0.8) + random.uniform(-0.2, 0.4))
            o3 = max(10.0, 35.0 + random.uniform(-10.0, 20.0))

            pollutants = {
                "pm25": round(pm25, 1),
                "pm10": round(pm10, 1),
                "no2": round(no2, 1),
                "so2": round(so2, 1),
                "co": round(co, 2),
                "o3": round(o3, 1)
            }
            
            aqi, cat, grap, dom = calculate_cpcb_aqi(pollutants)

            readings[s_id] = {
                "station_id": s_id,
                "name": station["name"],
                "zone": station["zone"],
                "type": station["type"],
                "lat": station["lat"],
                "lon": station["lon"],
                "hex_id": node.hex_id,
                "aqi": aqi,
                "category": cat,
                "grap_stage": grap,
                "dominant_pollutant": dom,
                "pollutants": pollutants
            }

        self.station_cache = readings
        return readings

# Global Sensor Engine Singleton
sensor_engine = CPCBSensorEngine()

