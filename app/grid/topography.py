"""
Topography, Elevation Resistance, Industrial Zones, and Sinks for Delhi-NCR
"""

import math
import numpy as np
from typing import Dict, List, Tuple
from app.grid.h3_grid import grid_manager, HexagonNode

# Geographic key features for distance calculations
RIDGE_CENTROIDS = [
    {"name": "Southern Ridge (Asola / Tughlaqabad)", "lat": 28.4800, "lon": 77.2300, "height": 310.0, "radius_km": 6.0},
    {"name": "Central Ridge (Dhaula Kuan / Chanakyapuri)", "lat": 28.5900, "lon": 77.1700, "height": 265.0, "radius_km": 3.5},
    {"name": "Northern Ridge (Delhi University / Kamla Nagar)", "lat": 28.6900, "lon": 77.2100, "height": 240.0, "radius_km": 2.5}
]

INDUSTRIAL_CENTROIDS = [
    {"name": "Mayapuri Metal Cluster", "lat": 28.6290, "lon": 77.1310, "intensity": 0.95},
    {"name": "Wazirpur Metal Finishing", "lat": 28.6997, "lon": 77.1654, "intensity": 0.98},
    {"name": "Mundka PVC / Industrial", "lat": 28.6840, "lon": 77.0340, "intensity": 0.92},
    {"name": "Bawana Industrial Estate", "lat": 28.7762, "lon": 77.0510, "intensity": 0.96},
    {"name": "Narela Industrial Estate", "lat": 28.8527, "lon": 77.0927, "intensity": 0.90},
    {"name": "Okhla Phase 1/2/3", "lat": 28.5308, "lon": 77.2712, "intensity": 0.94},
    {"name": "Patparganj Industrial Area", "lat": 28.6237, "lon": 77.2872, "intensity": 0.88},
    {"name": "Sahibabad Industrial Area (Ghaziabad)", "lat": 28.6700, "lon": 77.3700, "intensity": 0.90},
    {"name": "Udyog Vihar (Gurugram)", "lat": 28.5050, "lon": 77.0850, "intensity": 0.85}
]

LANDFILL_CENTROIDS = [
    {"name": "Ghazipur Landfill", "lat": 28.6240, "lon": 77.3280, "emission_intensity": 1.0, "radius_km": 4.0},
    {"name": "Bhalswa Landfill", "lat": 28.7410, "lon": 77.1540, "emission_intensity": 0.95, "radius_km": 4.0},
    {"name": "Okhla Landfill", "lat": 28.5150, "lon": 77.2820, "emission_intensity": 0.90, "radius_km": 3.5}
]

FREIGHT_ARTERIALS = [
    {"name": "Anand Vihar ISBT & Freight Corridor", "lat": 28.6468, "lon": 77.3160, "traffic_load": 1.0},
    {"name": "Punjabi Bagh Rohtak Road Junction", "lat": 28.6683, "lon": 77.1167, "traffic_load": 0.92},
    {"name": "ITO - Vikas Marg Hub", "lat": 28.6318, "lon": 77.2410, "traffic_load": 0.96},
    {"name": "NH-48 Mahipalpur / Airport Corridor", "lat": 28.5450, "lon": 77.1250, "traffic_load": 0.94},
    {"name": "GT Karnal Road / Alipur Arterial", "lat": 28.7972, "lon": 77.1331, "traffic_load": 0.90},
    {"name": "DND Flyway / Ashram Chowk", "lat": 28.5700, "lon": 77.2600, "traffic_load": 0.95},
    {"name": "Peeragarhi Chowk (Outer Ring Road)", "lat": 28.6780, "lon": 77.0920, "traffic_load": 0.91}
]

GREEN_SINKS = [
    {"name": "Asola Bhatti Sanctuary", "lat": 28.4550, "lon": 77.2400, "greenery_score": 0.95, "radius_km": 6.0},
    {"name": "Sanjay Van / Mehrauli Ridge", "lat": 28.5280, "lon": 77.1750, "greenery_score": 0.85, "radius_km": 3.5},
    {"name": "Central Vista & Lodhi Gardens", "lat": 28.5950, "lon": 77.2150, "greenery_score": 0.75, "radius_km": 3.0},
    {"name": "Yamuna Biodiversity Park", "lat": 28.7150, "lon": 77.2300, "greenery_score": 0.80, "radius_km": 3.0}
]

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance in kilometers between two GPS coordinates."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c

class TopographyManager:
    """
    Enriches each Hexagon Node in Delhi-NCR with altitude, industrial proximity,
    traffic artery weights, landfill smoldering risk, and green buffer indices.
    """
    def __init__(self):
        self._enrich_grid_nodes()

    def _enrich_grid_nodes(self):
        for hex_id, node in grid_manager.nodes.items():
            lat, lon = node.lat, node.lon
            
            # 1. Elevation Profile (Plains baseline = 212m, Ridge max = ~310m)
            elevation = 212.0
            for r in RIDGE_CENTROIDS:
                dist = haversine_km(lat, lon, r["lat"], r["lon"])
                if dist < r["radius_km"]:
                    fraction = math.exp(-0.5 * (dist / (r["radius_km"] * 0.5))**2)
                    elevation = max(elevation, 212.0 + (r["height"] - 212.0) * fraction)
            node.elevation = elevation

            # 2. Industrial Activity Weight
            ind_score = 0.05 # urban baseline
            for ind in INDUSTRIAL_CENTROIDS:
                dist = haversine_km(lat, lon, ind["lat"], ind["lon"])
                if dist < 6.0: # within 6 km zone
                    ind_score = max(ind_score, ind["intensity"] * math.exp(-dist / 2.5))
            node.industrial_weight = min(1.0, ind_score)

            # 3. Traffic Arterial Weight
            traffic_score = 0.25 # baseline city traffic
            for art in FREIGHT_ARTERIALS:
                dist = haversine_km(lat, lon, art["lat"], art["lon"])
                if dist < 5.0:
                    traffic_score = max(traffic_score, art["traffic_load"] * math.exp(-dist / 2.0))
            node.traffic_weight = min(1.0, traffic_score)

            # 4. Landfill Proximity and Smoldering Plume Risk
            landfill_score = 0.0
            for lf in LANDFILL_CENTROIDS:
                dist = haversine_km(lat, lon, lf["lat"], lf["lon"])
                if dist < lf["radius_km"]:
                    landfill_score = max(landfill_score, lf["emission_intensity"] * math.exp(-dist / 1.8))
            node.landfill_proximity = min(1.0, landfill_score)

            # 5. Green Buffer / Natural Sink Index
            green_score = 0.12 # dense urban baseline
            for grn in GREEN_SINKS:
                dist = haversine_km(lat, lon, grn["lat"], grn["lon"])
                if dist < grn["radius_km"]:
                    green_score = max(green_score, grn["greenery_score"] * math.exp(-dist / 2.2))
            node.greenery_index = min(1.0, green_score)

            # 6. Adjust baseline pollutant concentrations based on geography
            node.baseline_pm25 = 45.0 + (node.industrial_weight * 70.0) + (node.traffic_weight * 50.0) + (node.landfill_proximity * 60.0) - (node.greenery_index * 30.0)
            node.baseline_pm10 = 90.0 + (node.industrial_weight * 110.0) + (node.traffic_weight * 90.0) + (node.construction_weight * 80.0) - (node.greenery_index * 40.0)

# Initialize topography mapping
topography_manager = TopographyManager()

