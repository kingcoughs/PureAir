"""
Crowdsourced Citizen Incident Management, Transient Impulse Injection Engine,
DBSCAN Spatial Clustering, and Government Triage Queue.
"""

import time
import math
import uuid
import numpy as np
from typing import Dict, List, Any, Optional
from app.grid.h3_grid import grid_manager
from app.grid.topography import haversine_km

INCIDENT_EMISSION_PROFILES = {
    "garbage_burning": {
        "weights": {"pm25": 65.0, "pm10": 45.0, "no2": 15.0, "so2": 10.0, "co": 3.2, "o3": 0.0},
        "half_life_hours": 2.5,
        "label": "Open Waste / Garbage Burning"
    },
    "construction_dust": {
        "weights": {"pm25": 30.0, "pm10": 135.0, "no2": 5.0, "so2": 2.0, "co": 0.2, "o3": 0.0},
        "half_life_hours": 12.0,
        "label": "Uncovered Demolition & Construction Dust"
    },
    "industrial_exhaust": {
        "weights": {"pm25": 55.0, "pm10": 40.0, "no2": 45.0, "so2": 60.0, "co": 2.0, "o3": 0.0},
        "half_life_hours": 4.0,
        "label": "Illegal Industrial Chimney Exhaust"
    },
    "road_dust": {
        "weights": {"pm25": 20.0, "pm10": 85.0, "no2": 25.0, "so2": 5.0, "co": 0.5, "o3": 0.0},
        "half_life_hours": 3.0,
        "label": "Resuspended Heavy Road Dust"
    }
}

class IncidentReport:
    def __init__(
        self,
        report_id: str,
        lat: float,
        lon: float,
        incident_type: str,
        severity: int,
        description: str,
        timestamp: float,
        image_url: Optional[str] = None,
        confidence: float = 0.85
    ):
        self.report_id = report_id
        self.lat = lat
        self.lon = lon
        self.incident_type = incident_type
        self.severity = max(1, min(5, int(severity)))
        self.description = description
        self.timestamp = timestamp
        self.image_url = image_url or "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=400"
        self.confidence = confidence
        self.nearest_node = grid_manager.find_nearest_node(lat, lon)
        self.hex_id = self.nearest_node.hex_id
        self.status = "Action Required" # 'Action Required' | 'Dispatched (Enforcement En Route)' | 'Resolved'

    def get_current_impulse(self, current_time: float) -> Dict[str, float]:
        if self.confidence < 0.5:
            return {k: 0.0 for k in ["pm25", "pm10", "no2", "so2", "co", "o3"]}

        profile = INCIDENT_EMISSION_PROFILES.get(self.incident_type, INCIDENT_EMISSION_PROFILES["garbage_burning"])
        lambda_sec = profile["half_life_hours"] * 3600.0
        elapsed_sec = max(0.0, current_time - self.timestamp)
        
        decay = math.exp(-elapsed_sec / lambda_sec)
        if decay < 0.01:
            return {k: 0.0 for k in ["pm25", "pm10", "no2", "so2", "co", "o3"]}

        severity_multiplier = self.severity / 3.0
        impulse = {}
        for pollutant, base_val in profile["weights"].items():
            impulse[pollutant] = base_val * severity_multiplier * decay
        return impulse

    def to_dict(self, current_time: float = None) -> Dict[str, Any]:
        curr_t = current_time or time.time()
        elapsed_min = int((curr_t - self.timestamp) / 60)
        return {
            "report_id": self.report_id,
            "lat": round(self.lat, 5),
            "lon": round(self.lon, 5),
            "hex_id": self.hex_id,
            "locality": self.nearest_node.name,
            "zone": self.nearest_node.zone,
            "incident_type": self.incident_type,
            "type_label": INCIDENT_EMISSION_PROFILES.get(self.incident_type, {}).get("label", self.incident_type),
            "severity": self.severity,
            "description": self.description,
            "reported_ago_mins": elapsed_min,
            "status": self.status,
            "image_url": self.image_url,
            "confidence": round(self.confidence, 2),
            "active_impulse": {k: round(v, 1) for k, v in self.get_current_impulse(curr_t).items()}
        }

class IncidentStore:
    def __init__(self):
        self.reports: Dict[str, IncidentReport] = {}
        self._seed_sample_incidents()

    def _seed_sample_incidents(self):
        now = time.time()
        sample_data = [
            (28.6240, 77.3280, "garbage_burning", 5, "Smoldering plastic waste on Ghazipur landfill southern slope near border drain", now - 1200, "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=400"),
            (28.6250, 77.3290, "garbage_burning", 4, "Dense black smoke plume near Ghazipur Mandi bypass", now - 600, "https://images.unsplash.com/photo-1611273426858-450d8e3c9fce?w=400"),
            (28.6840, 77.0340, "industrial_exhaust", 4, "Illegal chemical plastic furnace exhaust venting in Mundka industrial lane 4", now - 2400, "https://images.unsplash.com/photo-1574943320219-553eb213f72d?w=400"),
            (28.7325, 77.1190, "construction_dust", 3, "Uncovered dry cement and sand transport at Rohini Sec 16 construction site", now - 3600, "https://images.unsplash.com/photo-1504307651254-35680f356dfd?w=400"),
            (28.6468, 77.3160, "road_dust", 4, "Resuspended road dust along Anand Vihar ISBT bus depot entry road", now - 1800, "https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=400")
        ]
        for lat, lon, itype, sev, desc, ts, img in sample_data:
            r_id = f"INC-{str(uuid.uuid4())[:8].upper()}"
            self.reports[r_id] = IncidentReport(
                report_id=r_id,
                lat=lat,
                lon=lon,
                incident_type=itype,
                severity=sev,
                description=desc,
                timestamp=ts,
                image_url=img,
                confidence=0.92
            )

    def add_report(self, lat: float, lon: float, incident_type: str, severity: int, description: str, image_url: Optional[str] = None) -> IncidentReport:
        r_id = f"INC-{str(uuid.uuid4())[:8].upper()}"
        report = IncidentReport(
            report_id=r_id,
            lat=lat,
            lon=lon,
            incident_type=incident_type,
            severity=severity,
            description=description,
            timestamp=time.time(),
            image_url=image_url,
            confidence=0.90
        )
        self.reports[r_id] = report
        return report

    def get_node_incident_impulses(self, current_time: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        curr_t = current_time or time.time()
        node_impulses = {h_id: {k: 0.0 for k in ["pm25", "pm10", "no2", "so2", "co", "o3"]} for h_id in grid_manager.hex_ids}
        for r in self.reports.values():
            if r.hex_id in node_impulses:
                impulse = r.get_current_impulse(curr_t)
                for pollutant, val in impulse.items():
                    node_impulses[r.hex_id][pollutant] += val
        return node_impulses

    def get_clustered_triage_queue(self, eps_km: float = 0.6, current_time: Optional[float] = None) -> List[Dict[str, Any]]:
        curr_t = current_time or time.time()
        active = [r for r in self.reports.values() if (curr_t - r.timestamp) < 43200]
        if not active:
            return []

        clusters = []
        visited = set()

        for i, r1 in enumerate(active):
            if r1.report_id in visited:
                continue
            
            cluster_members = [r1]
            visited.add(r1.report_id)

            for j, r2 in enumerate(active):
                if r2.report_id in visited:
                    continue
                d = haversine_km(r1.lat, r1.lon, r2.lat, r2.lon)
                if d <= eps_km:
                    cluster_members.append(r2)
                    visited.add(r2.report_id)

            total_reports = len(cluster_members)
            avg_lat = sum(m.lat for m in cluster_members) / total_reports
            avg_lon = sum(m.lon for m in cluster_members) / total_reports
            max_severity = max(m.severity for m in cluster_members)
            primary_type = cluster_members[0].incident_type
            nearest_node = grid_manager.find_nearest_node(avg_lat, avg_lon)

            is_dispatched = any("Dispatched" in m.status for m in cluster_members)
            status = "Dispatched (Enforcement En Route)" if is_dispatched else "Action Required"

            if total_reports >= 2 or max_severity >= 4:
                priority = "Priority 1 - High (Confirmed Hotspot)"
            else:
                priority = "Priority 2 - Moderate"

            # Pass all citizen descriptions & landmarks
            descriptions = " | ".join(list(dict.fromkeys(m.description for m in cluster_members if m.description)))
            photo_url = next((m.image_url for m in cluster_members if m.image_url), None)

            clusters.append({
                "cluster_id": f"CLUST-{r1.report_id[-4:]}",
                "report_count": total_reports,
                "lat": round(avg_lat, 5),
                "lon": round(avg_lon, 5),
                "hex_id": nearest_node.hex_id,
                "locality": nearest_node.name,
                "zone": nearest_node.zone,
                "incident_type": primary_type,
                "type_label": INCIDENT_EMISSION_PROFILES.get(primary_type, {}).get("label", primary_type),
                "description": descriptions,
                "image_url": photo_url,
                "max_severity": max_severity,
                "priority": priority,
                "status": status,
                "latest_report_ago_mins": int((curr_t - max(m.timestamp for m in cluster_members)) / 60),
                "reports": [m.to_dict(curr_t) for m in cluster_members]
            })

        # Sort: Action Required first, then high priority
        clusters.sort(key=lambda c: (
            0 if "Action Required" in c["status"] else 1,
            1 if "Priority 1" in c["priority"] else 2,
            -c["report_count"]
        ))
        return clusters

    def dispatch_cluster(self, cluster_id: str) -> bool:
        triage = self.get_clustered_triage_queue()
        for c in triage:
            if c["cluster_id"] == cluster_id:
                for r_dict in c["reports"]:
                    r_id = r_dict["report_id"]
                    if r_id in self.reports:
                        self.reports[r_id].status = "Dispatched (Enforcement En Route)"
                return True
        return False

# Global Singleton
incident_store = IncidentStore()
