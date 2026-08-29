"""
NASA FIRMS / ISRO Satellite Stubble Fire Inflow Modeling
Models regional transboundary biomass plumes from Punjab & Haryana into Delhi-NCR.
"""

import math
import random
from typing import Dict, Any

class StubbleFiresEngine:
    """
    Simulates / Ingests satellite thermal anomalies and computes
    transboundary smoke inflow vectors based on regional wind direction.
    """
    def __init__(self):
        self.active_fire_count = 1450 # Typical autumn peak active fires
        self.burn_intensity = 0.75

    def compute_stubble_inflow(self, wind_direction_deg: float, wind_speed_ms: float) -> Dict[str, Any]:
        """
        Calculates stubble smoke flux reaching Delhi-NCR based on NW wind alignment.
        
        Args:
            wind_direction_deg: Direction wind is coming FROM.
                               Punjab/Haryana is North-West (315°).
            wind_speed_ms: Wind transport velocity.
        """
        # North-Westerly alignment angle difference from 315°
        target_angle = 315.0
        angle_diff = abs((wind_direction_deg - target_angle + 180) % 360 - 180)
        
        # Gaussian directional alignment factor: 1.0 when wind is exactly 315° NW, drops to 0 at East/South
        alignment_factor = math.exp(-0.5 * (angle_diff / 30.0)**2)
        
        # Wind transport velocity factor: very high wind dilutes, very low wind doesn't travel far, sweet spot 2.5 - 5 m/s
        speed_factor = math.exp(-0.5 * ((wind_speed_ms - 3.5) / 2.0)**2)
        
        # Inflow PM2.5 concentration added to Delhi border nodes (µg/m³)
        base_smoke_pm25 = self.active_fire_count * 0.06 * self.burn_intensity
        inflow_pm25 = base_smoke_pm25 * alignment_factor * speed_factor

        return {
            "satellite_fire_count": self.active_fire_count,
            "regional_source": "Punjab / Haryana Agro-Belt",
            "alignment_factor": round(alignment_factor, 3),
            "is_plume_inbound": alignment_factor > 0.45 and inflow_pm25 > 15.0,
            "transboundary_pm25_inflow": round(inflow_pm25, 1),
            "smoke_contribution_pct": round(min(55.0, (inflow_pm25 / (inflow_pm25 + 120.0)) * 100.0), 1)
        }

# Global Singleton
stubble_engine = StubbleFiresEngine()

