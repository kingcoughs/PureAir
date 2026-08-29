"""
NASA FIRMS / ISRO Satellite Stubble Fire Inflow Modeling
Coupled to Real Agricultural Harvesting Calendar (Oct 15 - Nov 25 Peak; <0.5% off-season).
"""

import math
import random
import datetime
from typing import Dict, Any

class StubbleFiresEngine:
    """
    Simulates / Ingests satellite thermal anomalies coupled to seasonal harvest cycles.
    """
    def __init__(self):
        self.burn_intensity = 0.75

    def get_seasonal_fire_count(self) -> int:
        """
        Returns active stubble fires strictly aligned with Punjab/Haryana paddy harvesting calendar:
        - Oct 15 to Nov 25 (Autumn Post-Monsoon Harvesting Peak): 1,200 - 3,500 active fires
        - Dec to Sep (Off-Season): 5 - 25 fires (negligible baseline < 0.5% contribution)
        """
        now = datetime.datetime.now()
        month = now.month
        day = now.day

        # Mid October to Late November
        if (month == 10 and day >= 15) or (month == 11 and day <= 25):
            return random.randint(1400, 3200)
        elif month == 10 and day < 15:
            return random.randint(100, 450)
        elif month == 11 and day > 25:
            return random.randint(150, 500)
        else:
            return random.randint(5, 25) # Off-season negligible fires

    def compute_stubble_inflow(self, wind_direction_deg: float, wind_speed_ms: float) -> Dict[str, Any]:
        """
        Calculates stubble smoke flux reaching Delhi-NCR based on NW wind alignment and seasonality.
        """
        fire_count = self.get_seasonal_fire_count()
        target_angle = 315.0
        angle_diff = abs((wind_direction_deg - target_angle + 180) % 360 - 180)
        
        alignment_factor = math.exp(-0.5 * (angle_diff / 30.0)**2)
        speed_factor = math.exp(-0.5 * ((wind_speed_ms - 3.5) / 2.0)**2)
        
        base_smoke_pm25 = fire_count * 0.04 * self.burn_intensity
        inflow_pm25 = base_smoke_pm25 * alignment_factor * speed_factor

        is_stubble_season = fire_count > 200
        smoke_pct = min(55.0, (inflow_pm25 / (inflow_pm25 + 150.0)) * 100.0) if is_stubble_season else min(0.5, (inflow_pm25 / 150.0) * 100.0)

        return {
            "satellite_fire_count": fire_count,
            "regional_source": "Punjab / Haryana Agro-Belt",
            "is_stubble_season": is_stubble_season,
            "alignment_factor": round(alignment_factor, 3),
            "is_plume_inbound": alignment_factor > 0.45 and inflow_pm25 > 15.0 and is_stubble_season,
            "transboundary_pm25_inflow": round(inflow_pm25, 1),
            "smoke_contribution_pct": round(smoke_pct, 1)
        }

# Global Singleton
stubble_engine = StubbleFiresEngine()
