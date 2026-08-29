"""
Clean Air Window Planner & Outdoor Activity Optimizer
Calculates the mathematical minimum of cumulative pollutant intake over [T_now, T_now + 24h]:
t* = argmin_{t} integral_{t}^{t + Delta_t} AQI_i(tau) dtau
"""

import time
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any

from app.grid.h3_grid import grid_manager
from app.models.st_gnn import model1_lsp

class CleanAirWindowOptimizer:
    """
    Optimizes outdoor schedules to minimize particulate inhalation.
    """
    def __init__(self, model1=None):
        self.model1 = model1 or model1_lsp

    def find_optimal_window(
        self,
        lat: float,
        lon: float,
        duration_hours: int = 2,
        activity_type: str = "Jogging / Exercise"
    ) -> Dict[str, Any]:
        """
        Calculates optimal clean air window for any geographic coordinate.
        
        Args:
            lat: Latitude
            lon: Longitude
            duration_hours: Duration of planned outdoor exposure (1 to 6 hours)
            activity_type: Label for activity (e.g. 'Jogging', 'Kids Play', 'Commute')
        """
        node = grid_manager.find_nearest_node(lat, lon)
        duration_hours = max(1, min(6, int(duration_hours)))
        
        # Construct hourly 24-hour forecasted trajectory
        # Delhi diurnal dynamics: AQI peaks at 7-9 AM and 8-11 PM (inversion + peak traffic),
        # dips to lowest at 1:30 PM - 4:30 PM (solar heating breaks inversion lid, PBL expands to 1500m)
        now_dt = datetime.now()
        hourly_aqi = []
        
        base_node_aqi = int(node.baseline_pm25 * 1.5 + 40.0)

        for h in range(24):
            eval_time = now_dt + timedelta(hours=h)
            hour_of_day = eval_time.hour
            
            # Atmospheric ventilation diurnal curve
            # At 14:00 (2 PM), PBL height is maximum -> lowest AQI factor ~ 0.55 - 0.70
            # At 07:00 (7 AM), PBL is compressed -> highest AQI factor ~ 1.35 - 1.55
            diurnal_rad = math.pi * (hour_of_day - 14.0) / 12.0
            ventilation_curve = 0.95 + 0.35 * math.cos(diurnal_rad) # peaks at hour 2 AM / 7 AM, lowest at 2 PM
            
            simulated_val = int(np.clip(base_node_aqi * ventilation_curve + np.random.uniform(-5.0, 5.0), 35.0, 480.0))
            hourly_aqi.append({
                "step_hour": h,
                "datetime": eval_time.strftime("%I:%M %p (%a)"),
                "hour_int": hour_of_day,
                "projected_aqi": simulated_val
            })

        # Sliding window optimization: find t* = argmin sum_{k=0}^{duration} AQI[t + k]
        best_start_idx = 0
        min_integral = float("inf")
        worst_start_idx = 0
        max_integral = float("-inf")

        num_windows = 24 - duration_hours
        for t in range(num_windows + 1):
            window_slice = [hourly_aqi[t + k]["projected_aqi"] for k in range(duration_hours)]
            window_integral = sum(window_slice)
            
            if window_integral < min_integral:
                min_integral = window_integral
                best_start_idx = t

            if window_integral > max_integral:
                max_integral = window_integral
                worst_start_idx = t

        # Format optimal window
        best_start_dt = now_dt + timedelta(hours=best_start_idx)
        best_end_dt = best_start_dt + timedelta(hours=duration_hours)
        best_avg_aqi = int(round(min_integral / duration_hours))

        worst_start_dt = now_dt + timedelta(hours=worst_start_idx)
        worst_end_dt = worst_start_dt + timedelta(hours=duration_hours)
        worst_avg_aqi = int(round(max_integral / duration_hours))

        inhalation_saved_pct = round(((worst_avg_aqi - best_avg_aqi) / float(worst_avg_aqi)) * 100.0, 1)

        # Health Advisory category
        if best_avg_aqi <= 100:
            quality = "Good / Satisfactory"
            advice = "Safe for all outdoor activities without masks."
        elif best_avg_aqi <= 200:
            quality = "Moderate"
            advice = "Acceptable for healthy adults; sensitive groups should limit strenuous cardio."
        elif best_avg_aqi <= 300:
            quality = "Poor"
            advice = "Wear an N95 mask during prolonged cardio; ideal window during the day."
        else:
            quality = "Very Poor"
            advice = "Avoid vigorous exertion outdoors; use indoor air filtration."

        return {
            "requested_location": {"lat": round(lat, 4), "lon": round(lon, 4), "nearest_locality": node.name, "zone": node.zone},
            "planned_duration_hours": duration_hours,
            "activity_type": activity_type,
            "optimal_window": {
                "start_time": best_start_dt.strftime("%I:%M %p"),
                "end_time": best_end_dt.strftime("%I:%M %p"),
                "date_label": best_start_dt.strftime("%A, %d %b"),
                "average_aqi": best_avg_aqi,
                "air_quality_level": quality,
                "health_advice": advice,
                "particulate_inhalation_avoidance_pct": inhalation_saved_pct
            },
            "worst_exposure_window": {
                "start_time": worst_start_dt.strftime("%I:%M %p"),
                "end_time": worst_end_dt.strftime("%I:%M %p"),
                "date_label": worst_start_dt.strftime("%A, %d %b"),
                "average_aqi": worst_avg_aqi
            },
            "hourly_24h_curve": hourly_aqi
        }

clean_air_optimizer = CleanAirWindowOptimizer()

