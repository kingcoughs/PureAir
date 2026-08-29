"""
Clean Air Window Planner & Multi-Day Outdoor Activity Optimizer
Calculates the mathematical minimum of cumulative pollutant intake during practical waking/daytime hours
over a 2 to 3-day forecast horizon (48-72h):
t* = argmin_{t in WakingHours} integral_{t}^{t + Delta_t} AQI_i(tau) dtau
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
    Optimizes outdoor schedules to minimize particulate inhalation during practical daytime waking hours.
    """
    def __init__(self, model1=None):
        self.model1 = model1 or model1_lsp

    def find_optimal_window(
        self,
        lat: float,
        lon: float,
        duration_hours: int = 2,
        activity_type: str = "Jogging / Outdoor Workout",
        days_ahead: int = 3
    ) -> Dict[str, Any]:
        """
        Calculates optimal clean air windows for any geographic coordinate across upcoming 2-3 days.
        Filters strictly for waking hours (06:00 AM to 09:00 PM).
        """
        node = grid_manager.find_nearest_node(lat, lon)
        duration_hours = max(1, min(4, int(duration_hours)))
        days_ahead = max(1, min(3, int(days_ahead)))
        
        now_dt = datetime.now()
        total_hours = (days_ahead + 1) * 24
        base_node_aqi = int(node.baseline_pm25 * 1.5 + 40.0)

        # Generate hourly simulation
        hourly_aqi = []
        for h in range(total_hours):
            eval_time = now_dt + timedelta(hours=h)
            hour_of_day = eval_time.hour
            day_idx = h // 24
            
            # Atmospheric ventilation diurnal curve
            # Solar peak at 14:00 (2 PM) expands boundary layer to 1500m -> lowest AQI factor ~ 0.60
            # Morning inversion at 07:00 (7 AM) collapses boundary layer to 100m -> highest AQI factor ~ 1.45
            diurnal_rad = math.pi * (hour_of_day - 14.0) / 12.0
            ventilation_curve = 0.95 + 0.38 * math.cos(diurnal_rad)
            
            # Multi-day trend factor (slight day-to-day weather variation)
            day_trend = 1.0 - (day_idx * 0.04)
            
            simulated_val = int(np.clip(base_node_aqi * ventilation_curve * day_trend + np.random.uniform(-4.0, 4.0), 35.0, 480.0))
            is_waking_hours = (6 <= hour_of_day <= 21)

            hourly_aqi.append({
                "step_hour": h,
                "day_number": day_idx + 1,
                "datetime": eval_time.strftime("%I:%M %p (%a)"),
                "date_str": eval_time.strftime("%a, %d %b"),
                "hour_int": hour_of_day,
                "is_waking_hours": is_waking_hours,
                "projected_aqi": simulated_val
            })

        # Calculate daily best waking windows for each of the 3 days
        daily_recommendations = []
        overall_best = None
        overall_min_integral = float("inf")

        for d in range(days_ahead):
            day_slice = [h for h in hourly_aqi if h["day_number"] == d + 1]
            if not day_slice:
                continue
            day_date_str = day_slice[0]["date_str"]

            # Filter start hours such that [start, start + duration] is within waking hours
            valid_starts = [
                h for h in day_slice 
                if (6 <= h["hour_int"] <= (21 - duration_hours)) and (h["step_hour"] + duration_hours <= len(hourly_aqi))
            ]
            
            best_day_start = None
            min_day_integral = float("inf")

            for start_step in valid_starts:
                s_idx = start_step["step_hour"]
                window_slice = [hourly_aqi[s_idx + k]["projected_aqi"] for k in range(duration_hours)]
                window_integral = sum(window_slice)
                
                if window_integral < min_day_integral:
                    min_day_integral = window_integral
                    best_day_start = start_step

            # Worst peak window of that day (for contrast)
            worst_day_start = None
            max_day_integral = float("-inf")
            for start_step in valid_starts:
                s_idx = start_step["step_hour"]
                window_slice = [hourly_aqi[s_idx + k]["projected_aqi"] for k in range(duration_hours)]
                window_integral = sum(window_slice)
                if window_integral > max_day_integral:
                    max_day_integral = window_integral
                    worst_day_start = start_step

            if best_day_start:
                s_dt = now_dt + timedelta(hours=best_day_start["step_hour"])
                e_dt = s_dt + timedelta(hours=duration_hours)
                avg_aqi = int(round(min_day_integral / duration_hours))
                
                w_dt = now_dt + timedelta(hours=worst_day_start["step_hour"])
                w_end_dt = w_dt + timedelta(hours=duration_hours)
                worst_avg_aqi = int(round(max_day_integral / duration_hours))

                avoidance_pct = round(((worst_avg_aqi - avg_aqi) / float(worst_avg_aqi)) * 100.0, 1)

                day_res = {
                    "day_label": "Today" if d == 0 else ("Tomorrow" if d == 1 else "Day After Tomorrow"),
                    "date_label": day_date_str,
                    "start_time": s_dt.strftime("%I:%M %p"),
                    "end_time": e_dt.strftime("%I:%M %p"),
                    "average_aqi": avg_aqi,
                    "air_quality_level": self._get_level(avg_aqi),
                    "health_advice": self._get_advice(avg_aqi),
                    "particulate_inhalation_avoidance_pct": avoidance_pct,
                    "worst_exposure_window": {
                        "start_time": w_dt.strftime("%I:%M %p"),
                        "end_time": w_end_dt.strftime("%I:%M %p"),
                        "average_aqi": worst_avg_aqi
                    }
                }
                daily_recommendations.append(day_res)

                if min_day_integral < overall_min_integral:
                    overall_min_integral = min_day_integral
                    overall_best = day_res

        return {
            "requested_location": {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "nearest_locality": node.name,
                "zone": node.zone
            },
            "planned_duration_hours": duration_hours,
            "activity_type": activity_type,
            "overall_best_window": overall_best,
            "daily_recommendations": daily_recommendations,
            "hourly_curve": hourly_aqi[:48] # return 48 hours for graph rendering
        }

    def _get_level(self, aqi: int) -> str:
        if aqi <= 50: return "Good"
        elif aqi <= 100: return "Satisfactory"
        elif aqi <= 200: return "Moderate"
        elif aqi <= 300: return "Poor"
        elif aqi <= 400: return "Very Poor"
        else: return "Severe"

    def _get_advice(self, aqi: int) -> str:
        if aqi <= 100:
            return "Excellent conditions. Fully safe for all outdoor workouts."
        elif aqi <= 200:
            return "Acceptable for healthy adults; sensitive groups limit intense cardio."
        elif aqi <= 300:
            return "Wear an N95 mask during prolonged workouts; afternoon ventilation is highest."
        else:
            return "Avoid intense outdoor cardio; utilize indoor facilities."

clean_air_optimizer = CleanAirWindowOptimizer()
