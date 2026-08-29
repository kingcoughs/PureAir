"""
Open-Meteo Weather & Boundary Layer Dynamics Ingestion
Fetches or simulates temperature, humidity, wind vectors, PBL height, cloud cover, and rain.
"""

import time
import math
import random
import requests
from typing import Dict, Any, Optional

class WeatherEngine:
    """
    Ingests live meteorology for Delhi-NCR from Open-Meteo API
    with automatic realistic fallback simulation.
    """
    def __init__(self, lat: float = 28.6139, lon: float = 77.2090):
        self.lat = lat
        self.lon = lon
        self.last_fetch_time = 0.0
        self.cached_weather: Dict[str, Any] = {}

    def get_current_weather(self, lat: Optional[float] = None, lon: Optional[float] = None, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Returns atmospheric parameters:
        - temperature (°C)
        - humidity (%)
        - wind_speed (m/s)
        - wind_direction (degrees)
        - pbl_height (meters)
        - ventilation_index (m²/s)
        - cloud_cover (%)
        - rain (mm)
        - condition_label (str)
        """
        now = time.time()
        if not force_refresh and self.cached_weather and (now - self.last_fetch_time < 900): # 15 min cache
            return self.cached_weather

        weather = self._fetch_open_meteo()
        if not weather:
            weather = self._simulate_realistic_delhi_weather()

        self.cached_weather = weather
        self.last_fetch_time = now
        return weather

    def _fetch_open_meteo(self) -> Dict[str, Any] | None:
        """Attempts to fetch live meteorology from Open-Meteo API."""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": self.lat,
                "longitude": self.lon,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "boundary_layer_height"
                ],
                "wind_speed_unit": "ms"
            }
            resp = requests.get(url, params=params, timeout=4.0)
            if resp.status_code == 200:
                data = resp.json()
                current = data.get("current", {})
                
                temp = float(current.get("temperature_2m", 22.0))
                humidity = float(current.get("relative_humidity_2m", 65.0))
                wind_speed = float(current.get("wind_speed_10m", 2.2))
                wind_dir = float(current.get("wind_direction_10m", 310.0)) # Default NW
                cloud_cover = float(current.get("cloud_cover", 20.0))
                rain = float(current.get("precipitation", 0.0))
                
                # Boundary layer height in meters (if unavailable, estimate from solar angle/time)
                pbl = float(current.get("boundary_layer_height", 0.0))
                if pbl <= 0.0:
                    pbl = self._estimate_pbl_height(temp, wind_speed)

                vi = pbl * wind_speed
                return {
                    "temperature": round(temp, 1),
                    "humidity": round(humidity, 1),
                    "wind_speed": round(wind_speed, 2),
                    "wind_direction": round(wind_dir, 1),
                    "pbl_height": round(pbl, 1),
                    "ventilation_index": round(vi, 1),
                    "cloud_cover": round(cloud_cover, 1),
                    "rain": round(rain, 2),
                    "source": "Open-Meteo API",
                    "is_inversion_active": vi < 6000.0,
                    "condition_label": self._get_condition_label(wind_speed, pbl, vi)
                }
        except Exception:
            pass
        return None

    def _estimate_pbl_height(self, temp: float, wind_speed: float) -> float:
        """Estimates planetary boundary layer mixing height based on diurnal cycle."""
        # Delhi diurnal cycle: low at night/morning (100-300m), peaks at 2-3 PM (1200-1800m)
        local_hour = (time.localtime().tm_hour + 5.5) % 24 # IST estimate
        if 11 <= local_hour <= 16:
            base_pbl = 1200.0 + 300.0 * math.sin(math.pi * (local_hour - 11) / 5.0)
        elif 6 <= local_hour < 11:
            base_pbl = 250.0 + (local_hour - 6) * 190.0
        else:
            base_pbl = 150.0 + 50.0 * (temp / 30.0)
        return max(80.0, base_pbl + wind_speed * 40.0)

    def _simulate_realistic_delhi_weather(self) -> Dict[str, Any]:
        """Generates authentic Delhi-NCR atmospheric conditions."""
        # Simulated winter/post-monsoon conditions (common high-pollution regime)
        temp = 18.5 + random.uniform(-3.0, 3.0)
        humidity = 68.0 + random.uniform(-10.0, 15.0)
        wind_speed = round(random.uniform(1.2, 3.8), 2)
        wind_dir = round(random.uniform(295.0, 335.0), 1) # Classic North-Westerly plume from Punjab/Haryana
        cloud_cover = round(random.uniform(10.0, 45.0), 1)
        rain = 0.0
        pbl = self._estimate_pbl_height(temp, wind_speed)
        vi = pbl * wind_speed

        return {
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "wind_speed": wind_speed,
            "wind_direction": wind_dir,
            "pbl_height": round(pbl, 1),
            "ventilation_index": round(vi, 1),
            "cloud_cover": cloud_cover,
            "rain": rain,
            "source": "Delhi Atmospheric Simulator",
            "is_inversion_active": vi < 6000.0,
            "condition_label": self._get_condition_label(wind_speed, pbl, vi)
        }

    def _get_condition_label(self, wind_speed: float, pbl: float, vi: float) -> str:
        if vi < 2500.0:
            return "Severe Thermal Inversion (Trapped Box)"
        elif vi < 6000.0:
            return "Moderate Atmospheric Stagnation"
        elif wind_speed > 5.0:
            return "Breezy (High Ventilation)"
        else:
            return "Normal Airflow"

# Global Weather Engine Singleton
weather_engine = WeatherEngine()

