"""
Configuration and Physical Parameters for Delhi-NCR AQI AI Engine
"""

import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
STATIC_DIR = BASE_DIR / "app" / "static"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

class GridConfig(BaseModel):
    # Delhi-NCR Bounding Box
    MIN_LAT: float = 28.38
    MAX_LAT: float = 28.88
    MIN_LON: float = 76.84
    MAX_LON: float = 77.38
    
    # H3 Spatial Resolution (Res 7: ~5.16 km² per hexagon)
    H3_RESOLUTION: int = 7
    
    # Atmospheric Physics Constants
    KY_DISPERSION: float = 0.15           # Pasquill-Gifford dispersion parameter
    TAU_DECAY: float = 3600.0             # Particulate settling decay in seconds (1 hour)
    TAU_HEIGHT: float = 50.0              # Topographic elevation decay parameter (50 meters)
    VI_CRITICAL: float = 6000.0           # Critical Ventilation Index threshold (m²/s) for inversion lid
    
    # Clean air baseline for Integrated Gradients (µg/m³ for PM2.5, PM10, etc.)
    CLEAN_BASELINE_PM25: float = 15.0
    CLEAN_BASELINE_PM10: float = 25.0

class ModelConfig(BaseModel):
    NUM_NODE_FEATURES: int = 20
    HIDDEN_DIM: int = 64
    GNN_LAYERS: int = 2
    GRU_HIDDEN_DIM: int = 64
    FORECAST_HORIZONS: list[int] = [1, 3, 6, 12, 24, 48, 72] # hours
    DROPOUT: float = 0.15
    MC_SAMPLES: int = 15                  # Monte Carlo dropout passes for 90% confidence envelope
    LEARNING_RATE: float = 0.003
    WEIGHT_DECAY: float = 1e-4
    EPOCHS: int = 30
    BATCH_SIZE: int = 16
    SEQUENCE_LENGTH: int = 12             # 12 past steps (e.g. 12 past hours or 15-min intervals)

class AppSettings(BaseModel):
    APP_NAME: str = "Delhi-NCR AI Air Quality Engine (PureAir®)"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = int(os.environ.get("PORT", 8000))
    DEBUG: bool = False
    
    # Pollutant feature names
    FEATURE_NAMES: list[str] = [
        "pm25", "pm10", "no2", "so2", "co", "o3",
        "temperature", "humidity", "wind_speed", "wind_direction", "pbl_height", "ventilation_index", "cloud_cover", "rain",
        "traffic_density", "industrial_activity", "construction_activity", "landfill_proximity", "greenery_index", "elevation"
    ]
    
    # Category mappings for Model 2 Integrated Gradients source apportionment
    FACTOR_CATEGORIES: dict[str, list[str]] = {
        "Vehicular Traffic": ["traffic_density", "no2", "co"],
        "Stubble Burning / Inflow": ["pm25", "wind_direction", "wind_speed"],
        "Industrial Boilers & Plants": ["industrial_activity", "so2"],
        "Road & Construction Dust": ["construction_activity", "pm10"],
        "Atmospheric Inversion & Trapping": ["pbl_height", "ventilation_index", "temperature", "humidity", "cloud_cover"],
        "Landfills & Smoldering": ["landfill_proximity"],
        "Topography & Green Buffers": ["elevation", "greenery_index", "rain"]
    }

grid_config = GridConfig()
model_config = ModelConfig()
settings = AppSettings()

