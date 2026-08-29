"""
Spatio-Temporal Dataset Generation and Feature Normalization for Model Training
Builds multi-step graph sequences [Batch, Time, Nodes, Features] and target horizons.
"""

import time
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from app.config import grid_config, model_config, settings
from app.grid.h3_grid import grid_manager
from app.grid.dynamic_graph import dynamic_graph_engine
from app.data.open_meteo import weather_engine
from app.data.cpcb_sensors import sensor_engine
from app.data.stubble_firms import stubble_engine
from app.data.incidents_store import incident_store

class FeatureScaler:
    """Standard Min-Max / Z-Score feature normalizer for neural graph inputs."""
    def __init__(self):
        # Feature order matching settings.FEATURE_NAMES (20 features)
        # Approximate feature scales for standardization
        self.means = np.array([
            120.0, 220.0, 45.0, 18.0, 1.8, 40.0,  # pollutants
            22.0, 60.0, 2.8, 300.0, 600.0, 3500.0, 25.0, 0.1, # weather
            0.5, 0.4, 0.3, 0.2, 0.3, 220.0 # anthropogenic & terrain
        ], dtype=np.float32)
        
        self.stds = np.array([
            80.0, 130.0, 30.0, 15.0, 1.2, 25.0,
            8.0, 20.0, 1.8, 60.0, 500.0, 2800.0, 25.0, 1.0,
            0.3, 0.35, 0.25, 0.3, 0.25, 25.0
        ], dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.means) / (self.stds + 1e-6)

    def inverse_transform(self, X_norm: np.ndarray) -> np.ndarray:
        return (X_norm * (self.stds + 1e-6)) + self.means

scaler = FeatureScaler()

class DatasetBuilder:
    """
    Builds real-time and synthetic historical training datasets
    for the Spatiotemporal Graph Neural Network.
    """
    def __init__(self):
        self.num_nodes = grid_manager.num_nodes
        self.num_features = model_config.NUM_NODE_FEATURES

    def build_current_node_features(
        self,
        weather: Optional[Dict[str, Any]] = None,
        custom_traffic_factor: float = 1.0,
        custom_industrial_factor: float = 1.0,
        custom_construction_factor: float = 1.0,
        custom_dust_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the current physical feature matrix X(t) [N, F] and Adjacency matrix A(t) [N, N].
        Allows $do$-calculus policy modifications (traffic cuts, factory closures, etc.).
        """
        w = weather or weather_engine.get_current_weather()
        stubble_data = stubble_engine.compute_stubble_inflow(w["wind_direction"], w["wind_speed"])
        stubble_inflow = stubble_data["transboundary_pm25_inflow"]
        
        # Sensor readings
        sensors = sensor_engine.get_station_readings(w, stubble_inflow)
        
        # Transient incident impulses
        incident_impulses = incident_store.get_node_incident_impulses()

        N = self.num_nodes
        X = np.zeros((N, self.num_features), dtype=np.float32)

        for i, hex_id in enumerate(grid_manager.hex_ids):
            node = grid_manager.nodes[hex_id]
            impulse = incident_impulses.get(hex_id, {})
            
            # Policy-constrained factors
            traffic = min(1.0, node.traffic_weight * custom_traffic_factor)
            industry = min(1.0, node.industrial_weight * custom_industrial_factor)
            construction = min(1.0, node.construction_weight * custom_construction_factor)
            dust_suppression = custom_dust_factor # e.g. 0.7 for smog guns
            
            # Base pollutant calculation with inversion trapping
            vi = w.get("ventilation_index", 4500.0)
            trap_factor = 1.0 + max(0.0, (6000.0 - vi) / 4500.0)
            
            pm25 = (node.baseline_pm25 * trap_factor * (1.0 + (traffic - 0.3)*0.4 + (industry - 0.2)*0.5) 
                    + stubble_inflow * (1.2 if "North" in node.zone or "West" in node.zone else 0.8)
                    + impulse.get("pm25", 0.0)) * dust_suppression
            
            pm10 = (node.baseline_pm10 * trap_factor * (1.0 + (construction - 0.2)*0.6 + (traffic - 0.3)*0.4)
                    + (stubble_inflow * 0.6)
                    + impulse.get("pm10", 0.0)) * dust_suppression
            
            no2 = (20.0 + traffic * 80.0) * trap_factor + impulse.get("no2", 0.0)
            so2 = (8.0 + industry * 75.0) * trap_factor + impulse.get("so2", 0.0)
            co = (0.8 + traffic * 3.2 + node.landfill_proximity * 2.0) * (trap_factor * 0.8) + impulse.get("co", 0.0)
            o3 = max(10.0, 35.0 + (1.0 - w["cloud_cover"] / 100.0) * 20.0)

            # Feature vector assignment (order defined in settings.FEATURE_NAMES)
            X[i, 0] = max(5.0, pm25)
            X[i, 1] = max(10.0, pm10)
            X[i, 2] = max(5.0, no2)
            X[i, 3] = max(2.0, so2)
            X[i, 4] = max(0.2, co)
            X[i, 5] = max(5.0, o3)
            X[i, 6] = w["temperature"]
            X[i, 7] = w["humidity"]
            X[i, 8] = w["wind_speed"]
            X[i, 9] = w["wind_direction"]
            X[i, 10] = w["pbl_height"]
            X[i, 11] = w["ventilation_index"]
            X[i, 12] = w["cloud_cover"]
            X[i, 13] = w["rain"]
            X[i, 14] = traffic
            X[i, 15] = industry
            X[i, 16] = construction
            X[i, 17] = node.landfill_proximity
            X[i, 18] = node.greenery_index
            X[i, 19] = node.elevation

        # Compute dynamic adjacency A(t)
        A = dynamic_graph_engine.compute_adjacency(
            wind_speed_ms=w["wind_speed"],
            wind_direction_deg=w["wind_direction"],
            pbl_height_m=w["pbl_height"]
        )

        return X, A

    def generate_synthetic_training_batch(
        self,
        batch_size: int = 16,
        seq_len: int = 12
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates simulated spatiotemporal training batches:
        - X_seq: [Batch, seq_len, N, F]
        - A_seq: [Batch, N, N]
        - Y_targets: [Batch, Num_Horizons, N] (Predicted AQI for 1, 3, 6, 12, 24, 48, 72h)
        """
        N = self.num_nodes
        F = self.num_features
        H = len(model_config.FORECAST_HORIZONS)

        X_batch = np.zeros((batch_size, seq_len, N, F), dtype=np.float32)
        A_batch = np.zeros((batch_size, N, N), dtype=np.float32)
        Y_batch = np.zeros((batch_size, H, N), dtype=np.float32)

        for b in range(batch_size):
            # Sample random meteorological regime (winter inversion vs summer breezy)
            is_winter = random.random() > 0.3
            wind_dir = random.uniform(280.0, 340.0) if is_winter else random.uniform(60.0, 200.0)
            wind_spd = random.uniform(1.2, 3.2) if is_winter else random.uniform(3.5, 7.5)
            pbl_base = random.uniform(120.0, 450.0) if is_winter else random.uniform(800.0, 1600.0)
            stubble = random.uniform(25.0, 95.0) if (is_winter and wind_dir > 290) else 0.0

            for t in range(seq_len):
                diurnal_shift = math.sin(math.pi * (t % 24) / 12.0)
                curr_pbl = max(100.0, pbl_base + diurnal_shift * 300.0)
                curr_vi = curr_pbl * wind_spd
                
                weather = {
                    "temperature": 16.0 + diurnal_shift * 8.0,
                    "humidity": 70.0 - diurnal_shift * 25.0,
                    "wind_speed": wind_spd + random.uniform(-0.4, 0.4),
                    "wind_direction": wind_dir + random.uniform(-10.0, 10.0),
                    "pbl_height": curr_pbl,
                    "ventilation_index": curr_vi,
                    "cloud_cover": random.uniform(10.0, 40.0),
                    "rain": 0.0
                }
                X_t, A_t = self.build_current_node_features(weather)
                X_batch[b, t] = X_t
                if t == seq_len - 1:
                    A_batch[b] = A_t

            # Generate target AQI for each forecast horizon H = [1, 3, 6, 12, 24, 48, 72]
            last_pm25 = X_batch[b, -1, :, 0]
            for h_idx, horizon in enumerate(model_config.FORECAST_HORIZONS):
                # Physics forward evolution: diurnal cycle + advection dispersion
                horizon_trend = math.sin(math.pi * ((seq_len + horizon) % 24) / 12.0)
                # PM2.5 to AQI conversion approximation
                future_pm25 = last_pm25 * (1.0 + horizon_trend * 0.25) + random.uniform(-15.0, 15.0)
                future_aqi = np.clip(future_pm25 * 1.6 + 30.0, 30.0, 500.0)
                Y_batch[b, h_idx] = future_aqi

        return X_batch, A_batch, Y_batch

dataset_builder = DatasetBuilder()

