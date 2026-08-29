"""
Physics-Informed Dynamic Adjacency Matrix Computation A(t)
Implements Pasquill-Gifford Gaussian Plume Advection, Topographic Barrier Decay,
and Planetary Boundary Layer Thermal Inversion Trapping Gate.
"""

import math
import numpy as np
from typing import Tuple, Dict, Any, List
from app.config import grid_config
from app.grid.h3_grid import grid_manager

class DynamicGraphEngine:
    """
    Computes time-varying directed graph adjacency matrices reflecting
    real-time wind transport, terrain elevation barriers, and ventilation dynamics.
    """
    def __init__(self):
        self.num_nodes = grid_manager.num_nodes
        self.node_ids = grid_manager.hex_ids
        self._precompute_spatial_metrics()

    def _precompute_spatial_metrics(self):
        """
        Precomputes inter-node displacement vectors, distances, and elevation deltas.
        """
        N = self.num_nodes
        self.dx = np.zeros((N, N), dtype=np.float32) # East-West displacement in meters
        self.dy = np.zeros((N, N), dtype=np.float32) # North-South displacement in meters
        self.dist = np.zeros((N, N), dtype=np.float32) # Euclidean distance in meters
        self.dh = np.zeros((N, N), dtype=np.float32) # Height difference hj - hi

        coords = grid_manager.coords # [N, 2] (lat, lon)
        nodes = list(grid_manager.nodes.values())
        elevations = np.array([node.elevation for node in nodes], dtype=np.float32)

        for i in range(N):
            lat_i, lon_i = coords[i, 0], coords[i, 1]
            for j in range(N):
                if i == j:
                    continue
                lat_j, lon_j = coords[j, 0], coords[j, 1]
                
                # Convert lat/lon delta to metric displacement (meters)
                mean_lat_rad = math.radians((lat_i + lat_j) / 2.0)
                dy_m = (lat_j - lat_i) * 111139.0
                dx_m = (lon_j - lon_i) * 111320.0 * math.cos(mean_lat_rad)
                
                self.dx[i, j] = dx_m
                self.dy[i, j] = dy_m
                self.dist[i, j] = math.hypot(dx_m, dy_m)
                self.dh[i, j] = elevations[j] - elevations[i]

    def compute_adjacency(
        self,
        wind_speed_ms: float,
        wind_direction_deg: float,
        pbl_height_m: float,
        max_influence_radius_m: float = 35000.0 # 35 km max advection radius
    ) -> np.ndarray:
        """
        Calculates the directed, physics-informed adjacency matrix A(t) [N, N].
        
        Args:
            wind_speed_ms: Wind speed in m/s (e.g., 1.5 - 10.0 m/s)
            wind_direction_deg: Meteorological wind direction (deg 0-360 where wind is coming FROM)
            pbl_height_m: Planetary boundary layer height in meters (e.g., 100m in winter night, 1500m in summer)
            max_influence_radius_m: Cutoff distance for computation efficiency
        """
        N = self.num_nodes
        u_wind = max(0.5, float(wind_speed_ms))
        
        # Wind transport direction is 180 degrees opposite to "wind from" direction
        transport_angle_deg = (float(wind_direction_deg) + 180.0) % 360.0
        transport_rad = math.radians(transport_angle_deg)
        
        # Unit vector for wind transport: w_hat = (cos_theta, sin_theta) where x is East, y is North
        # In meteorology: 0° is North (+y), 90° is East (+x), 180° is South (-y), 270° is West (-x)
        # Math angle from East counter-clockwise: alpha = 90° - transport_angle
        math_angle_rad = math.radians(90.0 - transport_angle_deg)
        w_x = math.cos(math_angle_rad)
        w_y = math.sin(math_angle_rad)

        # 1. Downwind distance: x_parallel = r_ij . w_hat
        x_parallel = self.dx * w_x + self.dy * w_y
        
        # Downwind gate: transport only occurs downwind (x_parallel > 0)
        downwind_mask = (x_parallel > 10.0) & (self.dist <= max_influence_radius_m)

        # 2. Crosswind distance: x_perp = sqrt(max(0, dist^2 - x_parallel^2))
        x_perp_sq = np.maximum(0.0, self.dist**2 - x_parallel**2)

        # 3. Pasquill-Gifford lateral dispersion parameter sigma_y(x_parallel) = ky * (x_parallel)^0.89
        sigma_y = np.maximum(15.0, grid_config.KY_DISPERSION * np.power(np.maximum(10.0, x_parallel), 0.89))
        
        # Gaussian plume lateral spread term: exp(-x_perp^2 / (2 * sigma_y^2))
        lateral_decay = np.exp(-x_perp_sq / (2.0 * (sigma_y**2) + 1e-6))

        # 4. Particulate settling / advection distance decay: exp(-x_parallel / (u_wind * tau_decay))
        advection_decay = np.exp(-x_parallel / (u_wind * grid_config.TAU_DECAY + 1e-6))

        # 5. Topographic ridge barrier decay: exp(-max(0, hj - hi) / tau_h)
        elevation_barrier = np.exp(-np.maximum(0.0, self.dh) / grid_config.TAU_HEIGHT)

        # 6. Thermal Inversion Lid Gate: psi(VI)
        vi = max(100.0, pbl_height_m * u_wind)
        # When VI < 6000 m²/s, atmospheric trapping amplifies ground-level pollutant propagation
        psi_vi = 1.0 + (1.0 / (1.0 + (vi / grid_config.VI_CRITICAL)**2))

        # Combine terms
        A = np.zeros((N, N), dtype=np.float32)
        A[downwind_mask] = (
            lateral_decay[downwind_mask] *
            advection_decay[downwind_mask] *
            elevation_barrier[downwind_mask] *
            psi_vi
        )

        # Add self-loops (local retention)
        np.fill_diagonal(A, 1.0)

        # Row-normalize: A_norm = A_ij / (sum_k A_ik + epsilon)
        row_sums = A.sum(axis=1, keepdims=True) + 1e-7
        A_norm = A / row_sums
        return A_norm

    def get_top_downwind_neighbors(
        self,
        node_hex_id: str,
        wind_speed_ms: float,
        wind_direction_deg: float,
        pbl_height_m: float,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Returns the top downwind recipients of smoke/pollutants from a given source node."""
        if node_hex_id not in grid_manager.nodes:
            return []
        
        idx = self.node_ids.index(node_hex_id)
        A = self.compute_adjacency(wind_speed_ms, wind_direction_deg, pbl_height_m)
        row = A[idx, :]
        
        # Exclude self-loop for ranking
        row_no_self = row.copy()
        row_no_self[idx] = 0.0
        
        top_indices = np.argsort(row_no_self)[::-1][:top_k]
        results = []
        for target_idx in top_indices:
            weight = float(row[target_idx])
            if weight > 0.001:
                target_hex = self.node_ids[target_idx]
                target_node = grid_manager.nodes[target_hex]
                results.append({
                    "target_hex_id": target_hex,
                    "target_name": target_node.name,
                    "target_zone": target_node.zone,
                    "edge_weight": round(weight, 4),
                    "distance_km": round(float(self.dist[idx, target_idx]) / 1000.0, 2)
                })
        return results

# Global Dynamic Graph Instance
dynamic_graph_engine = DynamicGraphEngine()

