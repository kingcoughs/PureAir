"""
Uber H3 Spatial Hexagon Discretization for Delhi-NCR
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
try:
    import h3
except ImportError:
    h3 = None

from app.config import grid_config

# Representative CPCB / DPCC Monitoring Stations and Zones in Delhi-NCR
DELHI_HOTSPOTS = [
    {"id": "anand_vihar", "name": "Anand Vihar", "lat": 28.6468, "lon": 77.3160, "type": "traffic_industrial", "zone": "East Delhi"},
    {"id": "punjabi_bagh", "name": "Punjabi Bagh", "lat": 28.6683, "lon": 77.1167, "type": "traffic_residential", "zone": "West Delhi"},
    {"id": "rk_puram", "name": "R.K. Puram", "lat": 28.5632, "lon": 77.1869, "type": "residential_arterial", "zone": "South Delhi"},
    {"id": "dwarka_sec8", "name": "Dwarka Sector 8", "lat": 28.5710, "lon": 77.0694, "type": "residential_airport", "zone": "South West Delhi"},
    {"id": "mundka", "name": "Mundka Industrial", "lat": 28.6840, "lon": 77.0340, "type": "heavy_industrial", "zone": "West Delhi"},
    {"id": "wazirpur", "name": "Wazirpur Industrial", "lat": 28.6997, "lon": 77.1654, "type": "heavy_industrial", "zone": "North West Delhi"},
    {"id": "okhla_ph2", "name": "Okhla Phase 2", "lat": 28.5308, "lon": 77.2712, "type": "industrial_landfill", "zone": "South East Delhi"},
    {"id": "igi_airport", "name": "IGI Airport T3", "lat": 28.5562, "lon": 77.0999, "type": "aviation_corridor", "zone": "South West Delhi"},
    {"id": "ito_bsz", "name": "ITO - BSZ Marg", "lat": 28.6318, "lon": 77.2410, "type": "dense_traffic", "zone": "Central Delhi"},
    {"id": "rohini", "name": "Rohini Sector 16", "lat": 28.7325, "lon": 77.1190, "type": "residential_dense", "zone": "North West Delhi"},
    {"id": "bawana", "name": "Bawana Industrial", "lat": 28.7762, "lon": 77.0510, "type": "heavy_industrial", "zone": "North Delhi"},
    {"id": "narela", "name": "Narela", "lat": 28.8527, "lon": 77.0927, "type": "stubble_inflow_boundary", "zone": "North Delhi"},
    {"id": "patparganj", "name": "Patparganj Industrial", "lat": 28.6237, "lon": 77.2872, "type": "industrial_highway", "zone": "East Delhi"},
    {"id": "shadipur", "name": "Shadipur", "lat": 28.6515, "lon": 77.1581, "type": "industrial_dense", "zone": "West Delhi"},
    {"id": "jahangirpuri", "name": "Jahangirpuri", "lat": 28.7328, "lon": 77.1706, "type": "traffic_landfill_adjacent", "zone": "North Delhi"},
    {"id": "sonia_vihar", "name": "Sonia Vihar", "lat": 28.7105, "lon": 77.2494, "type": "riverbed_basin", "zone": "North East Delhi"},
    {"id": "mandir_marg", "name": "Mandir Marg (CP)", "lat": 28.6365, "lon": 77.2011, "type": "commercial_central", "zone": "Central Delhi"},
    {"id": "lodhi_road", "name": "Lodhi Road", "lat": 28.5883, "lon": 77.2215, "type": "green_buffer_institutional", "zone": "South Delhi"},
    {"id": "alipur", "name": "Alipur GT Karnal", "lat": 28.7972, "lon": 77.1331, "type": "freight_highway", "zone": "North Delhi"},
    {"id": "ashok_vihar", "name": "Ashok Vihar", "lat": 28.6954, "lon": 77.1817, "type": "dense_residential", "zone": "North Delhi"},
    {"id": "vivek_vihar", "name": "Vivek Vihar", "lat": 28.6723, "lon": 77.3153, "type": "residential_border", "zone": "East Delhi"},
    {"id": "ghazipur_landfill", "name": "Ghazipur Landfill Area", "lat": 28.6240, "lon": 77.3280, "type": "smoldering_landfill", "zone": "East Delhi"},
    {"id": "bhalswa_landfill", "name": "Bhalswa Landfill Area", "lat": 28.7410, "lon": 77.1540, "type": "smoldering_landfill", "zone": "North Delhi"},
    {"id": "mayapuri", "name": "Mayapuri Metal Scrap", "lat": 28.6290, "lon": 77.1310, "type": "heavy_metal_industrial", "zone": "West Delhi"},
    {"id": "asola_bhatti", "name": "Asola Bhatti Sanctuary", "lat": 28.4550, "lon": 77.2400, "type": "green_forest_sink", "zone": "South Delhi"},
    {"id": "sanjay_van", "name": "Sanjay Van / Ridge", "lat": 28.5280, "lon": 77.1750, "type": "green_forest_sink", "zone": "South Delhi"},
    {"id": "noida_sec62", "name": "Noida Sector 62", "lat": 28.6270, "lon": 77.3620, "type": "tech_corridor_border", "zone": "Gautam Buddha Nagar"},
    {"id": "gurugram_cybercity", "name": "Gurugram CyberCity", "lat": 28.4950, "lon": 77.0890, "type": "commercial_highway", "zone": "Gurugram"},
    {"id": "gurugram_sec51", "name": "Gurugram Sector 51", "lat": 28.4320, "lon": 77.0720, "type": "residential_arterial", "zone": "Gurugram"},
    {"id": "faridabad_sec16", "name": "Faridabad Sector 16A", "lat": 28.4110, "lon": 77.3180, "type": "industrial_residential", "zone": "Faridabad"},
    {"id": "ghaziabad_vasundhara", "name": "Ghaziabad Vasundhara", "lat": 28.6600, "lon": 77.3570, "type": "border_freight", "zone": "Ghaziabad"}
]

class HexagonNode:
    def __init__(self, hex_id: str, lat: float, lon: float, boundary: List[Tuple[float, float]], name: str, zone: str, node_type: str):
        self.hex_id = hex_id
        self.lat = lat
        self.lon = lon
        self.boundary = boundary # list of (lat, lon)
        self.name = name
        self.zone = zone
        self.node_type = node_type
        
        # Environmental and topographical parameters (populated by TopographyManager)
        self.elevation: float = 215.0 # meters above sea level
        self.industrial_weight: float = 0.1
        self.traffic_weight: float = 0.3
        self.construction_weight: float = 0.1
        self.landfill_proximity: float = 0.0
        self.greenery_index: float = 0.2
        self.baseline_pm25: float = 80.0
        self.baseline_pm10: float = 140.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hex_id": self.hex_id,
            "name": self.name,
            "zone": self.zone,
            "node_type": self.node_type,
            "centroid": {"lat": round(self.lat, 5), "lon": round(self.lon, 5)},
            "boundary": [{"lat": round(p[0], 5), "lon": round(p[1], 5)} for p in self.boundary],
            "elevation_m": round(self.elevation, 1),
            "industrial_weight": round(self.industrial_weight, 2),
            "traffic_weight": round(self.traffic_weight, 2),
            "construction_weight": round(self.construction_weight, 2),
            "landfill_proximity": round(self.landfill_proximity, 2),
            "greenery_index": round(self.greenery_index, 2)
        }

class DelhiNCRGridManager:
    """
    Manages the H3 Hexagonal Grid partitioning over Delhi-NCR.
    """
    def __init__(self, resolution: int = grid_config.H3_RESOLUTION):
        self.resolution = resolution
        self.nodes: Dict[str, HexagonNode] = {}
        self.hex_ids: List[str] = []
        self.coords: np.ndarray = np.empty((0, 2)) # [N, 2] lat, lon
        self._initialize_grid()

    def _generate_synthetic_hex_boundary(self, lat: float, lon: float, radius_km: float = 1.25) -> List[Tuple[float, float]]:
        """Generates standard 6-sided polygon vertices given center and radius."""
        points = []
        lat_deg_per_km = 1.0 / 110.574
        lon_deg_per_km = 1.0 / (111.320 * math.cos(math.radians(lat)))
        for i in range(6):
            angle_rad = math.radians(60 * i + 30)
            dx = radius_km * math.cos(angle_rad)
            dy = radius_km * math.sin(angle_rad)
            plat = lat + dy * lat_deg_per_km
            plon = lon + dx * lon_deg_per_km
            points.append((plat, plon))
        return points

    def _initialize_grid(self):
        """Generates resolution 7 H3 hexagons covering Delhi-NCR."""
        used_hex_ids = set()
        
        # 1. Seed hotspots first
        for item in DELHI_HOTSPOTS:
            lat, lon = item["lat"], item["lon"]
            if h3 is not None:
                try:
                    # h3 v4.x
                    if hasattr(h3, 'latlng_to_cell'):
                        h_id = h3.latlng_to_cell(lat, lon, self.resolution)
                        boundary = h3.cell_to_boundary(h_id)
                        c_lat, c_lon = h3.cell_to_latlng(h_id)
                    else:
                        h_id = h3.geo_to_h3(lat, lon, self.resolution)
                        boundary = h3.h3_to_geo_boundary(h_id)
                        c_lat, c_lon = h3.h3_to_geo(h_id)
                except Exception:
                    h_id = f"hex_seed_{item['id']}"
                    c_lat, c_lon = lat, lon
                    boundary = self._generate_synthetic_hex_boundary(lat, lon)
            else:
                h_id = f"hex_seed_{item['id']}"
                c_lat, c_lon = lat, lon
                boundary = self._generate_synthetic_hex_boundary(lat, lon)

            if h_id not in used_hex_ids:
                used_hex_ids.add(h_id)
                node = HexagonNode(
                    hex_id=h_id,
                    lat=c_lat,
                    lon=c_lon,
                    boundary=boundary,
                    name=item["name"],
                    zone=item["zone"],
                    node_type=item["type"]
                )
                self.nodes[h_id] = node

        # 2. Fill the Delhi-NCR bounding box with a regular lattice of hexagons
        lat_steps = np.linspace(grid_config.MIN_LAT, grid_config.MAX_LAT, 11)
        lon_steps = np.linspace(grid_config.MIN_LON, grid_config.MAX_LON, 10)
        
        for lat in lat_steps:
            for lon in lon_steps:
                if h3 is not None:
                    try:
                        if hasattr(h3, 'latlng_to_cell'):
                            h_id = h3.latlng_to_cell(lat, lon, self.resolution)
                            c_lat, c_lon = h3.cell_to_latlng(h_id)
                            boundary = h3.cell_to_boundary(h_id)
                        else:
                            h_id = h3.geo_to_h3(lat, lon, self.resolution)
                            c_lat, c_lon = h3.h3_to_geo(h_id)
                            boundary = h3.h3_to_geo_boundary(h_id)
                    except Exception:
                        h_id = f"hex_{round(lat, 3)}_{round(lon, 3)}"
                        c_lat, c_lon = lat, lon
                        boundary = self._generate_synthetic_hex_boundary(lat, lon)
                else:
                    h_id = f"hex_{round(lat, 3)}_{round(lon, 3)}"
                    c_lat, c_lon = lat, lon
                    boundary = self._generate_synthetic_hex_boundary(lat, lon)

                if h_id not in used_hex_ids:
                    used_hex_ids.add(h_id)
                    # Assign a descriptive zone
                    zone = self._estimate_zone(c_lat, c_lon)
                    node = HexagonNode(
                        hex_id=h_id,
                        lat=c_lat,
                        lon=c_lon,
                        boundary=boundary,
                        name=f"{zone} Sector ({round(c_lat, 2)}N, {round(c_lon, 2)}E)",
                        zone=zone,
                        node_type="urban_ambient"
                    )
                    self.nodes[h_id] = node

        self.hex_ids = list(self.nodes.keys())
        self.coords = np.array([[node.lat, node.lon] for node in self.nodes.values()])

    def _estimate_zone(self, lat: float, lon: float) -> str:
        """Estimates Delhi-NCR administrative sub-zone based on geographic coordinates."""
        if lon > 77.30 and lat < 28.65:
            return "Noida / East NCR"
        elif lon > 77.30 and lat >= 28.65:
            return "Ghaziabad / Trans-Yamuna"
        elif lon < 77.10 and lat < 28.52:
            return "Gurugram / South-West NCR"
        elif lat < 28.50 and lon >= 77.25:
            return "Faridabad / South NCR"
        elif lat >= 28.72:
            return "North Delhi Airshed"
        elif lon <= 77.10:
            return "West Delhi Airshed"
        elif lat <= 28.58:
            return "South Delhi Airshed"
        else:
            return "Central Delhi Airshed"

    @property
    def num_nodes(self) -> int:
        return len(self.hex_ids)

    def get_node(self, hex_id: str) -> Optional[HexagonNode]:
        return self.nodes.get(hex_id)

    def find_nearest_node(self, lat: float, lon: float) -> HexagonNode:
        """Finds the closest hexagon node to any arbitrary GPS lat/lon."""
        distances = np.hypot(self.coords[:, 0] - lat, self.coords[:, 1] - lon)
        nearest_idx = int(np.argmin(distances))
        nearest_hex_id = self.hex_ids[nearest_idx]
        return self.nodes[nearest_hex_id]

    def get_all_nodes_dict(self) -> List[Dict[str, Any]]:
        return [node.to_dict() for node in self.nodes.values()]

# Global Singleton Instance
grid_manager = DelhiNCRGridManager()

