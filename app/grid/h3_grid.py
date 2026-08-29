"""
Uber H3 Spatial Hexagon Discretization for Full Delhi-NCR Region
Covers Delhi NCT, Noida, Greater Noida, Ghaziabad, Gurugram, Faridabad, Sonipat, and Bahadurgarh.
"""

import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
try:
    import h3
except ImportError:
    h3 = None

from app.config import grid_config

# Comprehensive List of Key Hotspots & Municipal Wards across entire Delhi-NCR
DELHI_NCR_HOTSPOTS = [
    # --- Central & New Delhi ---
    {"id": "connaught_place", "name": "Connaught Place / Central Vista", "lat": 28.6315, "lon": 77.2167, "type": "commercial_central", "zone": "Central Delhi"},
    {"id": "mandir_marg", "name": "Mandir Marg", "lat": 28.6365, "lon": 77.2011, "type": "commercial_dense", "zone": "Central Delhi"},
    {"id": "ito_bsz", "name": "ITO - BSZ Marg", "lat": 28.6318, "lon": 77.2410, "type": "dense_traffic", "zone": "Central Delhi"},
    {"id": "chandni_chowk", "name": "Chandni Chowk / Old Delhi", "lat": 28.6562, "lon": 77.2300, "type": "dense_urban_commercial", "zone": "Central Delhi"},
    
    # --- East Delhi & Trans-Yamuna ---
    {"id": "anand_vihar", "name": "Anand Vihar ISBT & Freight Corridor", "lat": 28.6468, "lon": 77.3160, "type": "traffic_industrial", "zone": "East Delhi"},
    {"id": "patparganj", "name": "Patparganj Industrial Area", "lat": 28.6237, "lon": 77.2872, "type": "industrial_highway", "zone": "East Delhi"},
    {"id": "vivek_vihar", "name": "Vivek Vihar", "lat": 28.6723, "lon": 77.3153, "type": "residential_border", "zone": "East Delhi"},
    {"id": "ghazipur_landfill", "name": "Ghazipur Landfill & Mandi", "lat": 28.6240, "lon": 77.3280, "type": "smoldering_landfill", "zone": "East Delhi"},
    {"id": "mayur_vihar", "name": "Mayur Vihar Phase 1/2", "lat": 28.6050, "lon": 77.2950, "type": "dense_residential", "zone": "East Delhi"},
    {"id": "shahdara", "name": "Shahdara / Dilshad Garden", "lat": 28.6730, "lon": 77.2860, "type": "dense_residential", "zone": "Shahdara"},

    # --- North & North-West Delhi ---
    {"id": "wazirpur", "name": "Wazirpur Industrial Area", "lat": 28.6997, "lon": 77.1654, "type": "heavy_industrial", "zone": "North West Delhi"},
    {"id": "rohini_sec16", "name": "Rohini Sector 16", "lat": 28.7325, "lon": 77.1190, "type": "residential_dense", "zone": "North West Delhi"},
    {"id": "rohini_sec24", "name": "Rohini Sector 24/25", "lat": 28.7180, "lon": 77.0850, "type": "residential_construction", "zone": "North West Delhi"},
    {"id": "ashok_vihar", "name": "Ashok Vihar", "lat": 28.6954, "lon": 77.1817, "type": "dense_residential", "zone": "North Delhi"},
    {"id": "jahangirpuri", "name": "Jahangirpuri", "lat": 28.7328, "lon": 77.1706, "type": "traffic_landfill_adjacent", "zone": "North Delhi"},
    {"id": "bhalswa_landfill", "name": "Bhalswa Landfill & Mukarba", "lat": 28.7410, "lon": 77.1540, "type": "smoldering_landfill", "zone": "North Delhi"},
    {"id": "bawana", "name": "Bawana Industrial Estate", "lat": 28.7762, "lon": 77.0510, "type": "heavy_industrial", "zone": "North Delhi"},
    {"id": "narela", "name": "Narela Agro & Industrial Hub", "lat": 28.8527, "lon": 77.0927, "type": "stubble_inflow_boundary", "zone": "North Delhi"},
    {"id": "alipur", "name": "Alipur GT Karnal Arterial", "lat": 28.7972, "lon": 77.1331, "type": "freight_highway", "zone": "North Delhi"},
    {"id": "sonia_vihar", "name": "Sonia Vihar / Yamuna Floodplain", "lat": 28.7105, "lon": 77.2494, "type": "riverbed_basin", "zone": "North East Delhi"},
    {"id": "burari", "name": "Burari / Yamuna Biodiversity Park", "lat": 28.7550, "lon": 77.2000, "type": "green_buffer", "zone": "North Delhi"},

    # --- West & South-West Delhi ---
    {"id": "punjabi_bagh", "name": "Punjabi Bagh Rohtak Rd", "lat": 28.6683, "lon": 77.1167, "type": "traffic_residential", "zone": "West Delhi"},
    {"id": "shadipur", "name": "Shadipur / Naraina", "lat": 28.6515, "lon": 77.1581, "type": "industrial_dense", "zone": "West Delhi"},
    {"id": "mayapuri", "name": "Mayapuri Scrap & Fabrication", "lat": 28.6290, "lon": 77.1310, "type": "heavy_metal_industrial", "zone": "West Delhi"},
    {"id": "mundka", "name": "Mundka PVC Industrial Cluster", "lat": 28.6840, "lon": 77.0340, "type": "heavy_industrial", "zone": "West Delhi"},
    {"id": "peeragarhi", "name": "Peeragarhi Outer Ring Rd", "lat": 28.6780, "lon": 77.0920, "type": "freight_arterial", "zone": "West Delhi"},
    {"id": "janakpuri", "name": "Janakpuri District Center", "lat": 28.6210, "lon": 77.0870, "type": "residential_arterial", "zone": "West Delhi"},
    {"id": "dwarka_sec8", "name": "Dwarka Sector 8", "lat": 28.5710, "lon": 77.0694, "type": "residential_airport", "zone": "South West Delhi"},
    {"id": "dwarka_sec21", "name": "Dwarka Sector 21 / IICC", "lat": 28.5520, "lon": 77.0580, "type": "transport_hub", "zone": "South West Delhi"},
    {"id": "igi_airport", "name": "IGI Airport Terminal 3", "lat": 28.5562, "lon": 77.0999, "type": "aviation_corridor", "zone": "South West Delhi"},
    {"id": "najafgarh", "name": "Najafgarh / Jharoda Kalan", "lat": 28.6090, "lon": 76.9850, "type": "rural_urban_boundary", "zone": "South West Delhi"},

    # --- South & South-East Delhi ---
    {"id": "rk_puram", "name": "R.K. Puram / Munirka", "lat": 28.5632, "lon": 77.1869, "type": "residential_arterial", "zone": "South Delhi"},
    {"id": "lodhi_road", "name": "Lodhi Road / Central Vista Greens", "lat": 28.5883, "lon": 77.2215, "type": "green_buffer_institutional", "zone": "South Delhi"},
    {"id": "hauz_khas", "name": "Hauz Khas / IIT Delhi", "lat": 28.5450, "lon": 77.1920, "type": "institutional_residential", "zone": "South Delhi"},
    {"id": "sanjay_van", "name": "Sanjay Van / Mehrauli Ridge", "lat": 28.5280, "lon": 77.1750, "type": "green_forest_sink", "zone": "South Delhi"},
    {"id": "asola_bhatti", "name": "Asola Bhatti Wildlife Sanctuary", "lat": 28.4550, "lon": 77.2400, "type": "green_forest_sink", "zone": "South Delhi"},
    {"id": "saket", "name": "Saket District Centre", "lat": 28.5240, "lon": 77.2150, "type": "commercial_dense", "zone": "South Delhi"},
    {"id": "okhla_ph2", "name": "Okhla Phase 2 Industrial", "lat": 28.5308, "lon": 77.2712, "type": "industrial_landfill", "zone": "South East Delhi"},
    {"id": "okhla_landfill", "name": "Okhla Landfill & WTE Plant", "lat": 28.5150, "lon": 77.2820, "type": "smoldering_landfill", "zone": "South East Delhi"},
    {"id": "nehru_nagar", "name": "Nehru Nagar / Ring Road", "lat": 28.5680, "lon": 77.2510, "type": "dense_traffic", "zone": "South East Delhi"},

    # --- Noida & Greater Noida (NCR) ---
    {"id": "noida_sec62", "name": "Noida Sector 62", "lat": 28.6270, "lon": 77.3620, "type": "tech_corridor_border", "zone": "Noida NCR"},
    {"id": "noida_sec1", "name": "Noida Sector 1 / DND Gateway", "lat": 28.5880, "lon": 77.3150, "type": "highway_gateway", "zone": "Noida NCR"},
    {"id": "noida_sec125", "name": "Noida Sector 125 Expressway", "lat": 28.5450, "lon": 77.3320, "type": "expressway_arterial", "zone": "Noida NCR"},
    {"id": "greater_noida_pari_chowk", "name": "Greater Noida (Pari Chowk)", "lat": 28.4650, "lon": 77.5100, "type": "educational_corridor", "zone": "Greater Noida"},
    {"id": "greater_noida_knowledge_park", "name": "Greater Noida Knowledge Park", "lat": 28.4550, "lon": 77.4980, "type": "institutional_corridor", "zone": "Greater Noida"},

    # --- Ghaziabad (NCR) ---
    {"id": "ghaziabad_vasundhara", "name": "Ghaziabad Vasundhara", "lat": 28.6600, "lon": 77.3570, "type": "border_freight", "zone": "Ghaziabad NCR"},
    {"id": "ghaziabad_indirapuram", "name": "Indirapuram / NH-9 Hub", "lat": 28.6410, "lon": 77.3710, "type": "dense_highway", "zone": "Ghaziabad NCR"},
    {"id": "ghaziabad_sahibabad", "name": "Sahibabad Industrial Area", "lat": 28.6700, "lon": 77.3750, "type": "heavy_industrial", "zone": "Ghaziabad NCR"},
    {"id": "ghaziabad_sanjay_nagar", "name": "Ghaziabad Sanjay Nagar", "lat": 28.6910, "lon": 77.4450, "type": "dense_residential", "zone": "Ghaziabad NCR"},

    # --- Gurugram (NCR) ---
    {"id": "gurugram_cybercity", "name": "Gurugram CyberCity (DLF)", "lat": 28.4950, "lon": 77.0890, "type": "commercial_highway", "zone": "Gurugram NCR"},
    {"id": "gurugram_sec51", "name": "Gurugram Sector 51 / Artemis", "lat": 28.4320, "lon": 77.0720, "type": "residential_arterial", "zone": "Gurugram NCR"},
    {"id": "gurugram_udyog_vihar", "name": "Udyog Vihar Phase 1-5", "lat": 28.5080, "lon": 77.0780, "type": "heavy_industrial", "zone": "Gurugram NCR"},
    {"id": "gurugram_manesar", "name": "IMT Manesar Auto Cluster", "lat": 28.3580, "lon": 76.9280, "type": "heavy_manufacturing", "zone": "Gurugram NCR"},
    {"id": "gurugram_golf_course", "name": "Golf Course Extension Rd", "lat": 28.4100, "lon": 77.0980, "type": "construction_corridor", "zone": "Gurugram NCR"},

    # --- Faridabad (NCR) ---
    {"id": "faridabad_sec16", "name": "Faridabad Sector 16A", "lat": 28.4110, "lon": 77.3180, "type": "industrial_residential", "zone": "Faridabad NCR"},
    {"id": "faridabad_nit", "name": "Faridabad NIT Industrial", "lat": 28.3900, "lon": 77.2950, "type": "heavy_industrial", "zone": "Faridabad NCR"},
    {"id": "faridabad_ballabhgarh", "name": "Ballabhgarh Freight Corridor", "lat": 28.3380, "lon": 77.3240, "type": "freight_arterial", "zone": "Faridabad NCR"},

    # --- Sonipat & Bahadurgarh Boundary Corridors ---
    {"id": "sonipat_kundli", "name": "Kundli Industrial Area (Sonipat)", "lat": 28.8780, "lon": 77.1250, "type": "stubble_inflow_boundary", "zone": "Sonipat NCR"},
    {"id": "bahadurgarh_mandi", "name": "Bahadurgarh Bypass / Tikri", "lat": 28.6850, "lon": 76.9250, "type": "freight_highway", "zone": "Jhajjar NCR"}
]

# Alias for backward compatibility
DELHI_HOTSPOTS = DELHI_NCR_HOTSPOTS

class HexagonNode:
    def __init__(self, hex_id: str, lat: float, lon: float, boundary: List[Tuple[float, float]], name: str, zone: str, node_type: str):
        self.hex_id = hex_id
        self.lat = lat
        self.lon = lon
        self.boundary = boundary # list of (lat, lon)
        self.name = name
        self.zone = zone
        self.node_type = node_type
        
        # Environmental and topographical parameters
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
    Manages complete, contiguous H3 Hexagonal Grid partitioning over the entire Delhi-NCR airshed (~200+ nodes).
    """
    def __init__(self, resolution: int = grid_config.H3_RESOLUTION):
        self.resolution = resolution
        self.nodes: Dict[str, HexagonNode] = {}
        self.hex_ids: List[str] = []
        self.coords: np.ndarray = np.empty((0, 2))
        self._initialize_full_grid()

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

    def _initialize_full_grid(self):
        """Generates seamless resolution 7 H3 hexagons covering the whole Delhi-NCR region."""
        used_hex_ids = set()

        # 1. Seed prominent named hotspots first
        for item in DELHI_NCR_HOTSPOTS:
            lat, lon = item["lat"], item["lon"]
            if h3 is not None:
                try:
                    if hasattr(h3, 'latlng_to_cell'):
                        h_id = h3.latlng_to_cell(lat, lon, self.resolution)
                        boundary = h3.cell_to_boundary(h_id)
                        c_lat, c_lon = h3.cell_to_latlng(h_id)
                    else:
                        h_id = h3.geo_to_h3(lat, lon, self.resolution)
                        boundary = h3.h3_to_geo_boundary(h_id)
                        c_lat, c_lon = h3.h3_to_geo(h_id)
                except Exception:
                    h_id = f"hex_hotspot_{item['id']}"
                    c_lat, c_lon = lat, lon
                    boundary = self._generate_synthetic_hex_boundary(lat, lon)
            else:
                h_id = f"hex_hotspot_{item['id']}"
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

        # 2. Fill the expanded Delhi-NCR bounding box with a dense tessellation
        # Expanded bounds: 28.32° to 28.92° Lat (60km N-S) and 76.80° to 77.45° Lon (65km E-W)
        lat_steps = np.linspace(28.33, 28.90, 16)
        lon_steps = np.linspace(76.82, 77.44, 15)

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
                    zone = self._estimate_zone(c_lat, c_lon)
                    name = self._generate_sector_name(c_lat, c_lon, zone)
                    node = HexagonNode(
                        hex_id=h_id,
                        lat=c_lat,
                        lon=c_lon,
                        boundary=boundary,
                        name=name,
                        zone=zone,
                        node_type="urban_ambient"
                    )
                    self.nodes[h_id] = node

        self.hex_ids = list(self.nodes.keys())
        self.coords = np.array([[node.lat, node.lon] for node in self.nodes.values()])

    def _estimate_zone(self, lat: float, lon: float) -> str:
        """Determines regional administrative zone across Greater Delhi-NCR."""
        if lon >= 77.32 and lat <= 28.64:
            return "Noida NCR"
        elif lon >= 77.40 and lat <= 28.52:
            return "Greater Noida"
        elif lon >= 77.33 and lat > 28.64:
            return "Ghaziabad NCR"
        elif lon <= 77.10 and lat <= 28.52:
            return "Gurugram NCR"
        elif lat <= 28.45 and lon >= 77.24:
            return "Faridabad NCR"
        elif lat >= 28.84:
            return "Sonipat NCR"
        elif lon <= 76.96 and 28.60 <= lat <= 28.75:
            return "Bahadurgarh / West NCR"
        elif lat >= 28.72:
            return "North Delhi"
        elif lon <= 77.10:
            return "West Delhi"
        elif lat <= 28.56:
            return "South Delhi"
        elif lon >= 77.28:
            return "East Delhi"
        else:
            return "Central Delhi"

    def _generate_sector_name(self, lat: float, lon: float, zone: str) -> str:
        """Creates a readable landmark or sector name for grid hexagons."""
        # Find closest named hotspot for naming context
        closest_name = "Airshed Sector"
        min_dist = float("inf")
        for h in DELHI_NCR_HOTSPOTS:
            d = math.hypot(h["lat"] - lat, h["lon"] - lon)
            if d < min_dist:
                min_dist = d
                closest_name = h["name"]

        if min_dist < 0.025: # within ~2.5km
            return f"{closest_name} Sector"
        return f"{zone} - Sector ({round(lat, 2)}°N, {round(lon, 2)}°E)"

    @property
    def num_nodes(self) -> int:
        return len(self.hex_ids)

    def get_node(self, hex_id: str) -> Optional[HexagonNode]:
        return self.nodes.get(hex_id)

    def find_nearest_node(self, lat: float, lon: float) -> HexagonNode:
        distances = np.hypot(self.coords[:, 0] - lat, self.coords[:, 1] - lon)
        nearest_idx = int(np.argmin(distances))
        nearest_hex_id = self.hex_ids[nearest_idx]
        return self.nodes[nearest_hex_id]

    def get_all_nodes_dict(self) -> List[Dict[str, Any]]:
        return [node.to_dict() for node in self.nodes.values()]

# Global Singleton Instance
grid_manager = DelhiNCRGridManager()
