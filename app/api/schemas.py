"""
Pydantic Request & Response Schemas for Citizen and Government API Endpoints
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

# ==================== CITIZEN SCHEMAS ====================

class PollutantBreakdown(BaseModel):
    pm25: float = Field(..., description="PM2.5 concentration in µg/m³")
    pm10: float = Field(..., description="PM10 concentration in µg/m³")
    no2: float = Field(..., description="NO2 concentration in µg/m³")
    so2: float = Field(..., description="SO2 concentration in µg/m³")
    co: float = Field(..., description="CO concentration in mg/m³")
    o3: float = Field(..., description="Ozone concentration in µg/m³")

class LiveAQIResponse(BaseModel):
    hex_id: str
    locality: str
    zone: str
    centroid: Dict[str, float]
    aqi: int
    category: str
    grap_stage: str
    dominant_pollutant: str
    primary_driver: str
    driver_category: str
    driver_detail: str
    driver_confidence_pct: float
    pollutants: PollutantBreakdown
    weather: Dict[str, Any]
    stubble_smoke_inflow: Dict[str, Any]
    health_advisory: str

class ForecastStep(BaseModel):
    horizon_hours: int
    predicted_aqi: int
    lower_ci_90: int
    upper_ci_90: int
    uncertainty_range: int

class ForecastTrajectoryResponse(BaseModel):
    hex_id: str
    locality: str
    zone: str
    current_aqi: int
    forecast_trajectory: List[ForecastStep]

class CleanAirWindowRequest(BaseModel):
    lat: float = Field(28.6139, description="Target Latitude")
    lon: float = Field(77.2090, description="Target Longitude")
    duration_hours: int = Field(2, ge=1, le=6, description="Duration in hours (1-6)")
    activity_type: str = Field("Jogging / Exercise", description="Activity name")

class CleanAirWindowResponse(BaseModel):
    requested_location: Dict[str, Any]
    planned_duration_hours: int
    activity_type: str
    optimal_window: Dict[str, Any]
    worst_exposure_window: Dict[str, Any]
    hourly_24h_curve: List[Dict[str, Any]]

class IncidentReportRequest(BaseModel):
    lat: float = Field(..., description="Geotagged Latitude")
    lon: float = Field(..., description="Geotagged Longitude")
    incident_type: str = Field(..., description="garbage_burning | construction_dust | industrial_exhaust | road_dust")
    severity: int = Field(..., ge=1, le=5, description="Severity rating 1 to 5")
    description: str = Field(..., description="Incident description")
    image_url: Optional[str] = Field(None, description="Photo evidence URL")

class IncidentReportResponse(BaseModel):
    report_id: str
    hex_id: str
    locality: str
    zone: str
    incident_type: str
    type_label: str
    severity: int
    status: str
    timestamp: float
    message: str
    active_impulse: Dict[str, float]

class UserDigestResponse(BaseModel):
    hex_id: str
    locality: str
    zone: str
    digest_period: str
    weekly_average_aqi: int
    cleanest_day: str
    cleanest_day_aqi: int
    most_polluted_day: str
    most_polluted_day_aqi: int
    top_neighborhood_driver: str
    driver_percentage: float
    lifestyle_tips: List[str]

# ==================== GOVERNMENT SCHEMAS ====================

class HotspotCausalityItem(BaseModel):
    rank: int
    hex_id: str
    locality: str
    zone: str
    current_aqi: int
    grap_stage: str
    primary_contributor: str
    primary_pct: float
    secondary_contributor: str
    secondary_pct: float
    primary_recommended_action: str

class CausalityMatrixResponse(BaseModel):
    timestamp: float
    active_grap_regime: str
    top_impact_zones: List[HotspotCausalityItem]
    citywide_source_apportionment: Dict[str, float]

class PolicySimulationRequest(BaseModel):
    odd_even_active: bool = Field(False, description="Odd-Even vehicle rationing (50% traffic cut)")
    truck_diversion_active: bool = Field(False, description="Divert heavy freight to peripheral expressways (EPE/WPE)")
    construction_halt_active: bool = Field(False, description="Halt Tier-1 & Tier-2 construction works")
    industrial_curfew_active: bool = Field(False, description="Mandate 50% capacity on industrial boilers")
    smog_guns_units: int = Field(0, ge=0, le=200, description="Number of anti-smog mist cannon units deployed")

class PolicySimulationResponse(BaseModel):
    simulation_timestamp: float
    active_interventions: List[str]
    citywide_summary: Dict[str, Any]
    top_beneficiary_wards: List[Dict[str, Any]]
    all_ward_impacts: List[Dict[str, Any]]

class IncidentTriageResponse(BaseModel):
    total_active_clusters: int
    triage_queue: List[Dict[str, Any]]

class DispatchActionRequest(BaseModel):
    cluster_id: str = Field(..., description="Target Cluster ID to dispatch")

class DispatchActionResponse(BaseModel):
    cluster_id: str
    success: bool
    status: str
    message: str

class WeeklyAuditResponse(BaseModel):
    audit_timestamp: float
    audit_period: str
    executive_summary: str
    citywide_source_apportionment: Dict[str, float]
    model1_performance_metrics: Dict[str, Any]
    top_vulnerable_hotspots: List[Dict[str, Any]]
    policy_recommendations: List[str]

class RetrainRequest(BaseModel):
    epochs: int = Field(10, ge=1, le=50, description="Number of fine-tuning epochs")

class RetrainResponse(BaseModel):
    retraining_timestamp: float
    epochs_trained: int
    pre_training_rmse: float
    post_training_rmse: float
    rmse_improvement_pts: float
    final_r2: float
    status: str

