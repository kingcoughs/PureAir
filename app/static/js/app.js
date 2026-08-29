/**
 * Frontend Controller & API Integration Engine for Project Meswak
 */

let map = null;
let hexagonLayerGroup = null;
let edgeLayerGroup = null;
let incidentLayerGroup = null;
let selectedHexId = "anand_vihar";
let currentGridNodes = [];

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initMap();
  loadLiveAtmosphere();
  loadHexagonsAndAdjacency();
  loadCitizenLocality(28.6468, 77.3160); // Anand Vihar initial
  loadGovernmentCausalityMatrix();
  loadIncidentTriage();
  loadWeeklyAudit();
  bindEventHandlers();
});

// 1. Navigation Tab Switching
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
      
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.classList.add("active");
        if (targetId === "tab-map" && map) {
          setTimeout(() => map.invalidateSize(), 200);
        }
      }
    });
  });
}

// 2. Leaflet Map Initialization
function initMap() {
  const mapElement = document.getElementById("airshed-map");
  if (!mapElement) return;

  map = L.map("airshed-map", {
    center: [28.6250, 77.2100],
    zoom: 11,
    minZoom: 10,
    maxZoom: 15
  });

  // Dark Mode CartoDB Tile Layer
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; <a href="https://carto.com/">CARTO</a> | Delhi-NCR Airshed',
    subdomains: "abcd",
    maxZoom: 19
  }).addTo(map);

  hexagonLayerGroup = L.layerGroup().addTo(map);
  edgeLayerGroup = L.layerGroup().addTo(map);
  incidentLayerGroup = L.layerGroup().addTo(map);
}

// 3. Load Live Atmospheric Bar
async function loadLiveAtmosphere() {
  try {
    const res = await fetch("/api/aqi/live");
    if (!res.ok) return;
    const data = await res.json();
    
    document.getElementById("met-temp").textContent = `${data.weather.temperature}°C`;
    document.getElementById("met-wind").textContent = `${data.weather.wind_speed} m/s (${data.weather.wind_direction}°)`;
    document.getElementById("met-pbl").textContent = `${data.weather.pbl_height}m`;
    document.getElementById("met-vi").textContent = `${data.weather.ventilation_index} m²/s`;
    
    const grapBadge = document.getElementById("header-grap-badge");
    if (grapBadge) {
      grapBadge.textContent = data.grap_stage;
    }
    
    const invBadge = document.getElementById("header-inversion-badge");
    if (invBadge) {
      invBadge.style.display = data.weather.is_inversion_active ? "inline-flex" : "none";
    }
  } catch (err) {
    console.error("Failed to load atmospheric state:", err);
  }
}

// Helper: Color for AQI values
function getAQIColor(aqi) {
  if (aqi <= 50) return "#10b981";       // Good
  if (aqi <= 100) return "#84cc16";      // Satisfactory
  if (aqi <= 200) return "#f59e0b";      // Moderate
  if (aqi <= 300) return "#f97316";      // Poor
  if (aqi <= 400) return "#ef4444";      // Very Poor
  if (aqi <= 450) return "#a855f7";      // Severe
  return "#dc2626";                      // Severe+
}

// 4. Load H3 Hexagons & Dynamic Wind Adjacency Vectors
async function loadHexagonsAndAdjacency() {
  try {
    const hexRes = await fetch("/api/grid/hexagons");
    const hexData = await hexRes.json();
    currentGridNodes = hexData.nodes;

    // Populate Locality Select Dropdown
    const locSelect = document.getElementById("citizen-locality-select");
    if (locSelect) {
      locSelect.innerHTML = "";
      currentGridNodes.forEach(n => {
        const opt = document.createElement("option");
        opt.value = `${n.centroid.lat},${n.centroid.lon}`;
        opt.textContent = `${n.name} (${n.zone})`;
        if (n.name.includes("Anand Vihar")) opt.selected = true;
        locSelect.appendChild(opt);
      });
    }

    hexagonLayerGroup.clearLayers();

    // Render H3 Hexagons
    hexData.nodes.forEach(node => {
      const approxAQI = Math.round(node.industrial_weight * 160 + node.traffic_weight * 140 + 120);
      const hexColor = getAQIColor(approxAQI);

      const latlngs = node.boundary.map(p => [p.lat, p.lon]);
      const polygon = L.polygon(latlngs, {
        color: hexColor,
        weight: 1.5,
        fillColor: hexColor,
        fillOpacity: 0.35
      });

      polygon.bindTooltip(`
        <div style="font-size:12px; font-family:sans-serif; color:#fff;">
          <strong>${node.name}</strong><br/>
          Zone: ${node.zone}<br/>
          Est. AQI: <span style="color:${hexColor}; font-weight:bold;">${approxAQI}</span><br/>
          Traffic: ${(node.traffic_weight*100).toFixed(0)}% | Ind: ${(node.industrial_weight*100).toFixed(0)}%
        </div>
      `, { sticky: true });

      polygon.on("click", () => {
        loadCitizenLocality(node.centroid.lat, node.centroid.lon);
        // Switch to Citizen Inspector or update side card
        document.getElementById("map-quick-name").textContent = node.name;
        document.getElementById("map-quick-zone").textContent = node.zone;
        document.getElementById("map-quick-aqi").textContent = approxAQI;
        document.getElementById("map-quick-aqi").style.color = hexColor;
      });

      polygon.addTo(hexagonLayerGroup);
    });

    // Load Dynamic Adjacency Transport Vectors
    const adjRes = await fetch("/api/grid/adjacency");
    const adjData = await adjRes.json();

    edgeLayerGroup.clearLayers();
    adjData.edges.forEach(edge => {
      const line = L.polyline([edge.source_coord, edge.target_coord], {
        color: "#38bdf8",
        weight: Math.min(4, Math.max(1, edge.weight * 25)),
        opacity: Math.min(0.8, edge.weight * 5),
        dashArray: "4, 6"
      });
      line.addTo(edgeLayerGroup);
    });

    // Load Active Incidents onto Map
    loadMapIncidents();

  } catch (err) {
    console.error("Failed loading grid and adjacency:", err);
  }
}

// Load Active Incidents as Map Markers
async function loadMapIncidents() {
  try {
    const res = await fetch("/api/incidents/active");
    const data = await res.json();
    incidentLayerGroup.clearLayers();

    data.incidents.forEach(inc => {
      const iconHtml = inc.incident_type === "garbage_burning" ? "🔥" :
                       (inc.incident_type === "construction_dust" ? "🏗️" : "🏭");
      
      const customIcon = L.divIcon({
        html: `<div style="font-size:22px; filter:drop-shadow(0 0 6px rgba(239,68,68,0.8)); cursor:pointer;">${iconHtml}</div>`,
        className: "incident-map-icon",
        iconSize: [26, 26]
      });

      const marker = L.marker([inc.lat, inc.lon], { icon: customIcon });
      marker.bindPopup(`
        <div style="font-size:12px; color:#111827;">
          <strong style="color:#ef4444;">${inc.type_label}</strong><br/>
          ${inc.description}<br/>
          <strong>Severity:</strong> ${inc.severity}/5<br/>
          <strong>Reported:</strong> ${inc.reported_ago_mins} mins ago<br/>
          <strong>Status:</strong> ${inc.status}
        </div>
      `);
      marker.addTo(incidentLayerGroup);
    });
  } catch (err) {
    console.error("Failed loading active incidents on map:", err);
  }
}

// 5. Load Citizen Locality Details & Forecast
async function loadCitizenLocality(lat, lon) {
  try {
    const liveUrl = `/api/aqi/live?lat=${lat}&lon=${lon}`;
    const liveRes = await fetch(liveUrl);
    const liveData = await liveRes.json();

    // Populate Citizen View Hero
    document.getElementById("citizen-locality-name").textContent = liveData.locality;
    document.getElementById("citizen-zone-name").textContent = liveData.zone;
    
    const aqiCircle = document.getElementById("citizen-aqi-circle");
    aqiCircle.textContent = liveData.aqi;
    aqiCircle.style.background = getAQIColor(liveData.aqi);
    
    document.getElementById("citizen-aqi-category").textContent = liveData.category;
    document.getElementById("citizen-dominant-pollutant").textContent = liveData.dominant_pollutant;
    document.getElementById("citizen-health-advisory").textContent = liveData.health_advisory;

    // Primary Driver Attention Card
    document.getElementById("citizen-driver-title").textContent = liveData.primary_driver;
    document.getElementById("citizen-driver-detail").textContent = liveData.driver_detail;
    document.getElementById("citizen-driver-confidence").textContent = `${liveData.driver_confidence_pct}% Model Confidence`;

    // Sub-pollutants
    document.getElementById("val-pm25").textContent = liveData.pollutants.pm25;
    document.getElementById("val-pm10").textContent = liveData.pollutants.pm10;
    document.getElementById("val-no2").textContent = liveData.pollutants.no2;
    document.getElementById("val-so2").textContent = liveData.pollutants.so2;
    document.getElementById("val-co").textContent = liveData.pollutants.co;
    document.getElementById("val-o3").textContent = liveData.pollutants.o3;

    // Forecast Trajectory Chart
    const forecastUrl = `/api/aqi/forecast?lat=${lat}&lon=${lon}`;
    const forecastRes = await fetch(forecastUrl);
    const forecastData = await forecastRes.json();
    renderForecastChart("forecastChartCanvas", forecastData.forecast_trajectory);

    // Run Initial Clean Air Window
    runCleanAirOptimization(lat, lon);

    // Update JSON Diagnostics
    document.getElementById("json-output-viewer").textContent = JSON.stringify(liveData, null, 2);

  } catch (err) {
    console.error("Failed loading citizen locality details:", err);
  }
}

// 6. Clean Air Window Planner
async function runCleanAirOptimization(lat, lon) {
  try {
    const duration = parseInt(document.getElementById("opt-duration-select")?.value || "2");
    const activity = document.getElementById("opt-activity-select")?.value || "Jogging / Outdoor Workout";

    const res = await fetch(`/api/aqi/optimal-window?lat=${lat}&lon=${lon}&duration=${duration}&activity=${encodeURIComponent(activity)}`);
    const data = await res.json();

    document.getElementById("opt-window-time").textContent = `${data.optimal_window.start_time} - ${data.optimal_window.end_time}`;
    document.getElementById("opt-window-date").textContent = data.optimal_window.date_label;
    document.getElementById("opt-window-aqi").textContent = `Avg AQI: ${data.optimal_window.average_aqi} (${data.optimal_window.air_quality_level})`;
    document.getElementById("opt-saved-pct").textContent = `-${data.optimal_window.particulate_inhalation_avoidance_pct}% Particle Intake vs Peak Hours`;
    document.getElementById("opt-advice").textContent = data.optimal_window.health_advice;

    // Render 24h curve
    renderCleanAirChart("cleanAirChartCanvas", data.hourly_24h_curve, 14 - duration, duration);

  } catch (err) {
    console.error("Failed running clean air optimization:", err);
  }
}

// 7. Government Hotspot Causality Matrix
async function loadGovernmentCausalityMatrix() {
  try {
    const res = await fetch("/api/gov/causality-matrix");
    const data = await res.json();

    const tbody = document.getElementById("causality-matrix-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.top_impact_zones.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>#${item.rank}</strong></td>
        <td><strong>${item.locality}</strong><br/><span style="font-size:11px;color:#9ca3af;">${item.zone}</span></td>
        <td><span class="badge" style="background:${getAQIColor(item.current_aqi)}; color:#fff;">${item.current_aqi} (${item.grap_stage})</span></td>
        <td><span style="color:#fb923c; font-weight:600;">${item.primary_contributor} (${item.primary_pct}%)</span></td>
        <td><span style="color:#9ca3af;">${item.secondary_contributor} (${item.secondary_pct}%)</span></td>
        <td><span style="font-size:12px; color:#60a5fa;">${item.primary_recommended_action}</span></td>
      `;
      tbody.appendChild(tr);
    });

    // Render Integrated Gradients Source Apportionment Donut
    renderApportionmentDonut("apportionmentChartCanvas", data.citywide_source_apportionment);

  } catch (err) {
    console.error("Failed loading causality matrix:", err);
  }
}

// 8. Policy Simulation (do-calculus)
async function runPolicySimulation() {
  try {
    const oddEven = document.getElementById("policy-odd-even")?.checked || false;
    const truckDiv = document.getElementById("policy-truck-diversion")?.checked || false;
    const constHalt = document.getElementById("policy-construction-halt")?.checked || false;
    const indCurfew = document.getElementById("policy-industrial-curfew")?.checked || false;
    const smogGuns = parseInt(document.getElementById("policy-smog-guns")?.value || "0");

    const payload = {
      odd_even_active: oddEven,
      truck_diversion_active: truckDiv,
      construction_halt_active: constHalt,
      industrial_curfew_active: indCurfew,
      smog_guns_units: smogGuns
    };

    const res = await fetch("/api/gov/simulate-policy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    document.getElementById("policy-base-aqi").textContent = data.citywide_summary.baseline_mean_aqi_6h;
    document.getElementById("policy-proj-aqi").textContent = data.citywide_summary.projected_mean_aqi_6h;
    document.getElementById("policy-delta-pts").textContent = `-${data.citywide_summary.average_delta_reduction} pts`;
    document.getElementById("policy-delta-pct").textContent = `(${data.citywide_summary.average_percentage_drop}% Drop)`;

    // Render Policy Before-After Chart
    renderPolicyComparisonChart("policyComparisonCanvas", data.top_beneficiary_wards);

    // Update Beneficiary List
    const benList = document.getElementById("policy-beneficiaries-list");
    if (benList) {
      benList.innerHTML = "";
      data.top_beneficiary_wards.forEach(w => {
        const li = document.createElement("li");
        li.style.padding = "8px 0";
        li.style.borderBottom = "1px solid rgba(255,255,255,0.05)";
        li.innerHTML = `
          <strong>${w.name}</strong>: 
          <span style="color:#10b981; font-weight:bold;">-${w.delta_aqi_drop} AQI pts</span> (${w.percentage_reduction}% drop, ~${w.estimated_lag_hours}h lag)
        `;
        benList.appendChild(li);
      });
    }

  } catch (err) {
    console.error("Failed running policy simulation:", err);
  }
}

// 9. DBSCAN Incident Triage Queue & Squad Dispatch
async function loadIncidentTriage() {
  try {
    const res = await fetch("/api/gov/incidents/triage");
    const data = await res.json();

    const tbody = document.getElementById("incident-triage-tbody");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.triage_queue.forEach(clust => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${clust.cluster_id}</strong></td>
        <td><span class="badge ${clust.priority.includes('Priority 1') ? 'badge-grap4' : ''}">${clust.priority}</span></td>
        <td>${clust.locality} (${clust.zone})</td>
        <td><strong>${clust.type_label}</strong> (${clust.report_count} reports)</td>
        <td><span style="color:${clust.status.includes('Dispatched') ? '#10b981' : '#f59e0b'}; font-weight:600;">${clust.status}</span></td>
        <td>
          <button class="btn btn-danger" style="padding:4px 10px; font-size:11px; width:auto;" onclick="dispatchSquad('${clust.cluster_id}')">
            🚨 Dispatch Squad
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Failed loading incident triage queue:", err);
  }
}

async function dispatchSquad(clusterId) {
  try {
    const res = await fetch("/api/gov/incidents/dispatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cluster_id: clusterId })
    });
    const data = await res.json();
    alert(`🚨 SUCCESS: ${data.message}`);
    loadIncidentTriage();
    loadMapIncidents();
  } catch (err) {
    alert("Failed to dispatch squad.");
  }
}

// 10. Weekly Audit & Retraining Trigger
async function loadWeeklyAudit() {
  try {
    const res = await fetch("/api/gov/weekly-audit");
    const data = await res.json();

    document.getElementById("audit-summary-text").textContent = data.executive_summary;
    document.getElementById("metric-rmse").textContent = `${data.model1_performance_metrics.rmse_aqi_points} pts`;
    document.getElementById("metric-mae").textContent = `${data.model1_performance_metrics.mae_aqi_points} pts`;
    document.getElementById("metric-r2").textContent = data.model1_performance_metrics.r_squared;

  } catch (err) {
    console.error("Failed loading weekly audit:", err);
  }
}

async function triggerModelRetraining() {
  const btn = document.getElementById("btn-trigger-retrain");
  btn.disabled = true;
  btn.textContent = "⏳ Retraining Model 1 ST-GNN with Residuals...";

  try {
    const res = await fetch("/api/gov/retrain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 10 })
    });
    const data = await res.json();

    alert(`✅ SUCCESS: ${data.status}\nPre-training RMSE: ${data.pre_training_rmse}\nPost-training RMSE: ${data.post_training_rmse} (Improved by ${data.rmse_improvement_pts} pts)\nFinal R²: ${data.final_r2}`);
    loadWeeklyAudit();
  } catch (err) {
    alert("Retraining failed.");
  } finally {
    btn.disabled = false;
    btn.textContent = "🔄 Execute Closed-Loop Retraining";
  }
}

// 11. Event Handlers & Submissions
function bindEventHandlers() {
  // Locality Selector Change
  const locSelect = document.getElementById("citizen-locality-select");
  if (locSelect) {
    locSelect.addEventListener("change", (e) => {
      const [lat, lon] = e.target.value.split(",").map(Number);
      loadCitizenLocality(lat, lon);
    });
  }

  // Clean Air Planner Button
  const btnPlan = document.getElementById("btn-plan-clean-air");
  if (btnPlan) {
    btnPlan.addEventListener("click", () => {
      const locVal = document.getElementById("citizen-locality-select")?.value || "28.6468,77.3160";
      const [lat, lon] = locVal.split(",").map(Number);
      runCleanAirOptimization(lat, lon);
    });
  }

  // Incident Submit Form
  const incidentForm = document.getElementById("citizen-incident-form");
  if (incidentForm) {
    incidentForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const locVal = document.getElementById("citizen-locality-select")?.value || "28.6468,77.3160";
      const [lat, lon] = locVal.split(",").map(Number);
      const incidentType = document.getElementById("incident-type-select").value;
      const severity = parseInt(document.getElementById("incident-severity-slider").value);
      const description = document.getElementById("incident-desc-input").value;

      try {
        const res = await fetch("/api/incidents/report", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lat: lat,
            lon: lon,
            incident_type: incidentType,
            severity: severity,
            description: description
          })
        });
        const data = await res.json();
        alert(`✅ Citizen Report Submitted!\n${data.message}\nActive PM2.5 Impulse: +${data.active_impulse.pm25} µg/m³`);
        
        // Refresh views
        loadMapIncidents();
        loadCitizenLocality(lat, lon);
        loadIncidentTriage();
      } catch (err) {
        alert("Failed to submit incident report.");
      }
    });
  }

  // Policy Simulator Button
  const btnPolicy = document.getElementById("btn-run-policy-sim");
  if (btnPolicy) {
    btnPolicy.addEventListener("click", runPolicySimulation);
  }

  // Retraining Button
  const btnRetrain = document.getElementById("btn-trigger-retrain");
  if (btnRetrain) {
    btnRetrain.addEventListener("click", triggerModelRetraining);
  }
}

