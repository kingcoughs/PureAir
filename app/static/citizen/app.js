/**
 * Citizen Mobile Application Controller (Project Meswak)
 */

let apiBase = localStorage.getItem("meswak_api_base") || "";
let currentLat = 28.6468;
let currentLon = 77.3160;
let citizenMap = null;
let hexLayer = null;
let windLayer = null;
let incidentLayer = null;
let allHexNodes = [];
let forecastChart = null;
let cleanAirChart = null;

document.addEventListener("DOMContentLoaded", () => {
  initBottomNav();
  initTheme();
  loadAllHexagons();
  loadLiveAQI();
  loadForecast();
  initCleanAirPlanner();
  initIncidentForm();
  initSearchAutocomplete();
});

// 1. API Base URL Helper
function getApiUrl(path) {
  return `${apiBase}${path}`;
}

function updateApiBaseUrl(newUrl) {
  apiBase = newUrl.replace(/\/$/, "");
  localStorage.setItem("meswak_api_base", apiBase);
  alert(`API Base URL set to: ${apiBase || "Default (Local Origin)"}`);
  location.reload();
}

// 2. Bottom Navigation Switching
function initBottomNav() {
  const navBtns = document.querySelectorAll(".bottom-nav .nav-item");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      navBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".app-screen").forEach(s => s.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-target");
      const screen = document.getElementById(targetId);
      if (screen) {
        screen.classList.add("active");
        if (targetId === "screen-map") {
          initOrRefreshMap();
        }
      }
    });
  });
}

// 3. Theme Toggle
function initTheme() {
  const savedTheme = localStorage.getItem("meswak_theme") || "dark";
  document.body.setAttribute("data-theme", savedTheme);
}

function toggleTheme() {
  const current = document.body.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : (current === "light" ? "amoled" : "dark");
  document.body.setAttribute("data-theme", next);
  localStorage.setItem("meswak_theme", next);
}

// Helper: Color by AQI value
function getAQIColor(aqi) {
  if (aqi <= 50) return "#10b981";       // Good
  if (aqi <= 100) return "#84cc16";      // Satisfactory
  if (aqi <= 200) return "#f59e0b";      // Moderate
  if (aqi <= 300) return "#f97316";      // Poor
  if (aqi <= 400) return "#ef4444";      // Very Poor
  if (aqi <= 450) return "#a855f7";      // Severe
  return "#dc2626";                      // Severe+
}

// 4. Load Full H3 Hexagons Grid for Search and Map
async function loadAllHexagons() {
  try {
    const res = await fetch(getApiUrl("/api/grid/hexagons"));
    const data = await res.json();
    allHexNodes = data.nodes;
    populateLocalitySearchList(allHexNodes);
  } catch (err) {
    console.error("Failed to load hexagons:", err);
  }
}

function populateLocalitySearchList(nodes) {
  const datalist = document.getElementById("localities-datalist");
  if (!datalist) return;
  datalist.innerHTML = "";
  nodes.forEach(n => {
    const opt = document.createElement("option");
    opt.value = `${n.name} (${n.zone})`;
    opt.setAttribute("data-lat", n.centroid.lat);
    opt.setAttribute("data-lon", n.centroid.lon);
    opt.setAttribute("data-hex", n.hex_id);
    datalist.appendChild(opt);
  });
}

// 5. Load Live AQI & Primary Source Attribution
async function loadLiveAQI(lat = currentLat, lon = currentLon) {
  try {
    currentLat = lat;
    currentLon = lon;

    const res = await fetch(getApiUrl(`/api/aqi/live?lat=${lat}&lon=${lon}`));
    const data = await res.json();

    document.getElementById("current-locality-display").textContent = data.locality;
    document.getElementById("current-zone-display").textContent = data.zone;

    const aqiVal = data.aqi;
    const aqiColor = getAQIColor(aqiVal);

    const aqiNumElem = document.getElementById("main-aqi-number");
    aqiNumElem.textContent = aqiVal;
    aqiNumElem.style.color = aqiColor;

    const badgeElem = document.getElementById("main-aqi-badge");
    badgeElem.textContent = data.category;
    badgeElem.style.background = aqiColor;
    badgeElem.style.color = "#fff";

    document.getElementById("main-dominant-pollutant").textContent = data.dominant_pollutant;

    // Primary Source Attribution Tag
    document.getElementById("main-pollution-source").textContent = data.primary_driver;
    document.getElementById("main-source-detail").textContent = data.driver_detail;

    // Mini Bento Highlights
    document.getElementById("mini-pm25-val").textContent = `${data.pollutants.pm25} µg/m³`;
    document.getElementById("mini-temp-val").textContent = `${data.weather.temperature}°C`;
    document.getElementById("mini-wind-val").textContent = `${data.weather.wind_speed} m/s`;
    document.getElementById("mini-pbl-val").textContent = `${data.weather.pbl_height}m`;

    const invPill = document.getElementById("mini-inversion-status");
    if (data.weather.is_inversion_active) {
      invPill.textContent = "Trapped Inversion";
      invPill.style.color = "#fbbf24";
    } else {
      invPill.textContent = "Normal Airflow";
      invPill.style.color = "#34d399";
    }

    // Populate Detailed Drill-Down Sheet
    populateDetailsSheet(data);

  } catch (err) {
    console.error("Error loading live AQI:", err);
  }
}

function populateDetailsSheet(data) {
  document.getElementById("sheet-locality").textContent = `${data.locality} (${data.zone})`;
  document.getElementById("sheet-aqi-val").textContent = `${data.aqi} - ${data.category}`;
  document.getElementById("sheet-grap-stage").textContent = data.grap_stage;

  document.getElementById("sheet-pm25").textContent = `${data.pollutants.pm25} µg/m³`;
  document.getElementById("sheet-pm10").textContent = `${data.pollutants.pm10} µg/m³`;
  document.getElementById("sheet-no2").textContent = `${data.pollutants.no2} µg/m³`;
  document.getElementById("sheet-so2").textContent = `${data.pollutants.so2} µg/m³`;
  document.getElementById("sheet-co").textContent = `${data.pollutants.co} mg/m³`;
  document.getElementById("sheet-o3").textContent = `${data.pollutants.o3} µg/m³`;

  document.getElementById("sheet-temp").textContent = `${data.weather.temperature} °C`;
  document.getElementById("sheet-humidity").textContent = `${data.weather.humidity} %`;
  document.getElementById("sheet-wind").textContent = `${data.weather.wind_speed} m/s (${data.weather.wind_direction}°)`;
  document.getElementById("sheet-pbl").textContent = `${data.weather.pbl_height} m`;
  document.getElementById("sheet-vi").textContent = `${data.weather.ventilation_index} m²/s`;
  document.getElementById("sheet-stubble-inflow").textContent = `+${data.stubble_smoke_inflow.transboundary_pm25_inflow} µg/m³ (${data.stubble_smoke_inflow.smoke_contribution_pct}%)`;
}

function openDetailsModal() {
  document.getElementById("details-modal").classList.add("active");
}

function closeDetailsModal() {
  document.getElementById("details-modal").classList.remove("active");
}

// 6. Forecast Trajectory with 90% Confidence Envelopes
async function loadForecast(lat = currentLat, lon = currentLon) {
  try {
    const res = await fetch(getApiUrl(`/api/aqi/forecast?lat=${lat}&lon=${lon}`));
    const data = await res.json();
    renderForecastChart(data.forecast_trajectory);
  } catch (err) {
    console.error("Failed to load forecast trajectory:", err);
  }
}

function renderForecastChart(trajectory) {
  const ctx = document.getElementById("citizenForecastCanvas");
  if (!ctx) return;

  const labels = trajectory.map(d => `+${d.horizon_hours}h`);
  const means = trajectory.map(d => d.predicted_aqi);
  const uppers = trajectory.map(d => d.upper_ci_90);
  const lowers = trajectory.map(d => d.lower_ci_90);

  if (forecastChart) forecastChart.destroy();

  forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "90% Upper Bound",
          data: uppers,
          borderColor: "transparent",
          backgroundColor: "rgba(239, 68, 68, 0.15)",
          fill: "+1",
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: "90% Lower Bound",
          data: lowers,
          borderColor: "transparent",
          backgroundColor: "transparent",
          fill: false,
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: "Projected AQI",
          data: means,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.2)",
          borderWidth: 3,
          pointRadius: 4,
          pointBackgroundColor: "#60a5fa",
          tension: 0.35
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` AQI: ${c.raw}`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#94a3b8", font: { size: 10 } }
        },
        y: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#94a3b8", font: { size: 10 } }
        }
      }
    }
  });
}

// 7. Multi-Day Daytime Clean Air Planner
function initCleanAirPlanner() {
  const btn = document.getElementById("btn-calc-clean-air");
  if (btn) {
    btn.addEventListener("click", runCleanAirPlanner);
  }
  runCleanAirPlanner();
}

async function runCleanAirPlanner() {
  try {
    const duration = parseInt(document.getElementById("planner-duration")?.value || "2");
    const activity = document.getElementById("planner-activity")?.value || "Jogging / Outdoor Workout";

    const res = await fetch(getApiUrl(`/api/aqi/optimal-window?lat=${currentLat}&lon=${currentLon}&duration=${duration}&activity=${encodeURIComponent(activity)}&days=3`));
    const data = await res.json();

    // Render Daily Cards
    const container = document.getElementById("daily-windows-container");
    if (!container) return;
    container.innerHTML = "";

    data.daily_recommendations.forEach((d, idx) => {
      const card = document.createElement("div");
      card.className = "bento-card";
      card.style.marginBottom = "10px";
      card.style.borderLeft = `4px solid ${idx === 0 ? '#10b981' : '#3b82f6'}`;
      card.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-size:0.75rem; font-weight:700; color:#38bdf8;">${d.day_label} (${d.date_label})</span>
          <span class="badge" style="background:#10b981; color:#fff; font-size:0.7rem;">-${d.particulate_inhalation_avoidance_pct}% Exposure</span>
        </div>
        <h4 style="font-size:1.15rem; font-weight:800; color:#fff; margin:6px 0 2px;">${d.start_time} - ${d.end_time}</h4>
        <p style="font-size:0.8rem; color:#a7f3d0;">Average AQI: <strong>${d.average_aqi} (${d.air_quality_level})</strong></p>
        <p style="font-size:0.75rem; color:#94a3b8; margin-top:4px;">${d.health_advice}</p>
      `;
      container.appendChild(card);
    });

  } catch (err) {
    console.error("Clean air planner failed:", err);
  }
}

// 8. Incident Reporting with Photo Attachment
function initIncidentForm() {
  const photoInput = document.getElementById("incident-photo-input");
  const previewImg = document.getElementById("incident-photo-preview");
  let photoBase64 = null;

  if (photoInput) {
    photoInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (re) => {
          photoBase64 = re.target.result;
          if (previewImg) {
            previewImg.src = photoBase64;
            previewImg.style.display = "block";
          }
        };
        reader.readAsDataURL(file);
      }
    });
  }

  const form = document.getElementById("incident-submit-form");
  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const cat = document.getElementById("incident-cat").value;
      const sev = parseInt(document.getElementById("incident-sev").value);
      const desc = document.getElementById("incident-desc").value;

      try {
        const res = await fetch(getApiUrl("/api/incidents/report"), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lat: currentLat,
            lon: currentLon,
            incident_type: cat,
            severity: sev,
            description: desc,
            image_base64: photoBase64
          })
        });
        const data = await res.json();
        alert(`🚨 Incident Reported Successfully!\nReport ID: ${data.report_id}\nInstant PM2.5 impulse injected: +${data.active_impulse.pm25} µg/m³`);
        form.reset();
        if (previewImg) previewImg.style.display = "none";
        loadLiveAQI();
      } catch (err) {
        alert("Failed to submit incident report.");
      }
    });
  }
}

// 9. Interactive Airshed Map (Free Tile Layer, Full H3 Hexagons & Wind Vectors)
function initOrRefreshMap() {
  if (!citizenMap) {
    citizenMap = L.map("citizen-map", {
      center: [28.6139, 77.2090],
      zoom: 11,
      minZoom: 9,
      maxZoom: 15
    });

    // Free OpenStreetMap CartoDB Dark Matter tiles (No API key needed)
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> | Delhi-NCR Airshed',
      subdomains: "abcd",
      maxZoom: 19
    }).addTo(citizenMap);

    hexLayer = L.layerGroup().addTo(citizenMap);
    windLayer = L.layerGroup().addTo(citizenMap);
    incidentLayer = L.layerGroup().addTo(citizenMap);
  }

  setTimeout(() => {
    citizenMap.invalidateSize();
    renderMapHexagons();
    renderMapWindVectors();
  }, 250);
}

function renderMapHexagons() {
  if (!hexLayer || !allHexNodes.length) return;
  hexLayer.clearLayers();

  allHexNodes.forEach(node => {
    const approxAQI = Math.round(node.industrial_weight * 160 + node.traffic_weight * 140 + 120);
    const hexColor = getAQIColor(approxAQI);
    const latlngs = node.boundary.map(p => [p.lat, p.lon]);

    const poly = L.polygon(latlngs, {
      color: hexColor,
      weight: 1.2,
      fillColor: hexColor,
      fillOpacity: 0.35
    });

    poly.bindTooltip(`
      <div style="font-size:12px; font-family:sans-serif; color:#fff;">
        <strong>${node.name}</strong><br/>
        Zone: ${node.zone}<br/>
        Est. AQI: <span style="color:${hexColor}; font-weight:bold;">${approxAQI}</span>
      </div>
    `, { sticky: true });

    poly.on("click", () => {
      loadLiveAQI(node.centroid.lat, node.centroid.lon);
      loadForecast(node.centroid.lat, node.centroid.lon);
      document.querySelector('[data-target="screen-home"]').click();
    });

    poly.addTo(hexLayer);
  });
}

async function renderMapWindVectors() {
  if (!windLayer) return;
  try {
    const res = await fetch(getApiUrl("/api/grid/adjacency"));
    const data = await res.json();
    windLayer.clearLayers();

    data.edges.forEach(e => {
      const line = L.polyline([e.source_coord, e.target_coord], {
        color: "#38bdf8",
        weight: Math.min(3.5, Math.max(1, e.weight * 20)),
        opacity: Math.min(0.85, e.weight * 4),
        dashArray: "4, 6"
      });
      line.addTo(windLayer);
    });

    // Update Wind Compass Overlay
    document.getElementById("map-wind-speed-badge").textContent = `${data.wind_speed} m/s`;
    document.getElementById("map-wind-dir-badge").textContent = `${data.wind_direction}° (NW Inflow)`;
  } catch (err) {
    console.error("Failed to render wind vectors:", err);
  }
}

// 10. Search Autocomplete
function initSearchAutocomplete() {
  const searchInput = document.getElementById("locality-search-input");
  if (searchInput) {
    searchInput.addEventListener("change", (e) => {
      const val = e.target.value;
      const match = allHexNodes.find(n => `${n.name} (${n.zone})` === val || n.name === val);
      if (match) {
        loadLiveAQI(match.centroid.lat, match.centroid.lon);
        loadForecast(match.centroid.lat, match.centroid.lon);
        if (citizenMap) {
          citizenMap.setView([match.centroid.lat, match.centroid.lon], 13);
        }
      }
    });
  }
}

