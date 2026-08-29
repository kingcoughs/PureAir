/**
 * Government Command Center Application Controller (Project Meswak)
 */

let apiBase = localStorage.getItem("meswak_api_base") || "";
let govMap = null;
let govHexLayer = null;
let govWindLayer = null;
let govIncidentLayer = null;
let allHexNodes = [];
let apportionmentDonut = null;
let policyBarChart = null;

document.addEventListener("DOMContentLoaded", () => {
  initGovNav();
  initTheme();
  loadAtmosphericHeader();
  loadAllHexNodes();
  loadCausalityMatrix();
  initPolicySimulator();
  loadIncidentTriage();
  loadModelLab();
});

// Helper for API URLs
function getApiUrl(path) {
  return `${apiBase}${path}`;
}

function updateApiBaseUrl(newUrl) {
  apiBase = newUrl.replace(/\/$/, "");
  localStorage.setItem("meswak_api_base", apiBase);
  alert(`API Base URL set to: ${apiBase || "Default (Local Origin)"}`);
  location.reload();
}

// Navigation
function initGovNav() {
  const navBtns = document.querySelectorAll(".gov-nav-btn");
  navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      navBtns.forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".gov-screen").forEach(s => s.classList.remove("active"));

      btn.classList.add("active");
      const target = btn.getAttribute("data-screen");
      const screen = document.getElementById(target);
      if (screen) {
        screen.classList.add("active");
        if (target === "gov-screen-map") {
          initOrRefreshGovMap();
        }
      }
    });
  });
}

// Theme Toggle
function initTheme() {
  const saved = localStorage.getItem("meswak_gov_theme") || "dark";
  document.body.setAttribute("data-theme", saved);
}

function toggleTheme() {
  const current = document.body.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "slate" : (current === "slate" ? "light" : "dark");
  document.body.setAttribute("data-theme", next);
  localStorage.setItem("meswak_gov_theme", next);
}

function getAQIColor(aqi) {
  if (aqi <= 50) return "#10b981";
  if (aqi <= 100) return "#84cc16";
  if (aqi <= 200) return "#f59e0b";
  if (aqi <= 300) return "#f97316";
  if (aqi <= 400) return "#ef4444";
  if (aqi <= 450) return "#a855f7";
  return "#dc2626";
}

// 1. Load Live Atmosphere Header
async function loadAtmosphericHeader() {
  try {
    const res = await fetch(getApiUrl("/api/aqi/live"));
    const data = await res.json();
    document.getElementById("gov-header-temp").textContent = `${data.weather.temperature}°C`;
    document.getElementById("gov-header-wind").textContent = `${data.weather.wind_speed} m/s (${data.weather.wind_direction}°)`;
    document.getElementById("gov-header-pbl").textContent = `${data.weather.pbl_height}m`;
    document.getElementById("gov-header-vi").textContent = `${data.weather.ventilation_index} m²/s`;
    document.getElementById("gov-grap-badge").textContent = data.grap_stage;
  } catch (err) {
    console.error("Failed to load header atmosphere:", err);
  }
}

// 2. Load All Hexagon Nodes
async function loadAllHexNodes() {
  try {
    const res = await fetch(getApiUrl("/api/grid/hexagons"));
    const data = await res.json();
    allHexNodes = data.nodes;

    // Populate Policy Target Dropdown
    const sel = document.getElementById("policy-target-node");
    if (sel) {
      sel.innerHTML = `<option value="all">🌐 All Delhi-NCR Airshed (Citywide Aggregate)</option>`;
      allHexNodes.forEach(n => {
        const opt = document.createElement("option");
        opt.value = n.hex_id;
        opt.textContent = `${n.name} (${n.zone})`;
        sel.appendChild(opt);
      });
    }

    // Populate Map Search Datalist
    const datalist = document.getElementById("gov-localities-datalist");
    if (datalist) {
      datalist.innerHTML = "";
      allHexNodes.forEach(n => {
        const opt = document.createElement("option");
        opt.value = `${n.name} (${n.zone})`;
        datalist.appendChild(opt);
      });
    }

  } catch (err) {
    console.error("Failed loading hex nodes:", err);
  }
}

// 3. Hotspot Causality Matrix & Source Apportionment
async function loadCausalityMatrix() {
  try {
    const res = await fetch(getApiUrl("/api/gov/causality-matrix"));
    const data = await res.json();

    const tbody = document.getElementById("causality-matrix-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.top_impact_zones.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>#${item.rank}</strong></td>
        <td><strong>${item.locality}</strong><br/><span style="font-size:11px;color:#94a3b8;">${item.zone}</span></td>
        <td><span class="badge" style="background:${getAQIColor(item.current_aqi)}; color:#fff;">${item.current_aqi} (${item.grap_stage})</span></td>
        <td><span style="color:#fb923c; font-weight:700;">${item.primary_contributor} (${item.primary_pct}%)</span></td>
        <td><span style="color:#94a3b8;">${item.secondary_contributor} (${item.secondary_pct}%)</span></td>
        <td><span style="font-size:12px; color:#60a5fa; font-weight:500;">${item.primary_recommended_action}</span></td>
      `;
      tbody.appendChild(tr);
    });

    renderApportionmentDonut(data.citywide_source_apportionment);

  } catch (err) {
    console.error("Failed loading causality matrix:", err);
  }
}

function renderApportionmentDonut(apportionmentData) {
  const ctx = document.getElementById("govApportionmentDonut");
  if (!ctx) return;

  const labels = Object.keys(apportionmentData);
  const values = Object.values(apportionmentData);

  const colors = ['#f97316', '#ef4444', '#a855f7', '#eab308', '#3b82f6', '#10b981', '#6b7280'];

  if (apportionmentDonut) apportionmentDonut.destroy();

  apportionmentDonut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors.slice(0, labels.length),
        borderWidth: 1,
        borderColor: "#111827"
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "right",
          labels: { color: "#cbd5e1", font: { size: 11 }, boxWidth: 12 }
        }
      },
      cutout: "60%"
    }
  });
}

// 4. Hexagon-Specific Policy Simulator
function initPolicySimulator() {
  const btn = document.getElementById("btn-run-gov-policy");
  if (btn) {
    btn.addEventListener("click", runGovPolicySimulation);
  }
  runGovPolicySimulation();
}

async function runGovPolicySimulation() {
  try {
    const targetHex = document.getElementById("policy-target-node")?.value || "all";
    const oddEven = document.getElementById("gov-policy-odd-even")?.checked || false;
    const truckDiv = document.getElementById("gov-policy-truck-diversion")?.checked || false;
    const constHalt = document.getElementById("gov-policy-const-halt")?.checked || false;
    const indCurfew = document.getElementById("gov-policy-ind-curfew")?.checked || false;
    const smogGuns = parseInt(document.getElementById("gov-policy-smog-guns")?.value || "0");

    const payload = {
      target_hex_id: targetHex,
      odd_even_active: oddEven,
      truck_diversion_active: truckDiv,
      construction_halt_active: constHalt,
      industrial_curfew_active: indCurfew,
      smog_guns_units: smogGuns
    };

    const res = await fetch(getApiUrl("/api/gov/simulate-policy"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    // Check if target hexagon detail exists
    const targetCard = document.getElementById("policy-target-detail-card");
    if (data.target_node_detail && targetCard) {
      targetCard.style.display = "block";
      const td = data.target_node_detail;
      document.getElementById("td-locality").textContent = `${td.locality} (${td.zone})`;
      document.getElementById("td-base-aqi").textContent = td.baseline_aqi_6h;
      document.getElementById("td-proj-aqi").textContent = td.projected_aqi_6h;
      document.getElementById("td-delta").textContent = `-${td.delta_aqi_drop} pts (${td.percentage_reduction}%)`;
      document.getElementById("td-lag").textContent = `~${td.estimated_lag_hours} Hours`;
      document.getElementById("td-pm25-shift").textContent = `${td.local_pollutants.pm25_before} → ${td.local_pollutants.pm25_after} µg/m³`;
      document.getElementById("td-effectiveness").textContent = td.policy_effectiveness_label;
    } else if (targetCard) {
      targetCard.style.display = "none";
    }

    // Citywide Stats
    document.getElementById("gov-policy-city-base").textContent = data.citywide_summary.baseline_mean_aqi_6h;
    document.getElementById("gov-policy-city-proj").textContent = data.citywide_summary.projected_mean_aqi_6h;
    document.getElementById("gov-policy-city-delta").textContent = `-${data.citywide_summary.average_delta_reduction} pts (${data.citywide_summary.average_percentage_drop}%)`;

    renderPolicyComparisonChart(data.top_beneficiary_wards);

  } catch (err) {
    console.error("Policy simulation failed:", err);
  }
}

function renderPolicyComparisonChart(wards) {
  const ctx = document.getElementById("govPolicyBarCanvas");
  if (!ctx) return;

  const labels = wards.map(w => w.name.split(" ")[0]);
  const baseVals = wards.map(w => w.baseline_aqi_6h);
  const projVals = wards.map(w => w.projected_aqi_6h);

  if (policyBarChart) policyBarChart.destroy();

  policyBarChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Baseline AQI (+6h)",
          data: baseVals,
          backgroundColor: "rgba(239, 68, 68, 0.75)",
          borderRadius: 4
        },
        {
          label: "Policy Projected (+6h)",
          data: projVals,
          backgroundColor: "rgba(16, 185, 129, 0.85)",
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
        y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } }
      }
    }
  });
}

// 5. DBSCAN Incident Triage & Squad Dispatch
async function loadIncidentTriage() {
  try {
    const res = await fetch(getApiUrl("/api/gov/incidents/triage"));
    const data = await res.json();

    const tbody = document.getElementById("gov-triage-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    data.triage_queue.forEach(c => {
      const tr = document.createElement("tr");
      
      const photoHtml = (c.reports[0] && c.reports[0].image_url) ?
        `<button class="gov-btn" style="padding:4px 8px; font-size:11px;" onclick="viewPhotoModal('${c.reports[0].image_url}', '${c.locality}')">📷 View Photo</button>` :
        `<span style="color:#64748b; font-size:11px;">No Photo</span>`;

      tr.innerHTML = `
        <td><strong>${c.cluster_id}</strong></td>
        <td><span class="badge ${c.priority.includes('Priority 1') ? 'badge-danger' : ''}" style="background:${c.priority.includes('Priority 1') ? 'rgba(239,68,68,0.2)' : 'rgba(245,158,11,0.2)'}; color:${c.priority.includes('Priority 1') ? '#f87171' : '#fbbf24'};">${c.priority}</span></td>
        <td><strong>${c.locality}</strong> (${c.zone})</td>
        <td>${c.type_label} (<strong>${c.report_count}</strong> reports)</td>
        <td>${photoHtml}</td>
        <td><span style="color:${c.status.includes('Dispatched') ? '#10b981' : '#f59e0b'}; font-weight:600;">${c.status}</span></td>
        <td>
          <button class="gov-btn gov-btn-danger" style="padding:5px 10px; font-size:11px;" onclick="dispatchEnforcementSquad('${c.cluster_id}')">
            🚨 Dispatch Squad
          </button>
        </td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    console.error("Failed loading incident triage:", err);
  }
}

function viewPhotoModal(imgUrl, locality) {
  document.getElementById("modal-photo-img").src = imgUrl;
  document.getElementById("modal-photo-locality").textContent = locality;
  document.getElementById("photo-modal").style.display = "flex";
}

async function dispatchEnforcementSquad(clusterId) {
  try {
    const res = await fetch(getApiUrl("/api/gov/incidents/dispatch"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cluster_id: clusterId })
    });
    const data = await res.json();
    alert(`🚨 SUCCESS: ${data.message}`);
    loadIncidentTriage();
  } catch (err) {
    alert("Failed to dispatch squad.");
  }
}

// 6. Airshed Map
function initOrRefreshGovMap() {
  if (!govMap) {
    govMap = L.map("gov-airshed-map", {
      center: [28.6139, 77.2090],
      zoom: 11,
      minZoom: 9,
      maxZoom: 15
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> | Delhi-NCR Airshed',
      subdomains: "abcd",
      maxZoom: 19
    }).addTo(govMap);

    govHexLayer = L.layerGroup().addTo(govMap);
    govWindLayer = L.layerGroup().addTo(govMap);
  }

  setTimeout(() => {
    govMap.invalidateSize();
    renderGovHexagons();
    renderGovWindVectors();
  }, 200);
}

function renderGovHexagons() {
  if (!govHexLayer || !allHexNodes.length) return;
  govHexLayer.clearLayers();

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
        <strong>${node.name}</strong> (${node.zone})<br/>
        Est. AQI: <span style="color:${hexColor}; font-weight:bold;">${approxAQI}</span><br/>
        Elev: ${node.elevation_m}m | Traffic: ${(node.traffic_weight*100).toFixed(0)}%
      </div>
    `, { sticky: true });

    poly.addTo(govHexLayer);
  });
}

async function renderGovWindVectors() {
  if (!govWindLayer) return;
  try {
    const res = await fetch(getApiUrl("/api/grid/adjacency"));
    const data = await res.json();
    govWindLayer.clearLayers();

    data.edges.forEach(e => {
      const line = L.polyline([e.source_coord, e.target_coord], {
        color: "#38bdf8",
        weight: Math.min(3.5, Math.max(1, e.weight * 20)),
        opacity: Math.min(0.85, e.weight * 4),
        dashArray: "4, 6"
      });
      line.addTo(govWindLayer);
    });
  } catch (err) {
    console.error("Failed to render wind vectors:", err);
  }
}

// 7. AI Model Lab & Retraining Trigger
async function loadModelLab() {
  try {
    const res = await fetch(getApiUrl("/api/gov/weekly-audit"));
    const data = await res.json();

    document.getElementById("lab-rmse-val").textContent = `${data.model1_performance_metrics.rmse_aqi_points} pts`;
    document.getElementById("lab-mae-val").textContent = `${data.model1_performance_metrics.mae_aqi_points} pts`;
    document.getElementById("lab-r2-val").textContent = data.model1_performance_metrics.r_squared;
    document.getElementById("lab-nodes-val").textContent = data.model1_performance_metrics.total_monitored_nodes || allHexNodes.length;
  } catch (err) {
    console.error("Failed loading model lab:", err);
  }
}

async function triggerModelRetraining() {
  const btn = document.getElementById("btn-trigger-model-retrain");
  btn.disabled = true;
  btn.textContent = "⏳ Retraining ST-GNN with Residuals...";

  try {
    const res = await fetch(getApiUrl("/api/gov/retrain"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 12 })
    });
    const data = await res.json();

    alert(`✅ Active Learning Retraining Successful!\nPre-training RMSE: ${data.pre_training_rmse}\nPost-training RMSE: ${data.post_training_rmse} (Improved by ${data.rmse_improvement_pts} pts)\nFinal R²: ${data.final_r2}`);
    loadModelLab();
    loadCausalityMatrix();
  } catch (err) {
    alert("Retraining failed.");
  } finally {
    btn.disabled = false;
    btn.textContent = "🔄 Execute Closed-Loop Retraining";
  }
}

