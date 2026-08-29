/**
 * Charting & Visual Analytics Engine for Project Meswak (Delhi-NCR)
 * Uses Chart.js for forecast curves, confidence envelopes, and source apportionment.
 */

let forecastChartInstance = null;
let cleanAirChartInstance = null;
let apportionmentChartInstance = null;
let policyChartInstance = null;

function renderForecastChart(canvasId, trajectoryData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = trajectoryData.map(d => `+${d.horizon_hours}h`);
  const meanValues = trajectoryData.map(d => d.predicted_aqi);
  const upperValues = trajectoryData.map(d => d.upper_ci_90);
  const lowerValues = trajectoryData.map(d => d.lower_ci_90);

  if (forecastChartInstance) {
    forecastChartInstance.destroy();
  }

  forecastChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Upper 90% CI',
          data: upperValues,
          borderColor: 'transparent',
          backgroundColor: 'rgba(239, 68, 68, 0.12)',
          fill: '+1',
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: 'Lower 90% CI',
          data: lowerValues,
          borderColor: 'transparent',
          backgroundColor: 'transparent',
          fill: false,
          pointRadius: 0,
          tension: 0.35
        },
        {
          label: 'Projected AQI (Mean)',
          data: meanValues,
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.2)',
          borderWidth: 3,
          pointBackgroundColor: '#60a5fa',
          pointBorderColor: '#fff',
          pointRadius: 4,
          pointHoverRadius: 7,
          tension: 0.35
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          labels: { color: '#9ca3af', font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: 'rgba(17, 24, 39, 0.95)',
          titleColor: '#f3f4f6',
          bodyColor: '#e5e7eb',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af' },
          suggestedMin: 50,
          suggestedMax: 450
        }
      }
    }
  });
}

function renderCleanAirChart(canvasId, hourlyCurve, optimalStartIdx, duration) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = hourlyCurve.map(d => d.datetime.split(' ')[0] + ' ' + d.datetime.split(' ')[1]);
  const aqiVals = hourlyCurve.map(d => d.projected_aqi);
  
  // Highlight optimal window in green
  const pointColors = hourlyCurve.map((d, i) => {
    if (i >= optimalStartIdx && i < optimalStartIdx + duration) {
      return '#10b981';
    }
    return '#f59e0b';
  });

  if (cleanAirChartInstance) {
    cleanAirChartInstance.destroy();
  }

  cleanAirChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: '24h Inhalation Trajectory',
          data: aqiVals,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          fill: true,
          borderWidth: 2,
          pointBackgroundColor: pointColors,
          pointBorderColor: '#fff',
          pointRadius: 4,
          tension: 0.3
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
            label: function(context) {
              return `AQI: ${context.parsed.y}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af', maxTicksLimit: 8 }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af' }
        }
      }
    }
  });
}

function renderApportionmentDonut(canvasId, apportionmentData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = Object.keys(apportionmentData);
  const values = Object.values(apportionmentData);

  const colors = [
    '#f97316', // Traffic (orange)
    '#ef4444', // Stubble (red)
    '#a855f7', // Industry (purple)
    '#eab308', // Dust (yellow)
    '#3b82f6', // Inversion (blue)
    '#10b981', // Landfills (green)
    '#6b7280'  // Other
  ];

  if (apportionmentChartInstance) {
    apportionmentChartInstance.destroy();
  }

  apportionmentChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: colors.slice(0, labels.length),
        borderWidth: 1,
        borderColor: '#111827'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#e5e7eb', font: { size: 11 }, boxWidth: 14 }
        },
        tooltip: {
          callbacks: {
            label: function(c) {
              return ` ${c.label}: ${c.raw}%`;
            }
          }
        }
      },
      cutout: '65%'
    }
  });
}

function renderPolicyComparisonChart(canvasId, topWards) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  const labels = topWards.map(w => w.name.split(' ')[0]);
  const baseVals = topWards.map(w => w.baseline_aqi_6h);
  const projectedVals = topWards.map(w => w.projected_aqi_6h);

  if (policyChartInstance) {
    policyChartInstance.destroy();
  }

  policyChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Baseline AQI (+6h)',
          data: baseVals,
          backgroundColor: 'rgba(239, 68, 68, 0.75)',
          borderRadius: 4
        },
        {
          label: 'Policy Projected AQI (+6h)',
          data: projectedVals,
          backgroundColor: 'rgba(16, 185, 129, 0.85)',
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#e5e7eb', font: { size: 11 } }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af' }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#9ca3af' }
        }
      }
    }
  });
}

