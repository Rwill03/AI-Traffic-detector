/**
 * Traffic Ops Console - Dashboard JavaScript
 * Handles real-time monitoring, predictions, and UI interactions
 */

// ============================================
// Configuration
// ============================================
const API_URL = "/api/v1/status";
const PREDICTIONS_URL = "/api/v1/predictions";
const METRICS_URL = "/api/v1/model/metrics";
const REFRESH_INTERVAL = 5000; // 5 seconds

// ============================================
// State
// ============================================
let predictionChart = null;
let evalChart = null;
let selectedDate = new Date().toISOString().split('T')[0];

// ============================================
// Tab Navigation
// ============================================
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      // Remove active from all tabs
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      // Activate clicked tab
      btn.classList.add('active');
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');

      // Load predictions if switching to predictions tab
      if (btn.dataset.tab === 'predictions') {
        loadPredictions();
        loadMetrics();
      }
    });
  });
}

// ============================================
// Utility Functions
// ============================================
function statusToDotClass(status) {
  if (!status) return "dot danger";
  const s = status.toLowerCase();
  if (s === "online") return "dot";
  if (s === "lagging") return "dot warn";
  return "dot danger";
}

function formatTs(iso) {
  if (!iso) return "–";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return iso;
  }
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || isNaN(value)) return "--";
  return Number(value).toFixed(digits);
}

function setBodyState(rock5Status, isError) {
  document.body.classList.remove("state-lagging", "state-offline");

  if (isError) {
    document.body.classList.add("state-offline");
    return;
  }

  if (!rock5Status) return;

  const s = rock5Status.toLowerCase();
  if (s === "lagging") {
    document.body.classList.add("state-lagging");
  } else if (s === "offline") {
    document.body.classList.add("state-offline");
  }
}

// ============================================
// Status Refresh
// ============================================
async function refresh() {
  const statusMsg = document.getElementById("status-msg");
  try {
    const resp = await fetch(API_URL, { cache: "no-store" });
    if (!resp.ok) {
      statusMsg.textContent = "API error: " + resp.status;
      setBodyState(null, true);
      return;
    }
    const data = await resp.json();

    // Update timestamps
    document.getElementById("server-time").textContent =
      "Server: " + formatTs(data.server_time);
    document.getElementById("last-refresh").textContent =
      "Last refresh: " + formatTs(new Date().toISOString());

    // Update status indicators
    const rock5StatusRaw = data.rock5_status || "unknown";
    const rock5State = rock5StatusRaw.toUpperCase();
    const vmState = (data.vm_status || "unknown").toUpperCase();
    const dbState = (data.db_status || "unknown").toUpperCase();

    document.getElementById("rock5-state").textContent = rock5State;
    document.getElementById("vm-state").textContent = vmState;
    document.getElementById("db-state").textContent = dbState;
    document.getElementById("sample-age").textContent =
      data.sample_age_seconds != null
        ? Math.round(data.sample_age_seconds)
        : "–";

    // Update status dots
    const rock5Dot = document.getElementById("rock5-dot");
    const vmDot = document.getElementById("vm-dot");

    rock5Dot.className = statusToDotClass(rock5StatusRaw);
    vmDot.className = statusToDotClass(data.vm_status);

    setBodyState(rock5StatusRaw, false);

    // Update observations table
    updateObservationsTable(data.last_observations || []);

    statusMsg.textContent = "Live link OK · polling /api/v1/status";
  } catch (err) {
    console.error(err);
    statusMsg.textContent = "Connection issue – retrying…";
    setBodyState(null, true);
  }
}

function updateObservationsTable(observations) {
  const bodyEl = document.getElementById("obs-body");
  bodyEl.innerHTML = "";

  if (observations.length === 0) {
    const row = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.textContent = "No samples received yet.";
    td.style.color = "#9ca3af";
    td.style.paddingTop = "10px";
    row.appendChild(td);
    bodyEl.appendChild(row);
    return;
  }

  observations.forEach((o, idx) => {
    const tr = document.createElement("tr");
    if (idx === 0) tr.classList.add("highlight");

    // Timestamp column
    const tdTs = document.createElement("td");
    tdTs.textContent = formatTs(o.ts);

    // Total column
    const tdTotal = document.createElement("td");
    const total = o.total_vehicles ?? 0;
    tdTotal.textContent = `${total} voertuigen`;

    // Split column
    const tdSplit = document.createElement("td");
    const wrapper = document.createElement("div");
    wrapper.className = "tag-row";

    const keys = ["car", "truck", "bus", "motorcycle", "bicycle"];
    keys.forEach((k) => {
      const v = o[k] ?? 0;
      if (v > 0) {
        const chip = document.createElement("div");
        chip.className = "vehicle-chip";
        chip.innerHTML = `<strong>${v}</strong> ${k}`;
        wrapper.appendChild(chip);
      }
    });

    if (!wrapper.childNodes.length) {
      wrapper.textContent = "–";
    }
    tdSplit.appendChild(wrapper);

    // Snapshot column
    const tdSnap = document.createElement("td");
    if (o.snapshot_url) {
      const btn = document.createElement("button");
      btn.className = "snapshot-btn";
      btn.textContent = "View";
      btn.dataset.url = o.snapshot_url;
      tdSnap.appendChild(btn);
    } else {
      tdSnap.textContent = "–";
    }

    tr.appendChild(tdTs);
    tr.appendChild(tdTotal);
    tr.appendChild(tdSplit);
    tr.appendChild(tdSnap);

    bodyEl.appendChild(tr);
  });
}

// ============================================
// Day Selector
// ============================================
function initDaySelector() {
  const selector = document.getElementById('day-selector');
  selector.innerHTML = '';
  
  const today = new Date();
  for (let i = 0; i < 7; i++) {
    const date = new Date(today);
    date.setDate(date.getDate() + i);
    
    const btn = document.createElement('button');
    btn.className = 'day-btn' + (i === 0 ? ' active' : '');
    btn.dataset.date = date.toISOString().split('T')[0];
    
    if (i === 0) {
      btn.textContent = 'Today';
    } else if (i === 1) {
      btn.textContent = 'Tomorrow';
    } else {
      btn.textContent = date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    }
    
    btn.addEventListener('click', () => {
      document.querySelectorAll('.day-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      selectedDate = btn.dataset.date;
      loadPredictions();
    });
    
    selector.appendChild(btn);
  }
}

// ============================================
// Predictions
// ============================================
async function loadPredictions() {
  try {
    const resp = await fetch(`${PREDICTIONS_URL}?date=${selectedDate}`, { cache: "no-store" });
    if (!resp.ok) {
      console.error('Predictions API error:', resp.status);
      return;
    }
    
    const data = await resp.json();
    updatePredictionUI(data);
  } catch (err) {
    console.error('Error loading predictions:', err);
  }
}

function updatePredictionUI(data) {
  const predictions = data.predictions || [];
  const summary = data.summary || {};
  const modelReady = data.model_ready;
  
  // Update model status
  const modelStatus = document.getElementById('model-status');
  if (modelReady) {
    modelStatus.className = 'model-status ready';
    modelStatus.innerHTML = '<span class="dot"></span><span>Transformer model active</span>';
  } else {
    modelStatus.className = 'model-status dummy';
    modelStatus.innerHTML = '<span class="dot warn"></span><span>Using simulated predictions (model not trained)</span>';
  }
  
  // Update current prediction
  const currentHour = new Date().getHours();
  const currentPred = predictions.find(p => p.hour_of_day === currentHour) || predictions[0];
  
  if (currentPred) {
    document.getElementById('current-prediction-value').textContent = currentPred.predicted_cars;
    
    const rushIndicator = document.getElementById('rush-indicator');
    if (currentPred.is_rush_hour) {
      rushIndicator.className = 'rush-indicator rush';
      rushIndicator.innerHTML = '<span class="dot warn"></span><span>High density (rush hour)</span>';
    } else {
      rushIndicator.className = 'rush-indicator normal';
      rushIndicator.innerHTML = '<span class="dot"></span><span>Normal density</span>';
    }
  }
  
  // Update summary stats
  if (summary.peak_hour) {
    const peakHour = summary.peak_hour.split(' ')[1] || summary.peak_hour;
    document.getElementById('peak-hour').textContent = peakHour;
  }
  document.getElementById('peak-cars').textContent = summary.peak_cars || '--';
  const dailyAvg = summary.total_predicted ? Math.round(summary.total_predicted / 24) : '--';
  document.getElementById('daily-total').textContent = dailyAvg;
  document.getElementById('rush-avg').textContent = summary.rush_hour_average || '--';
  
  // Update chart
  updateChart(predictions);
  
  // Update prediction list
  updatePredictionList(predictions, currentHour);
}

function updateChart(predictions) {
  const ctx = document.getElementById('prediction-chart').getContext('2d');
  
  const labels = predictions.map(p => {
    const hour = p.hour_of_day;
    return `${hour.toString().padStart(2, '0')}:00`;
  });
  
  const values = predictions.map(p => p.predicted_cars);
  const rushHours = predictions.map(p => p.is_rush_hour);
  
  const backgroundColors = rushHours.map(isRush => 
    isRush ? 'rgba(55, 65, 81, 0.55)' : 'rgba(107, 114, 128, 0.35)'
  );
  
  const borderColors = rushHours.map(isRush => 
    isRush ? 'rgba(31, 41, 55, 0.9)' : 'rgba(75, 85, 99, 0.8)'
  );
  
  if (predictionChart) {
    predictionChart.destroy();
  }
  
  predictionChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Avg vehicles per snapshot',
        data: values,
        backgroundColor: backgroundColors,
        borderColor: borderColors,
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          backgroundColor: '#ffffff',
          titleColor: '#111827',
          bodyColor: '#111827',
          borderColor: '#e5e7eb',
          borderWidth: 1,
          callbacks: {
            label: function(context) {
              const isRush = rushHours[context.dataIndex];
              return `${context.parsed.y} vehicles per snapshot${isRush ? ' (rush hour)' : ''}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          },
          ticks: {
            color: '#9ca3af',
            font: { size: 10 }
          }
        },
        y: {
          grid: {
            color: 'rgba(148, 163, 184, 0.1)'
          },
          ticks: {
            color: '#9ca3af',
            font: { size: 10 }
          },
          beginAtZero: true
        }
      }
    }
  });
}

function updatePredictionList(predictions, currentHour) {
  const list = document.getElementById('prediction-list');
  list.innerHTML = '';
  
  predictions.forEach(pred => {
    const item = document.createElement('div');
    item.className = 'prediction-item';
    
    if (pred.hour_of_day === currentHour) {
      item.classList.add('current');
    }
    if (pred.is_rush_hour) {
      item.classList.add('rush-hour');
    }
    
    const hourStr = pred.hour_of_day.toString().padStart(2, '0') + ':00';
    
    item.innerHTML = `
      <span class="prediction-hour">${hourStr}</span>
      <span class="prediction-value">${pred.predicted_cars} cars/snapshot</span>
    `;
    
    list.appendChild(item);
  });
}

// ============================================
// Model Metrics
// ============================================
async function loadMetrics() {
  const meta = document.getElementById('eval-meta');
  meta.textContent = "Loading metrics…";
  try {
    const resp = await fetch(METRICS_URL, { cache: "no-store" });
    if (!resp.ok) {
      meta.textContent = "Geen metrics beschikbaar";
      return;
    }
    const data = await resp.json();
    document.getElementById('eval-mae').textContent = fmt(data.mae);
    document.getElementById('eval-rmse').textContent = fmt(data.rmse);
    document.getElementById('eval-smape').textContent = fmt(data.smape);
    document.getElementById('eval-seq').textContent = data.sequences ?? "--";
    meta.textContent = `Validatie op ${data.sequences ?? "–"} sequences · ${data.samples ?? "–"} samples`;
    renderEvalChart(data.per_hour_mae || []);
  } catch (e) {
    console.error("Error loading metrics", e);
    meta.textContent = "Kon metrics niet ophalen";
  }
}

function renderEvalChart(values) {
  const ctx = document.getElementById('eval-chart').getContext('2d');
  const labels = Array.from({ length: values.length || 24 }, (_, i) =>
    i.toString().padStart(2, '0') + ':00'
  );
  const dataVals = values.length ? values : Array.from({ length: labels.length }, () => 0);

  if (evalChart) evalChart.destroy();

  evalChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'MAE per hour',
        data: dataVals,
        borderColor: '#1f2937',
        backgroundColor: 'rgba(79, 70, 229, 0.12)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: '#6b7280', font: { size: 9 } },
          grid: { display: false },
        },
        y: {
          ticks: { color: '#6b7280', font: { size: 10 } },
          grid: { color: 'rgba(148, 163, 184, 0.2)' },
          beginAtZero: true
        }
      }
    }
  });
}

// ============================================
// Snapshot Modal
// ============================================
function initSnapshotModal() {
  const overlay = document.getElementById("snapshot-overlay");
  const overlayImg = document.getElementById("snapshot-img");
  const obsBody = document.getElementById("obs-body");

  // Open modal on snapshot button click
  obsBody.addEventListener("click", (e) => {
    const btn = e.target.closest(".snapshot-btn");
    if (!btn) return;
    const url = btn.dataset.url;
    if (!url) return;
    overlayImg.src = url;
    overlay.classList.remove("hidden");
  });

  // Close modal on overlay click
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
      overlay.classList.add("hidden");
      overlayImg.src = "";
    }
  });

  // Close modal on escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !overlay.classList.contains("hidden")) {
      overlay.classList.add("hidden");
      overlayImg.src = "";
    }
  });
}

// ============================================
// Initialization
// ============================================
function init() {
  initTabs();
  initDaySelector();
  initSnapshotModal();
  
  // Initial refresh
  refresh();
  
  // Set up auto-refresh
  setInterval(refresh, REFRESH_INTERVAL);
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
