const view = document.getElementById("view");
const title = document.getElementById("title");
const kicker = document.getElementById("kicker");
const asOf = document.getElementById("asOf");
let charts = [];
let simTimer = null;
let simPageActive = false;
let simPending = false;
let simSpeed = 1;
let simChart = null;
let simResourceChart = null;
let simLibraries = [];
let simState = null;
let activePage = "overview";

const titles = {
  overview: ["Campus occupancy", "Overview"],
  simulation: ["Digital twin", "Live simulation"],
  forecast: ["Demand ahead", "Forecast"],
  analytics: ["Patterns", "Analytics"],
  resources: ["When a zone fills", "Resources"],
  explain: ["Why the model moved", "Explainability"],
  performance: ["Holdout scores", "Model performance"],
  calendar: ["Term dates", "Academic calendar"],
  video: ["Demonstration", "Video detection"],
};

function destroyCharts() {
  charts.forEach((c) => c.destroy());
  charts = [];
  simChart = null;
  simResourceChart = null;
}

async function api(path, options) {
  const res = await fetch(path, options);
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; } catch { data = { error: text }; }
  if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
  return data;
}

function metric(label, value) {
  return `<article class="card"><h3>${label}</h3><p>${value}</p></article>`;
}

function table(rows, cols) {
  if (!rows?.length) return "<p class='muted'>No rows.</p>";
  const head = cols.map((c) => `<th>${c.label}</th>`).join("");
  const body = rows.map((r) => `<tr>${cols.map((c) => `<td>${c.fmt ? c.fmt(r[c.key]) : r[c.key] ?? ""}</td>`).join("")}</tr>`).join("");
  return `<div class="panel"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function pct(v) { return `${Math.round(Number(v) * 100)}%`; }
function num(v) { return Number(v).toFixed(1); }

function locationSelects(libs, prefix) {
  const first = libs[0];
  const zones = first?.zones || [];
  return `
    <div class="row">
      <div><label>Library</label>
        <select id="${prefix}-lib">${libs.map((l) => `<option value="${l.id}">${l.name}</option>`).join("")}</select>
      </div>
      <div><label>Zone</label>
        <select id="${prefix}-zone">${zones.map((z) => `<option value="${z.id}">${z.name} (${z.capacity})</option>`).join("")}</select>
      </div>
    </div>`;
}

function bindZoneSelect(libs, prefix) {
  const libEl = document.getElementById(`${prefix}-lib`);
  const zoneEl = document.getElementById(`${prefix}-zone`);
  libEl.addEventListener("change", () => {
    const lib = libs.find((l) => l.id === libEl.value);
    zoneEl.innerHTML = (lib?.zones || []).map((z) => `<option value="${z.id}">${z.name} (${z.capacity})</option>`).join("");
  });
}

function barChart(canvas, labels, values, color = "#3d6b54") {
  const chart = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: color }] },
    options: { plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
  });
  charts.push(chart);
}

function simSelectOptions(libraries) {
  return libraries.map((lib) => `<option value="${lib.id}">${lib.name}</option>`).join("");
}

function simZoneOptions(libraryId) {
  const lib = simLibraries.find((item) => item.id === libraryId);
  return (lib?.zones || []).map((zone) => `<option value="${zone.id}">${zone.name}</option>`).join("");
}

function simJson(method, body) {
  return api("/sim/" + (method.path || "state"), {
    method: method.method || "GET",
    ...(body ? { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}),
  });
}

function simLevelClass(level) {
  return `sim-level--${String(level || "low").toLowerCase()}`;
}

function simForecastFor(state, location) {
  return state.forecast_3h.find((row) => row.library_id === location.library_id && row.zone_id === location.zone_id && row.horizon_h === 1);
}

function renderSimSnapshot(state) {
  const grouped = simLibraries.map((library) => {
    const zones = state.snapshot.filter((zone) => zone.library_id === library.id);
    return `<section class="sim-library"><h3>${library.name}</h3><div class="sim-zone-grid">${zones.map((zone) => {
      const forecast = simForecastFor(state, zone);
      const delta = Number(forecast?.predicted_utilization || 0) - Number(zone.utilization || 0);
      return `<article class="sim-zone ${simLevelClass(zone.level)} ${zone.closed ? "sim-zone--closed" : ""}">
        <div class="sim-zone-head"><strong>${zone.zone_name}</strong>${zone.closed ? "<span class='sim-closed'>CLOSED</span>" : `<span>${zone.level}</span>`}</div>
        <div class="sim-zone-count">${zone.occupied}<small> / ${zone.total_capacity}</small></div>
        <div class="sim-meter"><i style="width:${Math.min(100, zone.utilization * 100)}%"></i></div>
        <div class="sim-zone-foot"><span>${pct(zone.utilization)}</span><span>Next ${pct(forecast?.predicted_utilization || 0)} ${delta >= 0 ? "↑" : "↓"}</span></div>
        <small>${zone.available} seats free · overflow ${num(zone.overflow_this_hour)}</small>
      </article>`;
    }).join("")}</div></section>`;
  }).join("");
  document.getElementById("sim-grid").innerHTML = grouped;
}

function renderSimCharts(state) {
  const libraryId = document.getElementById("sim-chart-library")?.value || simLibraries[0]?.id;
  const zones = state.snapshot.filter((item) => item.library_id === libraryId);
  const datasets = zones.map((zone, index) => {
    const key = `${zone.library_id}/${zone.zone_id}`;
    const history = state.series_24h[key] || [];
    const forecast = state.forecast_3h.filter((item) => item.library_id === zone.library_id && item.zone_id === zone.zone_id);
    const color = ["#3d6b54", "#b0783d", "#b85c38"][index % 3];
    const labels = history.map((item) => item.timestamp.slice(11, 16));
    return [
      { label: zone.zone_name, data: [...history.map((item) => item.utilization), null, null, null], borderColor: color, backgroundColor: color, pointRadius: 0, tension: 0.25 },
      { label: `${zone.zone_name} forecast`, data: [...Array(Math.max(0, history.length - 1)).fill(null), history.at(-1)?.utilization, ...forecast.map((item) => item.predicted_utilization)], borderColor: color, borderDash: [6, 4], pointRadius: 2, tension: 0.25 },
    ];
  }).flat();
  const baseLabels = zones.length ? (state.series_24h[`${libraryId}/${zones[0].zone_id}`] || []).map((item) => item.timestamp.slice(11, 16)) : [];
  const forecastLabels = state.forecast_3h.filter((item) => item.library_id === libraryId && item.zone_id === zones[0]?.zone_id).map((item) => `+${item.horizon_h}h`);
  const labels = [...baseLabels, ...forecastLabels];
  if (!simChart) {
    simChart = new Chart(document.getElementById("sim-chart"), {
      type: "line", data: { labels, datasets },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: { min: 0, max: 1, ticks: { callback: (v) => `${Math.round(v * 100)}%` } } } },
    });
    charts.push(simChart);
  } else {
    simChart.data.labels = labels;
    simChart.data.datasets = datasets;
    simChart.update("none");
  }

  const zoneId = document.getElementById("sim-resource-zone")?.value || zones[0]?.zone_id;
  const resourceRows = (state._resources || []).filter((item) => item.library_id === libraryId && item.zone_id === zoneId);
  const resourceNames = ["book_demand", "computer_usage", "study_room_usage", "charging_station_usage"];
  const resourceLabels = ["Books", "Computers", "Study rooms", "Charging"];
  const resourceDatasets = resourceNames.map((name, index) => ({
    label: resourceLabels[index],
    data: resourceRows.slice(0, 3).map((row) => row[name]),
    backgroundColor: ["#c9a36a", "#3d6b54", "#56828b", "#b85c38"][index],
  }));
  if (!simResourceChart) {
    simResourceChart = new Chart(document.getElementById("sim-resource-chart"), {
      type: "bar", data: { labels: resourceRows.slice(0, 3).map((item) => `+${item.horizon_h}h`), datasets: resourceDatasets },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } }, scales: { y: { beginAtZero: true } } },
    });
    charts.push(simResourceChart);
  } else {
    simResourceChart.data.labels = resourceRows.slice(0, 3).map((item) => `+${item.horizon_h}h`);
    simResourceChart.data.datasets = resourceDatasets;
    simResourceChart.update("none");
  }
}

function renderSimAlerts(state) {
  const alerts = state.recommendations || [];
  document.getElementById("sim-alerts").innerHTML = alerts.length ? alerts.map((item) => {
    const location = state.snapshot.find((zone) => zone.library_id === item.library_id && zone.zone_id === item.zone_id);
    const alternatives = item.alternatives || [];
    return `<article class="sim-alert"><strong>${location?.zone_name || item.zone_id} · ${pct(item.utilization)}</strong>
      <div>${alternatives.length ? alternatives.map((alt) => `<span>${alt.zone_name} · ${num(alt.remaining_seats)} seats free</span>`).join("") : "No open alternative zones."}</div></article>`;
  }).join("") : "<p class='muted'>No zones forecast above the alert threshold.</p>";
  const resourceRecommendations = (state._resources || []).flatMap((row) => ["computer_usage", "study_room_usage"].flatMap((resource) =>
    (row[`${resource}_recommendations`] || []).flatMap((recommendation) => recommendation.alternatives || []).map((alternative) =>
      `<p>${row.zone_name}: ${resource.replaceAll("_", " ")} → ${alternative.zone_name} (${num(alternative.resource_headroom)} free)</p>`
    )
  ));
  const bookAdvisories = (state._resources || []).flatMap((row) => (row.book_demand_recommendations || [])
    .filter((recommendation) => recommendation.advisory)
    .map((recommendation) => `<p>${row.zone_name}: ${recommendation.advisory}</p>`));
  document.getElementById("sim-resource-alerts").innerHTML = [...bookAdvisories, ...resourceRecommendations].slice(0, 8).join("") || "<p class='muted'>No resource redirects predicted.</p>";
}

function renderSimState(state) {
  if (!simPageActive || !state) return;
  simState = state;
  const date = new Date(state.clock.replace(" ", "T"));
  document.getElementById("sim-clock").textContent = date.toLocaleString([], { weekday: "long", year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  document.getElementById("sim-status").textContent = state.ete_in_days !== null && state.ete_in_days !== undefined && state.ete_in_days <= 14 && state.academic_status !== "exam"
    ? `ETE in ${state.ete_in_days} days · ${state.academic_status}` : state.academic_status;
  document.getElementById("sim-mae").textContent = state.live_mae.overall_rolling_24h === null ? "—" : num(state.live_mae.overall_rolling_24h);
  document.getElementById("sim-overflow").textContent = Math.round(state.counters.total_overflow);
  document.getElementById("sim-redirected").textContent = Math.round(state.counters.total_redirected);
  document.getElementById("sim-critical-hours").textContent = state.counters.hours_any_critical;
  document.getElementById("sim-scenarios").innerHTML = state.active_scenarios.length ? state.active_scenarios.map((item) =>
    `<span class="sim-scenario-chip">${item.kind.replaceAll("_", " ")} <button data-cancel="${item.id}" title="Cancel scenario" aria-label="Cancel scenario">×</button></span>`
  ).join("") : "<span class='muted'>No active scenarios</span>";
  document.getElementById("sim-log").innerHTML = state.event_log.map((item) => `<li><time>${item.timestamp.slice(5, 16)}</time>${item.message}</li>`).join("") || "<li class='muted'>No events recorded.</li>";
  renderSimSnapshot(state);
  renderSimAlerts(state);
  renderSimCharts(state);
}

async function loadSimState() {
  const [state, resources] = await Promise.all([
    api("/sim/state"),
    api("/sim/resources?horizon=3").catch(() => []),
  ]);
  state._resources = resources;
  renderSimState(state);
}

async function simAction(path, body, method = "POST") {
  try {
    const state = await api(path, { method, headers: { "Content-Type": "application/json" }, ...(body ? { body: JSON.stringify(body) } : {}) });
    if (path === "/sim/scenario") {
      state.state._resources = await api("/sim/resources?horizon=3").catch(() => []);
      renderSimState(state.state);
      return state;
    }
    if (path.endsWith(".csv")) return state;
    state._resources = await api("/sim/resources?horizon=3").catch(() => []);
    renderSimState(state);
    return state;
  } catch (error) {
    pauseSim();
    const message = document.getElementById("sim-message");
    if (message) { message.textContent = error.message; message.className = "sim-message error"; }
    throw error;
  }
}

function pauseSim() {
  if (simTimer) clearInterval(simTimer);
  simTimer = null;
  const play = document.getElementById("sim-play");
  if (play) play.textContent = "Play";
}

function startSim() {
  if (simTimer) return;
  const play = document.getElementById("sim-play");
  if (play) play.textContent = "Pause";
  simTimer = setInterval(async () => {
    if (simPending || !simPageActive) return;
    simPending = true;
    try { await simAction("/sim/step", { hours: simSpeed }); }
    catch { pauseSim(); }
    finally { simPending = false; }
  }, 1000);
}

async function renderSimulation() {
  simPageActive = true;
  const [{ libraries }] = await Promise.all([api("/libraries")]);
  if (activePage !== "simulation") return;
  simLibraries = libraries;
  view.innerHTML = `
    <div class="sim-toolbar">
      <div class="row"><button class="btn" id="sim-play">Play</button><button class="btn ghost" id="sim-step">Step +1h</button><button class="btn ghost" id="sim-reset">Reset</button>
      <label class="sim-inline">Speed<select id="sim-speed"><option value="1">1h / sec</option><option value="3">3h / sec</option><option value="6">6h / sec</option><option value="12">12h / sec</option><option value="24">24h / sec</option></select></label>
      <label class="sim-inline">Start preset<select id="sim-preset"><option value="normal_week">Normal week</option><option value="midterm">Midterm</option><option value="exam_week">Exam week</option><option value="pre_term_break">Pre-term break</option></select></label>
      <button class="btn ghost" id="sim-start-preset">Start preset</button><a class="btn ghost" href="/sim/export.csv">Export CSV</a></div>
      <div class="sim-recommender"><label><input id="sim-rec-on" type="checkbox" checked /> Recommender</label><label>Compliance <input id="sim-compliance" type="range" min="0" max="100" value="40" /><span id="sim-compliance-value">40%</span></label></div>
    </div>
    <div id="sim-message" class="sim-message" role="status"></div>
    <section class="sim-clock-band"><div><small>SIMULATED TIME</small><h2 id="sim-clock">Loading…</h2></div><strong id="sim-status"></strong></section>
    <div class="sim-kpis"><article><small>1h forecast MAE</small><strong id="sim-mae">—</strong></article><article><small>Turned away</small><strong id="sim-overflow">0</strong></article><article><small>Redirected</small><strong id="sim-redirected">0</strong></article><article><small>Critical hours</small><strong id="sim-critical-hours">0</strong></article></div>
    <section class="sim-scenarios"><div class="row"><button class="btn ghost" id="sim-exam">Start exam surge</button><div class="sim-close-controls"><select id="sim-close-lib">${simSelectOptions(libraries)}</select><select id="sim-close-zone">${simZoneOptions(libraries[0].id)}</select><input id="sim-close-hours" type="number" min="1" value="24" /><button class="btn ghost" id="sim-close">Close a zone</button></div><button class="btn ghost" id="sim-holiday">Declare holiday</button><button class="btn ghost" id="sim-event">Event spike</button><button class="btn ghost" id="sim-random">Random spike</button></div><div id="sim-scenarios" class="sim-scenario-list"></div></section>
    <div id="sim-grid" class="sim-grid"></div>
    <div class="sim-columns"><section class="panel sim-chart-panel"><div class="sim-panel-head"><h3>Utilization · last 24 hours + forecast</h3><select id="sim-chart-library">${simSelectOptions(libraries)}</select></div><div class="sim-chart-wrap"><canvas id="sim-chart"></canvas></div></section>
      <section class="panel sim-chart-panel"><div class="sim-panel-head"><h3>Resource demand · next 3 hours</h3><select id="sim-resource-zone">${simZoneOptions(libraries[0].id)}</select></div><div class="sim-chart-wrap"><canvas id="sim-resource-chart"></canvas></div></section></div>
    <div class="sim-columns"><section class="panel"><h3>Peak alerts and seat alternatives</h3><div id="sim-alerts"></div><h3>Resource alternatives</h3><div id="sim-resource-alerts"></div></section>
      <section class="panel"><div class="sim-panel-head"><h3>Event log</h3><button class="btn ghost" id="sim-compare">Run comparison</button></div><ol id="sim-log" class="sim-event-log"></ol><div id="sim-comparison"></div></section></div>`;

  const closeLibrary = document.getElementById("sim-close-lib");
  closeLibrary.onchange = () => { document.getElementById("sim-close-zone").innerHTML = simZoneOptions(closeLibrary.value); };
  document.getElementById("sim-chart-library").onchange = () => { if (simState) renderSimCharts(simState); };
  const resourceZone = document.getElementById("sim-resource-zone");
  document.getElementById("sim-chart-library").onchange = () => {
    resourceZone.innerHTML = simZoneOptions(document.getElementById("sim-chart-library").value);
    if (simState) renderSimCharts(simState);
  };
  resourceZone.onchange = () => { if (simState) renderSimCharts(simState); };
  document.getElementById("sim-play").onclick = () => simTimer ? pauseSim() : startSim();
  document.getElementById("sim-step").onclick = () => simAction("/sim/step", { hours: 1 });
  document.getElementById("sim-reset").onclick = () => simAction("/sim/reset", {});
  document.getElementById("sim-speed").onchange = (event) => { simSpeed = Number(event.target.value); };
  document.getElementById("sim-start-preset").onclick = () => simAction("/sim/start", { preset: document.getElementById("sim-preset").value });
  document.getElementById("sim-rec-on").onchange = (event) => simAction("/sim/recommender", { enabled: event.target.checked, compliance: Number(document.getElementById("sim-compliance").value) / 100 });
  document.getElementById("sim-compliance").oninput = (event) => { document.getElementById("sim-compliance-value").textContent = `${event.target.value}%`; };
  document.getElementById("sim-compliance").onchange = () => simAction("/sim/recommender", { enabled: document.getElementById("sim-rec-on").checked, compliance: Number(document.getElementById("sim-compliance").value) / 100 });
  document.getElementById("sim-exam").onclick = () => simAction("/sim/scenario", { kind: "exam_surge", days: 5 });
  document.getElementById("sim-holiday").onclick = () => simAction("/sim/scenario", { kind: "holiday", days: 1 });
  document.getElementById("sim-event").onclick = () => simAction("/sim/scenario", { kind: "event_spike", days: 2 });
  document.getElementById("sim-random").onclick = () => simAction("/sim/scenario", { kind: "random_spike" });
  document.getElementById("sim-close").onclick = () => simAction("/sim/scenario", { kind: "zone_closure", library_id: closeLibrary.value, zone_id: document.getElementById("sim-close-zone").value, hours: Number(document.getElementById("sim-close-hours").value) });
  document.getElementById("sim-scenarios").onclick = (event) => {
    const button = event.target.closest("button[data-cancel]");
    if (button) simAction(`/sim/scenario/${button.dataset.cancel}`, null, "DELETE");
  };
  document.getElementById("sim-compare").onclick = async () => {
    const target = document.getElementById("sim-comparison");
    target.textContent = "Running 14-day comparison…";
    try {
      const result = await api(`/sim/compare?days=14&compliance=${Number(document.getElementById("sim-compliance").value) / 100}`);
      const off = result.without_recommender;
      const on = result.with_recommender;
      target.innerHTML = `<p><strong>${num(result.overflow_reduction_percent)}% overflow reduction</strong></p>${table([
        { mode: "Recommender OFF", ...off }, { mode: "Recommender ON", ...on },
      ], [{ key: "mode", label: "Mode" }, { key: "total_overflow", label: "Overflow", fmt: num }, { key: "hours_any_critical", label: "Critical hours" }])}<div class="sim-chart-wrap"><canvas id="sim-compare-chart"></canvas></div>`;
      barChart(document.getElementById("sim-compare-chart"), ["Total overflow", "Critical hours"], [off.total_overflow, off.hours_any_critical], "#b85c38");
      const compareChart = charts[charts.length - 1];
      compareChart.data.datasets.push({ data: [on.total_overflow, on.hours_any_critical], backgroundColor: "#3d6b54" });
      compareChart.update();
    } catch (error) { target.textContent = error.message; }
  };
  document.getElementById("sim-log").closest(".panel").addEventListener("click", (event) => event.stopPropagation());
  await loadSimState();
}

async function renderOverview() {
  const status = await api("/status");
  if (activePage !== "overview") return;
  asOf.textContent = status.as_of ? `As of ${status.as_of}` : "";
  const loc = status.locations || [];
  const util = loc.reduce((s, r) => s + r.predicted_utilization, 0) / (loc.length || 1);
  view.innerHTML = `
    <div class="metrics">
      ${metric("Locations", loc.length)}
      ${metric("Mean utilization", pct(util))}
      ${metric("Seats occupied", loc.reduce((s, r) => s + r.occupied_seats, 0))}
      ${metric("Seats available", loc.reduce((s, r) => s + r.available_seats, 0))}
    </div>
    ${table(loc, [
      { key: "library_name", label: "Library" },
      { key: "zone_name", label: "Zone" },
      { key: "occupied_seats", label: "Occupied" },
      { key: "available_seats", label: "Available" },
      { key: "predicted_utilization", label: "Predicted util.", fmt: pct },
      { key: "occupancy_level", label: "Level" },
    ])}`;
}

async function renderForecast() {
  const { libraries } = await api("/libraries");
  if (activePage !== "forecast") return;
  view.innerHTML = `
    ${locationSelects(libraries, "fc")}
    <div><label>Horizon</label>
      <select id="fc-h"><option>1</option><option selected>3</option><option>6</option></select>
    </div>
    <button class="btn" id="fc-go">Run forecast</button>
    <div id="fc-out"></div>`;
  bindZoneSelect(libraries, "fc");
  document.getElementById("fc-go").onclick = async () => {
    const q = new URLSearchParams({
      horizon: document.getElementById("fc-h").value,
      library_id: document.getElementById("fc-lib").value,
      zone_id: document.getElementById("fc-zone").value,
    });
    const data = await api(`/forecast?${q}`);
    const rows = data.forecasts || [];
    document.getElementById("fc-out").innerHTML = `
      ${table(rows, [
        { key: "timestamp", label: "Time" },
        { key: "horizon_h", label: "Hours" },
        { key: "predicted_occupied", label: "Predicted seats", fmt: num },
        { key: "predicted_utilization", label: "Utilization", fmt: pct },
      ])}
      <div class="panel"><canvas id="fc-chart"></canvas></div>`;
    barChart(document.getElementById("fc-chart"), rows.map((r) => `+${r.horizon_h}h`), rows.map((r) => r.predicted_occupied), "#c9a36a");
  };
}

async function renderAnalytics() {
  const a = await api("/analytics");
  if (activePage !== "analytics") return;
  view.innerHTML = `
    <div class="grid2">
      <div class="panel"><h3>By hour</h3><canvas id="c-hour"></canvas></div>
      <div class="panel"><h3>By weekday</h3><canvas id="c-wd"></canvas></div>
      <div class="panel"><h3>Exam vs normal</h3><canvas id="c-ex"></canvas></div>
      <div class="panel"><h3>Recent daily mean</h3><canvas id="c-day"></canvas></div>
    </div>`;
  barChart(document.getElementById("c-hour"), Object.keys(a.by_hour), Object.values(a.by_hour));
  barChart(document.getElementById("c-wd"), Object.keys(a.by_weekday), Object.values(a.by_weekday));
  barChart(document.getElementById("c-ex"), ["Normal", "Exam"], [a.exam_vs_normal.normal, a.exam_vs_normal.exam], "#b85c38");
  const day = new Chart(document.getElementById("c-day"), {
    type: "line",
    data: { labels: a.daily.map((d) => d.date), datasets: [{ data: a.daily.map((d) => d.occupied_seats), borderColor: "#1c332b", tension: 0.2 }] },
    options: { plugins: { legend: { display: false } } },
  });
  charts.push(day);
}

async function renderResources() {
  const { libraries } = await api("/libraries");
  if (activePage !== "resources") return;
  view.innerHTML = `
    <p class="muted">When predicted utilization is near capacity, alternatives are ranked by remaining seats (same library first).</p>
    ${locationSelects(libraries, "rs")}
    <button class="btn" id="rs-go">Recommend</button>
    <div id="rs-out"></div>`;
  bindZoneSelect(libraries, "rs");
  document.getElementById("rs-go").onclick = async () => {
    const q = new URLSearchParams({
      library_id: document.getElementById("rs-lib").value,
      zone_id: document.getElementById("rs-zone").value,
      horizon: "1",
    });
    const data = await api(`/recommend?${q}`);
    const rec = (data.recommendations || [])[0] || {};
    document.getElementById("rs-out").innerHTML = `
      ${metric("Predicted utilization", pct(rec.utilization || 0))}
      <p>${rec.reason || ""}</p>
      ${table(rec.alternatives || [], [
        { key: "library_name", label: "Library" },
        { key: "zone_name", label: "Zone" },
        { key: "remaining_seats", label: "Remaining", fmt: num },
        { key: "predicted_utilization", label: "Util.", fmt: pct },
        { key: "same_library", label: "Same library" },
      ])}`;
  };
}

async function renderExplain() {
  const [{ libraries }, shap] = await Promise.all([api("/libraries"), api("/shap").catch(() => null)]);
  if (activePage !== "explain") return;
  const items = Object.entries(shap?.mean_abs_shap || {});
  view.innerHTML = `
    <p class="muted">Global mean |SHAP| from the trained tree model, plus a per-location breakdown.</p>
    <div class="panel"><canvas id="shap-g"></canvas></div>
    ${locationSelects(libraries, "ex")}
    <button class="btn" id="ex-go">Explain latest snapshot</button>
    <div id="ex-out"></div>`;
  if (items.length) barChart(document.getElementById("shap-g"), items.slice(0, 12).map((x) => x[0]), items.slice(0, 12).map((x) => x[1]), "#c9a36a");
  bindZoneSelect(libraries, "ex");
  document.getElementById("ex-go").onclick = async () => {
    const q = new URLSearchParams({
      library_id: document.getElementById("ex-lib").value,
      zone_id: document.getElementById("ex-zone").value,
    });
    const data = await api(`/explain?${q}`);
    const contrib = Object.entries(data.contributions || {}).slice(0, 12);
    document.getElementById("ex-out").innerHTML = `${metric("Prediction", num(data.prediction))} <div class="panel"><canvas id="shap-l"></canvas></div>`;
    barChart(document.getElementById("shap-l"), contrib.map((x) => x[0]), contrib.map((x) => x[1]));
  };
}

async function renderPerformance() {
  const [metrics, ablation] = await Promise.all([api("/metrics"), api("/ablation").catch(() => null)]);
  if (activePage !== "performance") return;
  const reg = Object.entries(metrics.regression || {}).map(([k, v]) => ({ model: k, ...v }));
  const base = Object.entries(metrics.baselines || {}).map(([k, v]) => ({ model: k, ...v }));
  const ab = Object.entries(ablation || {}).map(([k, v]) => ({ set: k, ...v }));
  view.innerHTML = `
    <p class="muted">Best regressor: <strong>${metrics.best_regressor || "—"}</strong></p>
    <h3>Regression</h3>
    ${table(reg, [{ key: "model", label: "Model" }, { key: "mae", label: "MAE", fmt: num }, { key: "rmse", label: "RMSE", fmt: num }, { key: "r2", label: "R²", fmt: num }])}
    <h3>Baselines</h3>
    ${table(base, [{ key: "model", label: "Baseline" }, { key: "mae", label: "MAE", fmt: num }, { key: "rmse", label: "RMSE", fmt: num }])}
    <h3>Peak classification</h3>
    <pre class="panel">${JSON.stringify(metrics.classification, null, 2)}</pre>
    <h3>Ablation A–D</h3>
    ${table(ab, [{ key: "set", label: "Set" }, { key: "n_features", label: "Features" }, { key: "mae", label: "MAE", fmt: num }, { key: "r2", label: "R²", fmt: num }])}`;
}

async function renderCalendar() {
  const cal = await api("/calendar");
  if (activePage !== "calendar") return;
  view.innerHTML = `
    <p class="muted">Anyone can read the published calendar. Saving requires the admin token.</p>
    <div class="panel"><pre>${JSON.stringify(cal.terms, null, 2)}</pre></div>
    <label>Admin token</label><input id="tok" type="password" />
    <label>Full calendar JSON</label>
    <textarea id="cal-json">${JSON.stringify(cal, null, 2)}</textarea>
    <button class="btn" id="cal-save">Save calendar</button>
    <p id="cal-msg" class="muted"></p>`;
  document.getElementById("cal-save").onclick = async () => {
    const msg = document.getElementById("cal-msg");
    try {
      const payload = JSON.parse(document.getElementById("cal-json").value);
      await api("/calendar", {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-Admin-Token": document.getElementById("tok").value },
        body: JSON.stringify(payload),
      });
      msg.textContent = "Saved. Retrain after material date changes.";
    } catch (err) {
      msg.textContent = err.message;
      msg.className = "error";
    }
  };
}

async function renderVideo() {
  const { libraries } = await api("/libraries");
  if (activePage !== "video") return;
  view.innerHTML = `
    <div class="banner">
      Demonstration and testing only. Upload a short clip you have the right to use (phone video of a study hall is enough).
      A pretrained YOLOv8 person detector draws boxes, a clutter grid, and an estimated empty-seat count from the selected zone capacity.
      This is not live CCTV, not identity tracking, and not a calibrated seat map.
    </div>
    ${locationSelects(libraries, "vd")}
    <div class="drop">
      <p>Upload library video (mp4 / mov / webm, up to 40 MB). The first ~180 sampled frames are processed.</p>
      <input id="vd-file" type="file" accept="video/*" />
    </div>
    <button class="btn" id="vd-go">Detect people</button>
    <p id="vd-msg" class="muted"></p>
    <div id="vd-out"></div>`;
  bindZoneSelect(libraries, "vd");
  document.getElementById("vd-go").onclick = async () => {
    const file = document.getElementById("vd-file").files[0];
    const msg = document.getElementById("vd-msg");
    if (!file) { msg.textContent = "Choose a video first."; return; }
    const body = new FormData();
    body.append("file", file);
    body.append("library_id", document.getElementById("vd-lib").value);
    body.append("zone_id", document.getElementById("vd-zone").value);
    msg.textContent = "Uploading and running person detection… first run may download YOLO weights.";
    try {
      const job = await api("/demo/video", { method: "POST", body });
      await pollJob(job.job_id, msg);
    } catch (err) {
      msg.textContent = err.message;
      msg.className = "error";
    }
  };
}

async function pollJob(id, msg) {
  for (let i = 0; i < 120; i += 1) {
    const job = await api(`/demo/jobs/${id}`);
    if (job.status === "running") {
      msg.textContent = "Detecting people in sampled frames…";
      await new Promise((r) => setTimeout(r, 1500));
      continue;
    }
    if (job.status === "error") throw new Error(job.error || "Detection failed");
    const s = job.summary;
    const peak = s.peak_occupancy || {};
    const mean = s.mean_occupancy || {};
    document.getElementById("vd-out").innerHTML = `
      <div class="metrics">
        ${metric("Peak people", peak.people_detected ?? s.peak_people)}
        ${metric("Empty seats (est.)", peak.empty_seats_estimate ?? "—")}
        ${metric("Peak clutter", pct(s.peak_clutter_ratio || 0))}
        ${metric("Level", peak.occupancy_level || "—")}
      </div>
      <p class="muted">${s.note || ""} Mean people ${s.mean_people} · ${s.library_name} / ${s.zone_name} (capacity ${mean.capacity}).</p>
      <video controls src="/demo/jobs/${id}/video"></video>
      <div class="film">${(s.previews || []).map((n) => `<img alt="annotated frame" src="/demo/jobs/${id}/frames/${n}" />`).join("")}</div>`;
    msg.textContent = "Done. Red-tinted cells are crowded; green-tinted cells had no person centroid.";
    return;
  }
  throw new Error("Timed out waiting for detection");
}

const pages = {
  overview: renderOverview,
  simulation: renderSimulation,
  forecast: renderForecast,
  analytics: renderAnalytics,
  resources: renderResources,
  explain: renderExplain,
  performance: renderPerformance,
  calendar: renderCalendar,
  video: renderVideo,
};

async function show(name) {
  activePage = name;
  if (simTimer) clearInterval(simTimer);
  simTimer = null;
  simPageActive = name === "simulation";
  destroyCharts();
  asOf.textContent = "";
  kicker.textContent = titles[name][0];
  title.textContent = titles[name][1];
  document.querySelectorAll(".rail button").forEach((b) => b.classList.toggle("active", b.dataset.page === name));
  view.innerHTML = "<p class='muted'>Loading…</p>";
  try {
    await pages[name]();
  } catch (err) {
    view.innerHTML = `<p class="error">${err.message}</p>`;
  }
}

document.getElementById("nav").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-page]");
  if (btn) show(btn.dataset.page);
});

show("overview");
