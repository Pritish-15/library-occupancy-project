const view = document.getElementById("view");
const title = document.getElementById("title");
const kicker = document.getElementById("kicker");
const asOf = document.getElementById("asOf");
let charts = [];

const titles = {
  overview: ["Campus occupancy", "Overview"],
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

async function renderOverview() {
  const status = await api("/status");
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
  forecast: renderForecast,
  analytics: renderAnalytics,
  resources: renderResources,
  explain: renderExplain,
  performance: renderPerformance,
  calendar: renderCalendar,
  video: renderVideo,
};

async function show(name) {
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
