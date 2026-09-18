const state = {
  flare: null,
  busy: false,
};

const colors = {
  mse: "#8bc5ff",
  mae: "#ffb86c",
  r2: "#7ee787",
  grid: "rgba(255, 255, 255, 0.10)",
  text: "#9aa7b4",
  objective: "#d2a8ff",
  sites: ["#ff9f7a", "#59d8a6", "#ffd166", "#b48cff", "#4dd6e7", "#f27cb3"],
};

const form = document.querySelector("#flare-form");
const startButton = document.querySelector("#start-flare-btn");
const resultOutput = document.querySelector("#result-output");
const eventLog = document.querySelector("#event-log");
const pollState = document.querySelector("#poll-state");
const clock = document.querySelector("#clock");
const chart = document.querySelector("#flare-chart");
const ctx = chart.getContext("2d");
const lossChart = document.querySelector("#loss-chart");
const lossCtx = lossChart.getContext("2d");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "na";
  return Number(value).toFixed(digits);
}

function fmtTime(timestamp) {
  if (!timestamp) return "";
  return new Date(timestamp * 1000).toLocaleTimeString();
}

function statusClass(status) {
  if (!status) return "healthy";
  return String(status).toLowerCase();
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body = {};
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { raw: text };
    }
  }
  if (!response.ok) {
    const message = body.detail || response.statusText;
    throw new Error(typeof message === "string" ? message : JSON.stringify(message));
  }
  return body;
}

function flarePayload() {
  return {
    n_sites: Number(document.querySelector("#sites-input").value || 2),
    rounds: Number(document.querySelector("#rounds-input").value || 2),
    local_epochs: Number(document.querySelector("#epochs-input").value || 1),
    split: document.querySelector("#split-input").value || "iid",
    loss: document.querySelector("#loss-input").value || "mse",
    mu: Number(document.querySelector("#mu-input").value || 0),
  };
}

function setBusy(isBusy) {
  state.busy = isBusy;
  startButton.disabled = isBusy || Boolean(state.flare?.job?.running);
}

function bestTarget(summary) {
  const targets = summary?.evaluation?.targets || [];
  if (!targets.length) return null;
  return targets.reduce((best, item) => (item.r2 > best.r2 ? item : best), targets[0]);
}

function renderStatus(payload) {
  const job = payload?.job || {};
  const summary = payload?.summary || {};
  const status = job.running ? "running" : summary.status || job.status || "idle";
  const best = bestTarget(summary);

  const pill = document.querySelector("#flare-status-pill");
  pill.textContent = status;
  pill.className = `status-pill ${statusClass(status)}`;
  document.querySelector("#run-status").textContent = status;
  document.querySelector("#run-id").textContent = summary.run_id || "na";
  document.querySelector("#best-r2").textContent = fmtNumber(best?.r2, 3);
  document.querySelector("#best-mse").textContent = fmtNumber(best?.mse);
  startButton.disabled = state.busy || Boolean(job.running);
}

function renderEvents(payload) {
  const events = payload?.events || [];
  if (!events.length) {
    eventLog.innerHTML = `<li><strong>No events yet</strong>Waiting for FLARE activity.</li>`;
    return;
  }
  eventLog.innerHTML = events
    .slice(0, 12)
    .map((item) => `<li><strong>${fmtTime(item.at)} · ${escapeHtml(item.level)}</strong>${escapeHtml(item.message)}</li>`)
    .join("");
}

function renderTable(summary) {
  const targets = summary?.evaluation?.targets || [];
  document.querySelector("#target-table").innerHTML =
    targets
      .map(
        (item) => `
          <tr>
            <td>${escapeHtml(item.target)}</td>
            <td>${fmtNumber(item.mse)}</td>
            <td>${fmtNumber(item.mae)}</td>
            <td>${fmtNumber(item.r2, 3)}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="4">No FLARE evaluation yet.</td></tr>`;
}

function resizeCanvas() {
  const box = chart.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  chart.width = Math.max(600, Math.floor(box.width * ratio));
  chart.height = Math.max(300, Math.floor(box.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width: box.width, height: box.height };
}

function resizeLossCanvas() {
  const box = lossChart.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  lossChart.width = Math.max(600, Math.floor(box.width * ratio));
  lossChart.height = Math.max(300, Math.floor(box.height * ratio));
  lossCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { width: box.width, height: box.height };
}

function nvflareOutput(summary) {
  return summary?.commands?.find((item) => item.name === "nvflare_sim")?.output || "";
}

function parseRoundLosses(summary) {
  const output = nvflareOutput(summary);
  const globals = [];
  const nodes = [];
  const globalRegex =
    /Round\s+(\d+)\s+global metrics:\s+\{[^}]*['"]val_mse['"]:\s*([-+0-9.eE]+)[^}]*['"]train_loss['"]:\s*([-+0-9.eE]+)/g;
  const nodeRegex =
    /\[(site-\d+)\]\s+round\s+(\d+):\s+global val MSE\s+([-+0-9.eE]+),\s+local train (?:MSE|loss)\s+([-+0-9.eE]+)(?:,\s+objective\s+([-+0-9.eE]+))?/g;

  for (const match of output.matchAll(globalRegex)) {
    globals.push({
      round: Number(match[1]),
      valMse: Number(match[2]),
      trainMse: Number(match[3]),
    });
  }
  for (const match of output.matchAll(nodeRegex)) {
    nodes.push({
      site: match[1],
      round: Number(match[2]),
      valMse: Number(match[3]),
      trainMse: Math.max(0, Number(match[4])),
      objective: match[5] === undefined ? null : Number(match[5]),
    });
  }
  return { globals, nodes };
}

function drawChart(summary) {
  const { width, height } = resizeCanvas();
  ctx.clearRect(0, 0, width, height);

  const targets = summary?.evaluation?.targets || [];
  document.querySelector("#flare-legend").innerHTML = `
    <span><i style="background:${colors.mse}"></i>MSE</span>
    <span><i style="background:${colors.mae}"></i>MAE</span>
    <span><i style="background:${colors.r2}"></i>R2</span>
  `;

  const padding = { left: 58, right: 26, top: 24, bottom: 48 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  ctx.strokeStyle = colors.grid;
  ctx.fillStyle = colors.text;
  ctx.font = "12px Inter, sans-serif";

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  if (!targets.length) {
    ctx.fillText("No held-out test metrics yet.", padding.left, padding.top + 34);
    return;
  }

  const bars = targets.flatMap((target) => [
    { target: target.target, label: "MSE", value: target.mse, color: colors.mse },
    { target: target.target, label: "MAE", value: target.mae, color: colors.mae },
    { target: target.target, label: "R2", value: target.r2, color: colors.r2 },
  ]);
  const maxY = Math.max(0.001, ...bars.map((item) => Math.max(0, Number(item.value)))) * 1.15;
  const groupW = plotW / targets.length;
  const barW = Math.min(38, groupW / 5);

  for (let i = 0; i <= 4; i += 1) {
    const value = maxY - (maxY * i) / 4;
    const y = padding.top + (plotH * i) / 4;
    ctx.fillText(value.toFixed(3), 8, y + 4);
  }

  targets.forEach((target, groupIndex) => {
    const groupX = padding.left + groupW * groupIndex + groupW / 2;
    const items = [
      { value: target.mse, color: colors.mse },
      { value: target.mae, color: colors.mae },
      { value: target.r2, color: colors.r2 },
    ];
    items.forEach((item, itemIndex) => {
      const value = Math.max(0, Number(item.value));
      const barH = (value / maxY) * plotH;
      const x = groupX + (itemIndex - 1) * (barW + 8) - barW / 2;
      const y = padding.top + plotH - barH;
      ctx.fillStyle = item.color;
      ctx.fillRect(x, y, barW, barH);
    });
    ctx.fillStyle = colors.text;
    ctx.fillText(target.target, groupX - 34, height - 18);
  });
}

function drawLossChart(summary) {
  const { width, height } = resizeLossCanvas();
  lossCtx.clearRect(0, 0, width, height);
  const { globals, nodes } = parseRoundLosses(summary);
  const sites = [...new Set(nodes.map((item) => item.site))].sort();

  const legendItems = [
    `<span><i style="background:${colors.mse}"></i>Global val MSE</span>`,
    ...sites.map(
      (site, index) =>
        `<span><i style="background:${colors.sites[index % colors.sites.length]}"></i>${escapeHtml(site)} train MSE</span>`,
    ),
  ];
  document.querySelector("#loss-legend").innerHTML = legendItems.join("");

  const padding = { left: 58, right: 26, top: 24, bottom: 48 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;
  lossCtx.strokeStyle = colors.grid;
  lossCtx.fillStyle = colors.text;
  lossCtx.font = "12px Inter, sans-serif";

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotH * i) / 4;
    lossCtx.beginPath();
    lossCtx.moveTo(padding.left, y);
    lossCtx.lineTo(width - padding.right, y);
    lossCtx.stroke();
  }

  const values = [
    ...globals.map((item) => item.valMse),
    ...nodes.map((item) => item.trainMse),
  ].filter((value) => Number.isFinite(value));

  if (!values.length) {
    lossCtx.fillText("No round loss data yet.", padding.left, padding.top + 34);
    document.querySelector("#node-loss-table").innerHTML = `<tr><td colspan="4">No node losses yet.</td></tr>`;
    return;
  }

  const rounds = [...new Set([...globals.map((item) => item.round), ...nodes.map((item) => item.round)])].sort((a, b) => a - b);
  const minRound = rounds[0];
  const maxRound = rounds[rounds.length - 1];
  const xForRound = (round) =>
    padding.left + (maxRound === minRound ? plotW / 2 : ((round - minRound) / (maxRound - minRound)) * plotW);
  const maxY = Math.max(0.001, ...values) * 1.15;
  const yForValue = (value) => padding.top + plotH - (Math.max(0, value) / maxY) * plotH;

  for (let i = 0; i <= 4; i += 1) {
    const value = maxY - (maxY * i) / 4;
    const y = padding.top + (plotH * i) / 4;
    lossCtx.fillStyle = colors.text;
    lossCtx.fillText(value.toFixed(3), 8, y + 4);
  }

  function drawSeries(points, valueKey, color) {
    const usable = points.filter((item) => Number.isFinite(item[valueKey])).sort((a, b) => a.round - b.round);
    if (!usable.length) return;
    lossCtx.strokeStyle = color;
    lossCtx.fillStyle = color;
    lossCtx.lineWidth = 2;
    lossCtx.beginPath();
    usable.forEach((point, index) => {
      const x = xForRound(point.round);
      const y = yForValue(point[valueKey]);
      if (index === 0) lossCtx.moveTo(x, y);
      else lossCtx.lineTo(x, y);
    });
    lossCtx.stroke();
    usable.forEach((point) => {
      const x = xForRound(point.round);
      const y = yForValue(point[valueKey]);
      lossCtx.beginPath();
      lossCtx.arc(x, y, 3.5, 0, Math.PI * 2);
      lossCtx.fill();
    });
  }

  drawSeries(globals, "valMse", colors.mse);
  sites.forEach((site, index) => {
    drawSeries(nodes.filter((item) => item.site === site), "trainMse", colors.sites[index % colors.sites.length]);
  });

  rounds.forEach((round) => {
    const x = xForRound(round);
    lossCtx.fillStyle = colors.text;
    lossCtx.fillText(`R${round}`, x - 8, height - 18);
  });

  const latestBySite = sites.map((site) =>
    nodes.filter((item) => item.site === site).sort((a, b) => b.round - a.round)[0],
  );
  document.querySelector("#node-loss-table").innerHTML =
    latestBySite
      .filter(Boolean)
      .map(
        (item) => `
          <tr>
            <td>${item.round}</td>
            <td>${escapeHtml(item.site)}</td>
            <td>${fmtNumber(item.trainMse)}</td>
            <td>${item.objective === null ? "same as MSE" : fmtNumber(item.objective)}</td>
          </tr>
        `,
      )
      .join("") || `<tr><td colspan="4">No node losses yet.</td></tr>`;
}

function renderDetails(payload) {
  const summary = payload?.summary || null;
  const details = {
    output_dir: payload?.output_dir,
    status: summary?.status || payload?.job?.status,
    run_id: summary?.run_id,
    parameters: summary?.parameters || payload?.job?.parameters,
    models: summary?.models,
    error: summary?.error || payload?.job?.error,
  };
  resultOutput.textContent = JSON.stringify(details, null, 2);
}

async function refresh() {
  try {
    const payload = await api("/api/flare/status");
    state.flare = payload;
    renderStatus(payload);
    renderEvents(payload);
    renderTable(payload.summary);
    drawChart(payload.summary);
    drawLossChart(payload.summary);
    renderDetails(payload);
    pollState.textContent = "Live";
    pollState.className = "status-pill healthy";
  } catch (error) {
    pollState.textContent = "Disconnected";
    pollState.className = "status-pill failed";
    resultOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  try {
    await api("/api/flare/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(flarePayload()),
    });
    await refresh();
  } catch (error) {
    resultOutput.textContent = JSON.stringify({ error: error.message }, null, 2);
  } finally {
    setBusy(false);
  }
});

function tickClock() {
  clock.textContent = new Date().toLocaleString();
}

window.addEventListener("resize", () => {
  drawChart(state.flare?.summary);
  drawLossChart(state.flare?.summary);
});

tickClock();
setInterval(tickClock, 1000);
refresh();
setInterval(refresh, 3000);
