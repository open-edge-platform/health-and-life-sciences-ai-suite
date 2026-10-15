const DEVICE_LABELS = { CPU: "CPU", GPU: "Intel iGPU", NPU: "Intel NPU" };

const els = {
  gate: document.getElementById("gate"),
  gateSub: document.getElementById("gate-sub"),
  conn: document.getElementById("conn"),
  usecase: document.getElementById("usecase"),
  ucNote: document.getElementById("uc-note"),
  device: document.getElementById("device"),
  precision: document.getElementById("precision"),
  video: document.getElementById("video"),
  start: document.getElementById("start"),
  stop: document.getElementById("stop"),
  reset: document.getElementById("reset"),
  err: document.getElementById("err"),
  statePill: document.getElementById("state-pill"),
  perfRow: document.getElementById("perf-row"),
  miModel: document.getElementById("mi-model"),
  miPrec: document.getElementById("mi-prec"),
  miTask: document.getElementById("mi-task"),
  miInput: document.getElementById("mi-input"),
  miDevice: document.getElementById("mi-device"),
  miDevname: document.getElementById("mi-devname"),
  miIr: document.getElementById("mi-ir"),
  cpuBar: document.getElementById("cpu-bar"),
  gpuBar: document.getElementById("gpu-bar"),
  npuBar: document.getElementById("npu-bar"),
  cpuPct: document.getElementById("cpu-pct"),
  gpuPct: document.getElementById("gpu-pct"),
  npuPct: document.getElementById("npu-pct"),
};

let usecases = [];

function fmt(x) {
  const n = Number(x);
  return Number.isFinite(n) ? n.toFixed(1) : "0.0";
}

function taskLabel(task) {
  if (task === "segmentation") return "LV Segmentation";
  if (task === "ejection_fraction") return "Ejection Fraction";
  if (task === "ovarian_sam") return "Ovarian Tumor Segmentation";
  return task || "-";
}

function setDashDefaults() {
  els.miModel.textContent = "—";
  els.miPrec.textContent = "—";
  els.miTask.textContent = "—";
  els.miInput.textContent = "—";
  els.miDevice.textContent = "—";
  els.miDevname.textContent = "—";
  els.miIr.textContent = "—";
  els.perfRow.innerHTML = '<td colspan="11" class="muted">Not running</td>';
}

function setUtil(barEl, pctEl, val, nullLabel) {
  if (val == null || Number.isNaN(Number(val))) {
    barEl.style.width = "0%";
    pctEl.textContent = nullLabel;
    return;
  }
  const pct = Math.max(0, Math.min(100, Number(val)));
  barEl.style.width = `${pct}%`;
  pctEl.textContent = `${Math.round(pct)}%`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  return res.json();
}

async function waitReadiness() {
  while (true) {
    try {
      const data = await api("/api/readiness");
      const names = (data.devices || []).map((d) => d.id).join(", ");
      els.gateSub.textContent = names ? `Detected devices: ${names}` : "Checking OpenVINO devices";
      if (data.ready) {
        els.gate.classList.add("hidden");
        return;
      }
    } catch (_e) {
      els.gateSub.textContent = "Backend not ready yet";
    }
    await new Promise((r) => setTimeout(r, 800));
  }
}

function fillSelect(sel, items) {
  sel.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = item.value;
    opt.textContent = item.label;
    opt.disabled = !!item.disabled;
    sel.appendChild(opt);
  }
}

function selectUsecase() {
  const uc = usecases.find((x) => x.key === els.usecase.value);
  if (!uc) {
    els.ucNote.textContent = "";
    fillSelect(els.device, []);
    fillSelect(els.precision, []);
    return;
  }
  els.ucNote.textContent = uc.note || "";
  fillSelect(
    els.device,
    (uc.devices || []).map((d) => ({ value: d, label: DEVICE_LABELS[d] || d }))
  );
  fillSelect(
    els.precision,
    (uc.precisions || []).map((p) => ({ value: p, label: String(p).toUpperCase() }))
  );
  els.video.placeholder = "upstream/UltraSam/sample_dataset/sample_images";
  els.video.value = "";
}

async function loadUsecases() {
  usecases = await api("/api/usecases");
  const options = usecases.map((uc) => {
    const disabled = !uc.enabled || !uc.devices || uc.devices.length === 0;
    return {
      value: uc.key,
      label: disabled ? `${uc.label} (coming soon)` : uc.label,
      disabled,
    };
  });
  fillSelect(els.usecase, options);
  const firstEnabled = options.find((o) => !o.disabled);
  if (firstEnabled) els.usecase.value = firstEnabled.value;
  selectUsecase();
}

function render(snap) {
  const state = snap.state || "idle";
  const metrics = snap.metrics || {};
  const platform = snap.platform || {};
  const cfg = snap.config;

  els.statePill.textContent = state;
  els.statePill.className = `pill ${state}`;
  els.err.textContent = snap.error || "";

  if (!cfg) {
    setDashDefaults();
  } else {
    let kpi = "—";
    if (Number(metrics.mean_ms) > 0) {
      const pass = Number(metrics.mean_ms) <= Number(cfg.kpi_latency_ms) &&
        Number(metrics.fps) >= Number(cfg.kpi_fps);
      kpi = pass
        ? '<span class="kpi-pass">PASS</span>'
        : '<span class="kpi-fail">FAIL</span>';
    }

    els.perfRow.innerHTML = `
      <td>${taskLabel(cfg.task)}</td>
      <td>${cfg.model || "-"}</td>
      <td><span class="pill dev-${cfg.device}">${DEVICE_LABELS[cfg.device] || cfg.device || "-"}</span></td>
      <td>${String(cfg.precision || "-").toUpperCase()}</td>
      <td>${fmt(metrics.fps)}</td>
      <td>${fmt(metrics.mean_ms)}</td>
      <td>${fmt(metrics.p50_ms)}</td>
      <td>${fmt(metrics.p90_ms)}</td>
      <td>${fmt(metrics.p95_ms)}</td>
      <td>${fmt(metrics.p99_ms)}</td>
      <td>${kpi}</td>
    `;

    els.miModel.textContent = cfg.model || "—";
    els.miPrec.textContent = `${String(cfg.precision || "-").toUpperCase()} OpenVINO IR`;
    els.miTask.textContent = taskLabel(cfg.task);
    els.miInput.textContent = (cfg.input_shape || []).join("×") || "—";
    els.miDevice.textContent = cfg.device || "—";
    els.miDevname.textContent = (snap.model_info || {}).device_name || "—";
    els.miIr.textContent = (snap.model_info || {}).ir || "—";
  }

  setUtil(els.cpuBar, els.cpuPct, platform.cpu_pct, "—");
  setUtil(els.gpuBar, els.gpuPct, platform.gpu_pct, "n/a");
  setUtil(els.npuBar, els.npuPct, platform.npu_pct, "n/a");
}

async function onStart() {
  const payload = {
    usecase: els.usecase.value,
    device: els.device.value,
    precision: els.precision.value,
    video: (els.video.value || "").trim() || null,
  };
  const out = await api("/api/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  els.err.textContent = out.ok ? "" : (out.error || "Failed to start");
}

async function onStop() {
  await api("/api/stop", { method: "POST" });
}

async function onReset() {
  await api("/api/reset", { method: "POST" });
}

function bind() {
  els.usecase.addEventListener("change", selectUsecase);
  els.start.addEventListener("click", () => onStart().catch((e) => {
    els.err.textContent = String(e);
  }));
  els.stop.addEventListener("click", () => onStop().catch((e) => {
    els.err.textContent = String(e);
  }));
  els.reset.addEventListener("click", () => onReset().catch((e) => {
    els.err.textContent = String(e);
  }));
}

function startEvents() {
  const es = new EventSource("/api/events");
  es.onopen = () => {
    els.conn.textContent = "connected";
  };
  es.onerror = () => {
    els.conn.textContent = "reconnecting…";
  };
  es.onmessage = (ev) => {
    try {
      render(JSON.parse(ev.data));
    } catch (_e) {
      // Ignore malformed transient event payloads.
    }
  };
}

async function main() {
  bind();
  setDashDefaults();
  await waitReadiness();
  await loadUsecases();
  startEvents();
}

main().catch((e) => {
  els.err.textContent = String(e);
});