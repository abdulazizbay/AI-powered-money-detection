const COLORS = {
  coin_10: "#f59e0b", coin_50: "#22c55e", coin_100: "#3b82f6", coin_500: "#c026d3",
  bill_1000: "#06b6d4", bill_5000: "#f43f5e", bill_10000: "#84cc16", bill_50000: "#ef4444",
};
const LABEL = (c) => `₩${Number(c.split("_")[1]).toLocaleString()} ${c.startsWith("coin") ? "coin" : "bill"}`;
const won = (n) => `₩${n.toLocaleString()}`;
const $ = (id) => document.getElementById(id);

const stage = $("stage"), photo = $("photo"), video = $("video"), overlay = $("overlay");
let mode = "upload", conf = 0.4, lastFile = null, lastResult = null, liveTimer = null, busy = false, rates = null, chart = null;

// ---------- detection ----------
async function detect(blob, { source = "upload", save = true } = {}) {
  const form = new FormData();
  form.append("file", blob, "image.jpg");
  const r = await fetch(`/api/detect?source=${source}&save=${save}&conf=${conf}`, { method: "POST", body: form });
  if (!r.ok) throw new Error(await r.text());
  const result = await r.json();
  render(result);
  if (save) loadHistory();
  return result;
}

function drawBoxes(result) {
  const rect = stage.getBoundingClientRect();
  overlay.width = rect.width * devicePixelRatio;
  overlay.height = rect.height * devicePixelRatio;
  const ctx = overlay.getContext("2d");
  ctx.scale(devicePixelRatio, devicePixelRatio);
  ctx.clearRect(0, 0, rect.width, rect.height);
  // the media is drawn with object-fit: contain
  const { width: iw, height: ih } = result.image;
  const s = Math.min(rect.width / iw, rect.height / ih);
  const ox = (rect.width - iw * s) / 2, oy = (rect.height - ih * s) / 2;
  ctx.font = "600 13px Inter, sans-serif";
  for (const d of result.detections) {
    const [x1, y1, x2, y2] = d.box.map((v, i) => v * s + (i % 2 ? oy : ox));
    const color = COLORS[d.label];
    ctx.strokeStyle = color; ctx.lineWidth = 2.5;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${won(d.value)} ${d.confidence.toFixed(2)}`;
    const tw = ctx.measureText(text).width + 10;
    const ty = y1 > 22 ? y1 - 22 : y1;
    ctx.fillStyle = color; ctx.fillRect(x1, ty, tw, 20);
    ctx.fillStyle = "#fff"; ctx.fillText(text, x1 + 5, ty + 14);
  }
}

function render(result) {
  lastResult = result;
  drawBoxes(result);
  $("ms").textContent = `${result.inference_ms} ms`;
  $("total").textContent = won(result.total);
  $("items").textContent = `${result.items} item${result.items === 1 ? "" : "s"} detected`;
  renderRates(result.total);

  const entries = Object.entries(result.counts).sort((a, b) => b[0].split("_")[1] - a[0].split("_")[1]);
  $("item-list").innerHTML = entries.length
    ? entries.map(([c, n]) => {
        const v = Number(c.split("_")[1]);
        return `<li><span class="dot" style="background:${COLORS[c]}"></span>
          <span class="name">${LABEL(c)}</span><span class="muted">× ${n}</span><b>${won(v * n)}</b></li>`;
      }).join("")
    : `<li class="empty">Nothing detected</li>`;

  const labels = entries.map(([c]) => LABEL(c));
  const data = entries.map(([c, n]) => Number(c.split("_")[1]) * n);
  const colors = entries.map(([c]) => COLORS[c]);
  if (!chart) {
    chart = new Chart($("chart"), {
      type: "doughnut",
      data: { labels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
      options: {
        maintainAspectRatio: false, cutout: "65%",
        plugins: { legend: { position: "right", labels: { color: "#cbd5e1", boxWidth: 12 } },
                   tooltip: { callbacks: { label: (c) => ` ${won(c.raw)}` } } },
      },
    });
  } else {
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.data.datasets[0].backgroundColor = colors;
    chart.update();
  }
}

function renderRates(total) {
  if (!rates) return;
  const symbols = { USD: "$", EUR: "€", JPY: "¥", CNY: "CN¥" };
  $("rates").innerHTML = Object.entries(rates.rates).map(([cur, r]) =>
    `<div><small>${cur}</small>${symbols[cur] || ""}${(total * r).toLocaleString(undefined, { maximumFractionDigits: 2 })}</div>`
  ).join("");
}

// ---------- upload ----------
function showImage(file) {
  lastFile = file;
  photo.src = URL.createObjectURL(file);
  photo.hidden = false;
  stage.classList.add("has-image");
  photo.onload = () => detect(file).catch(alert);
}

$("file").addEventListener("change", (e) => e.target.files[0] && showImage(e.target.files[0]));
const drop = $("drop");
["dragover", "dragenter"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); e.dataTransfer.files[0] && showImage(e.dataTransfer.files[0]); });

$("conf").addEventListener("input", (e) => {
  conf = Number(e.target.value);
  $("conf-val").textContent = conf.toFixed(2);
});
$("conf").addEventListener("change", () => mode === "upload" && lastFile && detect(lastFile, { save: false }));

// sample images bundled with the page
fetch("samples/index.json").then((r) => r.ok ? r.json() : []).then((files) => {
  $("samples").innerHTML = files.map((f) => `<img src="samples/${f}" alt="sample">`).join("");
  $("samples").querySelectorAll("img").forEach((img) => img.addEventListener("click", async () => {
    if (mode !== "upload") switchMode("upload");
    const blob = await (await fetch(img.src)).blob();
    showImage(new File([blob], "sample.jpg", { type: blob.type }));
  }));
}).catch(() => {});

// ---------- live camera ----------
function frameBlob() {
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  return new Promise((res) => c.toBlob(res, "image/jpeg", 0.8));
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720, facingMode: "environment" } });
  video.srcObject = stream;
  video.hidden = false;
  stage.classList.add("has-image");
  $("live-dot").hidden = false;
  $("capture").hidden = false;
  liveTimer = setInterval(async () => {
    if (busy || !video.videoWidth) return;
    busy = true;
    try { await detect(await frameBlob(), { source: "camera", save: false }); } finally { busy = false; }
  }, 350);
}

function stopCamera() {
  clearInterval(liveTimer);
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  video.hidden = true;
  $("live-dot").hidden = true;
  $("capture").hidden = true;
}

$("capture").addEventListener("click", async () => detect(await frameBlob(), { source: "camera", save: true }));

function switchMode(next) {
  mode = next;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === next));
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  if (next === "camera") {
    photo.hidden = true;
    drop.hidden = true;
    startCamera().catch((e) => { alert(`Camera unavailable: ${e.message}`); switchMode("upload"); });
  } else {
    stopCamera();
    drop.hidden = false;
    if (lastFile) { photo.hidden = false; } else { stage.classList.remove("has-image"); }
  }
}
document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchMode(t.dataset.mode)));

// ---------- history, rates, model ----------
async function loadHistory() {
  const rows = await (await fetch("/api/history?limit=8")).json();
  $("history").innerHTML = rows.length
    ? rows.map((r) => `<tr><td>${new Date(r.created_at * 1000).toLocaleTimeString()}</td><td>${r.source}</td>
        <td>${r.items}</td><td><b>${won(r.total)}</b></td><td>${r.avg_confidence.toFixed(2)}</td></tr>`).join("")
    : `<tr><td colspan="5" class="empty">No scans yet</td></tr>`;
}
$("clear").addEventListener("click", async () => { await fetch("/api/history", { method: "DELETE" }); loadHistory(); });

fetch("/api/rates").then((r) => r.ok ? r.json() : null).then((d) => { rates = d; renderRates(0); });

fetch("/api/model").then((r) => r.json()).then(({ metrics: m }) => {
  if (!m) return;
  $("badge-map").textContent = `${(m.test_mAP50 * 100).toFixed(1)}%`;
  $("badge-ms").textContent = `${Math.round(m.inference_ms)} ms`;
  $("metrics").innerHTML = [
    ["mAP50", m.test_mAP50], ["mAP50-95", m.test_mAP50_95], ["Precision", m.test_precision], ["Recall", m.test_recall],
  ].map(([k, v]) => `<div><b>${(v * 100).toFixed(1)}%</b><small>${k}</small></div>`).join("");
});

window.addEventListener("resize", () => lastResult && drawBoxes(lastResult));
loadHistory();
