const stateUrl = "/api/state";
const latestCanvas = document.getElementById("latest-canvas");
const latestCanvasContext = latestCanvas.getContext("2d");
const referenceImage = document.getElementById("reference-image");
const statusPill = document.getElementById("status-pill");
const statusText = document.getElementById("status-text");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const stepButton = document.getElementById("step-button");
const resetButton = document.getElementById("reset-button");
const slider = document.getElementById("epoch-slider");
const sliderValue = document.getElementById("slider-value");
const imagePicker = document.getElementById("image-picker");
const referenceMeta = document.getElementById("reference-meta");
const epochValue = document.getElementById("epoch-value");
const lossValue = document.getElementById("loss-value");
const psnrValue = document.getElementById("psnr-value");
const deviceValue = document.getElementById("device-value");
const outputMeta = document.getElementById("output-meta");
const timingValue = document.getElementById("timing-value");
const errorMessage = document.getElementById("error-message");
const lossChart = document.getElementById("loss-chart");
const chartContext = lossChart.getContext("2d");

let lastRevision = -1;
let busy = false;
let hasSyncedSlider = false;
let lastImageListKey = "";
let latestRunning = false;
let refreshTimer = null;

function previewCount() {
  return Number.parseInt(slider.value, 10);
}

function formatLoss(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  if (value === 0) {
    return "0";
  }
  return value < 0.001 ? value.toExponential(2) : value.toFixed(6);
}

function formatAxisLoss(value) {
  if (value >= 0.01) {
    return value.toFixed(2);
  }
  return value.toExponential(0);
}

function formatPsnr(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  return `${value.toFixed(2)} dB`;
}

function formatSeconds(value) {
  if (value === null || value === undefined) {
    return "--";
  }
  if (value < 1) {
    return `${Math.round(value * 1000)} ms`;
  }
  return `${value.toFixed(2)} s`;
}

async function postJson(url, payload = {}) {
  busy = true;
  renderBusy();
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const state = await response.json();
    renderState(state);
  } finally {
    busy = false;
    renderBusy();
  }
}

async function refreshState() {
  try {
    const response = await fetch(stateUrl, { cache: "no-store" });
    const state = await response.json();
    renderState(state);
  } catch (error) {
    showError(String(error));
  } finally {
    scheduleRefresh();
  }
}

function scheduleRefresh() {
  if (refreshTimer) {
    clearTimeout(refreshTimer);
  }
  refreshTimer = setTimeout(refreshState, latestRunning ? 50 : 700);
}

function renderState(state) {
  const running = Boolean(state.running);
  const training = Boolean(state.training);
  latestRunning = running || training;

  renderImageOptions(state.images || [], state.current_image);

  if (!hasSyncedSlider && state.epochs_per_preview) {
    slider.value = String(state.epochs_per_preview);
    sliderValue.textContent = String(previewCount());
    hasSyncedSlider = true;
  }

  statusPill.classList.toggle("is-running", running || training);
  statusText.textContent = running ? "Training" : training ? "Working" : "Idle";

  epochValue.textContent = String(state.epoch ?? 0);
  outputMeta.textContent = `epoch ${state.epoch ?? 0}`;
  referenceMeta.textContent = `${state.target_size ?? 256} x ${state.target_size ?? 256}`;
  lossValue.textContent = formatLoss(state.loss);
  psnrValue.textContent = formatPsnr(state.psnr);
  deviceValue.textContent = state.device ?? "--";
  timingValue.textContent = state.last_update_seconds
    ? `preview ${formatSeconds(state.last_update_seconds)}`
    : "--";

  if (state.image_revision !== lastRevision) {
    lastRevision = state.image_revision;
    referenceImage.src = `/media/reference.png?rev=${encodeURIComponent(lastRevision)}`;
    drawLatestRaw(lastRevision, state.target_size ?? 256);
  }

  if (state.error) {
    showError(state.error);
  } else {
    hideError();
  }

  startButton.disabled = busy;
  stopButton.disabled = busy || !running;
  stepButton.disabled = busy || running || training;
  resetButton.disabled = busy || training;
  setImageButtonsDisabled(busy || training);
  drawLossChart(state.loss_history || []);
}

async function drawLatestRaw(revision, size) {
  try {
    const response = await fetch(`/media/latest.raw?rev=${encodeURIComponent(revision)}`, {
      cache: "no-store",
    });
    const bytes = new Uint8ClampedArray(await response.arrayBuffer());
    const pixelCount = size * size;
    const rgba = new Uint8ClampedArray(pixelCount * 4);
    for (let source = 0, target = 0; source < bytes.length; source += 3, target += 4) {
      rgba[target] = bytes[source];
      rgba[target + 1] = bytes[source + 1];
      rgba[target + 2] = bytes[source + 2];
      rgba[target + 3] = 255;
    }
    if (latestCanvas.width !== size || latestCanvas.height !== size) {
      latestCanvas.width = size;
      latestCanvas.height = size;
    }
    latestCanvasContext.putImageData(new ImageData(rgba, size, size), 0, 0);
  } catch (error) {
    showError(`Could not draw preview: ${error}`);
  }
}

function renderImageOptions(images, currentImage) {
  const key = images.map((image) => `${image.name}:${image.display_name}`).join("|");
  if (key !== lastImageListKey) {
    imagePicker.replaceChildren();
    for (const image of images) {
      const button = document.createElement("button");
      button.className = "image-option";
      button.type = "button";
      button.dataset.image = image.name;
      button.title = image.display_name || image.name;
      button.setAttribute("aria-label", `Use ${image.display_name || image.name}`);

      const thumb = document.createElement("img");
      thumb.src = `/media/thumb/${encodeURIComponent(image.name)}`;
      thumb.alt = "";
      thumb.loading = "lazy";
      button.append(thumb);

      button.addEventListener("click", () => {
        lastRevision = -1;
        postJson("/api/select", { image: image.name });
      });
      imagePicker.append(button);
    }
    lastImageListKey = key;
  }

  for (const button of imagePicker.querySelectorAll(".image-option")) {
    button.classList.toggle("is-selected", button.dataset.image === currentImage);
    button.setAttribute("aria-pressed", String(button.dataset.image === currentImage));
  }
}

function setImageButtonsDisabled(disabled) {
  for (const button of imagePicker.querySelectorAll(".image-option")) {
    button.disabled = disabled;
  }
}

function renderBusy() {
  startButton.disabled = busy;
  stopButton.disabled = busy;
  stepButton.disabled = busy;
  resetButton.disabled = busy;
  setImageButtonsDisabled(busy);
}

function showError(message) {
  errorMessage.hidden = false;
  errorMessage.textContent = message;
}

function hideError() {
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function prepareCanvas() {
  const rect = lossChart.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.round(rect.width * dpr));
  const height = Math.max(190, Math.round(rect.height * dpr));
  if (lossChart.width !== width || lossChart.height !== height) {
    lossChart.width = width;
    lossChart.height = height;
  }
  chartContext.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width: width / dpr, height: height / dpr };
}

function drawLossChart(history) {
  const { width, height } = prepareCanvas();
  const left = 54;
  const right = 12;
  const top = 14;
  const bottom = 28;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;

  chartContext.clearRect(0, 0, width, height);
  chartContext.fillStyle = "#ffffff";
  chartContext.fillRect(0, 0, width, height);

  const points = history.filter((item) => item.loss > 0);
  if (points.length < 2) {
    chartContext.fillStyle = "#706b74";
    chartContext.font = "16px Google Sans Flex, Segoe UI, sans-serif";
    chartContext.fillText("Waiting for previews", left, height / 2);
    return;
  }

  const epochs = points.map((item) => item.epoch);
  const losses = points.map((item) => item.loss);
  const minEpoch = Math.min(...epochs);
  const maxEpoch = Math.max(...epochs);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const minLog = Math.floor(Math.log10(minLoss));
  const maxLog = Math.ceil(Math.log10(maxLoss));
  const logSpan = Math.max(1, maxLog - minLog);
  const epochSpan = Math.max(1, maxEpoch - minEpoch);

  const xOf = (epoch) => left + ((epoch - minEpoch) / epochSpan) * plotWidth;
  const yOf = (loss) => top + ((maxLog - Math.log10(loss)) / logSpan) * plotHeight;

  chartContext.strokeStyle = "#e7e0ea";
  chartContext.lineWidth = 1;
  chartContext.fillStyle = "#756f79";
  chartContext.font = "12px Google Sans Code, Consolas, monospace";
  chartContext.textAlign = "right";
  chartContext.textBaseline = "middle";

  for (let exponent = minLog; exponent <= maxLog; exponent += 1) {
    const value = 10 ** exponent;
    const y = yOf(value);
    chartContext.beginPath();
    chartContext.moveTo(left, y);
    chartContext.lineTo(width - right, y);
    chartContext.stroke();
    chartContext.fillText(formatAxisLoss(value), left - 8, y);
  }

  chartContext.strokeStyle = "#bcb4c1";
  chartContext.beginPath();
  chartContext.moveTo(left, top);
  chartContext.lineTo(left, height - bottom);
  chartContext.lineTo(width - right, height - bottom);
  chartContext.stroke();

  chartContext.textAlign = "center";
  chartContext.textBaseline = "top";
  chartContext.fillText(String(Math.round(minEpoch)), left, height - bottom + 8);
  chartContext.fillText(String(Math.round(maxEpoch)), width - right, height - bottom + 8);

  const maxDrawPoints = 1400;
  const stride = Math.max(1, Math.floor(points.length / maxDrawPoints));
  const sampled = points.filter((_, index) => index % stride === 0 || index === points.length - 1);

  chartContext.strokeStyle = "#2f5f4f";
  chartContext.lineWidth = 3;
  chartContext.lineJoin = "round";
  chartContext.lineCap = "round";
  chartContext.beginPath();
  sampled.forEach((item, index) => {
    const x = xOf(item.epoch);
    const y = yOf(item.loss);
    if (index === 0) {
      chartContext.moveTo(x, y);
    } else {
      chartContext.lineTo(x, y);
    }
  });
  chartContext.stroke();

  const last = points[points.length - 1];
  chartContext.fillStyle = "#2f5f4f";
  chartContext.beginPath();
  chartContext.arc(xOf(last.epoch), yOf(last.loss), 4.5, 0, Math.PI * 2);
  chartContext.fill();
}

slider.addEventListener("input", () => {
  sliderValue.textContent = String(previewCount());
});

startButton.addEventListener("click", () => {
  postJson("/api/start", { epochs_per_preview: previewCount() });
});

stopButton.addEventListener("click", () => {
  postJson("/api/stop");
});

stepButton.addEventListener("click", () => {
  postJson("/api/step", { steps: previewCount() });
});

resetButton.addEventListener("click", () => {
  postJson("/api/reset");
});

window.addEventListener("resize", () => refreshState());

sliderValue.textContent = String(previewCount());
refreshState();
