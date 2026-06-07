const TARGET_SIZE = 256;
const PIXEL_COUNT = TARGET_SIZE * TARGET_SIZE;
const MODEL = {
  featureDim: 256,
  hiddenDim: 128,
  layers: 2,
  freqScale: 35,
  learningRate: 0.01,
  lrDecay: 0.995,
  minLearningRate: 0.001,
  seed: 7,
};

const referenceCanvas = document.getElementById("reference-canvas");
const referenceContext = referenceCanvas.getContext("2d", { willReadFrequently: true });
const latestCanvas = document.getElementById("latest-canvas");
const latestCanvasContext = latestCanvas.getContext("2d");
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const stepButton = document.getElementById("step-button");
const resetButton = document.getElementById("reset-button");
const webcamButton = document.getElementById("webcam-button");
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
const chartEmpty = document.getElementById("chart-empty");

const experiment = {
  images: [],
  currentImage: null,
  ready: false,
  unsupported: false,
  running: false,
  training: false,
  stopRequested: false,
  pendingImageName: null,
  inputMode: "image",
  webcamStream: null,
  webcamVideo: null,
  webcamPreviewFrame: 0,
  epoch: 0,
  optimizerEpoch: 0,
  loss: null,
  psnr: null,
  lossHistory: [],
  lastUpdateSeconds: null,
  targetTensor: null,
  featureTensor: null,
  layers: [],
  variables: [],
  optimizer: null,
  currentLearningRate: MODEL.learningRate,
  tensorBaseline: 0,
  error: null,
};

function assetUrl(path) {
  return new URL(path, document.baseURI).toString();
}

function previewCount() {
  return Number.parseInt(slider.value, 10);
}

function updateSliderProgress() {
  const min = Number.parseFloat(slider.min || "0");
  const max = Number.parseFloat(slider.max || "100");
  const value = Number.parseFloat(slider.value || "0");
  const progress = ((value - min) / Math.max(1, max - min)) * 100;
  slider.style.setProperty("--slider-progress", `${progress}%`);
  sliderValue.textContent = String(previewCount());
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

function showError(message) {
  experiment.error = message;
  errorMessage.hidden = false;
  errorMessage.textContent = message;
}

function hideError() {
  experiment.error = null;
  errorMessage.hidden = true;
  errorMessage.textContent = "";
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return function random() {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

function makeNormalGenerator(seed) {
  const random = mulberry32(seed);
  let spare = null;
  return function normal() {
    if (spare !== null) {
      const value = spare;
      spare = null;
      return value;
    }
    const u = Math.max(random(), 1e-7);
    const v = random();
    const radius = Math.sqrt(-2 * Math.log(u));
    const theta = 2 * Math.PI * v;
    spare = radius * Math.sin(theta);
    return radius * Math.cos(theta);
  };
}

function createFloatTensor(shape, fill) {
  const size = shape.reduce((product, value) => product * value, 1);
  const data = new Float32Array(size);
  for (let index = 0; index < size; index += 1) {
    data[index] = fill(index);
  }
  return tf.tensor(data, shape);
}

function createVariable(shape, scale, seed, name) {
  const normal = makeNormalGenerator(seed);
  const tensor = createFloatTensor(shape, () => normal() * scale);
  const variable = tf.variable(tensor, true, name);
  tensor.dispose();
  return variable;
}

function disposeModel() {
  for (const variable of experiment.variables) {
    variable.dispose();
  }
  experiment.variables = [];
  experiment.layers = [];
  if (experiment.optimizer?.dispose) {
    experiment.optimizer.dispose();
  }
  experiment.optimizer = null;
}

function disposeTarget() {
  if (experiment.targetTensor) {
    experiment.targetTensor.dispose();
    experiment.targetTensor = null;
  }
}

function buildFeatureTensor() {
  if (experiment.featureTensor) {
    return;
  }

  experiment.featureTensor = tf.tidy(() => {
    const coords = createFloatTensor([PIXEL_COUNT, 2], (index) => {
      const pixelIndex = Math.floor(index / 2);
      const isX = index % 2 === 0;
      const x = pixelIndex % TARGET_SIZE;
      const y = Math.floor(pixelIndex / TARGET_SIZE);
      return (isX ? x : y) / (TARGET_SIZE - 1);
    });

    const normal = makeNormalGenerator(MODEL.seed);
    const freqs = createFloatTensor([2, MODEL.featureDim / 2], () => {
      return normal() * MODEL.freqScale;
    });

    const phase = coords.matMul(freqs).mul(2 * Math.PI);
    return tf.concat([phase.cos(), phase.sin()], 1);
  });
}

function buildModel() {
  disposeModel();
  experiment.layers = [];
  experiment.variables = [];
  experiment.currentLearningRate = MODEL.learningRate;
  experiment.optimizerEpoch = 0;

  let inputDim = MODEL.featureDim;
  for (let layerIndex = 0; layerIndex < MODEL.layers; layerIndex += 1) {
    const weight = createVariable(
      [inputDim, MODEL.hiddenDim],
      Math.sqrt(2 / inputDim),
      MODEL.seed + 101 + layerIndex,
      `hidden_${layerIndex}_weight`,
    );
    const bias = createVariable(
      [MODEL.hiddenDim],
      0.01,
      MODEL.seed + 201 + layerIndex,
      `hidden_${layerIndex}_bias`,
    );
    experiment.layers.push({ weight, bias, activation: "silu" });
    experiment.variables.push(weight, bias);
    inputDim = MODEL.hiddenDim;
  }

  const outputWeight = createVariable(
    [inputDim, 3],
    Math.sqrt(1 / inputDim),
    MODEL.seed + 301,
    "output_weight",
  );
  const outputBias = createVariable([3], 0.01, MODEL.seed + 302, "output_bias");
  experiment.layers.push({ weight: outputWeight, bias: outputBias, activation: "sigmoid" });
  experiment.variables.push(outputWeight, outputBias);
  experiment.optimizer = tf.train.adam(MODEL.learningRate);
}

function resetOptimizerState() {
  if (experiment.optimizer?.dispose) {
    experiment.optimizer.dispose();
  }
  experiment.currentLearningRate = MODEL.learningRate;
  experiment.optimizerEpoch = 0;
  experiment.optimizer = tf.train.adam(MODEL.learningRate);
}

function forward(features) {
  let value = features;
  for (const layer of experiment.layers) {
    value = value.matMul(layer.weight).add(layer.bias);
    if (layer.activation === "silu") {
      value = value.mul(value.sigmoid());
    } else if (layer.activation === "sigmoid") {
      value = value.sigmoid();
    }
  }
  return value;
}

function trainOneEpoch() {
  experiment.optimizer.minimize(() => {
    return tf.tidy(() => {
      const prediction = forward(experiment.featureTensor);
      return tf.mean(tf.squaredDifference(prediction, experiment.targetTensor));
    });
  }, false, experiment.variables);

  experiment.epoch += 1;
  experiment.optimizerEpoch += 1;
  experiment.currentLearningRate = Math.max(
    MODEL.minLearningRate,
    MODEL.learningRate * MODEL.lrDecay ** experiment.optimizerEpoch,
  );
  if (typeof experiment.optimizer.setLearningRate === "function") {
    experiment.optimizer.setLearningRate(experiment.currentLearningRate);
  }
}

async function publishPreview(startedAt, options = {}) {
  const { appendHistory = true } = options;
  const result = tf.tidy(() => {
    const prediction = forward(experiment.featureTensor);
    const loss = tf.mean(tf.squaredDifference(prediction, experiment.targetTensor));
    return { prediction, loss };
  });

  const [pixels, lossValues] = await Promise.all([
    result.prediction.data(),
    result.loss.data(),
  ]);
  result.prediction.dispose();
  result.loss.dispose();

  const loss = Number(lossValues[0]);
  const psnr = 10 * Math.log10(1 / Math.max(loss, 1e-12));
  drawOutput(pixels);

  experiment.loss = loss;
  experiment.psnr = psnr;
  if (appendHistory) {
    experiment.lossHistory.push({
      epoch: experiment.epoch,
      loss,
      psnr,
    });
  }
  experiment.lastUpdateSeconds = (performance.now() - startedAt) / 1000;

  console.debug("INR preview", {
    backend: tf.getBackend(),
    epoch: experiment.epoch,
    loss,
    tensors: tf.memory().numTensors,
    previewMs: Math.round(experiment.lastUpdateSeconds * 1000),
    lr: experiment.currentLearningRate,
  });
}

async function trainChunk(epochs) {
  if (!experiment.ready || experiment.training) {
    return;
  }

  const startedAt = performance.now();
  experiment.training = true;
  hideError();
  renderState();

  try {
    if (isWebcamActive()) {
      const sampled = await sampleWebcamTarget({
        resetOptimizer: true,
        appendHistory: false,
        publish: false,
      });
      if (!sampled) {
        throw new Error("Webcam frame is not ready yet.");
      }
    }
    for (let index = 0; index < epochs; index += 1) {
      trainOneEpoch();
    }
    await publishPreview(startedAt);
  } catch (error) {
    showError(String(error));
  } finally {
    experiment.training = false;
    renderState();
  }

  await applyPendingImageSwitch();
}

async function runContinuous() {
  if (!experiment.ready || experiment.running || experiment.training) {
    return;
  }

  experiment.running = true;
  experiment.stopRequested = false;
  hideError();
  renderState();

  try {
    while (!experiment.stopRequested) {
      await trainChunk(previewCount());
      await applyPendingImageSwitch();
      await tf.nextFrame();
    }
  } catch (error) {
    showError(String(error));
  } finally {
    experiment.running = false;
    experiment.training = false;
    experiment.stopRequested = false;
    renderState();
  }
}

function requestStop() {
  experiment.stopRequested = true;
  renderState();
}

function isWebcamActive() {
  return Boolean(experiment.webcamStream && experiment.webcamVideo);
}

async function applyPendingImageSwitch() {
  if (!experiment.pendingImageName || experiment.training) {
    return;
  }
  const imageName = experiment.pendingImageName;
  experiment.pendingImageName = null;
  await selectImage(imageName, { resetModel: false, force: true });
}

function drawWebcamFrameToReferenceCanvas() {
  const video = experiment.webcamVideo;
  if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return false;
  }

  const sourceWidth = video.videoWidth || TARGET_SIZE;
  const sourceHeight = video.videoHeight || TARGET_SIZE;
  const sourceSize = Math.min(sourceWidth, sourceHeight);
  const sourceX = Math.max(0, (sourceWidth - sourceSize) / 2);
  const sourceY = Math.max(0, (sourceHeight - sourceSize) / 2);

  referenceContext.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
  referenceContext.drawImage(
    video,
    sourceX,
    sourceY,
    sourceSize,
    sourceSize,
    0,
    0,
    TARGET_SIZE,
    TARGET_SIZE,
  );
  return true;
}

function updateTargetTensorFromReferenceCanvas() {
  const imageData = referenceContext.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
  const target = new Float32Array(PIXEL_COUNT * 3);
  for (let source = 0, targetIndex = 0; source < imageData.length; source += 4, targetIndex += 3) {
    target[targetIndex] = imageData[source] / 255;
    target[targetIndex + 1] = imageData[source + 1] / 255;
    target[targetIndex + 2] = imageData[source + 2] / 255;
  }
  const nextTargetTensor = tf.tensor2d(target, [PIXEL_COUNT, 3]);
  disposeTarget();
  experiment.targetTensor = nextTargetTensor;
}

async function sampleWebcamTarget(options = {}) {
  const {
    resetOptimizer = true,
    appendHistory = false,
    publish = false,
  } = options;

  if (!drawWebcamFrameToReferenceCanvas()) {
    return false;
  }

  if (experiment.ready) {
    updateTargetTensorFromReferenceCanvas();
    if (resetOptimizer && experiment.variables.length > 0) {
      resetOptimizerState();
    }
    if (publish && experiment.variables.length > 0) {
      await publishPreview(performance.now(), { appendHistory });
    }
  }
  return true;
}

function startWebcamPreviewLoop() {
  cancelAnimationFrame(experiment.webcamPreviewFrame);
  const draw = () => {
    if (!isWebcamActive()) {
      return;
    }
    drawWebcamFrameToReferenceCanvas();
    experiment.webcamPreviewFrame = requestAnimationFrame(draw);
  };
  experiment.webcamPreviewFrame = requestAnimationFrame(draw);
}

function stopWebcam() {
  cancelAnimationFrame(experiment.webcamPreviewFrame);
  experiment.webcamPreviewFrame = 0;
  if (experiment.webcamStream) {
    for (const track of experiment.webcamStream.getTracks()) {
      track.stop();
    }
  }
  if (experiment.webcamVideo) {
    experiment.webcamVideo.pause();
    experiment.webcamVideo.srcObject = null;
  }
  experiment.webcamStream = null;
  experiment.webcamVideo = null;
  experiment.inputMode = "image";
  renderState();
}

async function waitForVideoReady(video) {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.videoWidth > 0) {
    return;
  }
  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Timed out waiting for webcam video."));
    }, 10000);
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("error", onError);
    };
    const onReady = () => {
      if (video.videoWidth > 0) {
        cleanup();
        resolve();
      }
    };
    const onError = () => {
      cleanup();
      reject(new Error("Could not start webcam video."));
    };
    video.addEventListener("loadedmetadata", onReady);
    video.addEventListener("canplay", onReady);
    video.addEventListener("error", onError);
  });
}

async function startWebcam() {
  if (!experiment.ready || experiment.training || experiment.running) {
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    showError("Webcam input is not available in this browser.");
    return;
  }

  hideError();
  experiment.training = true;
  renderState();

  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 640 },
        height: { ideal: 640 },
        facingMode: "user",
      },
    });

    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    await waitForVideoReady(video);

    stopWebcam();
    experiment.inputMode = "webcam";
    experiment.webcamStream = stream;
    experiment.webcamVideo = video;
    experiment.currentImage = null;
    experiment.pendingImageName = null;

    startWebcamPreviewLoop();
    await sampleWebcamTarget({
      resetOptimizer: true,
      appendHistory: false,
      publish: true,
    });
  } catch (error) {
    if (stream && !isWebcamActive()) {
      for (const track of stream.getTracks()) {
        track.stop();
      }
    }
    showError(String(error));
  } finally {
    experiment.training = false;
    renderState();
  }

  if (isWebcamActive() && !experiment.running) {
    runContinuous();
  }
}

function toggleWebcam() {
  if (isWebcamActive()) {
    requestStop();
    stopWebcam();
  } else {
    startWebcam();
  }
}

function drawOutput(values) {
  const rgba = new Uint8ClampedArray(PIXEL_COUNT * 4);
  for (let source = 0, target = 0; source < values.length; source += 3, target += 4) {
    rgba[target] = Math.max(0, Math.min(255, Math.round(values[source] * 255)));
    rgba[target + 1] = Math.max(0, Math.min(255, Math.round(values[source + 1] * 255)));
    rgba[target + 2] = Math.max(0, Math.min(255, Math.round(values[source + 2] * 255)));
    rgba[target + 3] = 255;
  }
  latestCanvasContext.putImageData(new ImageData(rgba, TARGET_SIZE, TARGET_SIZE), 0, 0);
}

function clearOutput() {
  latestCanvasContext.fillStyle = "#777a73";
  latestCanvasContext.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load image: ${url}`));
    image.src = url;
  });
}

async function loadReferenceImage(imageInfo, createTensor) {
  const image = await loadImage(assetUrl(imageInfo.src));
  referenceContext.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
  referenceContext.drawImage(image, 0, 0, TARGET_SIZE, TARGET_SIZE);

  if (!createTensor) {
    disposeTarget();
    return;
  }

  updateTargetTensorFromReferenceCanvas();
}

async function selectImage(imageName, options = {}) {
  const { resetModel = false, force = false } = options;
  const imageInfo = experiment.images.find((image) => image.name === imageName) || experiment.images[0];
  if (!imageInfo) {
    return;
  }

  if (experiment.training && !force) {
    experiment.pendingImageName = imageInfo.name;
    renderState();
    return;
  }

  if (isWebcamActive()) {
    stopWebcam();
  }

  hideError();
  experiment.training = true;
  experiment.currentImage = imageInfo;
  experiment.inputMode = "image";
  const shouldResetModel = resetModel || experiment.variables.length === 0;
  if (shouldResetModel) {
    experiment.epoch = 0;
    experiment.loss = null;
    experiment.psnr = null;
    experiment.lossHistory = [];
    experiment.lastUpdateSeconds = null;
  }
  renderState();

  try {
    await loadReferenceImage(imageInfo, experiment.ready);
    if (experiment.ready) {
      if (shouldResetModel) {
        buildModel();
      } else {
        resetOptimizerState();
      }
      await publishPreview(performance.now());
    } else {
      clearOutput();
    }
  } catch (error) {
    showError(String(error));
  } finally {
    experiment.training = false;
    renderState();
  }
}

async function resetExperiment() {
  if (experiment.training || experiment.running) {
    return;
  }
  if (isWebcamActive()) {
    hideError();
    experiment.training = true;
    experiment.epoch = 0;
    experiment.optimizerEpoch = 0;
    experiment.loss = null;
    experiment.psnr = null;
    experiment.lossHistory = [];
    experiment.lastUpdateSeconds = null;
    renderState();
    try {
      buildModel();
      await sampleWebcamTarget({
        resetOptimizer: true,
        appendHistory: false,
        publish: true,
      });
    } catch (error) {
      showError(String(error));
    } finally {
      experiment.training = false;
      renderState();
    }
    return;
  }
  if (!experiment.currentImage && experiment.targetTensor) {
    hideError();
    experiment.training = true;
    experiment.epoch = 0;
    experiment.optimizerEpoch = 0;
    experiment.loss = null;
    experiment.psnr = null;
    experiment.lossHistory = [];
    experiment.lastUpdateSeconds = null;
    renderState();
    try {
      buildModel();
      await publishPreview(performance.now(), { appendHistory: false });
    } catch (error) {
      showError(String(error));
    } finally {
      experiment.training = false;
      renderState();
    }
    return;
  }
  if (!experiment.currentImage) {
    return;
  }
  await selectImage(experiment.currentImage.name, { resetModel: true });
}

async function loadManifest() {
  const response = await fetch(assetUrl("media/images.json"), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load media manifest (${response.status})`);
  }
  const manifest = await response.json();
  experiment.images = Array.isArray(manifest.images) ? manifest.images : [];
  if (experiment.images.length === 0) {
    throw new Error("media/images.json does not list any images.");
  }
}

async function initializeTensorFlow() {
  if (!window.tf) {
    throw new Error("TensorFlow.js did not load.");
  }
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser. Use current Chrome or Edge on desktop.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter was found. Check browser GPU settings and hardware acceleration.");
  }

  await tf.setBackend("webgpu");
  await tf.ready();
  buildFeatureTensor();
  experiment.ready = true;
  experiment.tensorBaseline = tf.memory().numTensors;
}

async function initializeApp() {
  updateSliderProgress();
  clearOutput();
  renderState();

  try {
    await loadManifest();
    renderImageOptions();
  } catch (error) {
    showError(String(error));
    renderState();
    return;
  }

  try {
    await initializeTensorFlow();
  } catch (error) {
    experiment.unsupported = true;
    showError(String(error));
  }

  await selectImage(experiment.images[0].name, { resetModel: true });
  renderState();
}

function renderState() {
  const webcamActive = isWebcamActive();
  document.body.dataset.state = experiment.running
    ? experiment.stopRequested
      ? "stopping"
      : "training"
    : experiment.training
      ? "working"
      : "idle";

  referenceMeta.textContent = webcamActive ? `webcam ${TARGET_SIZE}` : `${TARGET_SIZE} x ${TARGET_SIZE}`;
  epochValue.textContent = String(experiment.epoch);
  outputMeta.textContent = `epoch ${experiment.epoch}`;
  lossValue.textContent = formatLoss(experiment.loss);
  psnrValue.textContent = formatPsnr(experiment.psnr);
  deviceValue.textContent = experiment.ready ? tf.getBackend() : experiment.unsupported ? "No WebGPU" : "--";
  timingValue.textContent = experiment.lastUpdateSeconds
    ? `preview ${formatSeconds(experiment.lastUpdateSeconds)}`
    : "--";

  startButton.disabled = !experiment.ready || experiment.running || experiment.training;
  stopButton.disabled = !experiment.running || experiment.stopRequested;
  stepButton.disabled = !experiment.ready || experiment.running || experiment.training;
  resetButton.disabled = !experiment.ready || experiment.running || experiment.training || !experiment.targetTensor;
  webcamButton.textContent = webcamActive ? "Stop webcam" : "Use webcam";
  webcamButton.classList.toggle("is-active", webcamActive);
  webcamButton.disabled = !experiment.ready || (!webcamActive && (experiment.running || experiment.training));
  setImageButtonsDisabled(experiment.images.length === 0);
  renderImageOptions();
  drawLossChart(experiment.lossHistory);
}

function renderImageOptions() {
  const existing = new Set(Array.from(imagePicker.querySelectorAll(".image-option")).map((button) => button.dataset.image));
  const wanted = new Set(experiment.images.map((image) => image.name));
  const needsRebuild = existing.size !== wanted.size || [...wanted].some((name) => !existing.has(name));

  if (needsRebuild) {
    imagePicker.replaceChildren();
    for (const image of experiment.images) {
      const button = document.createElement("button");
      button.className = "image-option";
      button.type = "button";
      button.dataset.image = image.name;
      button.title = image.displayName || image.name;
      button.setAttribute("aria-label", `Use ${image.displayName || image.name}`);

      const thumb = document.createElement("img");
      thumb.src = assetUrl(image.thumb || image.src);
      thumb.alt = "";
      thumb.loading = "lazy";
      button.append(thumb);

      button.addEventListener("click", () => selectImage(image.name));
      imagePicker.append(button);
    }
  }

  for (const button of imagePicker.querySelectorAll(".image-option")) {
    const selected = button.dataset.image === experiment.currentImage?.name;
    const pending = button.dataset.image === experiment.pendingImageName;
    button.classList.toggle("is-selected", selected);
    button.classList.toggle("is-pending", pending);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function setImageButtonsDisabled(disabled) {
  for (const button of imagePicker.querySelectorAll(".image-option")) {
    button.disabled = disabled;
  }
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
  chartEmpty.hidden = points.length >= 2;
  if (points.length < 2) {
    return;
  }

  const epochs = points.map((item) => item.epoch);
  const losses = points.map((item) => item.loss);
  const minEpoch = Math.min(...epochs);
  const maxEpoch = Math.max(...epochs);
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const rawMinLog = Math.log10(minLoss);
  const rawMaxLog = Math.log10(maxLoss);
  const paddedLogSpan = Math.max(0.5, (rawMaxLog - rawMinLog) / 0.8);
  const minLog = rawMinLog - paddedLogSpan * 0.1;
  const maxLog = rawMaxLog + paddedLogSpan * 0.1;
  const logSpan = maxLog - minLog;
  const epochSpan = Math.max(1, maxEpoch - minEpoch);

  const xOf = (epoch) => left + ((epoch - minEpoch) / epochSpan) * plotWidth;
  const yOf = (loss) => top + ((maxLog - Math.log10(loss)) / logSpan) * plotHeight;

  chartContext.strokeStyle = "#e7e0ea";
  chartContext.lineWidth = 1;
  chartContext.fillStyle = "#756f79";
  chartContext.font = "12px Google Sans Code, Consolas, monospace";
  chartContext.textAlign = "right";
  chartContext.textBaseline = "middle";

  let tickCount = 0;
  for (let exponent = Math.ceil(minLog); exponent <= Math.floor(maxLog); exponent += 1) {
    const value = 10 ** exponent;
    const y = yOf(value);
    chartContext.beginPath();
    chartContext.moveTo(left, y);
    chartContext.lineTo(width - right, y);
    chartContext.stroke();
    chartContext.fillText(formatAxisLoss(value), left - 8, y);
    tickCount += 1;
  }

  if (tickCount === 0) {
    for (const value of [10 ** maxLog, 10 ** ((minLog + maxLog) / 2), 10 ** minLog]) {
      const y = yOf(value);
      chartContext.beginPath();
      chartContext.moveTo(left, y);
      chartContext.lineTo(width - right, y);
      chartContext.stroke();
      chartContext.fillText(formatAxisLoss(value), left - 8, y);
    }
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
  updateSliderProgress();
});

startButton.addEventListener("click", () => {
  runContinuous();
});

stopButton.addEventListener("click", () => {
  requestStop();
});

stepButton.addEventListener("click", () => {
  trainChunk(previewCount());
});

resetButton.addEventListener("click", () => {
  resetExperiment();
});

webcamButton.addEventListener("click", () => {
  toggleWebcam();
});

window.addEventListener("resize", () => drawLossChart(experiment.lossHistory));

initializeApp();
