const TARGET_SIZE = 256;
const PIXEL_COUNT = TARGET_SIZE * TARGET_SIZE;

const MODEL_PRESETS = {
  small: {
    label: "Small",
    featureDim: 128,
    hiddenDim: 64,
    layers: 2,
    featureType: "fourier",
    layerNorm: true,
    freqScale: 32,
    imageLearningRate: 0.012,
    imageLrDecay: 0.996,
    imageMinLearningRate: 0.0015,
    streamLearningRate: 0.006,
    seed: 7,
  },
  medium: {
    label: "Medium",
    featureDim: 256,
    hiddenDim: 128,
    layers: 2,
    featureType: "fourier",
    layerNorm: true,
    freqScale: 35,
    imageLearningRate: 0.01,
    imageLrDecay: 0.995,
    imageMinLearningRate: 0.001,
    streamLearningRate: 0.0045,
    seed: 7,
  },
  large: {
    label: "Large",
    featureDim: 384,
    hiddenDim: 256,
    layers: 3,
    featureType: "fourier",
    layerNorm: true,
    freqScale: 38,
    imageLearningRate: 0.0075,
    imageLrDecay: 0.996,
    imageMinLearningRate: 0.0008,
    streamLearningRate: 0.003,
    seed: 7,
  },
  xl: {
    label: "XL",
    featureDim: 512,
    hiddenDim: 384,
    layers: 4,
    featureType: "fourier",
    layerNorm: true,
    freqScale: 40,
    imageLearningRate: 0.005,
    imageLrDecay: 0.996,
    imageMinLearningRate: 0.0006,
    streamLearningRate: 0.0022,
    seed: 7,
  },
};

const DEFAULT_MODEL_KEY = "medium";
const STRUCTURAL_MODEL_KEYS = new Set(["featureDim", "hiddenDim", "layers", "featureType", "layerNorm"]);

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
const mediaPicker = document.getElementById("media-picker");
const modelToggle = document.getElementById("model-toggle");
const modelSummary = document.getElementById("model-summary");
const modelPicker = document.getElementById("model-picker");
const modelAdvanced = document.getElementById("model-advanced");
const modelField = modelToggle.closest(".model-field");
const hiddenDimSelect = document.getElementById("hidden-dim-select");
const hiddenLayersSelect = document.getElementById("hidden-layers-select");
const featureDimSelect = document.getElementById("feature-dim-select");
const featureTypeSelect = document.getElementById("feature-type-select");
const layerNormToggle = document.getElementById("layernorm-toggle");
const imageStartLrSlider = document.getElementById("image-start-lr-slider");
const imageEndLrSlider = document.getElementById("image-end-lr-slider");
const videoLrSlider = document.getElementById("video-lr-slider");
const imageStartLrValue = document.getElementById("image-start-lr-value");
const imageEndLrValue = document.getElementById("image-end-lr-value");
const videoLrValue = document.getElementById("video-lr-value");
const referenceMeta = document.getElementById("reference-meta");
const epochValue = document.getElementById("epoch-value");
const lossValue = document.getElementById("loss-value");
const psnrValue = document.getElementById("psnr-value");
const learningRateValue = document.getElementById("learning-rate-value");
const avgLossValue = document.getElementById("avg-loss-value");
const avgPsnrValue = document.getElementById("avg-psnr-value");
const deviceValue = document.getElementById("device-value");
const outputMeta = document.getElementById("output-meta");
const timingValue = document.getElementById("timing-value");
const errorMessage = document.getElementById("error-message");
const lossChart = document.getElementById("loss-chart");
const chartContext = lossChart.getContext("2d");
const chartEmpty = document.getElementById("chart-empty");

const experiment = {
  mediaItems: [],
  currentMedia: null,
  selectedModelKey: DEFAULT_MODEL_KEY,
  modelConfig: cloneModelConfig(MODEL_PRESETS[DEFAULT_MODEL_KEY]),
  modelExpanded: false,
  ready: false,
  unsupported: false,
  running: false,
  training: false,
  stopRequested: false,
  pendingMediaName: null,
  inputMode: "image",
  webcamStream: null,
  webcamVideo: null,
  webcamPreviewFrame: 0,
  webcamResumeOnFocus: false,
  videoElement: null,
  videoMediaName: null,
  videoStats: makeEmptyVideoStats(),
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
  currentLearningRate: MODEL_PRESETS[DEFAULT_MODEL_KEY].imageLearningRate,
  tensorBaseline: 0,
  error: null,
};

function makeEmptyVideoStats() {
  return {
    active: false,
    samples: 0,
    lossSum: 0,
    psnrSum: 0,
    currentTime: 0,
    duration: null,
  };
}

function cloneModelConfig(config) {
  return { ...config };
}

function assetUrl(path) {
  return new URL(path, document.baseURI).toString();
}

function currentModel() {
  return experiment.modelConfig || MODEL_PRESETS[DEFAULT_MODEL_KEY];
}

function isVideoSelected() {
  return experiment.currentMedia?.kind === "video";
}

function isVideoTarget() {
  return isVideoSelected() && Boolean(experiment.videoElement);
}

function isWebcamActive() {
  return Boolean(experiment.webcamStream && experiment.webcamVideo);
}

function isStreamTarget() {
  return isWebcamActive() || isVideoTarget();
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

function parameterCount(config) {
  let total = 0;
  let inputDim = config.featureDim;
  for (let layerIndex = 0; layerIndex < config.layers; layerIndex += 1) {
    total += inputDim * config.hiddenDim + config.hiddenDim;
    if (config.layerNorm) {
      total += config.hiddenDim * 2;
    }
    inputDim = config.hiddenDim;
  }
  total += inputDim * 3 + 3;
  return total;
}

function formatParameterCount(value) {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  return `${Math.round(value / 1000)}K`;
}

function formatLoss(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
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
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return `${value.toFixed(2)} dB`;
}

function formatLearningRate(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value < 0.001 ? value.toExponential(2) : value.toFixed(4);
}

function lrToSliderValue(value) {
  return Math.log10(Math.max(value, 1e-8));
}

function sliderValueToLr(value) {
  return 10 ** Number.parseFloat(value);
}

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  if (value < 1) {
    return `${Math.round(value * 1000)} ms`;
  }
  return `${value.toFixed(2)} s`;
}

function formatVideoTime(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }
  return `${value.toFixed(1)}s`;
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

function createConstantVariable(shape, value, name) {
  const tensor = tf.fill(shape, value);
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

function disposeFeatureTensor() {
  if (experiment.featureTensor) {
    experiment.featureTensor.dispose();
    experiment.featureTensor = null;
  }
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

  const config = currentModel();
  experiment.featureTensor = tf.tidy(() => {
    const coords = createFloatTensor([PIXEL_COUNT, 2], (index) => {
      const pixelIndex = Math.floor(index / 2);
      const isX = index % 2 === 0;
      const x = pixelIndex % TARGET_SIZE;
      const y = Math.floor(pixelIndex / TARGET_SIZE);
      return (isX ? x : y) / (TARGET_SIZE - 1);
    });

    const normal = makeNormalGenerator(config.seed);
    const atomCount = config.featureDim / 2;
    if (config.featureType === "gabor") {
      const centerRandom = mulberry32(config.seed + 17);
      const centers = createFloatTensor([atomCount, 2], () => centerRandom());
      const freqs = createFloatTensor([atomCount, 2], () => normal() * config.freqScale);
      const sigmas = tf.fill([atomCount, 2], 0.1 * Math.sqrt(256 / 2));
      const diff = coords.expandDims(1).sub(centers.expandDims(0));
      const envelope = diff.square().div(sigmas.expandDims(0).square()).sum(2).mul(-0.5).exp();
      const phase = diff.mul(freqs.expandDims(0)).sum(2).mul(2 * Math.PI);
      return tf.concat([envelope.mul(phase.cos()), envelope.mul(phase.sin())], 1);
    }

    const freqs = createFloatTensor([2, atomCount], () => normal() * config.freqScale);
    const phase = coords.matMul(freqs).mul(2 * Math.PI);
    return tf.concat([phase.cos(), phase.sin()], 1);
  });
}

function buildModel() {
  disposeModel();
  buildFeatureTensor();
  experiment.layers = [];
  experiment.variables = [];
  experiment.optimizerEpoch = 0;

  const config = currentModel();
  let inputDim = config.featureDim;
  for (let layerIndex = 0; layerIndex < config.layers; layerIndex += 1) {
    const weight = createVariable(
      [inputDim, config.hiddenDim],
      Math.sqrt(2 / inputDim),
      config.seed + 101 + layerIndex,
      `hidden_${layerIndex}_weight`,
    );
    const bias = createVariable(
      [config.hiddenDim],
      0.01,
      config.seed + 201 + layerIndex,
      `hidden_${layerIndex}_bias`,
    );
    let normGamma = null;
    let normBeta = null;
    if (config.layerNorm) {
      normGamma = createConstantVariable([config.hiddenDim], 1, `hidden_${layerIndex}_norm_gamma`);
      normBeta = createConstantVariable([config.hiddenDim], 0, `hidden_${layerIndex}_norm_beta`);
    }
    experiment.layers.push({ weight, bias, activation: "silu", normGamma, normBeta });
    experiment.variables.push(weight, bias);
    if (normGamma && normBeta) {
      experiment.variables.push(normGamma, normBeta);
    }
    inputDim = config.hiddenDim;
  }

  const outputWeight = createVariable(
    [inputDim, 3],
    Math.sqrt(1 / inputDim),
    config.seed + 301,
    "output_weight",
  );
  const outputBias = createVariable([3], 0.01, config.seed + 302, "output_bias");
  experiment.layers.push({ weight: outputWeight, bias: outputBias, activation: "sigmoid" });
  experiment.variables.push(outputWeight, outputBias);
  resetOptimizerState({ stream: isStreamTarget() });
}

function applyLayerNorm(value, gamma, beta) {
  const mean = value.mean(1, true);
  const centered = value.sub(mean);
  const variance = centered.square().mean(1, true);
  return centered.div(variance.add(1e-5).sqrt()).mul(gamma).add(beta);
}

function resetOptimizerState(options = {}) {
  const { stream = isStreamTarget() } = options;
  const config = currentModel();
  if (experiment.optimizer?.dispose) {
    experiment.optimizer.dispose();
  }
  experiment.optimizerEpoch = 0;
  experiment.currentLearningRate = stream ? config.streamLearningRate : config.imageLearningRate;
  experiment.optimizer = tf.train.adam(experiment.currentLearningRate);
}

function learningRateForEpoch() {
  const config = currentModel();
  if (isStreamTarget()) {
    return config.streamLearningRate;
  }
  return Math.max(
    config.imageMinLearningRate,
    config.imageLearningRate * config.imageLrDecay ** experiment.optimizerEpoch,
  );
}

function forward(features) {
  let value = features;
  for (const layer of experiment.layers) {
    value = value.matMul(layer.weight).add(layer.bias);
    if (layer.activation === "silu") {
      value = value.mul(value.sigmoid());
      if (layer.normGamma && layer.normBeta) {
        value = applyLayerNorm(value, layer.normGamma, layer.normBeta);
      }
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
  experiment.currentLearningRate = learningRateForEpoch();
  if (typeof experiment.optimizer.setLearningRate === "function") {
    experiment.optimizer.setLearningRate(experiment.currentLearningRate);
  }
}

function updateVideoAverage(loss, psnr) {
  if (!experiment.videoStats.active) {
    return;
  }
  experiment.videoStats.samples += 1;
  experiment.videoStats.lossSum += loss;
  experiment.videoStats.psnrSum += psnr;
  if (experiment.videoElement) {
    experiment.videoStats.currentTime = experiment.videoElement.currentTime || 0;
    experiment.videoStats.duration = Number.isFinite(experiment.videoElement.duration)
      ? experiment.videoElement.duration
      : null;
  }
}

async function publishPreview(startedAt, options = {}) {
  const { appendHistory = true, includeInVideoAverage = false } = options;
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
  if (includeInVideoAverage) {
    updateVideoAverage(loss, psnr);
  }
  experiment.lastUpdateSeconds = (performance.now() - startedAt) / 1000;

  console.debug("INR preview", {
    backend: tf.getBackend(),
    mode: experiment.inputMode,
    model: experiment.selectedModelKey,
    epoch: experiment.epoch,
    loss,
    tensors: tf.memory().numTensors,
    previewMs: Math.round(experiment.lastUpdateSeconds * 1000),
    lr: experiment.currentLearningRate,
  });
}

async function refreshDynamicTarget() {
  if (isWebcamActive()) {
    const sampled = await sampleWebcamTarget({ publish: false });
    if (!sampled) {
      throw new Error("Webcam frame is not ready yet.");
    }
  } else if (isVideoTarget()) {
    const sampled = await sampleVideoTarget({ publish: false });
    if (!sampled) {
      throw new Error("Video frame is not ready yet.");
    }
  }
}

async function trainChunk(epochs) {
  if (!experiment.ready || experiment.training || !experiment.targetTensor) {
    return;
  }
  if (isVideoSelected() && !experiment.running) {
    return;
  }

  const startedAt = performance.now();
  experiment.training = true;
  hideError();
  renderState();

  try {
    await refreshDynamicTarget();
    for (let index = 0; index < epochs; index += 1) {
      trainOneEpoch();
    }
    await publishPreview(startedAt, {
      appendHistory: true,
      includeInVideoAverage: isVideoTarget() && experiment.videoStats.active,
    });
  } catch (error) {
    showError(String(error));
  } finally {
    experiment.training = false;
    renderState();
  }

  await applyPendingMediaSwitch();
}

async function runContinuous() {
  if (isVideoSelected()) {
    await runVideoContinuous();
    return;
  }
  if (!experiment.ready || experiment.running || experiment.training || !experiment.targetTensor) {
    return;
  }

  experiment.running = true;
  experiment.stopRequested = false;
  hideError();
  renderState();

  try {
    while (!experiment.stopRequested) {
      await trainChunk(previewCount());
      await applyPendingMediaSwitch();
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

async function runVideoContinuous() {
  if (!experiment.ready || experiment.running || experiment.training || !isVideoTarget()) {
    return;
  }

  experiment.running = true;
  experiment.stopRequested = false;
  experiment.videoStats = makeEmptyVideoStats();
  experiment.videoStats.active = true;
  hideError();
  resetOptimizerState({ stream: true });
  renderState();

  try {
    await restartVideoPlayback();
    await sampleVideoTarget({ publish: true, appendHistory: false });
    while (!experiment.stopRequested && isVideoTarget() && !experiment.videoElement.ended) {
      await trainChunk(previewCount());
      await applyPendingMediaSwitch();
      if (!isVideoTarget()) {
        break;
      }
      await tf.nextFrame();
    }
  } catch (error) {
    showError(String(error));
  } finally {
    if (experiment.videoElement) {
      experiment.videoElement.pause();
    }
    experiment.videoStats.active = false;
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

async function applyPendingMediaSwitch() {
  if (!experiment.pendingMediaName || experiment.training) {
    return;
  }
  const mediaName = experiment.pendingMediaName;
  experiment.pendingMediaName = null;
  await selectMedia(mediaName, { resetModel: false, force: true });
}

function drawCanvasSourceToReference(source, sourceWidth, sourceHeight) {
  const sourceSize = Math.min(sourceWidth || TARGET_SIZE, sourceHeight || TARGET_SIZE);
  const sourceX = Math.max(0, ((sourceWidth || TARGET_SIZE) - sourceSize) / 2);
  const sourceY = Math.max(0, ((sourceHeight || TARGET_SIZE) - sourceSize) / 2);

  referenceContext.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
  referenceContext.drawImage(
    source,
    sourceX,
    sourceY,
    sourceSize,
    sourceSize,
    0,
    0,
    TARGET_SIZE,
    TARGET_SIZE,
  );
}

function drawWebcamFrameToReferenceCanvas() {
  const video = experiment.webcamVideo;
  if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return false;
  }
  drawCanvasSourceToReference(video, video.videoWidth, video.videoHeight);
  return true;
}

function drawVideoFrameToReferenceCanvas() {
  const video = experiment.videoElement;
  if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    return false;
  }
  drawCanvasSourceToReference(video, video.videoWidth, video.videoHeight);
  experiment.videoStats.currentTime = video.currentTime || 0;
  experiment.videoStats.duration = Number.isFinite(video.duration) ? video.duration : null;
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
  const { appendHistory = false, publish = false } = options;
  if (!drawWebcamFrameToReferenceCanvas()) {
    return false;
  }

  if (experiment.ready) {
    updateTargetTensorFromReferenceCanvas();
    if (publish && experiment.variables.length > 0) {
      await publishPreview(performance.now(), { appendHistory });
    }
  }
  return true;
}

async function sampleVideoTarget(options = {}) {
  const { appendHistory = false, publish = false } = options;
  if (!drawVideoFrameToReferenceCanvas()) {
    return false;
  }

  if (experiment.ready) {
    updateTargetTensorFromReferenceCanvas();
    if (publish && experiment.variables.length > 0) {
      await publishPreview(performance.now(), { appendHistory });
    }
  }
  return true;
}

function startWebcamPreviewLoop() {
  cancelAnimationFrame(experiment.webcamPreviewFrame);
  const draw = () => {
    if (!isWebcamActive() || document.hidden || !document.hasFocus()) {
      experiment.webcamPreviewFrame = 0;
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
  experiment.webcamResumeOnFocus = false;
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
  experiment.inputMode = experiment.currentMedia?.kind || "image";
  renderState();
}

function pauseWebcamForInactivePage() {
  if (!isWebcamActive()) {
    return;
  }
  experiment.webcamResumeOnFocus = experiment.running && !experiment.stopRequested;
  if (experiment.running) {
    requestStop();
  }
  cancelAnimationFrame(experiment.webcamPreviewFrame);
  experiment.webcamPreviewFrame = 0;
  if (experiment.webcamVideo) {
    experiment.webcamVideo.pause();
  }
  renderState();
}

function resumeWebcamForActivePage() {
  if (!isWebcamActive() || document.hidden || !document.hasFocus()) {
    return;
  }

  const restart = async () => {
    try {
      await experiment.webcamVideo.play();
    } catch (error) {
      showError(String(error));
      return;
    }
    startWebcamPreviewLoop();
    if (!experiment.webcamResumeOnFocus) {
      renderState();
      return;
    }
    if (experiment.running || experiment.training) {
      window.setTimeout(restart, 80);
      return;
    }
    experiment.webcamResumeOnFocus = false;
    runContinuous();
  };

  restart();
}

function disposeVideoElement() {
  if (experiment.videoElement) {
    experiment.videoElement.pause();
    experiment.videoElement.removeAttribute("src");
    experiment.videoElement.load();
  }
  experiment.videoElement = null;
  experiment.videoMediaName = null;
  experiment.videoStats = makeEmptyVideoStats();
}

async function waitForMediaReady(media, label) {
  if (media.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && media.videoWidth > 0) {
    return;
  }
  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error(`Timed out waiting for ${label}.`));
    }, 10000);
    const cleanup = () => {
      window.clearTimeout(timeout);
      media.removeEventListener("loadeddata", onReady);
      media.removeEventListener("loadedmetadata", onReady);
      media.removeEventListener("canplay", onReady);
      media.removeEventListener("error", onError);
    };
    const onReady = () => {
      if (media.videoWidth > 0) {
        cleanup();
        resolve();
      }
    };
    const onError = () => {
      cleanup();
      reject(new Error(`Could not start ${label}.`));
    };
    media.addEventListener("loadeddata", onReady);
    media.addEventListener("loadedmetadata", onReady);
    media.addEventListener("canplay", onReady);
    media.addEventListener("error", onError);
  });
}

async function seekVideo(video, time) {
  const duration = Number.isFinite(video.duration) ? video.duration : time;
  const targetTime = Math.max(0, Math.min(time, Math.max(0, duration - 0.02)));
  if (Math.abs((video.currentTime || 0) - targetTime) < 0.015 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    return;
  }
  await new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Timed out seeking video."));
    }, 6000);
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };
    const onSeeked = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Could not seek video."));
    };
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.currentTime = targetTime;
  });
}

async function loadVideoElement(mediaInfo) {
  disposeVideoElement();
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = assetUrl(mediaInfo.src);
  experiment.videoElement = video;
  experiment.videoMediaName = mediaInfo.name;
  video.load();
  await waitForMediaReady(video, "video");
  await seekVideo(video, 0);
  drawVideoFrameToReferenceCanvas();
  return video;
}

async function restartVideoPlayback() {
  if (!isVideoTarget()) {
    return;
  }
  const video = experiment.videoElement;
  video.pause();
  await seekVideo(video, 0);
  drawVideoFrameToReferenceCanvas();
  await video.play();
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
    await waitForMediaReady(video, "webcam video");

    stopWebcam();
    disposeVideoElement();
    experiment.inputMode = "webcam";
    experiment.webcamStream = stream;
    experiment.webcamVideo = video;
    experiment.currentMedia = null;
    experiment.pendingMediaName = null;
    experiment.videoStats = makeEmptyVideoStats();
    resetOptimizerState({ stream: true });

    startWebcamPreviewLoop();
    await sampleWebcamTarget({
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

async function loadReferenceImage(mediaInfo, createTensor) {
  const image = await loadImage(assetUrl(mediaInfo.src));
  referenceContext.clearRect(0, 0, TARGET_SIZE, TARGET_SIZE);
  referenceContext.drawImage(image, 0, 0, TARGET_SIZE, TARGET_SIZE);

  if (!createTensor) {
    disposeTarget();
    return;
  }

  updateTargetTensorFromReferenceCanvas();
}

function resetRunMetrics() {
  experiment.epoch = 0;
  experiment.optimizerEpoch = 0;
  experiment.loss = null;
  experiment.psnr = null;
  experiment.lossHistory = [];
  experiment.lastUpdateSeconds = null;
  experiment.videoStats = makeEmptyVideoStats();
}

function currentModelName() {
  if (MODEL_PRESETS[experiment.selectedModelKey]) {
    return MODEL_PRESETS[experiment.selectedModelKey].label;
  }
  return "Custom";
}

function syncCurrentLearningRate() {
  if (!experiment.optimizer) {
    return;
  }
  experiment.currentLearningRate = learningRateForEpoch();
  if (typeof experiment.optimizer.setLearningRate === "function") {
    experiment.optimizer.setLearningRate(experiment.currentLearningRate);
  }
}

async function rebuildForModelConfig(options = {}) {
  const { rebuildFeature = true, resetMetrics = true } = options;
  if (experiment.running || experiment.training) {
    return;
  }

  hideError();
  experiment.training = true;
  if (resetMetrics) {
    resetRunMetrics();
  }
  renderState();

  try {
    if (rebuildFeature) {
      disposeFeatureTensor();
    }
    if (experiment.ready) {
      buildFeatureTensor();
      buildModel();
      if (experiment.targetTensor) {
        await publishPreview(performance.now(), { appendHistory: false });
      }
    }
  } catch (error) {
    showError(String(error));
  } finally {
    experiment.training = false;
    renderState();
  }
}

function clampLearningRates() {
  const config = currentModel();
  if (config.imageMinLearningRate > config.imageLearningRate) {
    config.imageMinLearningRate = config.imageLearningRate;
  }
}

async function updateModelSetting(key, value) {
  if (experiment.running || experiment.training) {
    renderState();
    return;
  }
  const config = currentModel();
  if (config[key] === value) {
    return;
  }

  experiment.selectedModelKey = "custom";
  experiment.modelConfig = {
    ...config,
    [key]: value,
  };
  clampLearningRates();

  if (STRUCTURAL_MODEL_KEYS.has(key)) {
    await rebuildForModelConfig({
      rebuildFeature: key === "featureDim" || key === "featureType",
      resetMetrics: true,
    });
    return;
  }

  syncCurrentLearningRate();
  renderState();
}

async function selectMedia(mediaName, options = {}) {
  const { resetModel = false, force = false } = options;
  const mediaInfo = experiment.mediaItems.find((item) => item.name === mediaName) || experiment.mediaItems[0];
  if (!mediaInfo) {
    return;
  }

  if ((experiment.training || (experiment.running && isVideoTarget())) && !force) {
    experiment.pendingMediaName = mediaInfo.name;
    if (experiment.running) {
      requestStop();
    }
    renderState();
    return;
  }

  if (isWebcamActive()) {
    stopWebcam();
  }
  if (mediaInfo.kind !== "video" || experiment.videoMediaName !== mediaInfo.name) {
    disposeVideoElement();
  }

  hideError();
  experiment.training = true;
  experiment.currentMedia = mediaInfo;
  experiment.inputMode = mediaInfo.kind;
  const shouldResetModel = resetModel || experiment.variables.length === 0;
  if (shouldResetModel) {
    resetRunMetrics();
  } else if (mediaInfo.kind === "video") {
    experiment.videoStats = makeEmptyVideoStats();
  }
  renderState();

  try {
    if (mediaInfo.kind === "video") {
      await loadVideoElement(mediaInfo);
      if (experiment.ready) {
        updateTargetTensorFromReferenceCanvas();
      }
    } else {
      await loadReferenceImage(mediaInfo, experiment.ready);
    }

    if (experiment.ready) {
      if (shouldResetModel) {
        buildModel();
      } else {
        resetOptimizerState({ stream: mediaInfo.kind === "video" });
      }
      await publishPreview(performance.now(), {
        appendHistory: mediaInfo.kind !== "video",
      });
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

  hideError();
  experiment.training = true;
  resetRunMetrics();
  renderState();

  try {
    buildModel();
    if (isWebcamActive()) {
      resetOptimizerState({ stream: true });
      await sampleWebcamTarget({
        appendHistory: false,
        publish: true,
      });
    } else if (isVideoTarget()) {
      resetOptimizerState({ stream: true });
      await seekVideo(experiment.videoElement, 0);
      await sampleVideoTarget({
        appendHistory: false,
        publish: true,
      });
    } else if (experiment.currentMedia) {
      await loadReferenceImage(experiment.currentMedia, experiment.ready);
      resetOptimizerState({ stream: false });
      await publishPreview(performance.now(), { appendHistory: false });
    } else if (experiment.targetTensor) {
      await publishPreview(performance.now(), { appendHistory: false });
    }
  } catch (error) {
    showError(String(error));
  } finally {
    experiment.training = false;
    renderState();
  }
}

async function changeModelPreset(modelKey) {
  if (!MODEL_PRESETS[modelKey] || modelKey === experiment.selectedModelKey || experiment.running || experiment.training) {
    return;
  }

  experiment.selectedModelKey = modelKey;
  experiment.modelConfig = cloneModelConfig(MODEL_PRESETS[modelKey]);
  await rebuildForModelConfig({ rebuildFeature: true, resetMetrics: true });
}

async function loadManifest() {
  const response = await fetch(assetUrl("media/images.json"), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Could not load media manifest (${response.status})`);
  }
  const manifest = await response.json();
  const mediaItems = Array.isArray(manifest.media)
    ? manifest.media
    : (Array.isArray(manifest.images) ? manifest.images.map((item) => ({ ...item, kind: "image" })) : []);
  experiment.mediaItems = mediaItems.map((item) => ({
    ...item,
    kind: item.kind || item.type || "image",
  }));
  if (experiment.mediaItems.length === 0) {
    throw new Error("media/images.json does not list any media.");
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
  renderModelOptions();
  renderState();

  try {
    await loadManifest();
    renderMediaOptions();
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

  await selectMedia(experiment.mediaItems[0].name, { resetModel: true });
  renderState();
}

function referenceMetaText() {
  if (isWebcamActive()) {
    return `webcam ${TARGET_SIZE}`;
  }
  if (isVideoSelected()) {
    const current = experiment.videoStats.currentTime;
    const duration = experiment.videoStats.duration;
    return `${formatVideoTime(current)} / ${formatVideoTime(duration)}`;
  }
  if (!experiment.currentMedia && experiment.targetTensor) {
    return `still ${TARGET_SIZE}`;
  }
  return `${TARGET_SIZE} x ${TARGET_SIZE}`;
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

  referenceMeta.textContent = referenceMetaText();
  epochValue.textContent = String(experiment.epoch);
  outputMeta.textContent = `epoch ${experiment.epoch}`;
  lossValue.textContent = formatLoss(experiment.loss);
  psnrValue.textContent = formatPsnr(experiment.psnr);
  learningRateValue.textContent = formatLearningRate(experiment.currentLearningRate);
  if (experiment.videoStats.samples > 0) {
    avgLossValue.textContent = formatLoss(experiment.videoStats.lossSum / experiment.videoStats.samples);
    avgPsnrValue.textContent = formatPsnr(experiment.videoStats.psnrSum / experiment.videoStats.samples);
  } else {
    avgLossValue.textContent = "--";
    avgPsnrValue.textContent = "--";
  }
  deviceValue.textContent = experiment.ready ? tf.getBackend() : experiment.unsupported ? "No WebGPU" : "--";
  timingValue.textContent = experiment.lastUpdateSeconds
    ? `preview ${formatSeconds(experiment.lastUpdateSeconds)}`
    : "--";

  startButton.disabled = !experiment.ready || experiment.running || experiment.training || !experiment.targetTensor;
  stopButton.disabled = !experiment.running || experiment.stopRequested;
  stepButton.disabled = !experiment.ready || experiment.running || experiment.training || isVideoSelected() || !experiment.targetTensor;
  resetButton.disabled = !experiment.ready || experiment.running || experiment.training || !experiment.targetTensor;
  webcamButton.textContent = webcamActive ? "Stop webcam" : "Use webcam";
  webcamButton.classList.toggle("is-active", webcamActive);
  webcamButton.disabled = !experiment.ready || (!webcamActive && (experiment.running || experiment.training));
  setMediaButtonsDisabled(experiment.mediaItems.length === 0);
  renderMediaOptions();
  renderModelOptions();
  renderModelSettings();
  drawLossChart(experiment.lossHistory);
}

function renderMediaOptions() {
  const existing = new Set(Array.from(mediaPicker.querySelectorAll(".media-option")).map((button) => button.dataset.media));
  const wanted = new Set(experiment.mediaItems.map((item) => item.name));
  const needsRebuild = existing.size !== wanted.size || [...wanted].some((name) => !existing.has(name));

  if (needsRebuild) {
    mediaPicker.replaceChildren();
    for (const item of experiment.mediaItems) {
      const button = document.createElement("button");
      button.className = "media-option";
      button.type = "button";
      button.dataset.media = item.name;
      button.dataset.kind = item.kind || "image";
      button.title = item.displayName || item.name;
      button.setAttribute("aria-label", `Use ${item.displayName || item.name}`);

      const thumb = document.createElement("img");
      thumb.src = assetUrl(item.thumb || item.src);
      thumb.alt = "";
      thumb.loading = "lazy";
      button.append(thumb);

      button.addEventListener("click", () => selectMedia(item.name));
      mediaPicker.append(button);
    }
  }

  for (const button of mediaPicker.querySelectorAll(".media-option")) {
    const selected = button.dataset.media === experiment.currentMedia?.name;
    const pending = button.dataset.media === experiment.pendingMediaName;
    button.classList.toggle("is-selected", selected);
    button.classList.toggle("is-pending", pending);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function setMediaButtonsDisabled(disabled) {
  for (const button of mediaPicker.querySelectorAll(".media-option")) {
    button.disabled = disabled;
  }
}

function renderModelOptions() {
  if (modelPicker.children.length === 0) {
    for (const [key, config] of Object.entries(MODEL_PRESETS)) {
      const button = document.createElement("button");
      button.className = "model-option";
      button.type = "button";
      button.dataset.model = key;
      button.setAttribute("role", "radio");
      button.innerHTML = `<span>${config.label}</span><code>${formatParameterCount(parameterCount(config))}</code>`;
      button.addEventListener("click", () => changeModelPreset(key));
      modelPicker.append(button);
    }
  }

  for (const button of modelPicker.querySelectorAll(".model-option")) {
    const selected = button.dataset.model === experiment.selectedModelKey;
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-checked", String(selected));
    button.disabled = experiment.running || experiment.training;
  }
}

function renderModelSettings() {
  const config = currentModel();
  const params = formatParameterCount(parameterCount(config));
  modelSummary.textContent = `${currentModelName()} - ${params}`;
  modelToggle.setAttribute("aria-expanded", String(experiment.modelExpanded));
  modelAdvanced.setAttribute("aria-hidden", String(!experiment.modelExpanded));
  modelAdvanced.inert = !experiment.modelExpanded;
  modelField.classList.toggle("is-expanded", experiment.modelExpanded);

  hiddenDimSelect.value = String(config.hiddenDim);
  hiddenLayersSelect.value = String(config.layers);
  featureDimSelect.value = String(config.featureDim);
  featureTypeSelect.value = config.featureType || "fourier";
  layerNormToggle.checked = Boolean(config.layerNorm);

  imageStartLrSlider.value = String(lrToSliderValue(config.imageLearningRate));
  imageEndLrSlider.value = String(lrToSliderValue(config.imageMinLearningRate));
  videoLrSlider.value = String(lrToSliderValue(config.streamLearningRate));
  imageStartLrValue.textContent = formatLearningRate(config.imageLearningRate);
  imageEndLrValue.textContent = formatLearningRate(config.imageMinLearningRate);
  videoLrValue.textContent = formatLearningRate(config.streamLearningRate);

  const disabled = experiment.running || experiment.training;
  modelToggle.disabled = false;
  for (const control of [
    hiddenDimSelect,
    hiddenLayersSelect,
    featureDimSelect,
    featureTypeSelect,
    layerNormToggle,
    imageStartLrSlider,
    imageEndLrSlider,
    videoLrSlider,
  ]) {
    control.disabled = disabled;
  }
}

function prepareCanvas() {
  const rect = lossChart.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.round(rect.width * dpr));
  const height = Math.max(150, Math.round(rect.height * dpr));
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

  chartContext.strokeStyle = "#8f4d6b";
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
  chartContext.fillStyle = "#8f4d6b";
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

modelToggle.addEventListener("click", () => {
  experiment.modelExpanded = !experiment.modelExpanded;
  renderState();
});

hiddenDimSelect.addEventListener("change", () => {
  updateModelSetting("hiddenDim", Number.parseInt(hiddenDimSelect.value, 10));
});

hiddenLayersSelect.addEventListener("change", () => {
  updateModelSetting("layers", Number.parseInt(hiddenLayersSelect.value, 10));
});

featureDimSelect.addEventListener("change", () => {
  updateModelSetting("featureDim", Number.parseInt(featureDimSelect.value, 10));
});

featureTypeSelect.addEventListener("change", () => {
  updateModelSetting("featureType", featureTypeSelect.value);
});

layerNormToggle.addEventListener("change", () => {
  updateModelSetting("layerNorm", layerNormToggle.checked);
});

imageStartLrSlider.addEventListener("input", () => {
  updateModelSetting("imageLearningRate", sliderValueToLr(imageStartLrSlider.value));
});

imageEndLrSlider.addEventListener("input", () => {
  updateModelSetting("imageMinLearningRate", sliderValueToLr(imageEndLrSlider.value));
});

videoLrSlider.addEventListener("input", () => {
  updateModelSetting("streamLearningRate", sliderValueToLr(videoLrSlider.value));
});

window.addEventListener("resize", () => drawLossChart(experiment.lossHistory));
window.addEventListener("blur", pauseWebcamForInactivePage);
window.addEventListener("focus", resumeWebcamForActivePage);
window.addEventListener("pagehide", pauseWebcamForInactivePage);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    pauseWebcamForInactivePage();
  } else {
    resumeWebcamForActivePage();
  }
});

initializeApp();
