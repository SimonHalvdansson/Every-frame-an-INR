const TARGET_SIZES = [64, 128, 256];
const DEFAULT_TARGET_SIZE = 128;

const MODEL_PRESETS = {
  small: {
    label: "Small",
    featureDim: 128,
    hiddenDim: 64,
    layers: 2,
    featureType: "fourier",
    activation: "silu",
    rmsNorm: false,
    freqScale: 32,
    gaborAtomWidth: 0.1,
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
    activation: "silu",
    rmsNorm: false,
    freqScale: 35,
    gaborAtomWidth: 0.1,
    imageLearningRate: 0.01,
    imageLrDecay: 0.995,
    imageMinLearningRate: 0.001,
    streamLearningRate: 0.0045,
    seed: 7,
  },
  large: {
    label: "Large",
    featureDim: 256,
    hiddenDim: 256,
    layers: 3,
    featureType: "fourier",
    activation: "silu",
    rmsNorm: false,
    freqScale: 38,
    gaborAtomWidth: 0.1,
    imageLearningRate: 0.0075,
    imageLrDecay: 0.996,
    imageMinLearningRate: 0.0008,
    streamLearningRate: 0.003,
    seed: 7,
  },
  xl: {
    label: "XL",
    featureDim: 256,
    hiddenDim: 384,
    layers: 4,
    featureType: "fourier",
    activation: "silu",
    rmsNorm: false,
    freqScale: 40,
    gaborAtomWidth: 0.1,
    imageLearningRate: 0.005,
    imageLrDecay: 0.996,
    imageMinLearningRate: 0.0006,
    streamLearningRate: 0.0022,
    seed: 7,
  },
};

const DEFAULT_MODEL_KEY = "medium";
const STRUCTURAL_MODEL_KEYS = new Set(["featureDim", "hiddenDim", "layers", "featureType", "activation", "rmsNorm"]);

const referenceCanvas = document.getElementById("reference-canvas");
const referenceContext = referenceCanvas.getContext("2d", { willReadFrequently: true });
const latestCanvas = document.getElementById("latest-canvas");
const latestCanvasContext = latestCanvas.getContext("2d");
const startButton = document.getElementById("run-button");
const startButtonLabel = startButton.querySelector("span");
const runButtonIcon = document.getElementById("run-button-icon");
const stepButton = document.getElementById("step-button");
const resetButton = document.getElementById("reset-button");
const webcamButton = document.getElementById("webcam-button");
const slider = document.getElementById("epoch-slider");
const sliderValue = document.getElementById("slider-value");
const resolutionPicker = document.getElementById("resolution-picker");
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
const gaborAtomWidthSlider = document.getElementById("gabor-atom-width-slider");
const gaborAtomWidthValue = document.getElementById("gabor-atom-width-value");
const activationSelect = document.getElementById("activation-select");
const rmsNormToggle = document.getElementById("rmsnorm-toggle");
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
const weightsToggle = document.getElementById("weights-toggle");
const weightsSummary = document.getElementById("weights-summary");
const weightsContents = document.getElementById("weights-contents");
const weightsPanel = weightsToggle.closest(".weights-panel");
const weightsStrip = document.getElementById("weights-strip");

const experiment = {
  mediaItems: [],
  currentMedia: null,
  targetSize: DEFAULT_TARGET_SIZE,
  selectedModelKey: DEFAULT_MODEL_KEY,
  modelConfig: cloneModelConfig(MODEL_PRESETS[DEFAULT_MODEL_KEY]),
  modelExpanded: false,
  ready: false,
  unsupported: false,
  running: false,
  training: false,
  stopRequested: false,
  pendingMediaName: null,
  pendingOperation: null,
  inputMode: "image",
  webcamStream: null,
  webcamVideo: null,
  webcamPreviewFrame: 0,
  webcamResumeOnFocus: false,
  webcamResumeAvailable: false,
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
  weightsExpanded: false,
  weightRenderId: 0,
  weightRenderInFlight: false,
  weightRenderQueued: false,
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

function currentTargetSize() {
  return experiment.targetSize || DEFAULT_TARGET_SIZE;
}

function currentPixelCount() {
  const size = currentTargetSize();
  return size * size;
}

function configureCanvases() {
  const size = currentTargetSize();
  for (const canvas of [referenceCanvas, latestCanvas]) {
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
    }
  }
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

function isWebcamResumePending() {
  return experiment.inputMode === "webcam" && experiment.webcamResumeAvailable && !isWebcamActive();
}

function isStreamTarget() {
  return isWebcamActive() || isWebcamResumePending() || isVideoTarget();
}

function isOperationBusy() {
  return experiment.training || experiment.running;
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
    if (config.rmsNorm) {
      total += config.hiddenDim;
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

function formatAtomWidth(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "--";
  }
  return value.toFixed(3);
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

function formatPreviewRate(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "--";
  }
  return `${(1 / seconds).toFixed(1)} fps`;
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
  experiment.weightRenderId += 1;
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
  const size = currentTargetSize();
  const pixels = currentPixelCount();
  experiment.featureTensor = tf.tidy(() => {
    const coords = createFloatTensor([pixels, 2], (index) => {
      const pixelIndex = Math.floor(index / 2);
      const isX = index % 2 === 0;
      const x = pixelIndex % size;
      const y = Math.floor(pixelIndex / size);
      return (isX ? x : y) / (size - 1);
    });

    const normal = makeNormalGenerator(config.seed);
    const atomCount = config.featureDim / 2;
    if (config.featureType === "gabor") {
      const centerRandom = mulberry32(config.seed + 17);
      const centers = createFloatTensor([atomCount, 2], () => centerRandom());
      const freqs = createFloatTensor([atomCount, 2], () => normal() * config.freqScale);
      const sigmas = tf.fill([atomCount, 2], config.gaborAtomWidth ?? 0.1);
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
    let rmsScale = null;
    if (config.rmsNorm) {
      rmsScale = createConstantVariable([config.hiddenDim], 1, `hidden_${layerIndex}_rms_scale`);
    }
    experiment.layers.push({ weight, bias, activation: config.activation || "silu", rmsScale });
    experiment.variables.push(weight, bias);
    if (rmsScale) {
      experiment.variables.push(rmsScale);
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
  queueWeightRender();
}

function applyRmsNorm(value, scale) {
  const rms = value.square().mean(1, true).add(1e-5).sqrt();
  return value.div(rms).mul(scale);
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

function setOptimizerLearningRate(value) {
  experiment.currentLearningRate = value;
  if (!experiment.optimizer) {
    return;
  }
  if (typeof experiment.optimizer.setLearningRate === "function") {
    experiment.optimizer.setLearningRate(value);
  } else if ("learningRate" in experiment.optimizer) {
    experiment.optimizer.learningRate = value;
  }
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
      if (layer.rmsScale) {
        value = applyRmsNorm(value, layer.rmsScale);
      }
    } else if (layer.activation === "tanh") {
      value = value.tanh();
      if (layer.rmsScale) {
        value = applyRmsNorm(value, layer.rmsScale);
      }
    } else if (layer.activation === "relu") {
      value = value.relu();
      if (layer.rmsScale) {
        value = applyRmsNorm(value, layer.rmsScale);
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
  setOptimizerLearningRate(learningRateForEpoch());
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
  queueWeightRender();
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

  if (experiment.running) {
    await applyPendingMediaSwitch();
  } else {
    await applyPendingOperation();
  }
}

async function runContinuous() {
  if (isWebcamResumePending()) {
    await startWebcam();
    return;
  }
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
    const releaseWebcam = isWebcamActive() && experiment.stopRequested && !experiment.pendingOperation;
    experiment.running = false;
    experiment.training = false;
    experiment.stopRequested = false;
    if (releaseWebcam) {
      stopWebcam({ keepResumeMode: true });
    } else {
      renderState();
    }
    await applyPendingOperation();
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
    while (!experiment.stopRequested) {
      if (isVideoTarget() && experiment.videoElement.ended) {
        await restartVideoPlayback();
      }
      await trainChunk(previewCount());
      await applyPendingMediaSwitch();
      if (isVideoTarget() && experiment.videoElement.ended) {
        await restartVideoPlayback();
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
    await applyPendingOperation();
  }
}

function requestStop() {
  experiment.stopRequested = true;
  renderState();
}

async function applyPendingMediaSwitch() {
  if (experiment.pendingOperation || !experiment.pendingMediaName || experiment.training) {
    return;
  }
  const mediaName = experiment.pendingMediaName;
  experiment.pendingMediaName = null;
  await selectMedia(mediaName, { resetModel: false, force: true });
}

function drawCanvasSourceToReference(source, sourceWidth, sourceHeight) {
  const size = currentTargetSize();
  const sourceSize = Math.min(sourceWidth || size, sourceHeight || size);
  const sourceX = Math.max(0, ((sourceWidth || size) - sourceSize) / 2);
  const sourceY = Math.max(0, ((sourceHeight || size) - sourceSize) / 2);

  referenceContext.clearRect(0, 0, size, size);
  referenceContext.drawImage(
    source,
    sourceX,
    sourceY,
    sourceSize,
    sourceSize,
    0,
    0,
    size,
    size,
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
  const size = currentTargetSize();
  const pixels = currentPixelCount();
  const imageData = referenceContext.getImageData(0, 0, size, size).data;
  const target = new Float32Array(pixels * 3);
  for (let source = 0, targetIndex = 0; source < imageData.length; source += 4, targetIndex += 3) {
    target[targetIndex] = imageData[source] / 255;
    target[targetIndex + 1] = imageData[source + 1] / 255;
    target[targetIndex + 2] = imageData[source + 2] / 255;
  }
  const nextTargetTensor = tf.tensor2d(target, [pixels, 3]);
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

function stopWebcam(options = {}) {
  const { keepResumeMode = false, render = true } = options;
  cancelAnimationFrame(experiment.webcamPreviewFrame);
  experiment.webcamPreviewFrame = 0;
  if (!keepResumeMode) {
    experiment.webcamResumeOnFocus = false;
  }
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
  experiment.webcamResumeAvailable = keepResumeMode;
  experiment.inputMode = keepResumeMode ? "webcam" : experiment.currentMedia?.kind || "image";
  if (render) {
    renderState();
  }
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
  if (experiment.webcamResumeOnFocus && isWebcamResumePending() && !document.hidden && document.hasFocus()) {
    startWebcam();
    return;
  }

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
  video.loop = true;
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
  if (!experiment.ready) {
    return;
  }
  if (isOperationBusy()) {
    queuePendingOperation({ type: "webcamStart" });
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

    stopWebcam({ render: false });
    disposeVideoElement();
    experiment.inputMode = "webcam";
    experiment.webcamStream = stream;
    experiment.webcamVideo = video;
    experiment.webcamResumeAvailable = false;
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
    if (isOperationBusy()) {
      queuePendingOperation({ type: "webcamStop" });
    } else {
      stopWebcam({ keepResumeMode: false });
    }
  } else {
    startWebcam();
  }
}

function drawOutput(values) {
  const size = currentTargetSize();
  const rgba = new Uint8ClampedArray(currentPixelCount() * 4);
  for (let source = 0, target = 0; source < values.length; source += 3, target += 4) {
    rgba[target] = Math.max(0, Math.min(255, Math.round(values[source] * 255)));
    rgba[target + 1] = Math.max(0, Math.min(255, Math.round(values[source + 1] * 255)));
    rgba[target + 2] = Math.max(0, Math.min(255, Math.round(values[source + 2] * 255)));
    rgba[target + 3] = 255;
  }
  latestCanvasContext.putImageData(new ImageData(rgba, size, size), 0, 0);
}

function formatWeightName(name) {
  const hiddenMatch = name.match(/^hidden_(\d+)_(weight|bias|rms_scale)$/);
  if (hiddenMatch) {
    const layerNumber = Number.parseInt(hiddenMatch[1], 10) + 1;
    const kind = hiddenMatch[2] === "weight"
      ? "W"
      : hiddenMatch[2] === "bias"
        ? "b"
        : "RMS";
    return `H${layerNumber} ${kind}`;
  }
  if (name === "output_weight") {
    return "Out W";
  }
  if (name === "output_bias") {
    return "Out b";
  }
  return name.replaceAll("_", " ");
}

function weightShapeText(shape) {
  return shape.length > 0 ? shape.join("x") : "scalar";
}

function weightGridShape(shape) {
  if (shape.length === 1) {
    return { rows: Math.max(1, shape[0]), columns: 1 };
  }
  if (shape.length === 2) {
    return { rows: Math.max(1, shape[1]), columns: Math.max(1, shape[0]) };
  }

  const count = Math.max(1, shape.reduce((product, value) => product * value, 1));
  const columns = Math.ceil(Math.sqrt(count));
  return { rows: Math.ceil(count / columns), columns };
}

function renderWeightBlocks(variables) {
  const signature = variables
    .map((variable) => `${variable.name}:${weightShapeText(variable.shape)}`)
    .join("|");
  if (weightsStrip.dataset.signature === signature) {
    return;
  }

  weightsStrip.dataset.signature = signature;
  weightsStrip.replaceChildren();
  if (variables.length === 0) {
    const empty = document.createElement("div");
    empty.className = "weight-empty";
    empty.textContent = "No tensors";
    weightsStrip.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  variables.forEach((variable, index) => {
    const { rows, columns } = weightGridShape(variable.shape);
    const isVector = variable.shape.length === 1;
    const article = document.createElement("article");
    article.className = "weight-block";
    article.classList.toggle("is-vector", isVector);
    article.classList.toggle("is-matrix", !isVector);
    article.style.setProperty("--weight-aspect", `${columns} / ${rows}`);
    article.style.setProperty(
      "--weight-block-width",
      isVector
        ? "58px"
        : `${Math.min(380, Math.max(240, Math.round(columns * 1.25)))}px`,
    );
    article.style.setProperty("--weight-columns", String(columns));
    article.style.setProperty("--weight-rows", String(rows));
    article.title = `${variable.name} ${weightShapeText(variable.shape)}`;

    const header = document.createElement("header");
    const label = document.createElement("span");
    label.textContent = formatWeightName(variable.name);
    const shape = document.createElement("code");
    shape.textContent = weightShapeText(variable.shape);
    header.append(label, shape);

    const canvasWrap = document.createElement("div");
    canvasWrap.className = "weight-canvas-wrap";
    const canvas = document.createElement("canvas");
    canvas.width = columns;
    canvas.height = rows;
    canvas.dataset.weightIndex = String(index);
    canvas.setAttribute("aria-label", `${formatWeightName(variable.name)} weights`);
    canvasWrap.append(canvas);

    article.append(header, canvasWrap);
    fragment.append(article);
  });

  weightsStrip.append(fragment);
}

function weightColorScale(values) {
  let maxAbs = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (Number.isFinite(value)) {
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  }
  return maxAbs || 1;
}

function mixChannel(start, end, amount) {
  return Math.round(start + (end - start) * amount);
}

function weightColor(value, scale) {
  const neutral = [248, 246, 242];
  const negative = [47, 112, 168];
  const positive = [190, 61, 83];
  const amount = Math.sqrt(Math.min(1, Math.abs(value) / scale));
  const target = value < 0 ? negative : positive;
  return [
    mixChannel(neutral[0], target[0], amount),
    mixChannel(neutral[1], target[1], amount),
    mixChannel(neutral[2], target[2], amount),
  ];
}

function weightValueAt(values, shape, x, y, columns) {
  if (shape.length === 1) {
    return values[y] ?? 0;
  }
  if (shape.length === 2) {
    return values[x * shape[1] + y] ?? 0;
  }
  return values[y * columns + x] ?? 0;
}

function drawWeightCanvas(canvas, values, shape) {
  const { rows, columns } = weightGridShape(shape);
  if (canvas.width !== columns || canvas.height !== rows) {
    canvas.width = columns;
    canvas.height = rows;
  }

  const context = canvas.getContext("2d");
  const imageData = context.createImageData(columns, rows);
  const scale = weightColorScale(values);

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < columns; x += 1) {
      const rawValue = weightValueAt(values, shape, x, y, columns);
      const value = Number.isFinite(rawValue) ? rawValue : 0;
      const [red, green, blue] = weightColor(value, scale);
      const target = (y * columns + x) * 4;
      imageData.data[target] = red;
      imageData.data[target + 1] = green;
      imageData.data[target + 2] = blue;
      imageData.data[target + 3] = 255;
    }
  }

  context.putImageData(imageData, 0, 0);
}

async function renderWeights(renderId) {
  const variables = experiment.variables.slice();
  renderWeightBlocks(variables);
  if (variables.length === 0) {
    return;
  }

  for (let index = 0; index < variables.length; index += 1) {
    const variable = variables[index];
    let values;
    try {
      values = await variable.data();
    } catch (error) {
      if (renderId === experiment.weightRenderId && experiment.weightsExpanded) {
        window.setTimeout(() => {
          if (experiment.weightsExpanded) {
            queueWeightRender();
          }
        }, 80);
      }
      return;
    }

    if (renderId !== experiment.weightRenderId || !experiment.weightsExpanded) {
      return;
    }

    const canvas = weightsStrip.querySelector(`canvas[data-weight-index="${index}"]`);
    if (canvas) {
      drawWeightCanvas(canvas, values, variable.shape);
    }
    await tf.nextFrame();
  }
}

function queueWeightRender() {
  experiment.weightRenderId += 1;
  experiment.weightRenderQueued = true;
  if (!experiment.weightsExpanded) {
    return;
  }
  runQueuedWeightRender();
}

async function runQueuedWeightRender() {
  if (experiment.weightRenderInFlight || !experiment.weightsExpanded) {
    return;
  }

  experiment.weightRenderInFlight = true;
  try {
    while (experiment.weightRenderQueued && experiment.weightsExpanded) {
      experiment.weightRenderQueued = false;
      await renderWeights(experiment.weightRenderId);
    }
  } finally {
    experiment.weightRenderInFlight = false;
  }

  if (experiment.weightRenderQueued && experiment.weightsExpanded) {
    runQueuedWeightRender();
  }
}

async function forceWeightRender() {
  if (!experiment.weightsExpanded) {
    return;
  }
  experiment.weightRenderId += 1;
  experiment.weightRenderQueued = false;
  await renderWeights(experiment.weightRenderId);
}

function clearOutput() {
  const size = currentTargetSize();
  latestCanvasContext.fillStyle = "#777a73";
  latestCanvasContext.fillRect(0, 0, size, size);
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
  const size = currentTargetSize();
  const image = await loadImage(assetUrl(mediaInfo.src));
  referenceContext.clearRect(0, 0, size, size);
  referenceContext.drawImage(image, 0, 0, size, size);

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

function queuePendingOperation(operation) {
  if (operation.resume === undefined) {
    operation.resume = experiment.running && operation.type !== "webcamStop";
  }
  experiment.pendingOperation = operation;
  if (operation.type === "media") {
    experiment.pendingMediaName = operation.mediaName;
  }
  if (experiment.running) {
    requestStop();
  } else {
    renderState();
  }
}

async function applyPendingOperation() {
  if (!experiment.pendingOperation || experiment.training || experiment.running) {
    return;
  }

  const operation = experiment.pendingOperation;
  const shouldResume = Boolean(operation.resume);
  experiment.pendingOperation = null;
  if (operation.type !== "media") {
    experiment.pendingMediaName = null;
  }

  if (operation.type === "reset") {
    await resetExperiment();
  } else if (operation.type === "modelPreset") {
    experiment.selectedModelKey = operation.modelKey;
    experiment.modelConfig = cloneModelConfig(MODEL_PRESETS[operation.modelKey]);
    await rebuildForModelConfig({ rebuildFeature: true, resetMetrics: true });
  } else if (operation.type === "modelSetting") {
    await rebuildForModelConfig(operation.options);
  } else if (operation.type === "featureTensorSetting") {
    await refreshFeatureTensorForConfig(operation.options);
  } else if (operation.type === "targetSize") {
    await changeTargetSize(operation.size);
  } else if (operation.type === "media") {
    await selectMedia(operation.mediaName, {
      resetModel: operation.resetModel,
      force: true,
    });
  } else if (operation.type === "webcamStart") {
    await startWebcam();
  } else if (operation.type === "webcamStop") {
    stopWebcam({ keepResumeMode: false });
  }

  await forceWeightRender();

  if (shouldResume && experiment.ready && experiment.targetTensor && !experiment.running && !experiment.training) {
    runContinuous();
  }
}

function currentModelName() {
  if (MODEL_PRESETS[experiment.selectedModelKey]) {
    return MODEL_PRESETS[experiment.selectedModelKey].label;
  }
  return "Custom";
}

function syncCurrentLearningRate() {
  setOptimizerLearningRate(learningRateForEpoch());
}

async function rebuildForModelConfig(options = {}) {
  const { rebuildFeature = true, resetMetrics = true } = options;
  if (isOperationBusy()) {
    queuePendingOperation({
      type: "modelSetting",
      options: { rebuildFeature, resetMetrics },
    });
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

async function refreshFeatureTensorForConfig(options = {}) {
  const { resetMetrics = false } = options;
  if (isOperationBusy()) {
    queuePendingOperation({
      type: "featureTensorSetting",
      options: { resetMetrics },
    });
    return;
  }

  hideError();
  experiment.training = true;
  if (resetMetrics) {
    resetRunMetrics();
  }
  renderState();

  try {
    if (experiment.ready) {
      disposeFeatureTensor();
      buildFeatureTensor();
      if (experiment.variables.length === 0) {
        buildModel();
      }
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

  if (key === "gaborAtomWidth") {
    if (currentModel().featureType !== "gabor") {
      renderState();
      return;
    }
    await refreshFeatureTensorForConfig({ resetMetrics: false });
    return;
  }

  if (STRUCTURAL_MODEL_KEYS.has(key)) {
    if (isOperationBusy()) {
      queuePendingOperation({
        type: "modelSetting",
        options: {
          rebuildFeature: key === "featureDim" || key === "featureType",
          resetMetrics: true,
        },
      });
      return;
    }
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

  if (isOperationBusy() && !force) {
    queuePendingOperation({
      type: "media",
      mediaName: mediaInfo.name,
      resetModel,
    });
    renderState();
    return;
  }

  if (isWebcamActive() || isWebcamResumePending()) {
    stopWebcam({ keepResumeMode: false, render: false });
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
    experiment.videoStats.active = experiment.running;
  } else {
    experiment.videoStats = makeEmptyVideoStats();
  }
  renderState();

  try {
    if (mediaInfo.kind === "video") {
      await loadVideoElement(mediaInfo);
      if (experiment.ready) {
        updateTargetTensorFromReferenceCanvas();
      }
      if (experiment.running) {
        await restartVideoPlayback();
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
  if (isOperationBusy()) {
    queuePendingOperation({ type: "reset" });
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
  if (!MODEL_PRESETS[modelKey] || modelKey === experiment.selectedModelKey) {
    return;
  }

  experiment.selectedModelKey = modelKey;
  experiment.modelConfig = cloneModelConfig(MODEL_PRESETS[modelKey]);
  if (isOperationBusy()) {
    queuePendingOperation({ type: "modelPreset", modelKey });
    return;
  }
  await rebuildForModelConfig({ rebuildFeature: true, resetMetrics: true });
}

async function changeTargetSize(size) {
  if (!TARGET_SIZES.includes(size) || size === currentTargetSize()) {
    return;
  }
  if (isOperationBusy()) {
    queuePendingOperation({ type: "targetSize", size });
    return;
  }

  hideError();
  experiment.targetSize = size;
  configureCanvases();
  disposeFeatureTensor();
  experiment.training = true;
  renderState();

  try {
    if (experiment.ready) {
      buildFeatureTensor();
      if (experiment.variables.length === 0) {
        buildModel();
      }
      if (isWebcamActive()) {
        await sampleWebcamTarget({ appendHistory: false, publish: true });
      } else if (isVideoTarget()) {
        await sampleVideoTarget({ appendHistory: false, publish: true });
      } else if (experiment.currentMedia) {
        await loadReferenceImage(experiment.currentMedia, true);
        await publishPreview(performance.now(), { appendHistory: false });
      } else {
        clearOutput();
      }
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
  configureCanvases();
  updateSliderProgress();
  clearOutput();
  renderResolutionOptions();
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
  if (isWebcamActive() || isWebcamResumePending()) {
    return `webcam ${currentTargetSize()}`;
  }
  if (isVideoSelected()) {
    const current = experiment.videoStats.currentTime;
    const duration = experiment.videoStats.duration;
    return `${formatVideoTime(current)} / ${formatVideoTime(duration)}`;
  }
  if (!experiment.currentMedia && experiment.targetTensor) {
    return `still ${currentTargetSize()}`;
  }
  return `${currentTargetSize()} x ${currentTargetSize()}`;
}

function renderState() {
  const webcamActive = isWebcamActive();
  const canResume = experiment.epoch > 0;
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
    ? formatPreviewRate(experiment.lastUpdateSeconds)
    : "--";
  weightsSummary.textContent = experiment.variables.length > 0
    ? `${experiment.variables.length} tensors`
    : "--";
  weightsToggle.setAttribute("aria-expanded", String(experiment.weightsExpanded));
  weightsContents.setAttribute("aria-hidden", String(!experiment.weightsExpanded));
  weightsContents.inert = !experiment.weightsExpanded;
  weightsPanel.classList.toggle("is-expanded", experiment.weightsExpanded);

  startButtonLabel.textContent = experiment.running ? "Pause" : canResume ? "Resume" : "Start";
  startButton.setAttribute("aria-label", startButtonLabel.textContent);
  if (runButtonIcon) {
    runButtonIcon.setAttribute("d", experiment.running ? "M7 5h4v14H7zm6 0h4v14h-4z" : "M8 5v14l11-7z");
  }
  startButton.disabled = experiment.running
    ? experiment.stopRequested
    : !experiment.ready || experiment.training || !experiment.targetTensor;
  stepButton.disabled = !experiment.ready || experiment.running || experiment.training || isVideoSelected() || !experiment.targetTensor;
  resetButton.disabled = !experiment.ready || !experiment.targetTensor;
  webcamButton.textContent = webcamActive ? "Stop webcam" : "Use webcam";
  webcamButton.classList.toggle("is-active", webcamActive);
  webcamButton.disabled = !experiment.ready;
  setMediaButtonsDisabled(experiment.mediaItems.length === 0);
  renderResolutionOptions();
  renderMediaOptions();
  renderModelOptions();
  renderModelSettings();
  drawLossChart(experiment.lossHistory);
}

function renderResolutionOptions() {
  if (resolutionPicker.children.length === 0) {
    for (const size of TARGET_SIZES) {
      const button = document.createElement("button");
      button.className = "resolution-option";
      button.type = "button";
      button.dataset.size = String(size);
      button.setAttribute("role", "radio");
      button.textContent = `${size}x${size}`;
      button.addEventListener("click", () => changeTargetSize(size));
      resolutionPicker.append(button);
    }
  }

  for (const button of resolutionPicker.querySelectorAll(".resolution-option")) {
    const selected = Number.parseInt(button.dataset.size, 10) === currentTargetSize();
    button.classList.toggle("is-selected", selected);
    button.setAttribute("aria-checked", String(selected));
    button.disabled = false;
  }
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
    button.disabled = false;
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
  gaborAtomWidthSlider.value = String(config.gaborAtomWidth ?? 0.1);
  gaborAtomWidthValue.textContent = formatAtomWidth(config.gaborAtomWidth ?? 0.1);
  activationSelect.value = config.activation || "silu";
  rmsNormToggle.checked = Boolean(config.rmsNorm);

  imageStartLrSlider.value = String(lrToSliderValue(config.imageLearningRate));
  imageEndLrSlider.value = String(lrToSliderValue(config.imageMinLearningRate));
  videoLrSlider.value = String(lrToSliderValue(config.streamLearningRate));
  imageStartLrValue.textContent = formatLearningRate(config.imageLearningRate);
  imageEndLrValue.textContent = formatLearningRate(config.imageMinLearningRate);
  videoLrValue.textContent = formatLearningRate(config.streamLearningRate);

  modelToggle.disabled = false;
  modelField.classList.toggle("is-gabor", config.featureType === "gabor");
  for (const control of [
    hiddenDimSelect,
    hiddenLayersSelect,
    featureDimSelect,
    featureTypeSelect,
    gaborAtomWidthSlider,
    activationSelect,
    rmsNormToggle,
  ]) {
    control.disabled = false;
  }
  imageStartLrSlider.disabled = !experiment.ready;
  imageEndLrSlider.disabled = !experiment.ready;
  videoLrSlider.disabled = !experiment.ready;
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
  if (experiment.running) {
    requestStop();
    return;
  }
  runContinuous();
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

weightsToggle.addEventListener("click", () => {
  experiment.weightsExpanded = !experiment.weightsExpanded;
  renderState();
  queueWeightRender();
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

gaborAtomWidthSlider.addEventListener("input", () => {
  updateModelSetting("gaborAtomWidth", Number.parseFloat(gaborAtomWidthSlider.value));
});

activationSelect.addEventListener("change", () => {
  updateModelSetting("activation", activationSelect.value);
});

rmsNormToggle.addEventListener("change", () => {
  updateModelSetting("rmsNorm", rmsNormToggle.checked);
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

window.addEventListener("resize", () => {
  drawLossChart(experiment.lossHistory);
});
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
