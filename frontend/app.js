const drawCanvas = document.getElementById("draw-canvas");
const miniCanvas = document.getElementById("mini-canvas");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-btn");
const seedBtn = document.getElementById("seed-btn");
const strokeSizeInput = document.getElementById("stroke-size");
const scoreBars = document.getElementById("score-bars");
const predictedDigit = document.getElementById("predicted-digit");
const confidenceScore = document.getElementById("confidence-score");
const inputEnergy = document.getElementById("input-energy");
const topPrediction = document.getElementById("top-prediction");
const inferenceMode = document.getElementById("inference-mode");
const deviceLabel = document.getElementById("device-label");
const runtimeLabel = document.getElementById("runtime-label");
const modelLabel = document.getElementById("model-label");
const timelineList = document.getElementById("timeline-list");
const stageTitle = document.getElementById("stage-title");
const stageStatusChip = document.getElementById("stage-status-chip");
const stageSummary = document.getElementById("stage-summary");
const stageLiveStrip = document.getElementById("stage-live-strip");
const stageObjective = document.getElementById("stage-objective");
const stageFinish = document.getElementById("stage-finish");
const stageChecklist = document.getElementById("stage-checklist");
const stageCommand = document.getElementById("stage-command");
const stageReason = document.getElementById("stage-reason");
const stageCompleteBtn = document.getElementById("stage-complete-btn");
const stageResetBtn = document.getElementById("stage-reset-btn");
const liveEvalAccuracy = document.getElementById("live-eval-accuracy");
const liveEvalTotal = document.getElementById("live-eval-total");
const liveCompositionStatus = document.getElementById("live-composition-status");
const labView = document.getElementById("lab-view");
const taskView = document.getElementById("task-view");
const viewLabBtn = document.getElementById("view-lab-btn");
const viewTaskBtn = document.getElementById("view-task-btn");
const heroEyebrow = document.getElementById("hero-eyebrow");
const heroTitle = document.getElementById("hero-title");
const heroText = document.getElementById("hero-text");
const taskNameInput = document.getElementById("task-name");
const taskDatasetInput = document.getElementById("task-dataset");
const taskClassesInput = document.getElementById("task-classes");
const taskShapeInput = document.getElementById("task-shape");
const taskModelTagInput = document.getElementById("task-model-tag");
const taskScriptInput = document.getElementById("task-script");
const taskPositiveInput = document.getElementById("task-positive");
const taskNegativeInput = document.getElementById("task-negative");
const taskSaveBtn = document.getElementById("task-save-btn");
const taskNewBtn = document.getElementById("task-new-btn");
const taskPresets = document.getElementById("task-presets");
const taskClassCount = document.getElementById("task-class-count");
const taskLabelSlots = document.getElementById("task-label-slots");
const taskType = document.getElementById("task-type");
const taskTrainCommand = document.getElementById("task-train-command");
const taskOverlayPlan = document.getElementById("task-overlay-plan");
const taskConfigOutput = document.getElementById("task-config-output");

const drawCtx = drawCanvas ? drawCanvas.getContext("2d") : null;
const miniCtx = miniCanvas ? miniCanvas.getContext("2d") : null;
if (miniCtx) {
  miniCtx.imageSmoothingEnabled = false;
}

const STORAGE_KEY = "ff-timeline-progress-v1";
const TASK_STORAGE_KEY = "ff-task-studio-v1";
let isDrawing = false;
let backendInfo = null;
let currentView = window.location.pathname.endsWith("/tasks.html") || window.location.pathname.endsWith("tasks.html")
  ? "task"
  : "lab";
let selectedTaskId = null;

const stages = [
  {
    id: "runtime",
    title: "Boot the FF runtime",
    summary: "Start the backend and verify that Python, torch, and the selected device are flowing into the interface.",
    objective: "Replace browser fallback state with real backend-reported runtime information.",
    finish: "The header shows a real Device, Runtime, and Model value, and inference is no longer marked unavailable.",
    checklist: [
      "Run the backend from the project root.",
      "Open the UI from http://127.0.0.1:8000.",
      "Confirm the header shows actual backend values."
    ],
    command: "python3 ff_backend.py",
    reason: "Without the backend bridge, the UI cannot reflect the actual FF model or hardware state."
  },
  {
    id: "single-layer",
    title: "Train a single FF layer",
    summary: "Train the first backprop-free layer using positive and negative pairs with local Softplus goodness pressure.",
    objective: "Create ff_single_layer.pt and prove the heuristic scorer can be replaced with a learned checkpoint.",
    finish: "After training and backend restart, the model label changes to 1-layer FF checkpoint.",
    checklist: [
      "Run the single-layer trainer.",
      "Watch loss and test accuracy move above chance.",
      "Restart the backend after the checkpoint is saved."
    ],
    command: "python3 ff_mnist_single_layer.py",
    reason: "This isolates the local update rule before adding deeper feature hierarchies."
  },
  {
    id: "multilayer",
    title: "Train a stacked FF net",
    summary: "Build a real local hierarchy so each layer learns its own features and inference sums goodness across layers.",
    objective: "Produce ff_multilayer.pt and make the backend use a deeper FF checkpoint by default.",
    finish: "The header shows 2-layer FF checkpoint or deeper after restart.",
    checklist: [
      "Train the multi-layer script.",
      "Check that every layer prints its own loss.",
      "Restart the backend and verify the new checkpoint is loaded."
    ],
    command: "python3 ff_mnist_multilayer.py",
    reason: "This is the actual transition from a toy classifier to a reusable local feature stack."
  },
  {
    id: "evaluation",
    title: "Measure failure modes",
    summary: "Stop relying on visual demos alone and add evaluation for accuracy, confusion, and weak classes.",
    objective: "Know where the FF model works and where it is failing before making architectural changes.",
    finish: "You have a repeatable evaluation routine and can identify the worst digit confusions.",
    checklist: [
      "Run checkpoint evaluation on MNIST test data.",
      "Inspect class confusions and unstable cases.",
      "Use those results to tune threshold, width, or depth."
    ],
    command: "python3 ff_evaluate.py",
    reason: "Without evaluation, the timeline turns into theater instead of engineering."
  },
  {
    id: "composition",
    title: "Teach composition",
    summary: "Build a multi-digit route so the model can reuse local digit features for strings like 1356 instead of memorizing whole numbers.",
    objective: "Introduce a segmented or sliding-window FF pipeline for digit strings.",
    finish: "The system reuses learned digit features across positions and processes multi-digit inputs compositionally.",
    checklist: [
      "Generate multi-digit strings from MNIST digits.",
      "Apply a fixed-slot or sliding-window encoder.",
      "Add a second FF stage for sequence-level composition."
    ],
    command: "next: implement multi-digit FF pipeline",
    reason: "This is the path toward local feature reuse, which is the actual research goal."
  }
];

const defaultTaskPresets = [
  {
    id: "digits-mnist",
    name: "MNIST digits",
    dataset: "MNIST",
    classes: "0,1,2,3,4,5,6,7,8,9",
    shape: "28x28 grayscale",
    modelTag: "ff_digits_v1",
    script: "ff_mnist_multilayer.py",
    positive: "Real digit image paired with its correct digit label.",
    negative: "Real digit image paired with a wrong digit label from 0-9."
  },
  {
    id: "digits-letters",
    name: "Digits + letters",
    dataset: "EMNIST Balanced",
    classes: "0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z",
    shape: "28x28 grayscale",
    modelTag: "ff_alnum_v1",
    script: "ff_train_generic.py",
    positive: "Real symbol image paired with its true alphanumeric class.",
    negative: "Real symbol image paired with a wrong symbol class from the same vocabulary."
  },
  {
    id: "quickdraw-animals",
    name: "Sketch animals",
    dataset: "Quick, Draw! subset",
    classes: "cat,dog,bird,fish,rabbit",
    shape: "64x64 grayscale",
    modelTag: "ff_animals_v1",
    script: "ff_train_generic.py",
    positive: "Real sketch paired with the correct animal tag.",
    negative: "Real sketch paired with a different animal tag."
  }
];

function readProgress() {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) {
      return { activeIndex: 0, completed: [] };
    }

    const parsed = JSON.parse(saved);
    return {
      activeIndex: Number.isInteger(parsed.activeIndex) ? parsed.activeIndex : 0,
      completed: Array.isArray(parsed.completed) ? parsed.completed : []
    };
  } catch (error) {
    return { activeIndex: 0, completed: [] };
  }
}

let progress = readProgress();

function readTaskState() {
  try {
    const saved = window.localStorage.getItem(TASK_STORAGE_KEY);
    if (!saved) {
      return {
        presets: defaultTaskPresets,
        selectedTaskId: defaultTaskPresets[0].id
      };
    }

    const parsed = JSON.parse(saved);
    return {
      presets: Array.isArray(parsed.presets) && parsed.presets.length > 0 ? parsed.presets : defaultTaskPresets,
      selectedTaskId: typeof parsed.selectedTaskId === "string" ? parsed.selectedTaskId : defaultTaskPresets[0].id
    };
  } catch (error) {
    return {
      presets: defaultTaskPresets,
      selectedTaskId: defaultTaskPresets[0].id
    };
  }
}

const taskState = readTaskState();
let taskPresetsState = taskState.presets;
selectedTaskId = taskState.selectedTaskId;

function persistProgress() {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
}

function persistTaskState() {
  window.localStorage.setItem(
    TASK_STORAGE_KEY,
    JSON.stringify({
      presets: taskPresetsState,
      selectedTaskId,
    })
  );
}

function normalizeProgress() {
  const validIds = new Set(stages.map((stage) => stage.id));
  progress.completed = progress.completed.filter((id) => validIds.has(id));
  if (!Number.isInteger(progress.activeIndex) || progress.activeIndex < 0) {
    progress.activeIndex = 0;
  }
  if (progress.activeIndex >= stages.length) {
    progress.activeIndex = stages.length - 1;
  }
}

function setCompletedThrough(index) {
  for (let i = 0; i <= index && i < stages.length; i += 1) {
    if (!progress.completed.includes(stages[i].id)) {
      progress.completed.push(stages[i].id);
    }
  }
}

function lastCompletedIndex() {
  let index = -1;
  for (let i = 0; i < stages.length; i += 1) {
    if (progress.completed.includes(stages[i].id)) {
      index = i;
    } else {
      break;
    }
  }
  return index;
}

function syncAutoProgress() {
  normalizeProgress();

  if (backendInfo) {
    setCompletedThrough(0);

    const model = String(backendInfo.model || "");
    const runtime = String(backendInfo.runtime || "");
    const inference = String(backendInfo.inference_mode || "");
    const evaluation = backendInfo.evaluation || {};
    const composition = backendInfo.composition || {};

    if (model.includes("1-layer FF checkpoint") || model.includes("Single FFLayer checkpoint")) {
      setCompletedThrough(1);
    }

    const hasMultilayerCheckpoint = /([2-9]|[1-9][0-9]+)-layer FF checkpoint/.test(model);
    if (hasMultilayerCheckpoint) {
      setCompletedThrough(2);
    }

    const runtimeAvailable =
      runtime.includes("selected=") ||
      inference.toLowerCase().includes("forward-forward") ||
      inference.toLowerCase().includes("heuristic backend");
    if (runtimeAvailable) {
      setCompletedThrough(0);
    }

    if (evaluation.completed === true) {
      setCompletedThrough(3);
    }

    if (composition.completed === true) {
      setCompletedThrough(4);
    }
  }

  const highestComplete = lastCompletedIndex();
  const nextUnlocked = Math.min(highestComplete + 1, stages.length - 1);

  if (progress.activeIndex > nextUnlocked) {
    progress.activeIndex = nextUnlocked;
  }

  if (progress.activeIndex < 0) {
    progress.activeIndex = 0;
  }

  persistProgress();
}

function stageState(index) {
  if (progress.completed.includes(stages[index].id)) {
    return "complete";
  }
  if (index === progress.activeIndex) {
    return "active";
  }
  return "locked";
}

function renderTimeline() {
  timelineList.innerHTML = "";

  stages.forEach((stage, index) => {
    const state = stageState(index);
    const item = document.createElement("button");
    item.className = `timeline-step is-${state}`;
    item.disabled = state === "locked";
    item.innerHTML = `
      <div class="timeline-step-head">
        <div class="timeline-step-title">${index + 1}. ${stage.title}</div>
        <span class="timeline-state">${state}</span>
      </div>
      <div class="timeline-step-meta">${stage.summary}</div>
    `;
    item.addEventListener("click", () => {
      if (state === "locked") {
        return;
      }
      progress.activeIndex = index;
      persistProgress();
      renderWorkflow();
    });
    timelineList.appendChild(item);
  });
}

function renderStageDetail() {
  const stage = stages[progress.activeIndex];
  const state = stageState(progress.activeIndex);
  const evaluation = backendInfo?.evaluation || {};
  const composition = backendInfo?.composition || {};

  stageTitle.textContent = stage.title;
  stageStatusChip.textContent = state;
  stageSummary.textContent = stage.summary;
  stageObjective.textContent = stage.objective;
  stageFinish.textContent = stage.finish;
  stageCommand.textContent = stage.command;
  stageReason.textContent = stage.reason;
  stageChecklist.innerHTML = "";

  stage.checklist.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    stageChecklist.appendChild(li);
  });

  const showLiveStrip = progress.activeIndex >= 3;
  stageLiveStrip.hidden = !showLiveStrip;
  if (showLiveStrip) {
    liveEvalAccuracy.textContent =
      evaluation.completed === true && typeof evaluation.accuracy === "number"
        ? `${(evaluation.accuracy * 100).toFixed(2)}%`
        : "Not run";
    liveEvalTotal.textContent =
      evaluation.completed === true && typeof evaluation.total === "number"
        ? String(evaluation.total)
        : "-";
    liveCompositionStatus.textContent =
      composition.completed === true
        ? composition.mode || "Completed"
        : "Not built";
  }

  stageCompleteBtn.disabled = state === "complete";
  stageCompleteBtn.textContent = state === "complete" ? "Step completed" : "Mark step complete";
}

function renderWorkflow() {
  if (!timelineList || !stageTitle) {
    return;
  }
  syncAutoProgress();
  renderTimeline();
  renderStageDetail();
}

function taskClassesList(rawValue) {
  return rawValue
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function collectTaskForm() {
  return {
    id: selectedTaskId || `task-${Date.now()}`,
    name: taskNameInput.value.trim() || "Untitled FF task",
    dataset: taskDatasetInput.value.trim() || "Unknown dataset",
    classes: taskClassesInput.value.trim(),
    shape: taskShapeInput.value.trim() || "28x28 grayscale",
    modelTag: taskModelTagInput.value.trim() || "ff_task_v1",
    script: taskScriptInput.value.trim() || "ff_train_generic.py",
    positive: taskPositiveInput.value.trim() || "Real sample paired with its correct class.",
    negative: taskNegativeInput.value.trim() || "Real sample paired with a wrong class.",
  };
}

function fillTaskForm(task) {
  selectedTaskId = task.id;
  taskNameInput.value = task.name;
  taskDatasetInput.value = task.dataset;
  taskClassesInput.value = task.classes;
  taskShapeInput.value = task.shape;
  taskModelTagInput.value = task.modelTag;
  taskScriptInput.value = task.script;
  taskPositiveInput.value = task.positive;
  taskNegativeInput.value = task.negative;
  renderTaskStudio();
}

function newBlankTask() {
  selectedTaskId = `task-${Date.now()}`;
  taskNameInput.value = "";
  taskDatasetInput.value = "";
  taskClassesInput.value = "";
  taskShapeInput.value = "";
  taskModelTagInput.value = "";
  taskScriptInput.value = "";
  taskPositiveInput.value = "";
  taskNegativeInput.value = "";
  renderTaskStudio();
}

function renderTaskPresets() {
  taskPresets.innerHTML = "";
  taskPresetsState.forEach((task) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `task-preset${task.id === selectedTaskId ? " is-selected" : ""}`;
    button.innerHTML = `
      <strong>${task.name}</strong>
      <span>${task.dataset} · ${taskClassesList(task.classes).length} classes</span>
    `;
    button.addEventListener("click", () => fillTaskForm(task));
    taskPresets.appendChild(button);
  });
}

function renderTaskStudio() {
  if (!taskPresets || !taskNameInput) {
    return;
  }
  renderTaskPresets();
  const task = collectTaskForm();
  const classes = taskClassesList(task.classes);
  taskClassCount.textContent = String(classes.length);
  taskLabelSlots.textContent = String(classes.length);
  taskType.textContent = classes.length > 10 ? "Multi-class FF" : "Classifier";
  taskTrainCommand.textContent =
    `python3 ${task.script} --dataset "${task.dataset}" --classes ${classes.length} --task-tag ${task.modelTag}`;
  taskOverlayPlan.textContent =
    classes.length > 0
      ? `Reserve the first ${classes.length} input positions for label overlay. Positive pairs use the true class index. Negative pairs sample another class from the same list.`
      : "Add at least one class to define the label overlay width and FF positive/negative pairs.";
  taskConfigOutput.textContent = JSON.stringify(
    {
      name: task.name,
      dataset: task.dataset,
      class_count: classes.length,
      classes,
      input_shape: task.shape,
      model_tag: task.modelTag,
      train_script: task.script,
      positive_rule: task.positive,
      negative_rule: task.negative,
    },
    null,
    2
  );
}

function saveCurrentTask() {
  const task = collectTaskForm();
  const existingIndex = taskPresetsState.findIndex((item) => item.id === task.id);
  if (existingIndex >= 0) {
    taskPresetsState[existingIndex] = task;
  } else {
    taskPresetsState = [task, ...taskPresetsState];
  }
  selectedTaskId = task.id;
  persistTaskState();
  renderTaskStudio();
}

function switchView(view) {
  currentView = view;
  const showLab = view === "lab";
  if (labView) {
    labView.hidden = !showLab;
  }
  if (taskView) {
    taskView.hidden = showLab;
  }
  if (viewLabBtn) {
    viewLabBtn.classList.toggle("is-selected", showLab);
  }
  if (viewTaskBtn) {
    viewTaskBtn.classList.toggle("is-selected", !showLab);
  }
  if (heroEyebrow) {
    heroEyebrow.textContent = showLab
      ? "Backprop-Free / Local Goodness / MPS"
      : "Task Design / Class Layout / Positive-Negative Rules";
  }
  if (heroTitle) {
    heroTitle.textContent = showLab ? "Forward-Forward Visual Agent" : "Forward-Forward Task Studio";
  }
  if (heroText) {
    heroText.textContent = showLab
      ? "A front view for probing layer-local training, positive and negative passes, and digit inference without a global gradient."
      : "Define a new FF problem, choose the task classes, and generate the class-overlay plan and training command needed for another model.";
  }
}

function setupCanvas() {
  if (deviceLabel) {
    deviceLabel.textContent = "Backend unavailable";
  }
  if (runtimeLabel) {
    runtimeLabel.textContent = "Waiting for backend";
  }
  if (modelLabel) {
    modelLabel.textContent = "No backend connection";
  }
  if (drawCtx && drawCanvas) {
    drawCtx.fillStyle = "#111111";
    drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
    renderMiniGrid();
    renderScores(Array.from({ length: 10 }, () => 0));
  }
  renderWorkflow();
  if (taskNameInput) {
    const initialTask = taskPresetsState.find((task) => task.id === selectedTaskId) || taskPresetsState[0];
    fillTaskForm(initialTask);
  }
  switchView(currentView === "task" ? "task" : "lab");
}

async function loadBackendInfo() {
  try {
    const response = await fetch("/api/info");
    if (!response.ok) {
      throw new Error(`API responded with ${response.status}`);
    }

    const info = await response.json();
    backendInfo = info;
    if (deviceLabel) {
      deviceLabel.textContent = info.device || deviceLabel.textContent;
    }
    if (runtimeLabel) {
      runtimeLabel.textContent = info.runtime || runtimeLabel.textContent;
    }
    if (modelLabel) {
      modelLabel.textContent = info.model || modelLabel.textContent;
    }
    if (inferenceMode) {
      inferenceMode.textContent = info.inference_mode || "Live API";
    }
    renderWorkflow();
  } catch (error) {
    backendInfo = null;
    if (deviceLabel) {
      deviceLabel.textContent = "Backend unavailable";
    }
    if (runtimeLabel) {
      runtimeLabel.textContent = "Backend unavailable";
    }
    if (modelLabel) {
      modelLabel.textContent = "Backend unavailable";
    }
    if (inferenceMode) {
      inferenceMode.textContent = "Mock fallback active";
    }
    renderWorkflow();
  }
}

function drawStroke(x, y) {
  if (!drawCtx || !strokeSizeInput) {
    return;
  }
  const size = Number(strokeSizeInput.value);
  const gradient = drawCtx.createRadialGradient(x, y, 1, x, y, size);
  gradient.addColorStop(0, "rgba(255, 248, 236, 0.98)");
  gradient.addColorStop(1, "rgba(255, 248, 236, 0)");
  drawCtx.fillStyle = gradient;
  drawCtx.beginPath();
  drawCtx.arc(x, y, size, 0, Math.PI * 2);
  drawCtx.fill();
  renderMiniGrid();
}

function pointFromEvent(event) {
  if (!drawCanvas) {
    return { x: 0, y: 0 };
  }
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function renderMiniGrid() {
  if (!miniCtx || !drawCtx || !drawCanvas || !miniCanvas) {
    return [];
  }
  miniCtx.clearRect(0, 0, miniCanvas.width, miniCanvas.height);
  miniCtx.fillStyle = "#111111";
  miniCtx.fillRect(0, 0, miniCanvas.width, miniCanvas.height);

  const source = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height).data;
  const cell = miniCanvas.width / 28;
  const pixels = [];

  for (let row = 0; row < 28; row += 1) {
    for (let col = 0; col < 28; col += 1) {
      let sum = 0;
      for (let y = 0; y < 10; y += 1) {
        for (let x = 0; x < 10; x += 1) {
          const px = col * 10 + x;
          const py = row * 10 + y;
          const idx = (py * drawCanvas.width + px) * 4;
          sum += source[idx];
        }
      }
      const value = sum / 100 / 255;
      pixels.push(value);
      const shade = Math.round(value * 255);
      miniCtx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      miniCtx.fillRect(col * cell, row * cell, cell, cell);
    }
  }

  return pixels;
}

function renderScores(scores) {
  if (!scoreBars) {
    return;
  }
  const maxScore = Math.max(...scores, 1e-6);
  scoreBars.innerHTML = "";

  scores.forEach((score, digit) => {
    const row = document.createElement("div");
    row.className = "score-row";
    row.innerHTML = `
      <strong>${digit}</strong>
      <div class="score-track"><div class="score-fill" style="width:${(score / maxScore) * 100}%"></div></div>
      <span class="score-value">${score.toFixed(3)}</span>
    `;
    scoreBars.appendChild(row);
  });
}

function summarizeScores(scores, pixels) {
  if (!predictedDigit || !confidenceScore || !inputEnergy || !topPrediction) {
    return;
  }
  const bestScore = Math.max(...scores);
  const bestDigit = scores.indexOf(bestScore);
  const total = scores.reduce((sum, score) => sum + score, 0);
  const confidence = total > 0 ? bestScore / total : 0;
  const energy = pixels.reduce((sum, value) => sum + value * value, 0);

  predictedDigit.textContent = String(bestDigit);
  confidenceScore.textContent = confidence.toFixed(3);
  inputEnergy.textContent = energy.toFixed(2);
  topPrediction.textContent = String(bestDigit);
  renderScores(scores);
}

function mockPredict(pixels) {
  const energy = pixels.reduce((sum, value) => sum + value, 0);
  const centerBias = pixels
    .map((value, index) => {
      const row = Math.floor(index / 28);
      const col = index % 28;
      const dx = Math.abs(col - 13.5) / 13.5;
      const dy = Math.abs(row - 13.5) / 13.5;
      return value * (1 - (dx + dy) * 0.25);
    })
    .reduce((sum, value) => sum + value, 0);

  return Array.from({ length: 10 }, (_, digit) => {
    const phase = (digit + 1) * 0.61;
    const signal = Math.sin(energy * 0.037 + phase) + Math.cos(centerBias * 0.041 - phase);
    return Math.max(0.01, 0.8 + signal * 0.35 + digit * 0.015);
  });
}

async function predictScores() {
  if (!drawCanvas) {
    return;
  }
  const pixels = renderMiniGrid();
  const payload = { pixels, width: 28, height: 28 };

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API responded with ${response.status}`);
    }

    const result = await response.json();
    if (inferenceMode) {
      inferenceMode.textContent = result.inference_mode || "Live API";
    }
    if (deviceLabel && typeof result.device === "string" && result.device.trim()) {
      deviceLabel.textContent = result.device;
    }
    if (runtimeLabel && typeof result.runtime === "string" && result.runtime.trim()) {
      runtimeLabel.textContent = result.runtime;
    }
    if (modelLabel && typeof result.model === "string" && result.model.trim()) {
      modelLabel.textContent = result.model;
    }
    summarizeScores(result.scores, pixels);
  } catch (error) {
    if (inferenceMode) {
      inferenceMode.textContent = "Mock fallback active";
    }
    summarizeScores(mockPredict(pixels), pixels);
  }
}

function clearCanvas() {
  if (!drawCtx || !drawCanvas || !predictedDigit || !confidenceScore || !inputEnergy || !topPrediction) {
    return;
  }
  drawCtx.fillStyle = "#111111";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  predictedDigit.textContent = "-";
  confidenceScore.textContent = "-";
  inputEnergy.textContent = "-";
  topPrediction.textContent = "?";
  renderMiniGrid();
  renderScores(Array.from({ length: 10 }, () => 0));
}

function seedDigit() {
  if (!drawCtx) {
    return;
  }
  clearCanvas();
  drawCtx.strokeStyle = "rgba(255, 248, 236, 0.95)";
  drawCtx.lineWidth = 20;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";

  drawCtx.beginPath();
  drawCtx.moveTo(150, 50);
  drawCtx.lineTo(110, 70);
  drawCtx.lineTo(88, 112);
  drawCtx.lineTo(102, 165);
  drawCtx.lineTo(150, 210);
  drawCtx.lineTo(194, 170);
  drawCtx.lineTo(197, 110);
  drawCtx.lineTo(160, 78);
  drawCtx.stroke();

  renderMiniGrid();
}

if (drawCanvas) {
  drawCanvas.addEventListener("pointerdown", (event) => {
    isDrawing = true;
    const point = pointFromEvent(event);
    drawStroke(point.x, point.y);
  });

  drawCanvas.addEventListener("pointermove", (event) => {
    if (!isDrawing) {
      return;
    }
    const point = pointFromEvent(event);
    drawStroke(point.x, point.y);
  });
}

window.addEventListener("pointerup", () => {
  isDrawing = false;
});

if (predictBtn) {
  predictBtn.addEventListener("click", predictScores);
}
if (clearBtn) {
  clearBtn.addEventListener("click", clearCanvas);
}
if (seedBtn) {
  seedBtn.addEventListener("click", seedDigit);
}

if (stageCompleteBtn) {
  stageCompleteBtn.addEventListener("click", () => {
    const current = stages[progress.activeIndex];
    if (!progress.completed.includes(current.id)) {
      progress.completed.push(current.id);
    }
    if (progress.activeIndex < stages.length - 1) {
      progress.activeIndex += 1;
    }
    persistProgress();
    renderWorkflow();
  });
}

if (stageResetBtn) {
  stageResetBtn.addEventListener("click", () => {
    progress = { activeIndex: 0, completed: [] };
    persistProgress();
    renderWorkflow();
  });
}

if (taskSaveBtn) {
  taskSaveBtn.addEventListener("click", saveCurrentTask);
}
if (taskNewBtn) {
  taskNewBtn.addEventListener("click", newBlankTask);
}
[
  taskNameInput,
  taskDatasetInput,
  taskClassesInput,
  taskShapeInput,
  taskModelTagInput,
  taskScriptInput,
  taskPositiveInput,
  taskNegativeInput,
].filter(Boolean).forEach((input) => {
  input.addEventListener("input", renderTaskStudio);
});

setupCanvas();
loadBackendInfo();
