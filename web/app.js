const imageInput = document.querySelector("#imageInput");
const uploadButton = document.querySelector(".upload-button");
const debugInput = document.querySelector("#debugInput");
const debugToggle = document.querySelector(".debug-toggle");
const debugNotice = document.querySelector("#debugNotice");
const testImageChooserButton = document.querySelector("#testImageChooserButton");
const testImagePanel = document.querySelector("#testImagePanel");
const testImageSelect = document.querySelector("#testImageSelect");
const testImageMeta = document.querySelector("#testImageMeta");
const runTestImagesButton = document.querySelector("#runTestImagesButton");
const dropZone = document.querySelector("#dropZone");
const resultImage = document.querySelector("#resultImage");
const emptyState = document.querySelector(".empty-state");
const loadingOverlay = document.querySelector("#loadingOverlay");
const batchTabs = document.querySelector("#batchTabs");
const bookCount = document.querySelector("#bookCount");
const misplacedCount = document.querySelector("#misplacedCount");
const analysisStatus = document.querySelector("#analysisStatus");
const helperText = document.querySelector("#helperText");
const spineList = document.querySelector("#spineList");
const textDetectionPanel = document.querySelector("#textDetectionPanel");
const textDetectionHelper = document.querySelector("#textDetectionHelper");
const textDetectionList = document.querySelector("#textDetectionList");
const debugPanel = document.querySelector("#debugPanel");
const debugTabs = document.querySelector("#debugTabs");
const debugImage = document.querySelector("#debugImage");
const debugMeta = document.querySelector("#debugMeta");
const debugDetails = document.querySelector("#debugDetails");
const filterControls = document.querySelector("#filterControls");
const rerunButton = document.querySelector("#rerunButton");
const filterInputs = Array.from(document.querySelectorAll("[data-filter-param]"));
const analysisControls = [
  imageInput,
  debugInput,
  testImageChooserButton,
  testImageSelect,
  runTestImagesButton,
  rerunButton,
  ...filterInputs,
];
const visualAnalysisControls = [
  uploadButton,
  debugToggle,
  testImageChooserButton,
  runTestImagesButton,
  rerunButton,
];
const MAX_BATCH_IMAGES = 5;

let isAnalyzing = false;
let lastFile = null;
let lastFiles = [];
let lastRunWasTestImage = false;
let lastTestImageNames = [];
let lastPayload = null;
let testImagesLoaded = false;
let selectedDebugStageIndex = 0;

debugInput.addEventListener("change", () => {
  const isDebug = debugInput.checked;
  debugNotice.hidden = !isDebug;
  testImageChooserButton.hidden = !isDebug;
  testImagePanel.hidden = true;
});

imageInput.addEventListener("change", () => {
  if (isAnalyzing) return;
  const files = Array.from(imageInput.files);
  if (files.length === 1) {
    analyze(files[0]);
  } else if (files.length > 1) {
    analyzeBatch(files);
  }
});

rerunButton.addEventListener("click", () => {
  if (isAnalyzing) return;
  if (lastRunWasTestImage && lastTestImageNames.length > 0) {
    analyzeTestImages(lastTestImageNames);
  } else if (lastFiles.length > 1) {
    analyzeBatch(lastFiles);
  } else if (lastFile) {
    analyze(lastFile);
  }
});

testImageChooserButton.addEventListener("click", () => {
  if (isAnalyzing) return;
  testImagePanel.hidden = !testImagePanel.hidden;
  if (!testImagePanel.hidden) {
    loadTestImages();
  }
});

runTestImagesButton.addEventListener("click", () => {
  if (isAnalyzing) return;
  const selectedNames = getSelectedTestImageNames();
  if (selectedNames.length === 0) {
    setError("Select at least one test image.");
    return;
  }
  analyzeTestImages(selectedNames);
});

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragging");
});

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  if (isAnalyzing) return;
  dropZone.classList.remove("dragging");
  const files = Array.from(event.dataTransfer.files).filter((file) => file.type.startsWith("image/"));
  if (files.length === 1) {
    analyze(files[0]);
  } else if (files.length > 1) {
    analyzeBatch(files);
  }
});

async function analyze(file) {
  if (isAnalyzing) return;
  lastFile = file;
  lastFiles = [file];
  lastRunWasTestImage = false;
  lastTestImageNames = [];
  setLoading();

  try {
    const payload = await postAnalysis("/api/analyze", file);
    finishAnalysis(payload);
  } catch (error) {
    setError(`Analysis failed: ${error.message}`);
  }
}

async function analyzeBatch(files) {
  if (isAnalyzing) return;
  if (files.length > MAX_BATCH_IMAGES) {
    setError(`Please upload up to ${MAX_BATCH_IMAGES} images.`);
    return;
  }

  lastFile = files[0] || null;
  lastFiles = files;
  lastRunWasTestImage = false;
  lastTestImageNames = [];
  setLoading();

  try {
    const payload = await postBatchAnalysis(files);
    finishAnalysis(payload);
  } catch (error) {
    setError(`Batch analysis failed: ${error.message}`);
  }
}

async function analyzeTestImages(imageNames) {
  if (isAnalyzing) return;
  lastFile = null;
  lastFiles = [];
  lastRunWasTestImage = true;
  lastTestImageNames = imageNames;
  setLoading();

  try {
    const payload = await postTestImageAnalysis(imageNames);
    finishAnalysis(payload);
  } catch (error) {
    setError(`Test image analysis failed: ${error.message}`);
  }
}

async function loadTestImages() {
  if (testImagesLoaded) return;
  testImageMeta.textContent = "Loading test images...";

  try {
    const response = await fetch("/api/test-images");
    const payload = await response.json();
    if (!response.ok) {
      testImageMeta.textContent = payload.detail || "Failed to load test images.";
      return;
    }

    testImagesLoaded = true;
    testImageSelect.innerHTML = payload.images
      .map((image, index) => `<option value="${escapeHtml(image.name)}" ${index === 0 ? "selected" : ""}>${escapeHtml(image.name)}</option>`)
      .join("");
    testImageMeta.textContent =
      payload.count === 0
        ? "No test images found in the test_images folder."
        : `Found ${payload.count} test image${payload.count === 1 ? "" : "s"}. Select up to ${payload.maxImages}.`;
  } catch (error) {
    testImageMeta.textContent = `Failed to load test images: ${error.message}`;
  }
}

async function postBatchAnalysis(files) {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append("images", file);
  });
  formData.append("debug", debugInput.checked ? "true" : "false");
  appendFilterParams(formData);

  const response = await fetch("/api/analyze-batch", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    setError(payload.detail || payload.error || "Batch analysis failed.");
    return null;
  }

  return payload;
}

async function postTestImageAnalysis(imageNames) {
  const formData = new FormData();
  imageNames.forEach((imageName) => {
    formData.append("image_names", imageName);
  });
  formData.append("debug", debugInput.checked ? "true" : "false");
  appendFilterParams(formData);

  const response = await fetch("/api/analyze-test-images", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    setError(payload.detail || payload.error || "Test image analysis failed.");
    return null;
  }

  return payload;
}

async function postAnalysis(url, file = null) {
  const formData = new FormData();
  if (file) {
    formData.append("image", file);
  }
  formData.append("debug", debugInput.checked ? "true" : "false");
  appendFilterParams(formData);

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    setError(payload.detail || payload.error || "Analysis failed.");
    return null;
  }

  return payload;
}

function renderPayload(payload) {
  lastPayload = payload;
  if (payload.results) {
    renderBatchResult(payload);
  } else {
    batchTabs.hidden = true;
    batchTabs.innerHTML = "";
    renderResult(payload);
  }
}

function setLoading() {
  isAnalyzing = true;
  setAnalysisControlsDisabled(true);
  helperText.textContent = debugInput.checked
    ? "Analyzing with debug images..."
    : "Analyzing...";
  analysisStatus.textContent = "Running";
  dropZone.classList.add("is-loading");
  loadingOverlay.hidden = false;
  spineList.innerHTML = "";
  textDetectionPanel.hidden = true;
  textDetectionList.innerHTML = "";
  batchTabs.hidden = true;
  batchTabs.innerHTML = "";
  debugPanel.hidden = true;
  debugTabs.innerHTML = "";
  debugDetails.hidden = true;
  debugDetails.innerHTML = "";
  filterControls.hidden = true;
  debugImage.removeAttribute("src");
}

function setError(message) {
  helperText.textContent = message;
  analysisStatus.textContent = "Error";
  clearLoadingState();
}

function finishAnalysis(payload) {
  clearLoadingState();
  if (!payload) return;
  renderPayload(payload);
}

function clearLoadingState() {
  isAnalyzing = false;
  setAnalysisControlsDisabled(false);
  dropZone.classList.remove("is-loading");
  loadingOverlay.hidden = true;
}

function setAnalysisControlsDisabled(disabled) {
  analysisControls.forEach((control) => {
    control.disabled = disabled;
  });
  visualAnalysisControls.forEach((control) => {
    control.classList.toggle("is-disabled", disabled);
    control.setAttribute("aria-disabled", disabled ? "true" : "false");
  });
  document.body.classList.toggle("analysis-running", disabled);
}

function renderBatchResult(payload) {
  batchTabs.hidden = false;
  batchTabs.innerHTML = payload.results
    .map((item, index) => {
      const label = item.filename || `Image ${index + 1}`;
      const statusClass = item.status === "completed" ? "completed" : "failed";
      return `<button class="batch-tab ${index === 0 ? "active" : ""} ${statusClass}" type="button" data-index="${index}">
        ${index + 1}. ${label}
      </button>`;
    })
    .join("");

  batchTabs.querySelectorAll(".batch-tab").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.index);
      showBatchItem(payload, index);
      batchTabs.querySelectorAll(".batch-tab").forEach((tab) => tab.classList.remove("active"));
      button.classList.add("active");
    });
  });

  const firstCompletedIndex = payload.results.findIndex((item) => item.status === "completed");
  showBatchItem(payload, firstCompletedIndex >= 0 ? firstCompletedIndex : 0);
}

function showBatchItem(payload, index) {
  const item = payload.results[index];
  if (!item || item.status !== "completed") {
    renderFailedBatchItem(item, payload.summary);
    return;
  }

  renderResult(item.result);
  helperText.textContent = `${helperText.textContent} Batch ${payload.summary.completedCount}/${payload.summary.imageCount} completed.`;
}

function renderFailedBatchItem(item, summary) {
  emptyState.hidden = true;
  resultImage.hidden = true;
  resultImage.removeAttribute("src");
  bookCount.textContent = "0";
  misplacedCount.textContent = "0";
  analysisStatus.textContent = "Failed";
  helperText.textContent = `${item?.filename || "Image"} failed: ${item?.error || "Unknown error"}. Batch ${summary.completedCount}/${summary.imageCount} completed.`;
  spineList.innerHTML = "";
  textDetectionPanel.hidden = true;
  textDetectionList.innerHTML = "";
  debugPanel.hidden = true;
}

function renderResult(payload) {
  emptyState.hidden = true;
  resultImage.hidden = false;
  resultImage.src = payload.annotatedImage;

  bookCount.textContent = payload.summary.bookCount;
  misplacedCount.textContent = payload.summary.misplacedCount;
  analysisStatus.textContent = payload.summary.status === "ok" ? "OK" : "Needs review";
  helperText.textContent =
    payload.summary.status === "ok"
      ? "The current demo rule sees this order as correct."
      : "Red items are not in the expected order.";
  if (payload.summary.sourceImage) {
    helperText.textContent = `${helperText.textContent} Source: ${payload.summary.sourceImage}.`;
  }

  spineList.innerHTML = payload.spines.map(renderSpine).join("");
  renderDebug(payload.debug);
}

function renderSpine(spine) {
  const isMisplaced = spine.status !== "ok";
  const statusText = isMisplaced ? "Needs review" : "OK";
  return `
    <article class="spine-item ${isMisplaced ? "misplaced" : ""}">
      <span class="pill ${isMisplaced ? "warn" : ""}">${statusText}</span>
      <span class="label">${spine.call_number}</span>
      <span class="detail">Expected position ${spine.expected_rank}</span>
    </article>
  `;
}

function renderDebug(debug) {
  if (!debug || !debug.stages || debug.stages.length === 0) {
    debugPanel.hidden = true;
    textDetectionPanel.hidden = true;
    textDetectionList.innerHTML = "";
    return;
  }

  debugPanel.hidden = false;
  debugMeta.textContent = `Debug payload received: ${debug.boundaryCount} boundaries, ${debug.boxCount} boxes${
    debug.filteredOutCount ? `, ${debug.filteredOutCount} filtered out` : ""
  }${
    debug.usedFallback ? ", fallback used" : ""
  }.`;

  debugTabs.innerHTML = debug.stages
    .map(
      (stage, index) =>
        `<button class="debug-tab" type="button" data-index="${index}" aria-pressed="false">
          ${escapeHtml(stage.number || String(index + 1))}. ${escapeHtml(stage.label)}
        </button>`,
    )
    .join("");

  const stageIndex = Math.min(selectedDebugStageIndex, debug.stages.length - 1);
  if (debug.filterOptions) {
    syncFilterInputs(debug.filterOptions);
  }
  setDebugStage(debug, stageIndex);
  debugTabs.querySelectorAll(".debug-tab").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.index);
      setDebugStage(debug, index);
    });
  });

  const textRows = Array.isArray(debug.textDetectionRows) ? debug.textDetectionRows : [];
  textDetectionPanel.hidden = false;
  textDetectionHelper.textContent = debug.textDetection?.error
    ? `TEXT_DETECTION failed: ${debug.textDetection.error}`
    : `${textRows.length} contact-sheet row${textRows.length === 1 ? "" : "s"} processed without word confidence.`;
  const emptyTextDetectionCard = debug.textDetection?.error
    ? `<article class="spine-item text-detection-item misplaced"><span class="pill warn">Failed</span><span class="label">TEXT_DETECTION result unavailable</span><span class="detail">${escapeHtml(debug.textDetection.error)}</span></article>`
    : '<article class="spine-item text-detection-item misplaced"><span class="pill warn">Empty</span><span class="label">No TEXT_DETECTION rows</span><span class="detail">The contact sheet contained no OCR regions.</span></article>';
  textDetectionList.innerHTML = textRows.length
    ? textRows
        .map((row) => {
          const variants = Array.isArray(row.variantResults) ? row.variantResults : [];
          const selected = variants.find((variant) => variant.orientation === row.selectedOrientation);
          const stateText = row.eligible ? "Selected" : "Rejected";
          const callNumber = row.text || "No accepted call number";
          const orientation = selected ? selected.orientation : "No orientation selected";
          const rawVariants = variants
            .map((variant) => `${variant.orientation}: ${variant.text || "—"}`)
            .join(" · ");
          return `
            <article class="spine-item text-detection-item ${row.eligible ? "" : "misplaced"}">
              <span class="pill ${row.eligible ? "" : "warn"}">${stateText}</span>
              <span class="label">${escapeHtml(callNumber)}</span>
              <span class="detail">Row ${escapeHtml(String(row.index))} · ${escapeHtml(orientation)}</span>
              <span class="detail text-detection-raw">${escapeHtml(rawVariants || "No text annotations mapped")}</span>
            </article>
          `;
        })
        .join("")
    : emptyTextDetectionCard;
}

function setDebugStage(debug, index) {
  const stage = debug.stages[index];
  selectedDebugStageIndex = index;
  debugImage.src = stage.image;
  debugImage.classList.toggle(
    "wide-debug-image",
    stage.label === "OCR bounding boxes" || stage.label === "TEXT_DETECTION OCR",
  );
  filterControls.hidden = stage.label !== "Size filter";
  renderDebugDetails(stage.details || []);
  debugTabs.querySelectorAll(".debug-tab").forEach((tab) => {
    const isActive = Number(tab.dataset.index) === index;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function renderDebugDetails(details) {
  if (!details.length) {
    debugDetails.hidden = true;
    debugDetails.innerHTML = "";
    return;
  }
  debugDetails.hidden = false;
  debugDetails.innerHTML = details
    .map((detail) => `<div>${escapeHtml(detail)}</div>`)
    .join("");
}

function appendFilterParams(formData) {
  filterInputs.forEach((input) => {
    formData.append(input.dataset.filterParam, input.value);
  });
}

function syncFilterInputs(options) {
  const mapping = {
    max_box_width_ratio: "maxBoxWidthRatio",
    max_box_area_ratio: "maxBoxAreaRatio",
  };
  filterInputs.forEach((input) => {
    const key = mapping[input.dataset.filterParam];
    if (key && Object.prototype.hasOwnProperty.call(options, key)) {
      input.value = options[key];
    }
  });
}

function getSelectedTestImageNames() {
  return Array.from(testImageSelect.selectedOptions).map((option) => option.value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}
