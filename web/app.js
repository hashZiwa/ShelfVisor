const imageInput = document.querySelector("#imageInput");
const debugInput = document.querySelector("#debugInput");
const debugNotice = document.querySelector("#debugNotice");
const dropZone = document.querySelector("#dropZone");
const resultImage = document.querySelector("#resultImage");
const emptyState = document.querySelector(".empty-state");
const bookCount = document.querySelector("#bookCount");
const misplacedCount = document.querySelector("#misplacedCount");
const analysisStatus = document.querySelector("#analysisStatus");
const helperText = document.querySelector("#helperText");
const spineList = document.querySelector("#spineList");
const debugPanel = document.querySelector("#debugPanel");
const debugTabs = document.querySelector("#debugTabs");
const debugImage = document.querySelector("#debugImage");
const debugMeta = document.querySelector("#debugMeta");

debugInput.addEventListener("change", () => {
  debugNotice.hidden = !debugInput.checked;
});

imageInput.addEventListener("change", () => {
  const [file] = imageInput.files;
  if (file) analyze(file);
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
  dropZone.classList.remove("dragging");
  const [file] = event.dataTransfer.files;
  if (file) analyze(file);
});

async function analyze(file) {
  setLoading();

  const formData = new FormData();
  formData.append("image", file);
  formData.append("debug", debugInput.checked ? "true" : "false");

  const response = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    setError(payload.detail || payload.error || "Analysis failed.");
    return;
  }

  renderResult(payload);
}

function setLoading() {
  helperText.textContent = debugInput.checked
    ? "Analyzing with debug images..."
    : "Analyzing...";
  analysisStatus.textContent = "Running";
  spineList.innerHTML = "";
  debugPanel.hidden = true;
  debugTabs.innerHTML = "";
  debugImage.removeAttribute("src");
}

function setError(message) {
  helperText.textContent = message;
  analysisStatus.textContent = "Error";
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

  spineList.innerHTML = payload.spines.map(renderSpine).join("");
  renderDebug(payload.debug);
}

function renderSpine(spine) {
  const isMisplaced = spine.status !== "ok";
  const statusText = isMisplaced ? "Needs review" : "OK";
  return `
    <article class="spine-item ${isMisplaced ? "misplaced" : ""}">
      <span class="pill ${isMisplaced ? "warn" : ""}">${statusText}</span>
      <span class="label">${spine.index}. ${spine.call_number}</span>
      <span class="detail">Expected position ${spine.expected_rank}</span>
    </article>
  `;
}

function renderDebug(debug) {
  if (!debug || !debug.stages || debug.stages.length === 0) {
    debugPanel.hidden = true;
    return;
  }

  debugPanel.hidden = false;
  debugMeta.textContent = `Debug payload received: ${debug.boundaryCount} boundaries, ${debug.boxCount} boxes${
    debug.usedFallback ? ", fallback used" : ""
  }.`;

  debugTabs.innerHTML = debug.stages
    .map(
      (stage, index) =>
        `<button class="debug-tab ${index === 0 ? "active" : ""}" type="button" data-index="${index}">
          ${index + 1}. ${stage.label}
        </button>`,
    )
    .join("");

  debugImage.src = debug.stages[0].image;
  debugTabs.querySelectorAll(".debug-tab").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.index);
      debugImage.src = debug.stages[index].image;
      debugTabs.querySelectorAll(".debug-tab").forEach((tab) => tab.classList.remove("active"));
      button.classList.add("active");
    });
  });
}
