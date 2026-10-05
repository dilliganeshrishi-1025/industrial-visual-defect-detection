/* ==========================================================================
   Smart Triage — Industrial Visual Defect Detection Dashboard Logic
   Connects to FastAPI REST Backend (http://localhost:8000)
   ========================================================================== */

const API_BASE_URL = "http://localhost:8000";

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  checkBackendHealth();
  initInspectionStudio();
  initBatchInspection();
  initBenchmarkModule();
  initDatasetsExplorer();

  // Hide loading screen
  setTimeout(() => {
    document.getElementById("loading-overlay").classList.add("hidden");
  }, 500);
});

/* -------------------------------------------------------------------------- */
/* 1. Navigation & Page Switching                                             */
/* -------------------------------------------------------------------------- */
function navigateToPage(pageId) {
  const navLinks = document.querySelectorAll(".top-nav-item");
  const pages = document.querySelectorAll(".page");

  navLinks.forEach((l) => l.classList.remove("active"));
  pages.forEach((p) => p.classList.remove("active"));

  const activeLink = document.querySelector(`.top-nav-item[data-page="${pageId}"]`);
  if (activeLink) activeLink.classList.add("active");

  const targetPage = document.getElementById(`page-${pageId}`);
  if (targetPage) {
    targetPage.classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}
window.navigateToPage = navigateToPage;

function initNavigation() {
  const navLinks = document.querySelectorAll(".top-nav-item");

  navLinks.forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const pageId = link.getAttribute("data-page");
      navigateToPage(pageId);
    });
  });

  // Hero Quick Access Buttons
  const heroGlass = document.getElementById("btn-hero-glass");
  const heroDatasets = document.getElementById("btn-hero-datasets");

  if (heroGlass) heroGlass.addEventListener("click", () => navigateToPage("single"));
  if (heroDatasets) heroDatasets.addEventListener("click", () => navigateToPage("datasets"));
}

/* -------------------------------------------------------------------------- */
/* 2. Backend Health Check                                                    */
/* -------------------------------------------------------------------------- */
async function checkBackendHealth() {
  const statusEl = document.getElementById("api-status-text");
  const urlEl = document.getElementById("api-url-text");
  const coresetEl = document.getElementById("coreset-count-text");
  const threshEl = document.getElementById("threshold-val-text");

  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      statusEl.textContent = "🟢 ONLINE";
      statusEl.className = "kpi-value green";
      urlEl.textContent = `FastAPI REST API (${API_BASE_URL})`;
      coresetEl.textContent = data.coreset_size ? data.coreset_size.toLocaleString() : "7,840";
      threshEl.textContent = data.threshold ? data.threshold.toFixed(4) : "0.4263";
    } else {
      throw new Error("HTTP error");
    }
  } catch (err) {
    statusEl.textContent = "🔴 OFFLINE";
    statusEl.className = "kpi-value scrap";
    urlEl.textContent = "Please start backend_api.py on port 8000";
  }
}

/* -------------------------------------------------------------------------- */
/* 3. Single Bottle Inspection Studio                                         */
/* -------------------------------------------------------------------------- */
function initInspectionStudio() {
  const btnSample = document.getElementById("btn-input-sample");
  const btnUpload = document.getElementById("btn-input-upload");
  const formSample = document.getElementById("form-sample-select");
  const formUpload = document.getElementById("form-file-upload");
  const catSelect = document.getElementById("select-category");
  const fileSelect = document.getElementById("select-filename");

  // Toggle mode buttons
  btnSample.addEventListener("click", () => {
    btnSample.classList.add("active");
    btnUpload.classList.remove("active");
    formSample.classList.remove("hidden");
    formUpload.classList.add("hidden");
  });

  btnUpload.addEventListener("click", () => {
    btnUpload.classList.add("active");
    btnSample.classList.remove("active");
    formUpload.classList.remove("hidden");
    formSample.classList.add("hidden");
  });

  // Load sample categories
  loadCategories();

  catSelect.addEventListener("change", () => {
    updateSampleFilenames(catSelect.value);
  });

  // Inspect sample button click
  document.getElementById("btn-run-sample-inspect").addEventListener("click", async () => {
    const cat = catSelect.value;
    const fname = fileSelect.value;
    if (!cat || !fname) return;

    try {
      showLoadingButton("btn-run-sample-inspect", true);
      const res = await fetch(`${API_BASE_URL}/predict_sample?category=${cat}&filename=${fname}`);
      if (res.ok) {
        const data = await res.json();
        renderInspectionResults(data);
      } else {
        alert("Backend API prediction failed.");
      }
    } catch (e) {
      alert("Could not reach backend API at " + API_BASE_URL);
    } finally {
      showLoadingButton("btn-run-sample-inspect", false, "🔎 Run Inspection via API");
    }
  });

  // Custom file upload inspection click
  document.getElementById("btn-run-upload-inspect").addEventListener("click", async () => {
    const fileInput = document.getElementById("file-input-custom");
    if (!fileInput.files || fileInput.files.length === 0) {
      alert("Please select a bottle image file first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      showLoadingButton("btn-run-upload-inspect", true);
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        renderInspectionResults(data);
      } else {
        alert("File inspection failed.");
      }
    } catch (e) {
      alert("Error reaching API server: " + e.message);
    } finally {
      showLoadingButton("btn-run-upload-inspect", false, "🔎 Run Inspection via API");
    }
  });
}

let categoriesMap = {};

async function loadCategories() {
  try {
    const res = await fetch(`${API_BASE_URL}/categories`);
    if (res.ok) {
      const data = await res.json();
      categoriesMap = data.categories || {};
      updateSampleFilenames(document.getElementById("select-category").value);
    }
  } catch (e) {
    console.warn("Could not load categories from backend:", e);
  }
}

function updateSampleFilenames(category) {
  const fileSelect = document.getElementById("select-filename");
  fileSelect.innerHTML = "";
  const list = categoriesMap[category] || [];

  if (list.length === 0) {
    // fallback dummy options
    for (let i = 0; i < 5; i++) {
      const opt = document.createElement("option");
      opt.value = `${i.toString().padStart(3, '0')}.png`;
      opt.textContent = `${i.toString().padStart(3, '0')}.png`;
      fileSelect.appendChild(opt);
    }
  } else {
    list.forEach(fname => {
      const opt = document.createElement("option");
      opt.value = fname;
      opt.textContent = fname;
      fileSelect.appendChild(opt);
    });
  }
}

function renderInspectionResults(data) {
  const container = document.getElementById("inspection-results-container");
  container.classList.remove("hidden");

  document.getElementById("res-img-orig").src = `data:image/png;base64,${data.original_base64}`;
  document.getElementById("res-img-heatmap").src = `data:image/png;base64,${data.heatmap_base64}`;
  document.getElementById("res-img-polar").src = `data:image/png;base64,${data.polar_base64}`;

  document.getElementById("res-val-score").textContent = data.anomaly_score.toFixed(4);
  document.getElementById("res-val-dsar").textContent = `${data.dsar_percent.toFixed(2)}%`;
  document.getElementById("res-val-zone").textContent = data.defect_zone || "Rim Annulus";
  document.getElementById("res-val-verdict").textContent = data.verdict;

  const badgeEl = document.getElementById("result-grade-badge");
  const gradeStr = data.grade || "Grade A — Good";
  badgeEl.textContent = gradeStr;

  if (gradeStr.includes("Grade A")) {
    badgeEl.className = "badge-grade grade-a";
  } else if (gradeStr.includes("Grade B")) {
    badgeEl.className = "badge-grade grade-b";
  } else {
    badgeEl.className = "badge-grade grade-c";
  }

  const reasonEl = document.getElementById("res-val-reason");
  if (reasonEl) {
    reasonEl.textContent = data.grade_reason || "Bottle meets all quality compliance standards.";
  }

  document.getElementById("res-val-action").textContent = data.disposal_action || "Approved for Packaging";

  // Real PDF download handler
  document.getElementById("btn-download-pdf").onclick = async () => {
    const btn = document.getElementById("btn-download-pdf");
    const origText = btn.textContent;
    try {
      btn.textContent = "⏳ Generating QA PDF...";
      btn.disabled = true;

      const res = await fetch(`${API_BASE_URL}/generate_pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      if (res.ok) {
        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const cleanName = (data.filename || 'bottle').replace(/[^a-zA-Z0-9_-]/g, '_');
        a.download = `QA_Certificate_${cleanName}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } else {
        alert("Failed to generate QA PDF Certificate from server.");
      }
    } catch (e) {
      alert("Error generating QA PDF: " + e.message);
    } finally {
      btn.textContent = origText;
      btn.disabled = false;
    }
  };

  // Scroll to results
  container.scrollIntoView({ behavior: 'smooth' });
}

/* -------------------------------------------------------------------------- */
/* 4. Benchmark Results & Analytics                                           */
/* -------------------------------------------------------------------------- */
function initBenchmarkModule() {
  const btnRun = document.getElementById("btn-run-full-benchmark");
  const btnExport = document.getElementById("btn-export-csv");

  btnRun.addEventListener("click", async () => {
    try {
      showLoadingButton("btn-run-full-benchmark", true);
      const res = await fetch(`${API_BASE_URL}/benchmark`);
      if (res.ok) {
        const data = await res.json();
        renderBenchmarkAnalytics(data);
        btnExport.disabled = false;
      } else {
        alert("Benchmark request failed.");
      }
    } catch (e) {
      alert("Error reaching API server: " + e.message);
    } finally {
      showLoadingButton("btn-run-full-benchmark", false, "🚀 Trigger Full Backend Benchmark");
    }
  });

  btnExport.addEventListener("click", () => {
    window.open(`${API_BASE_URL}/download_csv`, '_blank');
  });
}

function renderBenchmarkAnalytics(data) {
  document.getElementById("benchmark-charts-container").classList.remove("hidden");
  document.getElementById("benchmark-count-badge").textContent = `Evaluated ${data.total_test_images} Images | Overall Accuracy: ${data.overall_accuracy}%`;

  // Render Plotly Boxplot
  const categories = ["good", "broken_large", "broken_small", "contamination"];
  const boxTraces = categories.map(cat => {
    const scores = data.detailed_results
      .filter(r => r.Category === cat)
      .map(r => r.Anomaly_Score);

    return {
      y: scores,
      name: cat,
      type: 'box',
      boxpoints: 'all',
      jitter: 0.3,
      pointpos: -1.8
    };
  });

  const boxLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8' },
    margin: { t: 20, b: 40, l: 40, r: 20 },
    yaxis: { title: 'Anomaly Score' },
    shapes: [{
      type: 'line',
      x0: -0.5,
      x1: 3.5,
      y0: 0.4263,
      y1: 0.4263,
      line: { color: '#ef4444', width: 2, dash: 'dash' }
    }]
  };

  Plotly.newPlot('chart-boxplot', boxTraces, boxLayout);

  // Render Accuracy Bar Chart
  const summary = data.category_summary || [];
  const barTrace = {
    x: summary.map(s => s.Category),
    y: summary.map(s => s.Accuracy),
    type: 'bar',
    marker: { color: ['#10b981', '#ef4444', '#f59e0b', '#a855f7'] }
  };

  const barLayout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#94a3b8' },
    margin: { t: 20, b: 40, l: 40, r: 20 },
    yaxis: { title: 'Accuracy (%)', range: [0, 105] }
  };

  Plotly.newPlot('chart-bar', [barTrace], barLayout);

  // Render Table Rows
  const tbody = document.getElementById("benchmark-table-body");
  tbody.innerHTML = "";

  data.detailed_results.forEach(row => {
    const tr = document.createElement("tr");
    const isGood = row.Predicted_Verdict === "GOOD";
    const tagClass = isGood ? "green" : "red";

    tr.innerHTML = `
      <td><b>${row.Filename}</b></td>
      <td>${row.Category}</td>
      <td>${row.True_Label}</td>
      <td>${row.Anomaly_Score.toFixed(4)}</td>
      <td><span class="tag ${tagClass}">${row.Predicted_Verdict}</span></td>
      <td>${row.Correct ? '✅ Match' : '❌ Mismatch'}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* -------------------------------------------------------------------------- */
/* 6. Batch & Multi-Image Inspection Module                                   */
/* -------------------------------------------------------------------------- */
function initBatchInspection() {
  const fileInput = document.getElementById("batch-file-input");
  const countBadge = document.getElementById("batch-file-count-badge");
  const btnRun = document.getElementById("btn-run-batch-inspect");
  const btnCsv = document.getElementById("btn-download-batch-csv");
  const statusSpan = document.getElementById("batch-process-status");

  if (!fileInput) return;

  fileInput.addEventListener("change", () => {
    const count = fileInput.files ? fileInput.files.length : 0;
    countBadge.textContent = `${count} files selected`;
    btnRun.disabled = count === 0;
    statusSpan.textContent = count > 0 ? `Ready to inspect ${count} image(s).` : "";
  });

  btnRun.addEventListener("click", async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
      alert("Please select images to analyze.");
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < fileInput.files.length; i++) {
      formData.append("files", fileInput.files[i]);
    }

    try {
      showLoadingButton("btn-run-batch-inspect", true);
      statusSpan.textContent = `Analyzing ${fileInput.files.length} images via PatchCore REST API...`;

      const res = await fetch(`${API_BASE_URL}/batch_upload_predict`, {
        method: "POST",
        body: formData
      });

      if (res.ok) {
        const data = await res.json();
        renderBatchResults(data);
        btnCsv.classList.remove("hidden");
        statusSpan.textContent = `✅ Successfully processed ${data.total_processed} images!`;
      } else {
        alert("Batch inspection failed on backend.");
        statusSpan.textContent = "❌ Batch inspection error.";
      }
    } catch (e) {
      alert("Could not connect to FastAPI backend: " + e.message);
      statusSpan.textContent = "❌ Network connection error.";
    } finally {
      showLoadingButton("btn-run-batch-inspect", false, "🚀 Analyze Batch via PatchCore AI");
    }
  });

  btnCsv.addEventListener("click", () => {
    window.open(`${API_BASE_URL}/download_csv`, '_blank');
  });
}

let currentBatchResults = [];

function renderBatchResults(data) {
  const container = document.getElementById("batch-results-container");
  container.classList.remove("hidden");
  currentBatchResults = data.results || [];

  // Summary counts
  const gradeACount = currentBatchResults.filter(r => r.Grade && r.Grade.includes("Grade A")).length;
  const gradeBCount = currentBatchResults.filter(r => r.Grade && r.Grade.includes("Grade B")).length;
  const gradeCCount = currentBatchResults.filter(r => r.Grade && (r.Grade.includes("Grade C") || (r.Verdict === "DEFECTIVE" && !r.Grade.includes("Grade B")))).length;

  document.getElementById("batch-stat-total").textContent = data.total_processed;
  document.getElementById("batch-stat-passed").textContent = gradeACount;
  
  const reworkEl = document.getElementById("batch-stat-rework");
  if (reworkEl) reworkEl.textContent = gradeBCount;

  const scrapEl = document.getElementById("batch-stat-scrap");
  if (scrapEl) scrapEl.textContent = gradeCCount;

  // Render Heatmap Gallery Grid
  const gallery = document.getElementById("batch-gallery-grid");
  gallery.innerHTML = "";

  currentBatchResults.forEach((item, idx) => {
    const card = document.createElement("div");
    card.className = "batch-gallery-card";

    const isGood = item.Verdict === "GOOD" || (item.Grade && item.Grade.includes("Grade A"));
    const isGradeB = item.Grade && item.Grade.includes("Grade B");
    const badgeClass = isGood ? "tag green" : (isGradeB ? "tag amber" : "tag red");

    card.innerHTML = `
      <div class="bg-header">
        <span title="${item.Filename}">${item.Filename.length > 15 ? item.Filename.substring(0, 13) + '...' : item.Filename}</span>
        <span class="${badgeClass}">${item.Verdict}</span>
      </div>
      <img src="data:image/png;base64,${item.Heatmap_Base64}" alt="Heatmap for ${item.Filename}">
      <div class="bg-info">
        <span>DSAR: <b>${item.DSAR_Percent}%</b></span>
        <span>Score: <b>${item.Anomaly_Score.toFixed(4)}</b></span>
      </div>
      <div style="font-size: 0.72rem; color: var(--text-muted); text-align: center;">
        ${item.Category || item.Defect_Type}
      </div>
      <button class="btn btn-secondary btn-sm" onclick="inspectSingleItem(${idx})" style="width: 100%; justify-content: center; font-size: 0.75rem; padding: 6px 10px; margin-top: 4px;">
        🔍 View in Single Inspection
      </button>
    `;
    gallery.appendChild(card);
  });

  // Render Detailed Table with dedicated Single Inspection Column
  const tbody = document.getElementById("batch-table-body");
  tbody.innerHTML = "";

  currentBatchResults.forEach((item, idx) => {
    const tr = document.createElement("tr");
    const isGood = item.Verdict === "GOOD" || (item.Grade && item.Grade.includes("Grade A"));
    const isGradeB = item.Grade && item.Grade.includes("Grade B");
    const verdictClass = isGood ? "tag green" : (isGradeB ? "tag amber" : "tag red");
    const catClass = isGood ? "tag green" : (item.Category && item.Category.includes("Broken Large") ? "tag red" : (item.Category && item.Category.includes("Broken Small") ? "tag amber" : "tag crimson"));

    tr.innerHTML = `
      <td>
        <img src="data:image/png;base64,${item.Heatmap_Base64}" style="width: 44px; height: 44px; border-radius: 6px; object-fit: cover; background: #000; cursor: pointer;" onclick="inspectSingleItem(${idx})" title="Click to view full diagnosis">
      </td>
      <td><b>${item.Filename}</b></td>
      <td><span class="${catClass}">${item.Category || 'Bottle Sample'}</span></td>
      <td>${item.Anomaly_Score.toFixed(4)}</td>
      <td><span class="${verdictClass}">${item.Verdict}</span></td>
      <td>${item.Grade}</td>
      <td><b>${item.DSAR_Percent}%</b></td>
      <td>${item.Defect_Zone}</td>
      <td>${item.Disposal_Action}</td>
      <td style="text-align: center;">
        <button class="btn btn-primary" onclick="inspectSingleItem(${idx})" style="padding: 6px 14px; font-size: 0.8rem; border-radius: 6px;">
          🔍 Inspect
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Scroll to results smoothly
  container.scrollIntoView({ behavior: 'smooth' });
}

// Global handler to transition any batch item into the Single Inspection Studio
window.inspectSingleItem = function(idx) {
  if (!currentBatchResults || !currentBatchResults[idx]) return;
  const item = currentBatchResults[idx];

  // Prepare payload for single inspection renderer
  const singleData = {
    filename: item.Filename,
    anomaly_score: item.Anomaly_Score,
    is_defective: item.Is_Defective,
    verdict: item.Verdict,
    grade: item.Grade,
    dsar_percent: item.DSAR_Percent,
    defect_zone: item.Defect_Zone,
    defect_type: item.Defect_Type,
    disposal_action: item.Disposal_Action,
    threshold: item.Threshold || 0.4263,
    original_base64: item.Original_Base64 || item.Heatmap_Base64,
    heatmap_base64: item.Heatmap_Base64,
    polar_base64: item.Polar_Base64
  };

  // Switch navigation tabs to Single Inspection
  const navLinks = document.querySelectorAll(".top-nav-item");
  const pages = document.querySelectorAll(".page");

  navLinks.forEach((l) => l.classList.remove("active"));
  pages.forEach((p) => p.classList.remove("active"));

  const singleNav = document.querySelector('.top-nav-item[data-page="single"]');
  const singlePage = document.getElementById("page-single");

  if (singleNav) singleNav.classList.add("active");
  if (singlePage) singlePage.classList.add("active");

  // Render the selected item in Single Inspection Studio
  renderInspectionResults(singleData);
};

function showLoadingButton(btnId, isLoading, originalText = "") {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  if (isLoading) {
    btn.disabled = true;
    btn.textContent = "⏳ Processing API Request...";
  } else {
    btn.disabled = false;
    btn.textContent = originalText;
  }
}

/* -------------------------------------------------------------------------- */
/* 7. MVTec AD Glass Bottle Dataset Explorer                                  */
/* -------------------------------------------------------------------------- */
function initDatasetsExplorer() {
  const gridMvtec = document.getElementById("grid-mvtec-gallery");

  // Fetch Dataset Summary from REST API
  fetch(`${API_BASE_URL}/api/datasets/summary`)
    .then(res => res.json())
    .then(data => {
      if (data && data.mvtec) {
        buildMvtecGallery(data.mvtec);
      }
    })
    .catch(err => console.warn("Dataset summary error:", err));

  function buildMvtecGallery(mvtec) {
    if (!gridMvtec || !mvtec || !mvtec.categories) return;
    gridMvtec.innerHTML = "";

    const filterPills = document.querySelectorAll("#filters-mvtec .category-pill");
    filterPills.forEach(pill => {
      pill.addEventListener("click", () => {
        filterPills.forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        const cat = pill.getAttribute("data-cat");
        const cards = gridMvtec.querySelectorAll(".dataset-sample-card");
        cards.forEach(card => {
          if (cat === "all" || card.getAttribute("data-cat") === cat) {
            card.style.display = "flex";
          } else {
            card.style.display = "none";
          }
        });
      });
    });

    const categoryBadges = {
      good: { label: "Good (Normal)", tag: "green" },
      broken_large: { label: "Broken Large", tag: "red" },
      broken_small: { label: "Broken Small", tag: "amber" },
      contamination: { label: "Contamination", tag: "crimson" }
    };

    Object.entries(mvtec.categories).forEach(([cat, files]) => {
      const badgeInfo = categoryBadges[cat] || { label: cat, tag: "cyan" };
      files.slice(0, 15).forEach(fname => {
        const card = document.createElement("div");
        card.className = "dataset-sample-card";
        card.setAttribute("data-cat", cat);

        card.innerHTML = `
          <img src="${API_BASE_URL}/api/dataset/image?dataset=mvtec&category=${encodeURIComponent(cat)}&filename=${encodeURIComponent(fname)}" alt="${fname}" loading="lazy">
          <div class="dataset-card-meta">
            <span class="tag ${badgeInfo.tag}">${badgeInfo.label}</span>
          </div>
          <div class="dataset-card-filename" title="${fname}">${fname}</div>
          <button class="dataset-card-btn">🔎 Inspect in Bottle Studio</button>
        `;

        const inspectBtn = card.querySelector(".dataset-card-btn");
        inspectBtn.addEventListener("click", () => {
          navigateToPage("single");
          const catSelect = document.getElementById("select-category");
          const fileSelect = document.getElementById("select-filename");
          const runBtn = document.getElementById("btn-run-sample-inspect");

          if (catSelect) {
            catSelect.value = cat;
            if (typeof updateSampleFilenames === "function") {
              updateSampleFilenames(cat);
            }
            if (fileSelect) {
              fileSelect.value = fname;
            }
            if (runBtn) {
              setTimeout(() => runBtn.click(), 100);
            }
          }
        });

        gridMvtec.appendChild(card);
      });
    });
  }
}


