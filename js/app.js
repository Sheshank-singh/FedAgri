/**
 * FedAgri Main Application Controller
 */

document.addEventListener("DOMContentLoaded", async () => {
  console.log("FedAgri initializing...");

  // State
  const state = {
    selectedModel: "mobilenetv3",
    modelsData: [],
    classesData: [],
    samplesData: [],
    experimentsData: [],
    checkpointsData: [],
    activeJob: null,
    selectedSampleFile: null,
    uploadedFile: null,
    currentTab: "studio"
  };

  // 1. Navigation Tab Switching
  const navTabs = document.querySelectorAll(".nav-tab-btn");
  const viewSections = document.querySelectorAll(".view-section");

  function switchTab(tabName) {
    state.currentTab = tabName;
    navTabs.forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });
    viewSections.forEach(section => {
      section.classList.toggle("active", section.id === `view-${tabName}`);
    });

    if (tabName === "monitor") {
      setTimeout(() => {
        if (!FedCharts.convergenceChart) FedCharts.initConvergenceChart();
        if (!FedCharts.robustnessChart) FedCharts.initRobustnessChart();
      }, 50);
    } else if (tabName === "results") {
      loadResultsView();
    } else if (tabName === "diagnostics") {
      loadDiagnosticsView();
    } else if (tabName === "dataset") {
      loadDatasetView();
    }
  }

  navTabs.forEach(btn => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // 2. Hardware Health & WebSocket Connection
  async function checkHealth() {
    try {
      const health = await API.getHealth();
      const dot = document.getElementById("systemStatusDot");
      const label = document.getElementById("systemStatusText");
      if (dot && label) {
        dot.classList.remove("offline");
        label.textContent = health.cuda_available 
          ? `GPU Active (${health.device_name})`
          : `CPU Mode (${health.platform.toUpperCase()})`;
      }
    } catch (e) {
      console.warn("Health check error:", e);
    }
  }
  checkHealth();

  fedWs.onStatusChange(connected => {
    const dot = document.getElementById("wsStatusDot");
    const label = document.getElementById("wsStatusText");
    if (dot && label) {
      dot.classList.toggle("offline", !connected);
      label.textContent = connected ? "Live Telemetry" : "Connecting...";
    }
  });
  fedWs.connect();

  // 3. Studio View Logic
  async function loadStudioView() {
    try {
      const models = await API.getModels();
      state.modelsData = models;
      const grid = document.getElementById("modelGrid");
      if (grid) {
        grid.innerHTML = "";
        models.forEach(m => {
          const card = document.createElement("div");
          card.className = `model-card ${m.Model === state.selectedModel ? "selected" : ""}`;
          card.dataset.model = m.Model;

          let badgeClass = "badge-edge";
          let badgeText = "Edge Optimized";
          if (m.Model === "resnet50") { badgeClass = "badge-deep"; badgeText = "Deep Residual"; }
          else if (m.Model === "convnext_tiny") { badgeClass = "badge-modern"; badgeText = "Modern Conv"; }
          else if (m.Model === "efficientnet_b0") { badgeClass = "badge-edge"; badgeText = "Compound Scaled"; }

          card.innerHTML = `
            <div class="model-card-header">
              <span class="model-card-title">${m.Model.toUpperCase()}</span>
              <span class="model-card-badge ${badgeClass}">${badgeText}</span>
            </div>
            <p style="font-size:0.8rem; color:#94a3b8; line-height:1.4;">${m.Description || ""}</p>
            <div class="model-specs-list">
              <span>Params: <b>${m["Parameters (M)"] || (m.Total_Params/1e6).toFixed(2)}M</b></span>
              <span>Size: <b>${m.Size_MB} MB</b></span>
              <span>FLOPs: <b>${m.GFLOPs || m.GFLOPs === 0 ? m.GFLOPs : 'N/A'} G</b></span>
              <span>Latency: <b>${m.Inference_Latency_ms ? m.Inference_Latency_ms + 'ms' : '~4ms'}</b></span>
            </div>
          `;

          card.addEventListener("click", () => {
            document.querySelectorAll(".model-card").forEach(c => c.classList.remove("selected"));
            card.classList.add("selected");
            state.selectedModel = m.Model;
          });

          grid.appendChild(card);
        });
      }
    } catch (e) {
      console.error("Error loading models:", e);
    }
  }

  // Sliders binding
  const setupSlider = (sliderId, badgeId, suffix = "") => {
    const slider = document.getElementById(sliderId);
    const badge = document.getElementById(badgeId);
    if (slider && badge) {
      slider.addEventListener("input", (e) => {
        badge.textContent = e.target.value + suffix;
      });
    }
  };
  setupSlider("sliderClients", "badgeClients");
  setupSlider("sliderRounds", "badgeRounds");
  setupSlider("sliderAlpha", "badgeAlpha");
  setupSlider("sliderMu", "badgeMu");

  // Algorithm select toggle mu slider
  const algoSelect = document.getElementById("selectAlgo");
  const muGroup = document.getElementById("muFormGroup");
  if (algoSelect && muGroup) {
    algoSelect.addEventListener("change", (e) => {
      muGroup.style.display = e.target.value === "fedprox" ? "block" : "none";
    });
  }

  // Quick mode toggle
  const quickToggle = document.getElementById("checkQuickMode");
  if (quickToggle) {
    quickToggle.addEventListener("change", (e) => {
      const isQuick = e.target.checked;
      const roundsSlider = document.getElementById("sliderRounds");
      const clientsSlider = document.getElementById("sliderClients");
      if (isQuick) {
        if (roundsSlider) { roundsSlider.value = 3; document.getElementById("badgeRounds").textContent = "3"; }
        if (clientsSlider) { clientsSlider.value = 3; document.getElementById("badgeClients").textContent = "3"; }
      } else {
        if (roundsSlider) { roundsSlider.value = 20; document.getElementById("badgeRounds").textContent = "20"; }
        if (clientsSlider) { clientsSlider.value = 5; document.getElementById("badgeClients").textContent = "5"; }
      }
    });
  }

  // Launch Form Submit
  const formLaunch = document.getElementById("formLaunchExperiment");
  if (formLaunch) {
    formLaunch.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btnLaunchExp");
      btn.disabled = true;
      btn.innerHTML = `<span>Launching simulation...</span>`;

      const config = {
        model: state.selectedModel,
        algorithm: document.getElementById("selectAlgo").value,
        alpha: parseFloat(document.getElementById("sliderAlpha").value),
        is_iid: document.getElementById("checkIid").checked,
        mu: parseFloat(document.getElementById("sliderMu").value),
        num_clients: parseInt(document.getElementById("sliderClients").value),
        num_rounds: parseInt(document.getElementById("sliderRounds").value),
        batch_size: parseInt(document.getElementById("selectBatchSize").value),
        local_epochs: parseInt(document.getElementById("selectLocalEpochs").value),
        client_lr: parseFloat(document.getElementById("inputLr").value),
        quick: document.getElementById("checkQuickMode").checked
      };

      try {
        const resp = await API.launchExperiment(config);
        console.log("Launched:", resp);
        btn.disabled = false;
        btn.innerHTML = `<span>🚀 Launch Federated Experiment</span>`;

        // Reset monitor charts and switch to monitor tab
        FedCharts.resetCharts();
        document.getElementById("liveEventsTerminal").innerHTML = "";
        if (resp.run_folder) {
          appendTerminal(new Date().toLocaleTimeString(), "RUN_INIT", `📁 Provisioned isolated run folder: results/runs/${resp.run_folder}`);
        }
        switchTab("monitor");
      } catch (err) {
        alert("Launch Failed: " + err.message);
        btn.disabled = false;
        btn.innerHTML = `<span>🚀 Launch Federated Experiment</span>`;
      }
    });
  }

  // 4. Monitor View & WebSocket Handlers
  const terminal = document.getElementById("liveEventsTerminal");
  function appendTerminal(timeStr, type, msg) {
    if (!terminal) return;
    const line = document.createElement("div");
    line.className = "event-line";
    line.innerHTML = `<span class="time">[${timeStr}]</span> <span class="type">${type}</span> <span class="msg">${msg}</span>`;
    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
  }

  // Stop experiment button
  const btnStop = document.getElementById("btnStopExperiment");
  if (btnStop) {
    btnStop.addEventListener("click", async () => {
      if (confirm("Are you sure you want to stop the ongoing federated simulation?")) {
        try {
          await API.stopExperiment();
          appendTerminal(new Date().toLocaleTimeString(), "CANCEL", "Stop signal dispatched to worker thread.");
        } catch (e) {
          alert("Error: " + e.message);
        }
      }
    });
  }

  // WebSocket Telemetry Listeners
  fedWs.on("INITIAL_STATE", (payload) => {
    if (payload.active_job) {
      handleJobStateUpdate(payload.active_job);
    }
  });

  fedWs.on("ROUND_START", (payload) => {
    const d = payload.data;
    document.getElementById("valSimStatus").textContent = `Round ${d.round}/${d.total_rounds}`;
    document.getElementById("valSimStatus").className = "metric-value accent";
    appendTerminal(new Date().toLocaleTimeString(), "ROUND", d.message);
  });

  fedWs.on("CLIENT_FIT_COMPLETE", (payload) => {
    const d = payload.data;
    updateClientCard(d.client_id, d.loss, d.accuracy, d.sample_count);
    appendTerminal(new Date().toLocaleTimeString(), `CLIENT ${d.client_id + 1}`, `Fit complete: Loss ${d.loss}, Acc ${d.accuracy}`);
  });

  fedWs.on("AGGREGATION_COMPLETE", (payload) => {
    const d = payload.data;
    appendTerminal(new Date().toLocaleTimeString(), "AGGREGATE", d.message);
  });

  fedWs.on("ROUND_COMPLETE", (payload) => {
    const r = payload.data.round_data;
    FedCharts.addRoundPoint(r);

    document.getElementById("valValAcc").textContent = `${(r.validation_accuracy * 100).toFixed(1)}%`;
    document.getElementById("valMacroF1").textContent = `${(r.validation_macro_f1 * 100).toFixed(1)}%`;
    document.getElementById("valWorstAcc").textContent = `${(r.worst_client_accuracy * 100).toFixed(1)}%`;
    document.getElementById("valDisparity").textContent = `±${r.client_accuracy_std.toFixed(3)}`;

    appendTerminal(new Date().toLocaleTimeString(), "EVAL", payload.data.message);
  });

  fedWs.on("EXPERIMENT_COMPLETE", (payload) => {
    const s = payload.data.summary;
    document.getElementById("valSimStatus").textContent = "COMPLETED";
    document.getElementById("valSimStatus").className = "metric-value accent";
    appendTerminal(new Date().toLocaleTimeString(), "COMPLETE", payload.data.message);
    loadResultsView();
  });

  fedWs.on("EXPERIMENT_CANCELLED", (payload) => {
    document.getElementById("valSimStatus").textContent = "CANCELLED";
    document.getElementById("valSimStatus").className = "metric-value amber";
    appendTerminal(new Date().toLocaleTimeString(), "CANCELLED", payload.data.message);
  });

  fedWs.on("EXPERIMENT_ERROR", (payload) => {
    document.getElementById("valSimStatus").textContent = "FAILED";
    document.getElementById("valSimStatus").className = "metric-value amber";
    appendTerminal(new Date().toLocaleTimeString(), "ERROR", payload.data.message);
  });

  function handleJobStateUpdate(job) {
    document.getElementById("valSimStatus").textContent = job.status;
    document.getElementById("valSimModel").textContent = `${job.model_name.toUpperCase()} (${job.algorithm.toUpperCase()})`;
    if (job.history && job.history.length > 0) {
      FedCharts.loadFullHistory(job.history);
      const last = job.history[job.history.length - 1];
      document.getElementById("valValAcc").textContent = `${(last.validation_accuracy * 100).toFixed(1)}%`;
      document.getElementById("valMacroF1").textContent = `${(last.validation_macro_f1 * 100).toFixed(1)}%`;
      document.getElementById("valWorstAcc").textContent = `${(last.worst_client_accuracy * 100).toFixed(1)}%`;
      document.getElementById("valDisparity").textContent = `±${last.client_accuracy_std.toFixed(3)}`;
    }
    setupClientFleet(job.num_clients);
  }

  function setupClientFleet(numClients) {
    const container = document.getElementById("clientFleetGrid");
    if (!container) return;
    container.innerHTML = "";
    for (let i = 0; i < numClients; i++) {
      const card = document.createElement("div");
      card.className = "client-node-card";
      card.id = `clientNodeCard_${i}`;
      card.innerHTML = `
        <div class="client-node-header">
          <span class="client-node-title">🚜 Client Node ${i + 1}</span>
          <span class="client-node-status" id="clientStatus_${i}">Idle</span>
        </div>
        <div class="client-stats">
          <div><span>Local Acc:</span> <b id="clientAcc_${i}">--</b></div>
          <div><span>Local Loss:</span> <b id="clientLoss_${i}">--</b></div>
          <div><span>Data Samples:</span> <b id="clientSamples_${i}">--</b></div>
        </div>
      `;
      container.appendChild(card);
    }
  }

  function updateClientCard(cid, loss, acc, samples) {
    const card = document.getElementById(`clientNodeCard_${cid}`);
    const badge = document.getElementById(`clientStatus_${cid}`);
    const accEl = document.getElementById(`clientAcc_${cid}`);
    const lossEl = document.getElementById(`clientLoss_${cid}`);
    const samplesEl = document.getElementById(`clientSamples_${cid}`);

    if (card) {
      card.classList.add("training");
      setTimeout(() => card.classList.remove("training"), 1200);
    }
    if (badge) {
      badge.textContent = "Synced";
      badge.className = "client-node-status active";
    }
    if (accEl) accEl.textContent = `${(acc * 100).toFixed(1)}%`;
    if (lossEl) lossEl.textContent = loss.toFixed(4);
    if (samplesEl) samplesEl.textContent = samples.toLocaleString();
  }

  // 5. Results & Benchmark Explorer View
  async function loadResultsView() {
    try {
      const [experiments, figures, checkpoints] = await Promise.all([
        API.getExperiments(),
        API.getFigures(),
        API.getCheckpoints()
      ]);

      state.experimentsData = experiments;
      state.checkpointsData = checkpoints;

      // Table render
      const tbody = document.getElementById("resultsTableBody");
      if (tbody) {
        tbody.innerHTML = "";
        experiments.forEach(exp => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><b>${exp.model.toUpperCase()}</b></td>
            <td><span class="tab-badge" style="background:${exp.algorithm === 'FEDPROX' ? 'rgba(16,185,129,0.2)' : 'rgba(6,182,212,0.2)'}">${exp.algorithm}</span></td>
            <td>${exp.alpha === 'IID' ? 'IID' : 'α = ' + exp.alpha}</td>
            <td>${exp.mu ? exp.mu : '--'}</td>
            <td>${exp.rounds_completed}</td>
            <td><b style="color:#10b981">${(exp.val_accuracy * 100).toFixed(1)}%</b></td>
            <td><b style="color:#06b6d4">${(exp.val_macro_f1 * 100).toFixed(1)}%</b></td>
            <td>${(exp.worst_client_accuracy * 100).toFixed(1)}%</td>
            <td>±${exp.client_accuracy_std.toFixed(3)}</td>
            <td>${exp.total_time_s}s</td>
            <td>
              ${exp.has_checkpoint ? `<a href="/api/checkpoints/download/${exp.checkpoint_name}" class="btn btn-secondary" style="padding:0.25rem 0.6rem; font-size:0.75rem;">Download .pt</a>` : '<span style="color:#64748b">No ckpt</span>'}
            </td>
          `;
          tbody.appendChild(tr);
        });
      }

      // Pareto Scatter Chart
      FedCharts.renderParetoChart("chartPareto", experiments);

      // Figures Gallery
      const figContainer = document.getElementById("figuresGallery");
      if (figContainer) {
        figContainer.innerHTML = "";
        figures.forEach(fig => {
          const card = document.createElement("div");
          card.className = "figure-thumb-card";
          card.innerHTML = `
            <img src="${fig.url}" alt="${fig.title}" class="figure-thumb-img" loading="lazy">
            <div class="figure-thumb-meta">
              <div class="figure-thumb-title">${fig.title}</div>
              <div class="figure-thumb-desc">${fig.description}</div>
            </div>
          `;
          card.addEventListener("click", () => openFigureModal(fig.url, fig.title, fig.description));
          figContainer.appendChild(card);
        });
      }
    } catch (e) {
      console.error("Error loading results:", e);
    }
  }

  // Modal logic
  const modal = document.getElementById("figureModal");
  const modalImg = document.getElementById("modalFigureImg");
  const modalTitle = document.getElementById("modalFigureTitle");
  const modalDesc = document.getElementById("modalFigureDesc");
  const modalClose = document.getElementById("modalCloseBtn");

  function openFigureModal(url, title, desc) {
    if (!modal) return;
    modalImg.src = url;
    modalTitle.textContent = title;
    modalDesc.textContent = desc;
    modal.classList.add("active");
  }

  if (modalClose) {
    modalClose.addEventListener("click", () => modal.classList.remove("active"));
  }
  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.remove("active");
    });
  }

  // 6. Diagnostics & Plant Disease Classifier View
  async function loadDiagnosticsView() {
    try {
      const [checkpoints, samples] = await Promise.all([
        API.getCheckpoints(),
        API.getSampleImages()
      ]);

      // Populate Checkpoint Select
      const ckptSelect = document.getElementById("selectDiagnosticCkpt");
      if (ckptSelect) {
        ckptSelect.innerHTML = "";
        checkpoints.forEach(c => {
          const opt = document.createElement("option");
          opt.value = c.filename;
          opt.textContent = `${c.filename} (${c.size_mb} MB)`;
          if (c.filename.includes("fedprox_mobilenetv3_alpha_0.1_mu_0.01")) {
            opt.selected = true;
          }
          ckptSelect.appendChild(opt);
        });
      }

      // Populate Sample Thumbs
      const samplesStrip = document.getElementById("samplesStrip");
      if (samplesStrip) {
        samplesStrip.innerHTML = "";
        samples.forEach(s => {
          const img = document.createElement("img");
          img.src = s.url;
          img.alt = s.title;
          img.title = s.title;
          img.className = "sample-thumb";
          img.addEventListener("click", () => {
            document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("selected"));
            img.classList.add("selected");
            state.selectedSampleFile = s.filename;
            state.uploadedFile = null;

            // Update preview
            const preview = document.getElementById("previewImg");
            if (preview) {
              preview.src = s.url;
              preview.style.display = "block";
              document.getElementById("dropzonePlaceholder").style.display = "none";
            }
          });
          samplesStrip.appendChild(img);
        });
      }
    } catch (e) {
      console.error("Error loading diagnostics view:", e);
    }
  }

  // File Upload Dropzone
  const dropzone = document.getElementById("imageDropzone");
  const fileInput = document.getElementById("fileInput");

  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageFile(e.target.files[0]);
      }
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--accent-primary)";
    });
    dropzone.addEventListener("dragleave", () => {
      dropzone.style.borderColor = "var(--border-medium)";
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--border-medium)";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleImageFile(e.dataTransfer.files[0]);
      }
    });
  }

  function handleImageFile(file) {
    state.uploadedFile = file;
    state.selectedSampleFile = null;
    document.querySelectorAll(".sample-thumb").forEach(t => t.classList.remove("selected"));

    const reader = new FileReader();
    reader.onload = (e) => {
      const preview = document.getElementById("previewImg");
      if (preview) {
        preview.src = e.target.result;
        preview.style.display = "block";
        document.getElementById("dropzonePlaceholder").style.display = "none";
      }
    };
    reader.readAsDataURL(file);
  }

  // Run Diagnosis Button
  const btnDiagnose = document.getElementById("btnRunDiagnosis");
  if (btnDiagnose) {
    btnDiagnose.addEventListener("click", async () => {
      const ckpt = document.getElementById("selectDiagnosticCkpt").value;
      if (!state.uploadedFile && !state.selectedSampleFile) {
        alert("Please select a sample leaf photo or upload your own leaf image first.");
        return;
      }

      btnDiagnose.disabled = true;
      btnDiagnose.innerHTML = `<span>Evaluating model...</span>`;

      try {
        const result = await API.predictLeaf(state.uploadedFile, state.selectedSampleFile, ckpt);
        renderDiagnosisResult(result);
      } catch (err) {
        alert("Diagnostic failed: " + err.message);
      } finally {
        btnDiagnose.disabled = false;
        btnDiagnose.innerHTML = `<span>🔬 Diagnose Crop Leaf</span>`;
      }
    });
  }

  function renderDiagnosisResult(res) {
    const card = document.getElementById("diagnosisResultCard");
    if (!card) return;
    card.style.display = "block";

    const top = res.top_prediction;
    const advisory = res.advisory;

    document.getElementById("diagCropName").textContent = top.crop;
    document.getElementById("diagDiseaseName").textContent = top.disease;
    document.getElementById("diagConfidence").textContent = top.percentage;
    document.getElementById("diagLatency").textContent = `${res.latency_ms} ms (${res.device.toUpperCase()})`;

    const sevBadge = document.getElementById("diagSeverityBadge");
    sevBadge.textContent = advisory.severity;
    sevBadge.className = `severity-badge severity-${advisory.severity.replace(/\s+/g, '')}`;

    document.getElementById("diagDiseaseDesc").textContent = advisory.description;
    document.getElementById("diagRemedy").textContent = advisory.remedy;

    // Top-3 Bar Chart
    const barList = document.getElementById("diagTopKList");
    if (barList) {
      barList.innerHTML = "";
      res.all_predictions.slice(0, 3).forEach(p => {
        const row = document.createElement("div");
        row.innerHTML = `
          <div class="conf-row">
            <span>${p.crop} — ${p.disease}</span>
            <b>${p.percentage}</b>
          </div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width: ${p.percentage}"></div>
          </div>
        `;
        barList.appendChild(row);
      });
    }

    card.scrollIntoView({ behavior: 'smooth' });
  }

  // 7. Dataset & Partition View
  async function loadDatasetView() {
    try {
      const [summary, classes] = await Promise.all([
        API.getDatasetSummary(),
        API.getClasses()
      ]);

      // Summary Pills
      document.getElementById("dsTotalImages").textContent = summary.total_images.toLocaleString();
      document.getElementById("dsClassesCount").textContent = summary.num_classes;
      document.getElementById("dsCropsCount").textContent = summary.num_crops;

      // Partition Chart
      updatePartitionChart(0.1);

      // Search & Classes List
      const classesList = document.getElementById("classesGrid");
      if (classesList) {
        renderClassesGrid(classes, classesList);
        const searchInput = document.getElementById("classSearchInput");
        if (searchInput) {
          searchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase();
            const filtered = classes.filter(c => 
              c.crop.toLowerCase().includes(query) || 
              c.disease.toLowerCase().includes(query)
            );
            renderClassesGrid(filtered, classesList);
          });
        }
      }
    } catch (e) {
      console.error("Error loading dataset view:", e);
    }
  }

  async function updatePartitionChart(alphaVal) {
    try {
      const isIid = alphaVal === "iid";
      const partData = await API.getPartitionDistribution(isIid ? 0.1 : alphaVal, isIid);
      FedCharts.renderPartitionChart("chartPartition", partData.client_sample_counts);
    } catch (e) {
      console.error("Error updating partition chart:", e);
    }
  }

  const partAlphaSelect = document.getElementById("selectPartitionAlpha");
  if (partAlphaSelect) {
    partAlphaSelect.addEventListener("change", (e) => {
      updatePartitionChart(e.target.value);
    });
  }

  function renderClassesGrid(classes, container) {
    container.innerHTML = "";
    classes.forEach(c => {
      const item = document.createElement("div");
      item.className = "model-card";
      item.innerHTML = `
        <div class="model-card-header">
          <span style="font-weight:600; font-size:0.92rem;">${c.crop}</span>
          <span class="severity-badge ${c.is_healthy ? 'severity-None' : 'severity-High'}" style="font-size:0.7rem; padding:0.1rem 0.45rem;">
            ${c.is_healthy ? 'Healthy' : 'Diseased'}
          </span>
        </div>
        <div style="font-size:0.85rem; color:#cbd5e1; margin-bottom:0.4rem;">${c.disease}</div>
        <div style="font-size:0.75rem; color:#94a3b8;">Severity: <b style="color:#e2e8f0">${c.severity}</b></div>
      `;
      container.appendChild(item);
    });
  }

  // Initial load
  loadStudioView();
  setupClientFleet(5);
  FedCharts.initConvergenceChart();
  FedCharts.initRobustnessChart();
});
