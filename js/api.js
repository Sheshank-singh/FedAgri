/**
 * FedAgri API Client
 * Clean async wrappers around backend REST endpoints
 */

const API = {
  baseUrl: window.location.origin,

  async getHealth() {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error("Failed to fetch system health");
    return await res.json();
  },

  async getModels() {
    const res = await fetch(`${this.baseUrl}/api/models`);
    if (!res.ok) throw new Error("Failed to fetch model registry");
    return await res.json();
  },

  async getDatasetSummary() {
    const res = await fetch(`${this.baseUrl}/api/dataset/summary`);
    if (!res.ok) throw new Error("Failed to fetch dataset summary");
    return await res.json();
  },

  async getClasses() {
    const res = await fetch(`${this.baseUrl}/api/dataset/classes`);
    if (!res.ok) throw new Error("Failed to fetch dataset classes");
    return await res.json();
  },

  async getSampleImages() {
    const res = await fetch(`${this.baseUrl}/api/dataset/samples`);
    if (!res.ok) throw new Error("Failed to fetch sample images");
    return await res.json();
  },

  async getPartitionDistribution(alpha = 0.1, isIid = false) {
    const res = await fetch(`${this.baseUrl}/api/dataset/partitions?alpha=${alpha}&is_iid=${isIid}`);
    if (!res.ok) throw new Error("Failed to fetch partition distribution");
    return await res.json();
  },

  async getExperiments() {
    const res = await fetch(`${this.baseUrl}/api/experiments`);
    if (!res.ok) throw new Error("Failed to fetch experiments list");
    return await res.json();
  },

  async getActiveExperiment() {
    const res = await fetch(`${this.baseUrl}/api/experiments/active`);
    if (!res.ok) throw new Error("Failed to fetch active experiment status");
    return await res.json();
  },

  async getExperimentDetails(expId) {
    const res = await fetch(`${this.baseUrl}/api/experiments/${encodeURIComponent(expId)}`);
    if (!res.ok) throw new Error(`Failed to fetch experiment details for ${expId}`);
    return await res.json();
  },

  async launchExperiment(config) {
    const res = await fetch(`${this.baseUrl}/api/experiments/launch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to launch experiment");
    }
    return await res.json();
  },

  async stopExperiment() {
    const res = await fetch(`${this.baseUrl}/api/experiments/stop`, {
      method: "POST"
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to stop experiment");
    }
    return await res.json();
  },

  async getCheckpoints() {
    const res = await fetch(`${this.baseUrl}/api/checkpoints`);
    if (!res.ok) throw new Error("Failed to fetch checkpoints list");
    return await res.json();
  },

  async getFigures() {
    const res = await fetch(`${this.baseUrl}/api/figures`);
    if (!res.ok) throw new Error("Failed to fetch publication figures");
    return await res.json();
  },

  async predictLeaf(imageFileOrBlob, sampleFilename, checkpoint) {
    const formData = new FormData();
    if (imageFileOrBlob) {
      formData.append("file", imageFileOrBlob);
    }
    if (sampleFilename) {
      formData.append("sample_filename", sampleFilename);
    }
    formData.append("checkpoint", checkpoint);

    const res = await fetch(`${this.baseUrl}/api/predict`, {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Inference failed");
    }
    return await res.json();
  }
};

window.API = API;
