/**
 * FedAgri Frontend API Client
 */

export const api = {
  async getHealth() {
    const res = await fetch('/api/health');
    if (!res.ok) throw new Error('Health check failed');
    return res.json();
  },

  async getModels() {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error('Failed to fetch models');
    return res.json();
  },

  async getDatasetSummary() {
    const res = await fetch('/api/dataset/summary');
    if (!res.ok) throw new Error('Failed to fetch dataset summary');
    return res.json();
  },

  async getClasses() {
    const res = await fetch('/api/dataset/classes');
    if (!res.ok) throw new Error('Failed to fetch classes');
    return res.json();
  },

  async getSampleImages() {
    const res = await fetch('/api/dataset/samples');
    if (!res.ok) throw new Error('Failed to fetch sample images');
    return res.json();
  },

  async getPartitionDistribution(alpha = 0.1, isIid = false) {
    const res = await fetch(`/api/dataset/partitions?alpha=${alpha}&is_iid=${isIid}`);
    if (!res.ok) throw new Error('Failed to fetch partition data');
    return res.json();
  },

  async getExperiments() {
    const res = await fetch('/api/experiments');
    if (!res.ok) throw new Error('Failed to fetch experiments');
    return res.json();
  },

  async getActiveExperiment() {
    const res = await fetch('/api/experiments/active');
    if (!res.ok) throw new Error('Failed to fetch active experiment');
    return res.json();
  },

  async getExperimentDetails(expId) {
    const res = await fetch(`/api/experiments/${encodeURIComponent(expId)}`);
    if (!res.ok) throw new Error(`Failed to fetch experiment details: ${expId}`);
    return res.json();
  },

  async launchExperiment(config) {
    const res = await fetch('/api/experiments/launch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to launch experiment');
    }
    return res.json();
  },

  async stopExperiment() {
    const res = await fetch('/api/experiments/stop', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to stop experiment');
    }
    return res.json();
  },

  async getCheckpoints() {
    const res = await fetch('/api/checkpoints');
    if (!res.ok) throw new Error('Failed to fetch checkpoints');
    return res.json();
  },

  async getRuns() {
    const res = await fetch('/api/runs');
    if (!res.ok) throw new Error('Failed to fetch runs');
    return res.json();
  },

  async getFigures() {
    const res = await fetch('/api/figures');
    if (!res.ok) throw new Error('Failed to fetch figures');
    return res.json();
  },

  async predictLeaf(imageFileOrBlob, sampleFilename, checkpoint) {
    const formData = new FormData();
    if (imageFileOrBlob) {
      formData.append('file', imageFileOrBlob);
    }
    if (sampleFilename) {
      formData.append('sample_filename', sampleFilename);
    }
    formData.append('checkpoint', checkpoint);

    const res = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Prediction failed');
    }
    return res.json();
  },
};
