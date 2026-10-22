import React, { useState } from 'react';
import { 
  Play, 
  Zap, 
  Cpu, 
  Network, 
  Sliders, 
  Flame, 
  CheckCircle2, 
  Layers, 
  Gauge, 
  Info,
  Sparkles
} from 'lucide-react';

export function StudioView({ models, onLaunchExperiment, isLaunching }) {
  const [selectedModel, setSelectedModel] = useState('mobilenetv3');
  const [algorithm, setAlgorithm] = useState('fedprox');
  const [alpha, setAlpha] = useState(0.1);
  const [isIid, setIsIid] = useState(false);
  const [mu, setMu] = useState(0.01);
  const [numClients, setNumClients] = useState(3);
  const [numRounds, setNumRounds] = useState(3);
  const [batchSize, setBatchSize] = useState(32);
  const [localEpochs, setLocalEpochs] = useState(1);
  const [clientLr, setClientLr] = useState(0.001);
  const [quickMode, setQuickMode] = useState(true);

  const handleQuickToggle = (checked) => {
    setQuickMode(checked);
    if (checked) {
      setNumClients(3);
      setNumRounds(3);
    } else {
      setNumClients(5);
      setNumRounds(20);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onLaunchExperiment({
      model: selectedModel,
      algorithm,
      alpha: parseFloat(alpha),
      is_iid: isIid,
      mu: parseFloat(mu),
      num_clients: parseInt(numClients, 10),
      num_rounds: parseInt(numRounds, 10),
      batch_size: parseInt(batchSize, 10),
      local_epochs: parseInt(localEpochs, 10),
      client_lr: parseFloat(clientLr),
      quick: quickMode,
    });
  };

  const getModelBadge = (modelName) => {
    switch (modelName) {
      case 'mobilenetv3':
        return { label: 'Edge Champion', color: '#10b981', bg: 'rgba(16, 185, 129, 0.15)' };
      case 'resnet50':
        return { label: 'Deep Residual', color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)' };
      case 'efficientnet_b0':
        return { label: 'Optimal Pareto', color: '#06b6d4', bg: 'rgba(6, 182, 212, 0.15)' };
      case 'convnext_tiny':
        return { label: 'Modern Pure-Conv', color: '#a855f7', bg: 'rgba(168, 85, 247, 0.15)' };
      default:
        return { label: 'CNN Backbone', color: '#94a3b8', bg: 'rgba(255, 255, 255, 0.08)' };
    }
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      
      {/* 1. Model Architecture Selector Grid */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <Layers size={20} color="#34d399" />
              <span>1. Select CNN Backbone Architecture</span>
            </h2>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              All backbones are instantiated with ImageNet weights and customized to classify the 38 PlantVillage crop pathologies.
            </p>
          </div>
          <span className="status-pill">ImageNet Pretrained</span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
          gap: '1.1rem',
        }}>
          {models.map((m) => {
            const isSelected = selectedModel === m.Model;
            const badge = getModelBadge(m.Model);
            const params = m['Parameters (M)'] || (m.Total_Params ? (m.Total_Params / 1e6).toFixed(2) : '2.0');
            const latency = m.Inference_Latency_ms ? `${m.Inference_Latency_ms} ms` : '~4.0 ms';

            return (
              <div
                key={m.Model}
                onClick={() => setSelectedModel(m.Model)}
                className={`model-card-item ${isSelected ? 'selected' : ''}`}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontFamily: 'var(--font-heading)', fontWeight: 700, fontSize: '1.05rem', color: isSelected ? 'var(--emerald-300)' : '#fff' }}>
                    {m.Model.toUpperCase()}
                  </span>
                  <span style={{
                    fontSize: '0.68rem',
                    fontWeight: 700,
                    padding: '0.15rem 0.5rem',
                    borderRadius: 'var(--radius-full)',
                    color: badge.color,
                    background: badge.bg,
                    border: `1px solid ${badge.color}33`,
                  }}>
                    {badge.label}
                  </span>
                </div>

                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.4, minHeight: '36px' }}>
                  {m.Description || 'Convolutional neural network for distributed feature extraction.'}
                </p>

                <div style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '0.5rem',
                  marginTop: '0.85rem',
                  paddingTop: '0.85rem',
                  borderTop: '1px solid rgba(255, 255, 255, 0.05)',
                  fontSize: '0.78rem',
                  color: 'var(--text-secondary)',
                }}>
                  <div>Params: <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{params}M</b></div>
                  <div>Size: <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{m.Size_MB} MB</b></div>
                  <div>GFLOPs: <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{m.GFLOPs ?? '0.82'}</b></div>
                  <div>Latency: <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>{latency}</b></div>
                </div>

                {isSelected && (
                  <div style={{
                    position: 'absolute',
                    top: '8px',
                    right: '8px',
                    color: 'var(--emerald-400)',
                  }}>
                    <CheckCircle2 size={16} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* 2. Federated Optimization & Topology Parameters Grid */}
      <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr', gap: '1.75rem' }}>
        
        {/* Left Column: Heterogeneity & Algorithm */}
        <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.35rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Flame size={18} color="#f59e0b" />
              <span>2. Statistical Heterogeneity & Optimization</span>
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
              Configure label non-IID distribution skew and proximal drift regularizer.
            </p>
          </div>

          {/* Algorithm Selection */}
          <div>
            <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              <span>Aggregation Algorithm</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--emerald-400)', fontFamily: 'var(--font-mono)' }}>
                {algorithm === 'fedprox' ? 'Loss + (μ/2)||w - w_t||²' : 'Σ (n_k / N) w_k'}
              </span>
            </label>
            <select
              className="form-field-input"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
            >
              <option value="fedprox">FedProx — Proximal Regularized (Mitigates Client Drift)</option>
              <option value="fedavg">FedAvg — Classical Sample-Weighted Averaging</option>
            </select>
          </div>

          {/* FedProx Mu Slider */}
          {algorithm === 'fedprox' && (
            <div style={{
              background: 'rgba(16, 185, 129, 0.05)',
              border: '1px solid rgba(16, 185, 129, 0.2)',
              borderRadius: 'var(--radius-md)',
              padding: '1rem',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--emerald-300)' }}>
                  Proximal Penalty Strength (μ)
                </span>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  background: 'rgba(16, 185, 129, 0.2)',
                  color: 'var(--emerald-300)',
                  padding: '0.15rem 0.5rem',
                  borderRadius: 'var(--radius-sm)'
                }}>
                  {mu}
                </span>
              </div>
              <input
                type="range"
                className="slider-styled"
                min="0.001"
                max="0.1"
                step="0.001"
                value={mu}
                onChange={(e) => setMu(e.target.value)}
              />
              <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
                Controls constraint restricting local client weights from diverging away from global weights under skewed label distributions.
              </p>
            </div>
          )}

          {/* Dirichlet Alpha Slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
                Dirichlet Concentration Parameter (α)
              </span>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                fontSize: '0.85rem',
                background: 'rgba(6, 182, 212, 0.15)',
                color: 'var(--cyan-400)',
                padding: '0.15rem 0.5rem',
                borderRadius: 'var(--radius-sm)'
              }}>
                {isIid ? 'IID' : `α = ${alpha}`}
              </span>
            </div>
            
            <input
              type="range"
              className="slider-styled"
              min="0.1"
              max="10.0"
              step="0.1"
              disabled={isIid}
              value={alpha}
              onChange={(e) => setAlpha(e.target.value)}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.35rem' }}>
              <span style={{ color: '#f43f5e' }}>Severe Non-IID (0.1)</span>
              <span>Moderate (1.0)</span>
              <span style={{ color: '#10b981' }}>Near-IID (10.0)</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.85rem' }}>
              <input
                type="checkbox"
                id="checkIidInput"
                checked={isIid}
                onChange={(e) => setIsIid(e.target.checked)}
                style={{ accentColor: 'var(--emerald-500)', width: '16px', height: '16px', cursor: 'pointer' }}
              />
              <label htmlFor="checkIidInput" style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                Uniform IID Partition Baseline (No class skew across clients)
              </label>
            </div>
          </div>

          {/* Quick Mode Toggle */}
          <div style={{
            background: 'rgba(6, 182, 212, 0.06)',
            border: '1px dashed rgba(6, 182, 212, 0.35)',
            borderRadius: 'var(--radius-md)',
            padding: '1rem',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <input
                type="checkbox"
                id="quickModeToggle"
                checked={quickMode}
                onChange={(e) => handleQuickToggle(e.target.checked)}
                style={{ accentColor: 'var(--cyan-500)', width: '18px', height: '18px', cursor: 'pointer' }}
              />
              <label htmlFor="quickModeToggle" style={{ fontWeight: 700, color: 'var(--cyan-300)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Zap size={16} /> Quick Verification Mode (Fast ~60s Demo)
              </label>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.3rem', marginLeft: '1.75rem' }}>
              Subsamples dataset to 3 clients & 3 rounds for rapid pipeline verification. Uncheck to run full 20-round scientific simulation.
            </p>
          </div>
        </div>

        {/* Right Column: Client Fleet & Hyperparameters */}
        <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.35rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Network size={18} color="#06b6d4" />
              <span>3. Client Fleet Topology & Training</span>
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.15rem' }}>
              Simulated agricultural edge devices with private isolated data loaders.
            </p>
          </div>

          {/* Virtual Clients Slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Simulated Virtual Clients (K)</span>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                fontSize: '0.85rem',
                background: 'rgba(255, 255, 255, 0.08)',
                color: '#fff',
                padding: '0.15rem 0.5rem',
                borderRadius: 'var(--radius-sm)'
              }}>
                {numClients} Clients
              </span>
            </div>
            <input
              type="range"
              className="slider-styled"
              min="3"
              max="10"
              step="1"
              value={numClients}
              onChange={(e) => setNumClients(e.target.value)}
            />
          </div>

          {/* Total Rounds Slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.45rem' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Federated Communication Rounds (T)</span>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                fontSize: '0.85rem',
                background: 'rgba(255, 255, 255, 0.08)',
                color: '#fff',
                padding: '0.15rem 0.5rem',
                borderRadius: 'var(--radius-sm)'
              }}>
                {numRounds} Rounds
              </span>
            </div>
            <input
              type="range"
              className="slider-styled"
              min="2"
              max="50"
              step="1"
              value={numRounds}
              onChange={(e) => setNumRounds(e.target.value)}
            />
          </div>

          {/* Batch Size & Local Epochs */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.35rem' }}>Batch Size</label>
              <select
                className="form-field-input"
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
              >
                <option value="16">16 images</option>
                <option value="32">32 images</option>
                <option value="64">64 images</option>
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.35rem' }}>Local Epochs (E)</label>
              <select
                className="form-field-input"
                value={localEpochs}
                onChange={(e) => setLocalEpochs(e.target.value)}
              >
                <option value="1">1 Epoch / Round</option>
                <option value="2">2 Epochs / Round</option>
                <option value="3">3 Epochs / Round</option>
              </select>
            </div>
          </div>

          {/* Optimizer Learning Rate */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.35rem' }}>
              Local Client Learning Rate (AdamW)
            </label>
            <input
              type="number"
              className="form-field-input"
              value={clientLr}
              step="0.0001"
              min="0.00001"
              max="0.05"
              onChange={(e) => setClientLr(e.target.value)}
            />
          </div>

          {/* One-Click Launch Button */}
          <div style={{ marginTop: 'auto', paddingTop: '0.5rem' }}>
            <button
              type="submit"
              disabled={isLaunching}
              className="btn-emerald"
              style={{ width: '100%', padding: '0.95rem 1.5rem', fontSize: '1rem' }}
            >
              {isLaunching ? (
                <>
                  <Sparkles size={18} className="animate-spin" />
                  <span>Initializing Distributed Nodes...</span>
                </>
              ) : (
                <>
                  <Play size={18} />
                  <span>Launch Federated Simulation</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
