import React, { useState } from 'react';
import { 
  Download, 
  Search, 
  Layers, 
  TrendingUp, 
  Image, 
  Maximize2, 
  ExternalLink,
  Filter,
  CheckCircle2,
  HardDrive
} from 'lucide-react';
import { Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend);

export function BenchmarkView({ experiments, figures, checkpoints, onOpenFigureModal }) {
  const [filterType, setFilterType] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Filtering
  const filteredExperiments = experiments.filter((exp) => {
    const matchesSearch = 
      exp.model.toLowerCase().includes(searchQuery.toLowerCase()) ||
      exp.algorithm.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (exp.run_folder && exp.run_folder.toLowerCase().includes(searchQuery.toLowerCase())) ||
      String(exp.alpha).includes(searchQuery);

    if (!matchesSearch) return false;

    if (filterType === 'runs') return exp.is_isolated_run;
    if (filterType === 'fedprox') return exp.algorithm.toLowerCase().includes('prox');
    if (filterType === 'fedavg') return !exp.algorithm.toLowerCase().includes('prox');
    if (filterType === 'severe') return String(exp.alpha) === '0.1';
    return true;
  });

  // Pareto Chart Data Preparation
  const sizeMap = {
    mobilenetv3: 5.94,
    resnet50: 89.97,
    efficientnet_b0: 15.47,
    convnext_tiny: 106.24
  };

  const fedavgPoints = [];
  const fedproxPoints = [];

  experiments.forEach((exp) => {
    const mSize = sizeMap[exp.model.toLowerCase()] || 20;
    const rounds = exp.rounds_completed || 20;
    const commMB = Math.round(2 * 5 * rounds * mSize);
    const point = {
      x: commMB,
      y: (exp.val_macro_f1 || 0) * 100,
      label: `${exp.model.toUpperCase()} (α=${exp.alpha})`,
    };

    if (exp.algorithm.toLowerCase().includes('prox')) {
      fedproxPoints.push(point);
    } else {
      fedavgPoints.push(point);
    }
  });

  const paretoChartData = {
    datasets: [
      {
        label: 'FedProx (μ=0.01)',
        data: fedproxPoints,
        backgroundColor: '#10b981',
        borderColor: '#34d399',
        pointRadius: 6,
        pointHoverRadius: 9,
      },
      {
        label: 'FedAvg Baseline',
        data: fedavgPoints,
        backgroundColor: '#06b6d4',
        borderColor: '#67e8f9',
        pointRadius: 6,
        pointHoverRadius: 9,
      },
    ],
  };

  const paretoOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: { color: '#cbd5e1', font: { family: 'Plus Jakarta Sans', size: 11 }, usePointStyle: true },
      },
      tooltip: {
        backgroundColor: 'rgba(6, 12, 9, 0.95)',
        titleColor: '#f0fdf4',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(52, 211, 153, 0.3)',
        borderWidth: 1,
        padding: 8,
        callbacks: {
          label: (ctx) => `${ctx.raw.label}: ${ctx.raw.y.toFixed(1)}% F1 (${ctx.raw.x.toLocaleString()} MB)`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
        title: { display: true, text: 'Total Communication Volume (MB)', color: '#64748b' },
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
        title: { display: true, text: 'Macro-F1 Score (%)', color: '#64748b' },
        min: 65,
        max: 102,
      },
    },
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      
      {/* 1. Benchmark Results Table Panel */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={20} color="#10b981" />
              <span>Empirical Research Experiments (Tables 4 & 5)</span>
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Evaluated on the strictly isolated 38-class test split (8,146 images).
            </p>
          </div>

          {/* Filter Pills & Search */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', background: 'rgba(255, 255, 255, 0.04)', borderRadius: 'var(--radius-md)', padding: '2px', border: '1px solid var(--border-subtle)', flexWrap: 'wrap' }}>
              {[
                { id: 'all', label: 'All Experiments' },
                { id: 'runs', label: '📁 Custom Runs (results/runs/)' },
                { id: 'severe', label: 'Severe Non-IID (α=0.1)' },
                { id: 'fedprox', label: 'FedProx Only' },
                { id: 'fedavg', label: 'FedAvg Only' },
              ].map((pill) => (
                <button
                  key={pill.id}
                  onClick={() => setFilterType(pill.id)}
                  style={{
                    background: filterType === pill.id ? 'rgba(16, 185, 129, 0.2)' : 'transparent',
                    color: filterType === pill.id ? 'var(--emerald-300)' : 'var(--text-secondary)',
                    border: 'none',
                    padding: '0.35rem 0.75rem',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: '0.78rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                >
                  {pill.label}
                </button>
              ))}
            </div>

            <div style={{ position: 'relative', width: '220px' }}>
              <Search size={14} style={{ position: 'absolute', left: '10px', top: '10px', color: 'var(--text-tertiary)' }} />
              <input
                type="text"
                placeholder="Filter models, alphas..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="form-field-input"
                style={{ padding: '0.45rem 0.75rem 0.45rem 2rem', fontSize: '0.82rem' }}
              />
            </div>
          </div>
        </div>

        {/* Table Container */}
        <div style={{ overflowX: 'auto', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <table className="fed-table">
            <thead>
              <tr>
                <th>Model / Run Folder</th>
                <th>Algorithm</th>
                <th>Heterogeneity</th>
                <th>μ (FedProx)</th>
                <th>Rounds</th>
                <th>Test Accuracy</th>
                <th>Macro-F1</th>
                <th>Worst Client</th>
                <th>Client Std</th>
                <th>Time</th>
                <th>Checkpoint</th>
              </tr>
            </thead>
            <tbody>
              {filteredExperiments.length === 0 ? (
                <tr>
                  <td colSpan="11" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-tertiary)' }}>
                    No experiments match your search criteria.
                  </td>
                </tr>
              ) : (
                filteredExperiments.map((exp) => (
                  <tr key={exp.experiment_id}>
                    <td>
                      <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <div style={{ display: 'flex', alignItems: 'center' }}>
                          <b style={{ color: '#fff' }}>{exp.model.toUpperCase()}</b>
                          {exp.is_isolated_run && (
                            <span style={{
                              fontSize: '0.62rem',
                              fontWeight: 700,
                              background: 'rgba(16, 185, 129, 0.2)',
                              color: 'var(--emerald-300)',
                              padding: '0.05rem 0.35rem',
                              borderRadius: '4px',
                              marginLeft: '0.4rem',
                              border: '1px solid rgba(16, 185, 129, 0.3)'
                            }}>
                              CUSTOM RUN
                            </span>
                          )}
                        </div>
                        {exp.run_folder && (
                          <span style={{
                            fontFamily: 'var(--font-mono)',
                            fontSize: '0.68rem',
                            color: 'var(--text-tertiary)',
                            marginTop: '0.15rem'
                          }}>
                            {exp.run_folder}
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span style={{
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        padding: '0.15rem 0.5rem',
                        borderRadius: 'var(--radius-full)',
                        background: exp.algorithm === 'FEDPROX' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(6, 182, 212, 0.15)',
                        color: exp.algorithm === 'FEDPROX' ? 'var(--emerald-300)' : 'var(--cyan-400)',
                        border: `1px solid ${exp.algorithm === 'FEDPROX' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(6, 182, 212, 0.3)'}`,
                      }}>
                        {exp.algorithm}
                      </span>
                    </td>
                    <td>{exp.alpha === 'IID' ? 'IID' : `α = ${exp.alpha}`}</td>
                    <td>{exp.mu ? exp.mu : '--'}</td>
                    <td>{exp.rounds_completed}</td>
                    <td><b style={{ color: 'var(--emerald-400)' }}>{(exp.val_accuracy * 100).toFixed(1)}%</b></td>
                    <td><b style={{ color: 'var(--cyan-400)' }}>{(exp.val_macro_f1 * 100).toFixed(1)}%</b></td>
                    <td><b style={{ color: 'var(--amber-400)' }}>{(exp.worst_client_accuracy * 100).toFixed(1)}%</b></td>
                    <td style={{ color: 'var(--purple-400)' }}>±{exp.client_accuracy_std.toFixed(3)}</td>
                    <td>{exp.total_time_s}s</td>
                    <td>
                      {exp.has_checkpoint ? (
                        <a
                          href={exp.checkpoint_download_url || `/api/checkpoints/download/${exp.checkpoint_name}`}
                          className="btn-outline"
                          style={{ padding: '0.25rem 0.6rem', fontSize: '0.72rem', textDecoration: 'none' }}
                        >
                          <Download size={12} />
                          <span>.pt</span>
                        </a>
                      ) : (
                        <span style={{ color: 'var(--text-tertiary)', fontSize: '0.75rem' }}>--</span>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Pareto Chart & Checkpoints Explorer Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.75rem' }}>
        
        {/* Pareto Efficiency Scatter Chart */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                <TrendingUp size={18} color="#06b6d4" />
                <span>Communication vs Performance (Figure 8 Pareto)</span>
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Pareto frontier demonstrating edge efficiency vs conventional backbones
              </p>
            </div>
          </div>

          <div style={{ height: '320px' }}>
            <Scatter data={paretoChartData} options={paretoOptions} />
          </div>
        </div>

        {/* Checkpoint Exporter Panel */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
              <HardDrive size={18} color="#10b981" />
              <span>Model Checkpoint Repository</span>
            </h3>
            <span className="status-pill">{checkpoints.length} Checkpoints</span>
          </div>

          <div style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '0.6rem',
            overflowY: 'auto',
            maxHeight: '320px',
            paddingRight: '0.3rem',
          }}>
            {checkpoints.map((ckpt) => (
              <div
                key={ckpt.filename}
                style={{
                  background: 'rgba(8, 14, 11, 0.7)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '0.75rem 1rem',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.84rem', fontWeight: 600, color: '#f0fdf4' }}>{ckpt.filename}</div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.15rem' }}>
                    Size: <b style={{ color: '#fff' }}>{ckpt.size_mb} MB</b> | Saved: {ckpt.modified}
                  </div>
                </div>

                <a
                  href={ckpt.download_url}
                  className="btn-outline"
                  style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', textDecoration: 'none' }}
                >
                  <Download size={13} />
                  <span>Download</span>
                </a>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 3. Scientific Publication Figures Gallery */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ marginBottom: '1.25rem' }}>
          <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Image size={19} color="#34d399" />
            <span>Scientific Publication Gallery (Figures 1 to 9)</span>
          </h3>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Click on any figure to examine in full resolution with high-dpi rendering.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: '1.25rem',
        }}>
          {figures.map((fig) => (
            <div
              key={fig.filename}
              onClick={() => onOpenFigureModal(fig)}
              className="figure-thumb-card"
              style={{
                background: 'rgba(8, 14, 11, 0.85)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                overflow: 'hidden',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
              }}
            >
              <div style={{ position: 'relative', height: '160px', background: '#000', overflow: 'hidden' }}>
                <img
                  src={fig.url}
                  alt={fig.title}
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  loading="lazy"
                />
                <div style={{
                  position: 'absolute',
                  top: '8px',
                  right: '8px',
                  background: 'rgba(0,0,0,0.6)',
                  color: '#fff',
                  borderRadius: 'var(--radius-sm)',
                  padding: '4px',
                }}>
                  <Maximize2 size={13} />
                </div>
              </div>

              <div style={{ padding: '0.85rem' }}>
                <div style={{ fontSize: '0.88rem', fontWeight: 600, color: '#f0fdf4' }}>{fig.title}</div>
                <div style={{ fontSize: '0.74rem', color: 'var(--text-secondary)', marginTop: '0.2rem', lineHeight: 1.4 }}>
                  {fig.description}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
