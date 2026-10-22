import React, { useRef, useEffect } from 'react';
import { 
  Square, 
  Activity, 
  TrendingUp, 
  ShieldCheck, 
  AlertTriangle, 
  Server, 
  Terminal, 
  Radio, 
  HardDriveUpload, 
  Clock,
  Folder,
  FolderCheck,
  Download
} from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export function LiveMonitorView({ activeJob, onStopExperiment }) {
  const terminalEndRef = useRef(null);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [activeJob?.log_events]);

  const history = activeJob?.history || [];
  const currentRound = activeJob?.current_round || 0;
  const totalRounds = activeJob?.total_rounds || 20;
  const numClients = activeJob?.num_clients || 5;
  const status = activeJob?.status || 'IDLE';
  const recentClients = activeJob?.recent_client_states || {};

  const latest = history.length > 0 ? history[history.length - 1] : null;

  // Calculate approximate communication volume
  const modelSizeMB = activeJob?.model_name === 'convnext_tiny' ? 106.24 :
                      activeJob?.model_name === 'resnet50' ? 89.97 :
                      activeJob?.model_name === 'efficientnet_b0' ? 15.47 : 5.94;
  const commMB = Math.round(2 * numClients * Math.max(currentRound, 1) * modelSizeMB);

  // Chart 1: Global Convergence
  const convergenceData = {
    labels: history.map((h) => `R${h.round}`),
    datasets: [
      {
        label: 'Validation Accuracy',
        data: history.map((h) => h.validation_accuracy),
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        borderWidth: 2.5,
      },
      {
        label: 'Validation Macro-F1',
        data: history.map((h) => h.validation_macro_f1),
        borderColor: '#06b6d4',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 4,
        borderWidth: 2,
      },
      {
        label: 'Train Loss (Scaled)',
        data: history.map((h) => h.train_loss),
        borderColor: '#f59e0b',
        backgroundColor: 'transparent',
        borderDash: [4, 4],
        tension: 0.3,
        pointRadius: 3,
        borderWidth: 1.5,
        yAxisID: 'yLoss'
      }
    ],
  };

  // Chart 2: Client Robustness & Drift
  const robustnessData = {
    labels: history.map((h) => `R${h.round}`),
    datasets: [
      {
        label: 'Mean Client Acc',
        data: history.map((h) => h.mean_client_accuracy),
        borderColor: '#34d399',
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 4,
      },
      {
        label: 'Worst-Client Acc',
        data: history.map((h) => h.worst_client_accuracy),
        borderColor: '#f43f5e',
        borderWidth: 2,
        tension: 0.25,
        pointRadius: 4,
      },
      {
        label: 'Client Disparity (Std)',
        data: history.map((h) => h.client_accuracy_std),
        borderColor: '#c084fc',
        backgroundColor: 'rgba(192, 132, 252, 0.1)',
        fill: true,
        borderDash: [3, 3],
        tension: 0.25,
        pointRadius: 2,
        yAxisID: 'yStd'
      }
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
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
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
        min: 0,
        max: 1.05,
      },
      yLoss: {
        position: 'right',
        grid: { drawOnChartArea: false },
        ticks: { color: '#f59e0b', font: { size: 9 } },
        min: 0,
      },
      yStd: {
        position: 'right',
        grid: { drawOnChartArea: false },
        ticks: { color: '#c084fc', font: { size: 9 } },
        min: 0,
      }
    },
  };

  const progressPercent = totalRounds > 0 ? Math.min(100, Math.round((currentRound / totalRounds) * 100)) : 0;

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* 1. Top Telemetry Ribbon */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
        gap: '1rem',
      }}>
        {/* Status Card */}
        <div className="glass-panel" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Status</span>
            <Radio size={14} color={status === 'RUNNING' ? '#10b981' : '#94a3b8'} />
          </div>
          <div style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '0.35rem', color: status === 'RUNNING' ? 'var(--emerald-400)' : status === 'COMPLETED' ? 'var(--cyan-400)' : '#f0fdf4' }}>
            {status}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
            {activeJob ? `${activeJob.model_name.toUpperCase()} (${activeJob.algorithm.toUpperCase()})` : 'Idle'}
          </div>
        </div>

        {/* Progress Card */}
        <div className="glass-panel" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Round Progress</span>
            <Clock size={14} color="#06b6d4" />
          </div>
          <div style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '0.35rem', color: 'var(--cyan-400)', fontFamily: 'var(--font-mono)' }}>
            {currentRound} / {totalRounds}
          </div>
          <div style={{
            height: '4px',
            background: 'rgba(255, 255, 255, 0.1)',
            borderRadius: '2px',
            overflow: 'hidden',
            marginTop: '0.45rem',
          }}>
            <div style={{ width: `${progressPercent}%`, height: '100%', background: 'linear-gradient(90deg, #06b6d4, #10b981)', transition: 'width 0.4s ease' }} />
          </div>
        </div>

        {/* Val Accuracy */}
        <div className="glass-panel" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Val Accuracy</span>
            <TrendingUp size={14} color="#10b981" />
          </div>
          <div style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '0.35rem', color: 'var(--emerald-400)', fontFamily: 'var(--font-mono)' }}>
            {latest ? `${(latest.validation_accuracy * 100).toFixed(1)}%` : '--'}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>Global Test Split</div>
        </div>

        {/* Macro-F1 */}
        <div className="glass-panel" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Macro-F1</span>
            <ShieldCheck size={14} color="#06b6d4" />
          </div>
          <div style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '0.35rem', color: 'var(--cyan-400)', fontFamily: 'var(--font-mono)' }}>
            {latest ? `${(latest.validation_macro_f1 * 100).toFixed(1)}%` : '--'}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>Class-Balanced</div>
        </div>

        {/* Worst-Client Accuracy */}
        <div className="glass-panel" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Worst Client</span>
            <AlertTriangle size={14} color="#f59e0b" />
          </div>
          <div style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '0.35rem', color: 'var(--amber-400)', fontFamily: 'var(--font-mono)' }}>
            {latest ? `${(latest.worst_client_accuracy * 100).toFixed(1)}%` : '--'}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>Resilience Floor</div>
        </div>

        {/* Disparity & Comm */}
        <div className="glass-panel" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Traffic</span>
            <HardDriveUpload size={14} color="#c084fc" />
          </div>
          <div style={{ fontSize: '1.45rem', fontWeight: 800, marginTop: '0.35rem', color: 'var(--purple-400)', fontFamily: 'var(--font-mono)' }}>
            {commMB.toLocaleString()} MB
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
            Std: {latest ? `±${latest.client_accuracy_std.toFixed(3)}` : '--'}
          </div>
        </div>
      </div>

      {/* 1.5 Dedicated Run Directory Banner (Zero Overwrite Guarantee) */}
      {activeJob?.run_folder && (
        <div className="glass-panel" style={{
          padding: '0.9rem 1.25rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '0.75rem',
          borderLeft: '4px solid var(--color-emerald)',
          background: 'linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.4) 100%)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '36px',
              height: '36px',
              borderRadius: '8px',
              background: 'rgba(16, 185, 129, 0.15)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'var(--color-emerald)'
            }}>
              <Folder size={20} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-emerald)', fontWeight: 700 }}>
                  Active Run Output Directory
                </span>
                <span style={{
                  fontSize: '0.65rem',
                  padding: '0.1rem 0.45rem',
                  borderRadius: 'var(--radius-full)',
                  background: 'rgba(16, 185, 129, 0.2)',
                  color: 'var(--emerald-300)',
                  fontWeight: 600
                }}>
                  Zero Overwrite Guarantee
                </span>
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.88rem', color: '#fff', fontWeight: 600, marginTop: '0.15rem' }}>
                results/runs/{activeJob.run_folder}
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
            <span>Artifacts: <code>config.json</code> • <code>rounds.csv</code> • <code>summary.json</code> • <code>checkpoint.pt</code></span>
            {activeJob.status === 'COMPLETED' && (
              <a
                href={`/api/runs/${activeJob.run_folder}/checkpoint`}
                className="btn-emerald"
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem', textDecoration: 'none' }}
              >
                <Download size={13} />
                <span>Download Checkpoint</span>
              </a>
            )}
          </div>
        </div>
      )}

      {/* 2. Federated Animated Topology Map & Client Matrix */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.8fr', gap: '1.5rem' }}>
        
        {/* Topology Map */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Server size={18} color="#10b981" />
                <span>Federated Topology & Aggregation Map</span>
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                Private edge clients communicating model weights with central aggregator
              </p>
            </div>

            {status === 'RUNNING' && (
              <button
                onClick={onStopExperiment}
                className="btn-danger-outline"
              >
                <Square size={13} fill="currentColor" />
                <span>Halt</span>
              </button>
            )}
          </div>

          {/* SVG Federated Topology Diagram */}
          <div style={{
            flex: 1,
            minHeight: '260px',
            background: 'radial-gradient(circle, rgba(16,185,129,0.05) 0%, rgba(6,10,8,0.8) 70%)',
            border: '1px solid var(--border-subtle)',
            borderRadius: 'var(--radius-md)',
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            overflow: 'hidden',
          }}>
            <svg width="100%" height="260" viewBox="0 0 400 240">
              {/* Central Server Node */}
              <circle cx="200" cy="120" r="28" fill="#0d2818" stroke="#10b981" strokeWidth="2.5" />
              <circle cx="200" cy="120" r="36" fill="none" stroke="rgba(16,185,129,0.25)" strokeWidth="1.5" strokeDasharray="3 3">
                <animateTransform attributeName="transform" type="rotate" from="0 200 120" to="360 200 120" dur="15s" repeatCount="indefinite" />
              </circle>
              <text x="200" y="117" textAnchor="middle" fill="#f0fdf4" fontSize="10" fontWeight="700">SERVER</text>
              <text x="200" y="130" textAnchor="middle" fill="#34d399" fontSize="8" fontWeight="600">{activeJob?.algorithm?.toUpperCase() || 'FEDPROX'}</text>

              {/* Edge Clients around the server */}
              {Array.from({ length: numClients }).map((_, i) => {
                const angle = (i / numClients) * 2 * Math.PI - Math.PI / 2;
                const cx = 200 + 130 * Math.cos(angle);
                const cy = 120 + 80 * Math.sin(angle);
                const clientState = recentClients[i];
                const isTraining = status === 'RUNNING';

                return (
                  <g key={i}>
                    {/* Beam to Server */}
                    <line
                      x1="200" y1="120" x2={cx} y2={cy}
                      stroke={isTraining ? 'rgba(52,211,153,0.35)' : 'rgba(255,255,255,0.08)'}
                      strokeWidth="1.5"
                      strokeDasharray={isTraining ? "4 4" : "none"}
                    >
                      {isTraining && (
                        <animate attributeName="stroke-dashoffset" from="20" to="0" dur="1s" repeatCount="indefinite" />
                      )}
                    </line>

                    {/* Client Circle */}
                    <circle cx={cx} cy={cy} r="18" fill="#0b1712" stroke={isTraining ? '#34d399' : 'rgba(255,255,255,0.2)'} strokeWidth="2" />
                    <text x={cx} y={cy + 3} textAnchor="middle" fill="#f0fdf4" fontSize="9" fontWeight="700">C{i + 1}</text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Client Fleet Matrix Cards */}
        <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ fontSize: '1.1rem', marginBottom: '0.85rem' }}>
            Simulated Edge Client Fleet Status
          </h3>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: '0.85rem',
            overflowY: 'auto',
            maxHeight: '260px',
          }}>
            {Array.from({ length: numClients }).map((_, i) => {
              const cState = recentClients[i];
              const isUpdated = cState && cState.round === currentRound;

              return (
                <div
                  key={i}
                  style={{
                    background: 'rgba(8, 16, 12, 0.75)',
                    border: `1px solid ${isUpdated ? 'var(--emerald-500)' : 'var(--border-subtle)'}`,
                    borderRadius: 'var(--radius-md)',
                    padding: '0.85rem',
                    transition: 'all 0.3s ease',
                    boxShadow: isUpdated ? '0 0 15px -3px var(--emerald-glow)' : 'none',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700 }}>🚜 Client {i + 1}</span>
                    <span style={{
                      fontSize: '0.68rem',
                      padding: '0.1rem 0.4rem',
                      borderRadius: 'var(--radius-full)',
                      fontWeight: 600,
                      background: isUpdated ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.06)',
                      color: isUpdated ? 'var(--emerald-300)' : 'var(--text-tertiary)',
                    }}>
                      {isUpdated ? 'Active Fit' : 'Synced'}
                    </span>
                  </div>

                  <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Acc:</span>
                      <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>
                        {cState?.accuracy !== undefined ? `${(cState.accuracy * 100).toFixed(1)}%` : '--'}
                      </b>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Loss:</span>
                      <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>
                        {cState?.loss !== undefined ? cState.loss.toFixed(4) : '--'}
                      </b>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Private Data:</span>
                      <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>
                        {cState?.samples ? `${cState.samples} samples` : 'Dirichlet α'}
                      </b>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* 3. Dual Real-Time Charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
        
        {/* Convergence Chart */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <h4 style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <TrendingUp size={16} color="#10b981" />
              <span>Global Convergence (Val Accuracy & F1)</span>
            </h4>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>Isolated Val Set</span>
          </div>
          <div style={{ height: '260px' }}>
            <Line data={convergenceData} options={chartOptions} />
          </div>
        </div>

        {/* Robustness Chart */}
        <div className="glass-panel" style={{ padding: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <h4 style={{ fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <ShieldCheck size={16} color="#34d399" />
              <span>Client Robustness (Mean vs Worst & Drift)</span>
            </h4>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>Section 18 Metrics</span>
          </div>
          <div style={{ height: '260px' }}>
            <Line data={robustnessData} options={chartOptions} />
          </div>
        </div>
      </div>

      {/* 4. Live Event Stream Terminal */}
      <div className="glass-panel" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <h4 style={{ fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Terminal size={16} color="#06b6d4" />
            <span>Real-Time Simulation Telemetry Log</span>
          </h4>
          <span className="status-pill">Streaming</span>
        </div>

        <div style={{
          background: '#040705',
          border: '1px solid rgba(16, 185, 129, 0.2)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem',
          height: '180px',
          overflowY: 'auto',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.78rem',
          lineHeight: 1.6,
        }}>
          {(activeJob?.log_events || []).length === 0 ? (
            <div style={{ color: 'var(--text-tertiary)' }}>No events recorded yet. Ready to launch.</div>
          ) : (
            activeJob.log_events.map((evt, idx) => (
              <div key={idx} style={{ display: 'flex', gap: '0.75rem', marginBottom: '0.25rem' }}>
                <span style={{ color: 'var(--text-tertiary)' }}>[{evt.time}]</span>
                <span style={{ color: 'var(--emerald-400)', fontWeight: 700, minWidth: '90px' }}>
                  {evt.type}
                </span>
                <span style={{ color: '#e2e8f0' }}>{evt.message}</span>
              </div>
            ))
          )}
          <div ref={terminalEndRef} />
        </div>
      </div>
    </div>
  );
}
