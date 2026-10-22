import React from 'react';
import { Cpu, Activity, Sprout } from 'lucide-react';

export function Header({ health, isWsConnected }) {
  const isCuda = health?.cuda_available;
  const hardwareLabel = isCuda
    ? `GPU: ${health.device_name || 'NVIDIA CUDA'}`
    : `CPU Context (${health?.platform?.toUpperCase() || 'HOST'})`;

  return (
    <header style={{
      position: 'sticky',
      top: 0,
      zIndex: 50,
      backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      background: 'rgba(5, 8, 7, 0.88)',
      borderBottom: '1px solid var(--border-subtle)',
      padding: '0.85rem 2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
    }}>
      {/* Brand Identity */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div style={{
          width: '42px',
          height: '42px',
          borderRadius: 'var(--radius-md)',
          background: 'linear-gradient(135deg, #059669 0%, #10b981 60%, #06b6d4 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 20px -2px var(--emerald-glow)',
          color: '#fff',
        }}>
          <Sprout size={24} strokeWidth={2.2} />
        </div>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <h1 className="gradient-text-emerald" style={{ fontSize: '1.35rem', lineHeight: 1.1 }}>
              FedAgri
            </h1>
            <span style={{
              background: 'rgba(16, 185, 129, 0.15)',
              border: '1px solid rgba(16, 185, 129, 0.3)',
              color: 'var(--emerald-300)',
              fontSize: '0.68rem',
              fontWeight: 700,
              padding: '0.15rem 0.5rem',
              borderRadius: 'var(--radius-full)',
              letterSpacing: '0.05em',
            }}>
              RESEARCH PLATFORM
            </span>
          </div>
          <p style={{
            fontSize: '0.72rem',
            color: 'var(--text-tertiary)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            fontWeight: 600,
            marginTop: '0.15rem',
          }}>
            Robust Federated Learning for Precision Crop Pathology
          </p>
        </div>
      </div>

      {/* System Status Indicators */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {/* Hardware Status Pill */}
        <div className="status-pill" style={{
          background: isCuda ? 'rgba(16, 185, 129, 0.08)' : 'rgba(255, 255, 255, 0.04)',
          borderColor: isCuda ? 'rgba(16, 185, 129, 0.25)' : 'var(--border-subtle)',
          color: isCuda ? 'var(--emerald-300)' : 'var(--text-secondary)'
        }}>
          <Cpu size={14} color={isCuda ? '#34d399' : '#94a3b8'} />
          <span>{hardwareLabel}</span>
        </div>

        {/* WebSocket Connection Pill */}
        <div className="status-pill" style={{
          background: isWsConnected ? 'rgba(6, 182, 212, 0.08)' : 'rgba(244, 63, 94, 0.08)',
          borderColor: isWsConnected ? 'rgba(6, 182, 212, 0.3)' : 'rgba(244, 63, 94, 0.3)',
          color: isWsConnected ? 'var(--cyan-400)' : 'var(--rose-400)',
        }}>
          <span className="pulsing-indicator" style={{
            backgroundColor: isWsConnected ? 'var(--cyan-500)' : 'var(--rose-500)',
            boxShadow: isWsConnected ? '0 0 8px var(--cyan-500)' : '0 0 8px var(--rose-500)',
            animation: isWsConnected ? 'radar-pulse 2s infinite' : 'none',
          }} />
          <Activity size={13} />
          <span>{isWsConnected ? 'Live Telemetry Active' : 'Connecting Stream...'}</span>
        </div>
      </div>
    </header>
  );
}
