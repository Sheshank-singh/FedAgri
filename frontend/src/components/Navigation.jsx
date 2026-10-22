import React from 'react';
import { Sliders, Activity, Layers, Sparkles, Database } from 'lucide-react';

export function Navigation({ activeTab, onSelectTab, isTrainingActive }) {
  const tabs = [
    { id: 'studio', label: 'Experiment Studio', icon: Sliders },
    { id: 'monitor', label: 'Live Telemetry & Fleet', icon: Activity, badge: isTrainingActive ? 'RUNNING' : 'LIVE' },
    { id: 'results', label: 'Benchmark Hub & Pareto', icon: Layers },
    { id: 'diagnostics', label: 'Crop Diagnostic Lab', icon: Sparkles, badge: 'PyTorch' },
    { id: 'dataset', label: 'PlantVillage & Partitions', icon: Database },
  ];

  return (
    <nav style={{
      background: 'rgba(8, 14, 11, 0.95)',
      borderBottom: '1px solid var(--border-subtle)',
      padding: '0.45rem 2rem',
      display: 'flex',
      gap: '0.5rem',
      overflowX: 'auto',
    }}>
      {tabs.map((tab) => {
        const Icon = tab.icon;
        const isActive = activeTab === tab.id;

        return (
          <button
            key={tab.id}
            onClick={() => onSelectTab(tab.id)}
            className={`tab-button ${isActive ? 'active' : ''}`}
          >
            <Icon size={16} />
            <span>{tab.label}</span>
            {tab.badge && (
              <span style={{
                background: isActive ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255, 255, 255, 0.08)',
                color: isActive ? 'var(--emerald-300)' : 'var(--text-tertiary)',
                fontSize: '0.65rem',
                fontWeight: 700,
                padding: '0.1rem 0.45rem',
                borderRadius: 'var(--radius-full)',
                marginLeft: '0.2rem',
                letterSpacing: '0.04em',
              }}>
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}
