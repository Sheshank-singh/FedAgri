import React, { useState, useEffect } from 'react';
import { Database, Search, Filter, Sprout, BarChart3, Layers, CheckCircle, AlertCircle } from 'lucide-react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend
} from 'chart.js';
import { api } from '../services/api';

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

export function DatasetExplorerView({ summary, classes }) {
  const [selectedAlpha, setSelectedAlpha] = useState('0.1');
  const [partitionData, setPartitionData] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [cropFilter, setCropFilter] = useState('all');

  useEffect(() => {
    async function loadPartition() {
      try {
        const isIid = selectedAlpha === 'iid';
        const data = await api.getPartitionDistribution(isIid ? 0.1 : parseFloat(selectedAlpha), isIid);
        setPartitionData(data);
      } catch (e) {
        console.error('Error loading partition data:', e);
      }
    }
    loadPartition();
  }, [selectedAlpha]);

  const crops = summary?.crops || [];

  const filteredClasses = classes.filter((c) => {
    const matchesSearch = c.crop.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          c.disease.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCrop = cropFilter === 'all' || c.crop.toLowerCase() === cropFilter.toLowerCase();
    return matchesSearch && matchesCrop;
  });

  const chartData = {
    labels: partitionData?.client_sample_counts
      ? Object.keys(partitionData.client_sample_counts).map((cid) => `Client ${parseInt(cid, 10) + 1}`)
      : [],
    datasets: [
      {
        label: 'Assigned Training Images',
        data: partitionData?.client_sample_counts
          ? Object.values(partitionData.client_sample_counts)
          : [],
        backgroundColor: [
          'rgba(16, 185, 129, 0.7)',
          'rgba(6, 182, 212, 0.7)',
          'rgba(245, 158, 11, 0.7)',
          'rgba(168, 85, 247, 0.7)',
          'rgba(244, 63, 94, 0.7)',
        ],
        borderColor: ['#10b981', '#06b6d4', '#f59e0b', '#a855f7', '#f43f5e'],
        borderWidth: 1.5,
        borderRadius: 8,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
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
        ticks: { color: '#94a3b8', font: { size: 11 } },
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
        title: { display: true, text: 'Sample Count', color: '#64748b' },
      },
    },
  };

  return (
    <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      
      {/* 1. Dataset Top Summary Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1.25rem',
      }}>
        <div className="glass-panel" style={{ padding: '1.35rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Total Images</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--emerald-400)', fontFamily: 'var(--font-mono)', marginTop: '0.2rem' }}>
            {summary?.total_images?.toLocaleString() || '54,305'}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>PlantVillage Color Dataset</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.35rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Pathology Classes</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--cyan-400)', fontFamily: 'var(--font-mono)', marginTop: '0.2rem' }}>
            {summary?.num_classes || 38}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>Healthy & Diseased Conditions</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.35rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Crops Represented</div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--amber-400)', fontFamily: 'var(--font-mono)', marginTop: '0.2rem' }}>
            {summary?.num_crops || 14}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>Apple, Corn, Tomato, Grape, etc.</div>
        </div>

        <div className="glass-panel" style={{ padding: '1.35rem' }}>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Split Protocol</div>
          <div style={{ fontSize: '1.35rem', fontWeight: 700, color: '#f0fdf4', marginTop: '0.4rem' }}>
            70% / 15% / 15%
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>Train / Val / Isolated Test</div>
        </div>
      </div>

      {/* 2. Non-IID Dirichlet Partition Skew Visualizer */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <BarChart3 size={19} color="#34d399" />
              <span>Dirichlet Non-IID Client Data Skew (α)</span>
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Samples partitioned across simulated client nodes using fixed Dirichlet random seeds.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Distribution Level:</span>
            <select
              className="form-field-input"
              value={selectedAlpha}
              onChange={(e) => setSelectedAlpha(e.target.value)}
              style={{ width: '200px', padding: '0.4rem 0.8rem', fontSize: '0.82rem' }}
            >
              <option value="0.1">Severe Non-IID (α = 0.1)</option>
              <option value="0.5">Strong Non-IID (α = 0.5)</option>
              <option value="1.0">Moderate Non-IID (α = 1.0)</option>
              <option value="10.0">Near-IID (α = 10.0)</option>
              <option value="iid">Uniform IID Partition</option>
            </select>
          </div>
        </div>

        <div style={{ height: '240px' }}>
          <Bar data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* 3. Searchable 38-Class Pathology Catalog */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Sprout size={19} color="#10b981" />
              <span>PlantVillage 38 Pathologies Catalog</span>
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
              Browse crop pathologies, biological causal agents, and health classifications.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', flexWrap: 'wrap' }}>
            {/* Crop Filter Dropdown */}
            <select
              className="form-field-input"
              value={cropFilter}
              onChange={(e) => setCropFilter(e.target.value)}
              style={{ width: '160px', padding: '0.45rem 0.75rem', fontSize: '0.82rem' }}
            >
              <option value="all">All 14 Crops</option>
              {crops.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>

            {/* Search Input */}
            <div style={{ position: 'relative', width: '220px' }}>
              <Search size={14} style={{ position: 'absolute', left: '10px', top: '10px', color: 'var(--text-tertiary)' }} />
              <input
                type="text"
                placeholder="Search diseases, blights..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="form-field-input"
                style={{ padding: '0.45rem 0.75rem 0.45rem 2rem', fontSize: '0.82rem' }}
              />
            </div>
          </div>
        </div>

        {/* Classes Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
          gap: '1rem',
          maxHeight: '420px',
          overflowY: 'auto',
          paddingRight: '0.25rem',
        }}>
          {filteredClasses.map((cls) => (
            <div
              key={cls.idx}
              style={{
                background: 'rgba(8, 14, 11, 0.75)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '1rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.4rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.92rem', fontWeight: 700, color: '#f0fdf4' }}>{cls.crop}</span>
                <span style={{
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  padding: '0.15rem 0.5rem',
                  borderRadius: 'var(--radius-full)',
                  background: cls.is_healthy ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                  color: cls.is_healthy ? 'var(--emerald-300)' : 'var(--rose-400)',
                  border: `1px solid ${cls.is_healthy ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
                }}>
                  {cls.is_healthy ? 'Healthy' : 'Diseased'}
                </span>
              </div>

              <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
                {cls.disease}
              </div>

              <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                Class Index: <b style={{ color: '#fff', fontFamily: 'var(--font-mono)' }}>#{cls.idx}</b>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
