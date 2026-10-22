import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Navigation } from './components/Navigation';
import { StudioView } from './components/StudioView';
import { LiveMonitorView } from './components/LiveMonitorView';
import { BenchmarkView } from './components/BenchmarkView';
import { DiagnosticLabView } from './components/DiagnosticLabView';
import { DatasetExplorerView } from './components/DatasetExplorerView';
import { api } from './services/api';
import { wsManager } from './services/websocket';
import { X, Maximize2 } from 'lucide-react';

export function App() {
  const [activeTab, setActiveTab] = useState('studio');
  const [health, setHealth] = useState(null);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [models, setModels] = useState([]);
  const [datasetSummary, setDatasetSummary] = useState(null);
  const [classes, setClasses] = useState([]);
  const [samples, setSamples] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const [figures, setFigures] = useState([]);
  const [activeJob, setActiveJob] = useState(null);
  const [isLaunching, setIsLaunching] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [modalFigure, setModalFigure] = useState(null);

  // Initialize data on mount
  useEffect(() => {
    async function loadInitialData() {
      try {
        const [h, m, ds, cls, s, exp, ckpt, fig, act] = await Promise.all([
          api.getHealth().catch(() => null),
          api.getModels().catch(() => []),
          api.getDatasetSummary().catch(() => null),
          api.getClasses().catch(() => []),
          api.getSampleImages().catch(() => []),
          api.getExperiments().catch(() => []),
          api.getCheckpoints().catch(() => []),
          api.getFigures().catch(() => []),
          api.getActiveExperiment().catch(() => ({ active: false })),
        ]);

        if (h) setHealth(h);
        if (m) setModels(m);
        if (ds) setDatasetSummary(ds);
        if (cls) setClasses(cls);
        if (s) setSamples(s);
        if (exp) setExperiments(exp);
        if (ckpt) setCheckpoints(ckpt);
        if (fig) setFigures(fig);
        if (act?.active && act.job) {
          setActiveJob(act.job);
        }
      } catch (err) {
        console.error('Initial data load error:', err);
      }
    }

    loadInitialData();

    // Connect WebSocket
    wsManager.connect();
    const unsubStatus = wsManager.subscribeStatus(setIsWsConnected);

    const unsubEvents = wsManager.subscribe((payload) => {
      const type = payload.type;
      const data = payload.data;

      if (type === 'INITIAL_STATE' && payload.active_job) {
        setActiveJob(payload.active_job);
      } else if (type === 'ROUND_START') {
        setActiveJob((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            status: 'RUNNING',
            current_round: data.round,
            total_rounds: data.total_rounds,
            log_events: [
              ...(prev.log_events || []),
              { time: new Date().toLocaleTimeString(), type: 'ROUND', message: data.message },
            ],
          };
        });
      } else if (type === 'CLIENT_FIT_COMPLETE') {
        setActiveJob((prev) => {
          if (!prev) return prev;
          const updatedClients = { ...(prev.recent_client_states || {}) };
          updatedClients[data.client_id] = {
            client_id: data.client_id,
            round: data.round,
            loss: data.loss,
            accuracy: data.accuracy,
            samples: data.sample_count,
            updated_at: Date.now(),
          };
          return {
            ...prev,
            recent_client_states: updatedClients,
            log_events: [
              ...(prev.log_events || []),
              { time: new Date().toLocaleTimeString(), type: `CLIENT ${data.client_id + 1}`, message: data.message },
            ],
          };
        });
      } else if (type === 'ROUND_COMPLETE') {
        setActiveJob((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            history: [...(prev.history || []), data.round_data],
            log_events: [
              ...(prev.log_events || []),
              { time: new Date().toLocaleTimeString(), type: 'EVAL', message: data.message },
            ],
          };
        });
      } else if (type === 'EXPERIMENT_COMPLETE') {
        setActiveJob((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            status: 'COMPLETED',
            summary: data.summary,
            log_events: [
              ...(prev.log_events || []),
              { time: new Date().toLocaleTimeString(), type: 'COMPLETE', message: data.message },
            ],
          };
        });
        // Refresh benchmarks and checkpoints
        api.getExperiments().then(setExperiments).catch(() => {});
        api.getCheckpoints().then(setCheckpoints).catch(() => {});
      } else if (type === 'EXPERIMENT_CANCELLED') {
        setActiveJob((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            status: 'CANCELLED',
            log_events: [
              ...(prev.log_events || []),
              { time: new Date().toLocaleTimeString(), type: 'CANCEL', message: data.message },
            ],
          };
        });
      } else if (type === 'EXPERIMENT_ERROR') {
        setActiveJob((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            status: 'FAILED',
            error: data.error,
            log_events: [
              ...(prev.log_events || []),
              { time: new Date().toLocaleTimeString(), type: 'ERROR', message: data.message },
            ],
          };
        });
      }
    });

    return () => {
      unsubStatus();
      unsubEvents();
    };
  }, []);

  // Launch Experiment
  const handleLaunchExperiment = async (config) => {
    setIsLaunching(true);
    try {
      const resp = await api.launchExperiment(config);
      setIsLaunching(false);
      setActiveTab('monitor');
      // Create initial active job representation
      setActiveJob({
        job_id: resp.job_id,
        experiment_id: resp.experiment_id,
        model_name: config.model,
        algorithm: config.algorithm,
        alpha: config.alpha,
        is_iid: config.is_iid,
        mu: config.mu,
        num_clients: config.num_clients,
        num_rounds: config.num_rounds,
        status: 'RUNNING',
        current_round: 0,
        total_rounds: config.num_rounds,
        history: [],
        recent_client_states: {},
        log_events: [
          { time: new Date().toLocaleTimeString(), type: 'LAUNCH', message: `Experiment started for ${config.model.toUpperCase()}` },
        ],
      });
    } catch (err) {
      alert(`Launch error: ${err.message}`);
      setIsLaunching(false);
    }
  };

  // Stop Experiment
  const handleStopExperiment = async () => {
    if (confirm('Halt the active simulation between rounds?')) {
      try {
        await api.stopExperiment();
      } catch (err) {
        alert(err.message);
      }
    }
  };

  // Run Diagnosis
  const handleRunDiagnosis = async ({ file, sampleFilename, checkpoint }) => {
    setIsPredicting(true);
    try {
      const result = await api.predictLeaf(file, sampleFilename, checkpoint);
      setPredictionResult(result);
    } catch (err) {
      alert(`Diagnosis failed: ${err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  const isTrainingActive = activeJob?.status === 'RUNNING';

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Top Header */}
      <Header health={health} isWsConnected={isWsConnected} />

      {/* Navigation Tab Bar */}
      <Navigation
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isTrainingActive={isTrainingActive}
      />

      {/* Main Workspaces */}
      <main style={{ flex: 1, padding: '2rem', maxWidth: '1600px', margin: '0 auto', width: '100%' }}>
        {activeTab === 'studio' && (
          <StudioView
            models={models}
            onLaunchExperiment={handleLaunchExperiment}
            isLaunching={isLaunching}
          />
        )}

        {activeTab === 'monitor' && (
          <LiveMonitorView
            activeJob={activeJob}
            onStopExperiment={handleStopExperiment}
          />
        )}

        {activeTab === 'results' && (
          <BenchmarkView
            experiments={experiments}
            figures={figures}
            checkpoints={checkpoints}
            onOpenFigureModal={setModalFigure}
          />
        )}

        {activeTab === 'diagnostics' && (
          <DiagnosticLabView
            checkpoints={checkpoints}
            samples={samples}
            onRunDiagnosis={handleRunDiagnosis}
            isPredicting={isPredicting}
            predictionResult={predictionResult}
          />
        )}

        {activeTab === 'dataset' && (
          <DatasetExplorerView
            summary={datasetSummary}
            classes={classes}
          />
        )}
      </main>

      {/* High-Resolution Scientific Figure Zoom Modal */}
      {modalFigure && (
        <div
          onClick={() => setModalFigure(null)}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.88)',
            backdropFilter: 'blur(10px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
          }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            style={{
              background: '#09120e',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-xl)',
              maxWidth: '920px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              padding: '1.75rem',
              position: 'relative',
              boxShadow: '0 20px 50px rgba(0,0,0,0.8)',
            }}
          >
            <button
              onClick={() => setModalFigure(null)}
              style={{
                position: 'absolute',
                top: '1.25rem',
                right: '1.25rem',
                background: 'rgba(255, 255, 255, 0.08)',
                border: 'none',
                color: '#fff',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <X size={16} />
            </button>

            <h3 style={{ fontSize: '1.2rem', marginBottom: '0.4rem' }}>{modalFigure.title}</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
              {modalFigure.description}
            </p>

            <div style={{ background: '#000', borderRadius: 'var(--radius-md)', padding: '0.5rem', textAlign: 'center' }}>
              <img
                src={modalFigure.url}
                alt={modalFigure.title}
                style={{ maxWidth: '100%', maxHeight: '65vh', objectFit: 'contain' }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
