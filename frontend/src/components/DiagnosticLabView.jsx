import React, { useState } from 'react';
import { 
  Sparkles, 
  Upload, 
  CheckCircle2, 
  AlertOctagon, 
  ShieldCheck, 
  Clock, 
  Layers, 
  Leaf, 
  Stethoscope, 
  Zap 
} from 'lucide-react';

export function DiagnosticLabView({ checkpoints, samples, onRunDiagnosis, isPredicting, predictionResult }) {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState(
    checkpoints[0]?.filename || 'fedprox_mobilenetv3_alpha_0.1_mu_0.01.pt'
  );
  const [selectedSample, setSelectedSample] = useState(samples[0]?.filename || null);
  const [previewUrl, setPreviewUrl] = useState(samples[0]?.url || null);
  const [uploadedFile, setUploadedFile] = useState(null);

  const handleSampleClick = (sample) => {
    setSelectedSample(sample.filename);
    setUploadedFile(null);
    setPreviewUrl(sample.url);
  };

  const handleFileUpload = (file) => {
    if (!file) return;
    setUploadedFile(file);
    setSelectedSample(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreviewUrl(e.target.result);
    reader.readAsDataURL(file);
  };

  const handleDiagnose = () => {
    onRunDiagnosis({
      file: uploadedFile,
      sampleFilename: selectedSample,
      checkpoint: selectedCheckpoint,
    });
  };

  const getSeverityBadgeClass = (severity) => {
    const s = (severity || '').toLowerCase();
    if (s.includes('critical')) return { bg: 'rgba(244, 63, 94, 0.2)', border: '#f43f5e', text: '#fb7185' };
    if (s.includes('high') || s.includes('severe')) return { bg: 'rgba(245, 158, 11, 0.2)', border: '#f59e0b', text: '#fcd34d' };
    if (s.includes('moderate')) return { bg: 'rgba(6, 182, 212, 0.2)', border: '#06b6d4', text: '#67e8f9' };
    return { bg: 'rgba(16, 185, 129, 0.2)', border: '#10b981', text: '#6ee7b7' };
  };

  return (
    <div className="animate-fade-in" style={{ display: 'grid', gridTemplateColumns: '1fr 1.25fr', gap: '1.75rem' }}>
      
      {/* Left Column: Image Selection & Controls */}
      <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        <div>
          <h2 style={{ fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Stethoscope size={20} color="#34d399" />
            <span>Plant Disease Diagnostic Lab</span>
          </h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
            Evaluate trained federated checkpoints on high-resolution leaf imagery.
          </p>
        </div>

        {/* Checkpoint Selector */}
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>
            Select Trained PyTorch Model Checkpoint
          </label>
          <select
            className="form-field-input"
            value={selectedCheckpoint}
            onChange={(e) => setSelectedCheckpoint(e.target.value)}
          >
            {checkpoints.map((ckpt) => (
              <option key={ckpt.filename} value={ckpt.filename}>
                {ckpt.filename} ({ckpt.size_mb} MB)
              </option>
            ))}
          </select>
        </div>

        {/* Upload Dropzone */}
        <div>
          <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.4rem' }}>
            Leaf Image Input
          </label>
          <div
            onClick={() => document.getElementById('diagnosticFileInput').click()}
            style={{
              border: '2px dashed var(--border-medium)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.5rem',
              textAlign: 'center',
              background: 'rgba(6, 12, 9, 0.6)',
              cursor: 'pointer',
              transition: 'all 0.2s ease',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              minHeight: '180px',
            }}
          >
            <input
              type="file"
              id="diagnosticFileInput"
              accept="image/*"
              style={{ display: 'none' }}
              onChange={(e) => handleFileUpload(e.target.files[0])}
            />

            {previewUrl ? (
              <div style={{ width: '100%', maxHeight: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', borderRadius: 'var(--radius-md)' }}>
                <img src={previewUrl} alt="Leaf Preview" style={{ maxWidth: '100%', maxHeight: '200px', objectFit: 'contain' }} />
              </div>
            ) : (
              <>
                <Upload size={32} color="#10b981" />
                <p style={{ fontSize: '0.88rem', fontWeight: 600, marginTop: '0.6rem' }}>
                  Click to browse or drop leaf image
                </p>
                <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '0.2rem' }}>
                  Accepts JPG, PNG RGB leaf photographs
                </p>
              </>
            )}
          </div>
        </div>

        {/* Quick Sample Selector Strip */}
        <div>
          <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.4rem', color: 'var(--text-secondary)' }}>
            Or Pick a Sample PlantVillage Leaf:
          </label>
          <div style={{
            display: 'flex',
            gap: '0.6rem',
            overflowX: 'auto',
            paddingBottom: '0.4rem',
          }}>
            {samples.map((s) => {
              const isSelected = selectedSample === s.filename;
              return (
                <img
                  key={s.filename}
                  src={s.url}
                  alt={s.title}
                  title={s.title}
                  onClick={() => handleSampleClick(s)}
                  style={{
                    width: '56px',
                    height: '56px',
                    borderRadius: 'var(--radius-sm)',
                    objectFit: 'cover',
                    cursor: 'pointer',
                    border: `2px solid ${isSelected ? 'var(--emerald-400)' : 'var(--border-subtle)'}`,
                    boxShadow: isSelected ? '0 0 12px var(--emerald-glow)' : 'none',
                    transform: isSelected ? 'scale(1.05)' : 'none',
                    transition: 'all 0.2s ease',
                    flexShrink: 0,
                  }}
                />
              );
            })}
          </div>
        </div>

        {/* Diagnose Trigger Button */}
        <button
          onClick={handleDiagnose}
          disabled={isPredicting || (!uploadedFile && !selectedSample)}
          className="btn-emerald"
          style={{ width: '100%', padding: '0.9rem', fontSize: '0.95rem', marginTop: '0.5rem' }}
        >
          {isPredicting ? (
            <>
              <Sparkles size={18} className="animate-spin" />
              <span>Running PyTorch Inference...</span>
            </>
          ) : (
            <>
              <Zap size={18} />
              <span>Diagnose Leaf Pathology</span>
            </>
          )}
        </button>
      </div>

      {/* Right Column: Diagnostic Assessment & Agronomic Advisory */}
      <div className="glass-panel" style={{ padding: '1.75rem', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', paddingBottom: '0.85rem', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Leaf size={19} color="#10b981" />
            <span>Pathology Diagnosis & Agronomic Advisory</span>
          </h3>

          {predictionResult && (
            <span className="status-pill" style={{ color: 'var(--cyan-400)' }}>
              <Clock size={12} />
              <span>{predictionResult.latency_ms} ms</span>
            </span>
          )}
        </div>

        {predictionResult ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            
            {/* Primary Diagnosis Header */}
            <div style={{
              background: 'rgba(10, 18, 14, 0.85)',
              border: '1px solid var(--border-medium)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.25rem',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
            }}>
              <div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Target Crop
                </span>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#f0fdf4' }}>
                  {predictionResult.top_prediction.crop}
                </div>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: 'var(--emerald-400)', marginTop: '0.15rem' }}>
                  {predictionResult.top_prediction.disease}
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                {(() => {
                  const s = getSeverityBadgeClass(predictionResult.advisory.severity);
                  return (
                    <span style={{
                      fontSize: '0.75rem',
                      fontWeight: 700,
                      padding: '0.25rem 0.75rem',
                      borderRadius: 'var(--radius-full)',
                      background: s.bg,
                      color: s.text,
                      border: `1px solid ${s.border}`,
                    }}>
                      Severity: {predictionResult.advisory.severity}
                    </span>
                  );
                })()}

                <div style={{
                  fontSize: '1.6rem',
                  fontWeight: 800,
                  color: 'var(--emerald-400)',
                  fontFamily: 'var(--font-mono)',
                  marginTop: '0.35rem',
                }}>
                  {predictionResult.top_prediction.percentage}
                </div>
              </div>
            </div>

            {/* Candidate Logits Probabilities */}
            <div>
              <label style={{ display: 'block', fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.6rem', color: 'var(--text-secondary)' }}>
                Candidate Predictions (Top Softmax Output):
              </label>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                {predictionResult.all_predictions.slice(0, 4).map((p, idx) => (
                  <div key={idx}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', marginBottom: '0.25rem' }}>
                      <span style={{ color: idx === 0 ? '#fff' : 'var(--text-secondary)' }}>
                        {p.crop} — {p.disease}
                      </span>
                      <b style={{ color: idx === 0 ? 'var(--emerald-300)' : 'var(--text-tertiary)', fontFamily: 'var(--font-mono)' }}>
                        {p.percentage}
                      </b>
                    </div>
                    <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: p.percentage,
                          background: idx === 0 ? 'linear-gradient(90deg, #059669, #10b981)' : 'rgba(255, 255, 255, 0.25)',
                          borderRadius: 'var(--radius-full)',
                          transition: 'width 0.5s ease',
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Agronomic Advisory Box */}
            <div style={{
              background: 'rgba(16, 185, 129, 0.06)',
              borderLeft: '4px solid var(--emerald-500)',
              borderRadius: '0 var(--radius-md) var(--radius-md) 0',
              padding: '1.15rem',
            }}>
              <h4 style={{ fontSize: '0.92rem', color: 'var(--emerald-300)', display: 'flex', alignItems: 'center', gap: '0.45rem', marginBottom: '0.4rem' }}>
                <ShieldCheck size={16} />
                <span>Pathology & Biological Management</span>
              </h4>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.5, marginBottom: '0.6rem' }}>
                {predictionResult.advisory.description}
              </p>
              <div style={{ fontSize: '0.86rem', color: '#f0fdf4', lineHeight: 1.5, fontWeight: 500 }}>
                <b>Recommended Treatment:</b> {predictionResult.advisory.remedy}
              </div>
            </div>

            <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', display: 'flex', justifyContent: 'space-between' }}>
              <span>Architecture: <b>{predictionResult.model_architecture.toUpperCase()}</b></span>
              <span>Loaded Checkpoint: <b>{predictionResult.checkpoint_used}</b></span>
            </div>
          </div>
        ) : (
          <div style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '3rem 1.5rem',
            textAlign: 'center',
            color: 'var(--text-tertiary)',
          }}>
            <Leaf size={48} strokeWidth={1.5} color="rgba(16, 185, 129, 0.3)" style={{ marginBottom: '1rem' }} />
            <h4 style={{ fontSize: '1.05rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Awaiting Leaf Imagery</h4>
            <p style={{ fontSize: '0.82rem', maxWidth: '340px', marginTop: '0.35rem', lineHeight: 1.4 }}>
              Select a sample leaf or upload your farm photo, pick a trained model checkpoint, and click <b>Diagnose Leaf Pathology</b> to evaluate in real time.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
