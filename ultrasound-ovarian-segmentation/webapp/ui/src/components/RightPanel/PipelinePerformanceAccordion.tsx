import React from 'react';
import { useAppSelector } from '../../redux/hooks';
import Accordion from '../common/Accordion';
import '../../assets/css/RightPanel.css';

const DEVICE_COLORS: Record<string, string> = {
  GPU: '#1565c0',
  CPU: '#2e7d32',
  NPU: '#6a1b9a',
};

const StatCard: React.FC<{ label: string; value: string; sub?: string }> = ({ label, value, sub }) => (
  <div style={{
    flex: '1 1 0',
    minWidth: 92,
    background: '#fff',
    border: '1px solid #e6e7e8',
    borderRadius: 8,
    padding: '10px 12px',
  }}>
    <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.4, textTransform: 'uppercase', color: '#8a9099' }}>{label}</div>
    <div style={{ fontSize: 20, fontWeight: 700, color: '#24292f', lineHeight: 1.3 }}>{value}</div>
    {sub && <div style={{ fontSize: 10, color: '#8a9099' }}>{sub}</div>}
  </div>
);

export function PipelinePerformanceAccordion() {
  const lifecycle = useAppSelector((s) => s.pipeline.lifecycle);
  const metrics = useAppSelector((s) => s.pipeline.metrics);
  const runningConfig = useAppSelector((s) => s.pipeline.runningConfig);
  const selection = useAppSelector((s) => s.pipeline.selection);
  const useCases = useAppSelector((s) => s.pipeline.useCases);

  const uc = useCases.find((u) => u.key === selection.usecase) || useCases[0];
  const running = lifecycle === 'running' || lifecycle === 'loading';

  const device = runningConfig?.device || selection.device || 'CPU';
  const precision = (runningConfig?.precision || selection.precision || 'fp16').toUpperCase();
  const model = runningConfig?.model || uc?.model || 'UltraSAM';
  const label = runningConfig?.label || uc?.label || 'Ovarian Tumor Segmentation';
  const kpiLat = runningConfig?.kpi_latency_ms ?? uc?.kpi_latency_ms ?? 0;
  const kpiFps = runningConfig?.kpi_fps ?? uc?.kpi_fps ?? 0;

  const hasData = metrics.window > 0;
  const fps = metrics.fps;
  const mean = metrics.mean_ms;
  const devColor = DEVICE_COLORS[device] || '#555';

  const statusInfo = lifecycle === 'error'
    ? { color: '#f44336', label: 'Error' }
    : running
      ? { color: '#4caf50', label: lifecycle === 'loading' ? 'Loading' : 'Running' }
      : { color: '#9e9e9e', label: 'Idle' };

  const thStyle: React.CSSProperties = {
    padding: '8px 10px', color: '#fff', fontWeight: 600, fontSize: '11px',
    textTransform: 'uppercase', letterSpacing: '0.4px', textAlign: 'left', border: '1px solid #888',
  };
  const cellStyle: React.CSSProperties = { padding: '8px 10px', border: '1px solid #bbb', verticalAlign: 'middle' };

  return (
    <Accordion title="Pipeline Performance" defaultOpen>
      <div className="pipeline-perf">
        {/* Live stat tiles */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
          <StatCard label="FPS" value={hasData ? fps.toFixed(1) : '—'} sub={`target ≥ ${kpiFps}`} />
          <StatCard label="Latency" value={hasData ? `${mean.toFixed(1)} ms` : '—'} sub={`target ≤ ${kpiLat} ms`} />
          <StatCard label="P90" value={hasData ? `${metrics.p90_ms.toFixed(1)} ms` : '—'} />
          <StatCard label="P99" value={hasData ? `${metrics.p99_ms.toFixed(1)} ms` : '—'} />
        </div>

        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', border: '2px solid #888' }}>
          <thead>
            <tr style={{ background: '#3a3f47' }}>
              <th style={thStyle}>Workload</th>
              <th style={thStyle}>Model</th>
              <th style={thStyle}>Device</th>
              <th style={thStyle}>Status</th>
            </tr>
          </thead>
          <tbody>
            <tr style={{ background: '#fff' }}>
              <td style={{ ...cellStyle, fontWeight: 500, color: '#24292f' }}>{label}</td>
              <td style={{ ...cellStyle, fontSize: '10px', color: '#888', fontFamily: 'monospace' }}>{model} · {precision}</td>
              <td style={cellStyle}>
                <span style={{
                  display: 'inline-block', padding: '2px 10px', border: '1px solid',
                  borderRadius: '10px', fontFamily: 'monospace', fontWeight: 700, fontSize: '10px',
                  backgroundColor: devColor + '14', color: devColor, borderColor: devColor + '40',
                }}>{device}</span>
              </td>
              <td style={cellStyle}>
                <span style={{
                  display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
                  marginRight: 6, verticalAlign: 'middle', backgroundColor: statusInfo.color,
                }} />
                <span style={{ fontSize: 11, color: '#555', fontWeight: 500 }}>{statusInfo.label}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </Accordion>
  );
}

export default PipelinePerformanceAccordion;
