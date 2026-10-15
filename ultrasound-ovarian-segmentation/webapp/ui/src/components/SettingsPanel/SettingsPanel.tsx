import React, { useEffect } from 'react';
import { useAppDispatch, useAppSelector } from '../../redux/hooks';
import { setUseCases, setDevices, setSelection } from '../../redux/slices/pipelineSlice';
import { api } from '../../services/api';
import Accordion from '../common/Accordion';
import '../../components/ConfigModal/ConfigModal.css';

// Left-panel settings for the ovarian pipeline. Replaces the NICU video panel:
// the inference video is shown in a separate OpenCV pop-up window on start.
const SettingsPanel: React.FC = () => {
  const dispatch = useAppDispatch();
  const useCases = useAppSelector((s) => s.pipeline.useCases);
  const devices = useAppSelector((s) => s.pipeline.devices);
  const selection = useAppSelector((s) => s.pipeline.selection);
  const isProcessing = useAppSelector((s) => s.app.isProcessing);

  // Load use-cases + devices once and seed the default selection.
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [ucs, devs] = await Promise.all([api.getUseCases(), api.getDevices()]);
        if (!active) return;
        dispatch(setUseCases(ucs));
        dispatch(setDevices(devs));
        const uc = ucs.find((u) => u.enabled) || ucs[0];
        if (uc) {
          dispatch(setSelection({
            usecase: uc.key,
            device: uc.devices[0] || 'CPU',
            precision: uc.precisions[0] || 'fp16',
          }));
        }
      } catch {
        /* backend not ready yet; App gate handles retries */
      }
    })();
    return () => { active = false; };
  }, [dispatch]);

  const uc = useCases.find((u) => u.key === selection.usecase) || useCases[0];
  const deviceLabel = (id: string) => devices.find((d) => d.id === id)?.label || id;
  const selReady = !!uc?.ir_available?.[selection.precision];
  const buildCmd = uc?.task === 'ovarian_sam'
    ? '.\\prepare_model.ps1 -Arch sam'
    : '.\\prepare_model.ps1 -Train -Verify';

  if (!uc) {
    return (
      <div className="config-section">
        <div className="config-loading">Loading configuration…</div>
      </div>
    );
  }

  return (
    <Accordion title="Configuration" defaultOpen>
      <div className="config-section">
        {/* Model / use-case — selected at launch on the CLI (run_webapp.ps1
            -UseCase); read-only here. The Start button runs this model. */}
        <div className="config-field-group">
          <label className="config-label">Model</label>
          <div className="config-active-video">
            <span className="config-video-badge">{uc.label}</span>
          </div>
        </div>

        {uc.enabled && !selReady && (
          <p className="config-hint" style={{ color: '#c62828', marginBottom: 12 }}>
            ⚠️ This model isn't built on this box yet. Build it with{' '}
            <code>{buildCmd}</code> (see Model Preparation), then Start.
          </p>
        )}

        {/* Device */}
        <div className="config-field-group">
          <label className="config-label">Device</label>
          <select
            className="config-select"
            style={{ width: '100%' }}
            value={selection.device}
            disabled={isProcessing}
            onChange={(e) => dispatch(setSelection({ device: e.target.value }))}
          >
            {uc.devices.map((d) => (
              <option key={d} value={d}>{deviceLabel(d)}</option>
            ))}
          </select>
        </div>

        {/* Precision */}
        <div className="config-field-group">
          <label className="config-label">Precision</label>
          <select
            className="config-select"
            style={{ width: '100%' }}
            value={selection.precision}
            disabled={isProcessing}
            onChange={(e) => dispatch(setSelection({ precision: e.target.value }))}
          >
            {uc.precisions.map((p) => (
              <option key={p} value={p}>{p.toUpperCase()}</option>
            ))}
          </select>
        </div>

        {/* Video source */}
        <div className="config-field-group">
          <label className="config-label">Video / image source</label>
          <input
            className="config-input"
            type="text"
            placeholder="leave blank to use the sample dataset"
            value={selection.video}
            disabled={isProcessing}
            onChange={(e) => dispatch(setSelection({ video: e.target.value }))}
          />
          <p className="config-hint" style={{ marginTop: 8 }}>
            Inference opens in a separate pop-up window (Original | Segmentation).
            Press <strong>ESC</strong> in that window to close it. Use <strong>Start</strong> in
            the top bar to run.
          </p>
        </div>
      </div>
    </Accordion>
  );
};

export default SettingsPanel;
