import React, { useState } from 'react';
import { useAppDispatch, useAppSelector } from '../../redux/hooks';
import { startProcessing, stopProcessing } from '../../redux/slices/appSlice';
import { api } from '../../services/api';
import '../../assets/css/TopPanel.css';

const TopPanel: React.FC = () => {
  const dispatch = useAppDispatch();
  const isProcessing = useAppSelector((s) => s.app.isProcessing);
  const selection = useAppSelector((s) => s.pipeline.selection);
  const useCases = useAppSelector((s) => s.pipeline.useCases);
  const lifecycle = useAppSelector((s) => s.pipeline.lifecycle);
  const error = useAppSelector((s) => s.pipeline.error);
  const [notification, setNotification] = useState('');
  const [busy, setBusy] = useState(false);

  const flash = (msg: string, ms = 4000) => {
    setNotification(msg);
    window.setTimeout(() => setNotification(''), ms);
  };

  const handleStart = async () => {
    if (busy || isProcessing) return;
    if (!selection.usecase || !selection.device) {
      flash('⚠️ Pick a device first');
      return;
    }
    const uc = useCases.find((u) => u.key === selection.usecase);
    if (uc && !uc.ir_available?.[selection.precision]) {
      const cmd = uc.task === 'ovarian_sam'
        ? '.\\prepare_model.ps1 -Arch sam'
        : '.\\prepare_model.ps1 -Train -Verify';
      flash(`❌ ${uc.label} isn't built on this box — run ${cmd} first`, 6000);
      return;
    }
    try {
      setBusy(true);
      dispatch(startProcessing());
      await api.start({
        usecase: selection.usecase,
        device: selection.device,
        precision: selection.precision,
        video: selection.video || null,
      });
      dispatch({ type: 'sse/connect', payload: { url: api.getEventsUrl() } });
      flash('✅ Running');
    } catch (e: any) {
      dispatch(stopProcessing());
      flash(`❌ ${e?.message || 'Start failed'}`, 6000);
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    if (busy || !isProcessing) return;
    try {
      setBusy(true);
      dispatch({ type: 'sse/disconnect' });
      dispatch(stopProcessing());
      await api.stop();
      flash('✅ Stopped');
    } catch (e: any) {
      flash(`❌ ${e?.message || 'Stop failed'}`);
    } finally {
      setBusy(false);
    }
  };

  const statusText = isProcessing
    ? (lifecycle === 'loading' ? 'Loading…' : 'Running')
    : lifecycle === 'error' ? 'Error'
      : lifecycle === 'stopped' ? 'Stopped' : 'Idle';
  const statusColor = isProcessing ? '#2e7d32' : lifecycle === 'error' ? '#c62828' : '#888';

  return (
    <div className="top-panel">
      <div className="action-buttons">
        <button onClick={handleStart} disabled={busy || isProcessing} className="start-button">
          {busy && !isProcessing ? '⏳ Starting…' : isProcessing ? '✅ Running' : '▶️ Start'}
        </button>
        <button
          onClick={handleStop}
          disabled={busy || !isProcessing}
          className="stop-button"
          title={!isProcessing ? 'No pipeline running' : 'Stop pipeline'}
        >
          ⏹ Stop
        </button>
      </div>

      <div className="notification-center">
        {notification && (
          <span style={{
            padding: '8px 16px',
            background: notification.includes('❌') ? '#fee' : notification.includes('⚠️') ? '#ffc' : '#efe',
            borderRadius: '4px',
            fontSize: '13px',
            border: `1px solid ${notification.includes('❌') ? '#fcc' : notification.includes('⚠️') ? '#fc6' : '#cfc'}`,
          }}>
            {notification}
          </span>
        )}
        {!notification && error && (
          <span style={{ padding: '8px 16px', background: '#fee', border: '1px solid #fcc', borderRadius: 4, fontSize: 13 }}>
            ❌ {error}
          </span>
        )}
      </div>

      <div className="spacer"></div>

      <div style={{ fontSize: 13, color: '#555', fontWeight: 600, paddingRight: 8 }}>
        Status: <span style={{ color: statusColor }}>{statusText}</span>
      </div>
    </div>
  );
};

export default TopPanel;
