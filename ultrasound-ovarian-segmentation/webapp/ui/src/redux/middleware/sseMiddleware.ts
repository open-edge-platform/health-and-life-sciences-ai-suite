// src/redux/middleware/sseMiddleware.ts — subscribe to the Flask /api/events
// SSE stream and fan the live snapshot into the pipeline slice.
import type { Middleware } from '@reduxjs/toolkit';
import { setSnapshot, resetPipeline } from '../slices/pipelineSlice';
import { startProcessing, stopProcessing } from '../slices/appSlice';

export const sseMiddleware: Middleware = (store) => {
  let es: EventSource | null = null;

  return (next) => (action: any) => {
    if (action?.type === 'sse/connect') {
      const url = action.payload?.url;
      if (!url) return next(action);
      if (es) { es.close(); es = null; }

      es = new EventSource(url);
      es.onmessage = (ev: MessageEvent) => {
        try {
          const snap = JSON.parse(ev.data);
          store.dispatch(setSnapshot(snap));
          const life = snap.lifecycle || snap.state;
          if (life === 'running' || life === 'loading') {
            store.dispatch(startProcessing());
          } else {
            store.dispatch(stopProcessing());
          }
        } catch {
          /* ignore malformed frames */
        }
      };
      es.onerror = () => {
        // EventSource reconnects automatically; nothing to do here.
      };
    }

    if (action?.type === 'sse/disconnect') {
      if (es) { es.close(); es = null; }
      store.dispatch(resetPipeline());
    }

    return next(action);
  };
};
