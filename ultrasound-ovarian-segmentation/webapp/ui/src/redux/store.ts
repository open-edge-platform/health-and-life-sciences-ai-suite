// src/redux/store.ts
import { configureStore } from '@reduxjs/toolkit';
import appReducer from './slices/appSlice';
import metricsReducer from './slices/metricsSlice';
import pipelineReducer from './slices/pipelineSlice';
import { sseMiddleware } from './middleware/sseMiddleware';

export const store = configureStore({
  reducer: {
    app: appReducer,
    metrics: metricsReducer,
    pipeline: pipelineReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
      immutableCheck: false,
    }).concat(sseMiddleware),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
