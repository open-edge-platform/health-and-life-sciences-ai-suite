// src/redux/slices/pipelineSlice.ts — selection + live pipeline snapshot.
import { createSlice } from '@reduxjs/toolkit';
import type { PayloadAction } from '@reduxjs/toolkit';
import type { DeviceInfo, UseCase, StatusSnapshot, Metrics } from '../../services/api';

export interface Selection {
  usecase: string;
  device: string;
  precision: string;
  video: string;
}

interface PipelineState {
  useCases: UseCase[];
  devices: DeviceInfo[];
  selection: Selection;
  lifecycle: string; // idle | loading | running | stopped | error
  error: string | null;
  metrics: Metrics;
  modelInfo: Record<string, string>;
  runningConfig: any | null;
}

const emptyMetrics: Metrics = {
  count: 0, window: 0, mean_ms: 0, p50_ms: 0, p90_ms: 0, p95_ms: 0, p99_ms: 0, fps: 0,
};

const initialState: PipelineState = {
  useCases: [],
  devices: [],
  selection: { usecase: '', device: '', precision: 'fp16', video: '' },
  lifecycle: 'idle',
  error: null,
  metrics: emptyMetrics,
  modelInfo: {},
  runningConfig: null,
};

const pipelineSlice = createSlice({
  name: 'pipeline',
  initialState,
  reducers: {
    setUseCases: (s, a: PayloadAction<UseCase[]>) => { s.useCases = a.payload; },
    setDevices: (s, a: PayloadAction<DeviceInfo[]>) => { s.devices = a.payload; },
    setSelection: (s, a: PayloadAction<Partial<Selection>>) => {
      s.selection = { ...s.selection, ...a.payload };
    },
    setSnapshot: (s, a: PayloadAction<StatusSnapshot>) => {
      const p = a.payload;
      s.lifecycle = p.lifecycle || p.state || 'idle';
      s.error = p.error ?? null;
      if (p.metrics) s.metrics = p.metrics;
      if (p.model_info) s.modelInfo = p.model_info;
      if (p.config) s.runningConfig = p.config;
    },
    resetPipeline: (s) => {
      s.lifecycle = 'idle';
      s.error = null;
      s.metrics = emptyMetrics;
      s.modelInfo = {};
      s.runningConfig = null;
    },
  },
});

export const { setUseCases, setDevices, setSelection, setSnapshot, resetPipeline } =
  pipelineSlice.actions;
export default pipelineSlice.reducer;
