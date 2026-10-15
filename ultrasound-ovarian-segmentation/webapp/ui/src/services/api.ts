// src/services/api.ts — client for the Flask control plane of the ovarian app.
// All calls are same-origin under /api (Flask serves the built UI; the Vite dev
// server proxies /api to Flask — see vite.config.ts).
const BASE_URL = '/api';

export interface DeviceInfo {
  id: string;
  label: string;
  name: string;
}

export interface UseCase {
  key: string;
  label: string;
  modality?: string;
  task: string;
  model: string;
  input_shape: number[];
  precisions: string[];
  devices: string[];
  ir_available: Record<string, boolean>;
  kpi_latency_ms: number;
  kpi_fps: number;
  enabled: boolean;
  note: string;
}

export interface ReadinessResponse {
  ready: boolean;
  lifecycle?: string;
  devices: DeviceInfo[];
}

export interface Metrics {
  count: number;
  window: number;
  mean_ms: number;
  p50_ms: number;
  p90_ms: number;
  p95_ms: number;
  p99_ms: number;
  fps: number;
}

export interface StatusSnapshot {
  state: string;
  error: string | null;
  config: any;
  model_info: Record<string, string>;
  metrics: Metrics;
  platform?: { cpu_pct: number | null; gpu_pct: number | null; npu_pct: number | null };
  lifecycle?: string;
}

export interface StartRequest {
  usecase: string;
  device: string;
  precision: string;
  video?: string | null;
}

export interface PlatformInfo {
  Processor?: string;
  NPU?: string;
  iGPU?: string;
  Memory?: string;
  Storage?: string;
  OS?: string;
}

export interface ResourceMetrics {
  cpu_utilization: Array<[string, number]>;
  gpu_utilization: Array<[string, number]>;
  memory: Array<[string, number, number, number, number]>;
  power: Array<[string, ...number[]]>;
  npu_utilization: Array<[string, number]>;
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

export async function pingBackend(): Promise<boolean> {
  try {
    const r = await getJSON<ReadinessResponse>(`${BASE_URL}/readiness`);
    return r.ready === true;
  } catch {
    return false;
  }
}

export async function getReadiness(): Promise<ReadinessResponse> {
  return getJSON<ReadinessResponse>(`${BASE_URL}/readiness`);
}

export async function getUseCases(): Promise<UseCase[]> {
  return getJSON<UseCase[]>(`${BASE_URL}/usecases`);
}

export async function getDevices(): Promise<DeviceInfo[]> {
  return getJSON<DeviceInfo[]>(`${BASE_URL}/devices`);
}

export async function getStatusSnapshot(): Promise<StatusSnapshot> {
  return getJSON<StatusSnapshot>(`${BASE_URL}/status`);
}

export async function startPipeline(req: StartRequest): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${BASE_URL}/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `Start failed: ${res.status}`);
  }
  return data;
}

export async function stopPipeline(): Promise<void> {
  await fetch(`${BASE_URL}/stop`, { method: 'POST' });
}

export async function resetPipeline(): Promise<void> {
  await fetch(`${BASE_URL}/reset`, { method: 'POST' });
}

export function getEventsUrl(): string {
  return `${BASE_URL}/events`;
}

export async function getPlatformInfo(): Promise<PlatformInfo> {
  return getJSON<PlatformInfo>(`${BASE_URL}/platform-info`);
}

export async function getResourceMetrics(): Promise<ResourceMetrics> {
  return getJSON<ResourceMetrics>(`${BASE_URL}/hardware-metrics`);
}

export const api = {
  pingBackend,
  getReadiness,
  getUseCases,
  getDevices,
  getStatusSnapshot,
  start: startPipeline,
  stop: stopPipeline,
  reset: resetPipeline,
  getEventsUrl,
  getPlatformInfo,
  getResourceMetrics,
};

export default api;
