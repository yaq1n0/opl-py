const BASE = "/api";

export interface LifterSummary {
  name: string;
  competition_count: number;
  best_total_kg: number | null;
}

export interface EntryOut {
  date: string;
  federation: string;
  meet_name: string;
  equipment: string;
  event: string;
  bodyweight_kg: number | null;
  weight_class_kg: string | null;
  best3_squat_kg: number | null;
  best3_bench_kg: number | null;
  best3_deadlift_kg: number | null;
  total_kg: number | null;
  place: string;
  dots: number | null;
  wilks: number | null;
  age: number | null;
  tested: boolean | null;
}

export interface LifterDetail {
  name: string;
  competition_count: number;
  best_total_kg: number | null;
  best_squat_kg: number | null;
  best_bench_kg: number | null;
  best_deadlift_kg: number | null;
  entries: EntryOut[];
}

export interface PredictionOut {
  next_total_kg: number | null;
  next_squat_kg: number | null;
  next_bench_kg: number | null;
  next_deadlift_kg: number | null;
  confidence_interval: [number, number] | null;
  trajectory_curve: [number, number][];
  target_date: string | null;
  target_bodyweight_kg: number | null;
  approach: string;
}

export interface Approach {
  name: string;
  display_name: string;
  description: string;
  available: boolean;
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export function searchLifters(query: string): Promise<LifterSummary[]> {
  return fetchJSON(`${BASE}/search?q=${encodeURIComponent(query)}`);
}

export function getLifter(name: string): Promise<LifterDetail> {
  return fetchJSON(`${BASE}/lifter/${encodeURIComponent(name)}`);
}

export function fetchApproaches(): Promise<Approach[]> {
  return fetchJSON(`${BASE}/approaches`);
}

export function predict(
  lifterName: string,
  targetDate?: string,
  targetBodyweightKg?: number,
  approach?: string,
): Promise<PredictionOut> {
  return fetchJSON(`${BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lifter_name: lifterName,
      target_date: targetDate || null,
      target_bodyweight_kg: targetBodyweightKg || null,
      approach: approach || "gradient_boosting",
    }),
  });
}
