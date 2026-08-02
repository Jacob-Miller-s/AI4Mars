import type { Overview, Provenance, RunCard, RunDetail, SampleRecord } from "./types";

export const apiBase = import.meta.env.VITE_API_BASE ?? "/api";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBase}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}): ${path}`);
  }
  return response.json() as Promise<T>;
}

export function getOverview(): Promise<Overview> {
  return request<Overview>("/overview");
}

export function getRuns(): Promise<{ runs: RunCard[]; warnings: string[] }> {
  return request<{ runs: RunCard[]; warnings: string[] }>("/runs?limit=250");
}

export function getRun(runId: string): Promise<RunDetail> {
  return request<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
}

export function runStreamUrl(runId: string, after: number): string {
  return `${apiBase}/runs/${encodeURIComponent(runId)}/stream?${new URLSearchParams({ after: String(after) })}`;
}

export function getProvenance(): Promise<Provenance> {
  return request<Provenance>("/provenance");
}

export function getComparison(runIds: string[]): Promise<{ warnings: string[]; config_diff: Record<string, unknown[]>; runs: RunDetail[] }> {
  const query = runIds.map((runId) => `run_id=${encodeURIComponent(runId)}`).join("&");
  return request(`/compare?${query}`);
}

export function getSamples(
  runId: string,
  filters: { bigRockFalseNegative: boolean; bigRockToSoil: boolean; sortBy: string; split: string },
  offset = 0,
  limit = 4
): Promise<{ samples: SampleRecord[]; available: boolean; total: number; offset: number; limit: number; available_splits: string[] }> {
  const query = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
    sort_by: filters.sortBy,
    big_rock_false_negative: String(filters.bigRockFalseNegative),
    big_rock_to_soil: String(filters.bigRockToSoil)
  });
  if (filters.split) query.set("split", filters.split);
  return request(`/runs/${encodeURIComponent(runId)}/samples?${query}`);
}

export function artifactUrl(runId: string, artifactPath: string): string {
  return `${apiBase}/runs/${encodeURIComponent(runId)}/artifacts/${artifactPath.split("/").map(encodeURIComponent).join("/")}`;
}