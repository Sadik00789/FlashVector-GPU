export interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export interface GpuSearchResult {
  id: number;
  distance: number;
}

export interface TraversalHop3D {
  step: number;
  from_node: number;
  to_node: number;
  distance: number;
  from_pos: Vector3D;
  to_pos: Vector3D;
}

export interface LatencyStats {
  count: number;
  p50_us: number;
  p90_us: number;
  p99_us: number;
  p999_us: number;
  min_us: number;
  max_us: number;
  mean_us: number;
  qps: number;
}

export interface IndexStats {
  num_vectors: number;
  dim: number;
  free_vram_mb: number;
  total_vram_mb: number;
  stats: LatencyStats;
}

export interface SearchResponse {
  results: GpuSearchResult[];
  hops: TraversalHop3D[];
  latency_us: number;
  stats: LatencyStats;
}

export interface QueryParams {
  top_k: number;
  ef_search: number;
  nprobe: number;
  use_ivf: boolean;
}

export interface Vectors3DResponse {
  points: Vector3D[];
  clusters: number[];
  count: number;
}

export interface PqDistortionResponse {
  cluster_id: number;
  avg_reconstruction_error: number;
  vector_count: number;
}
