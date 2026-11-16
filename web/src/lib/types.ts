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

export type Algorithm = 'hnsw' | 'cagra' | 'ivf_pq' | 'ivf_sq8' | 'ivf_rabitq';

export type ProjectionMethod = 'pca' | 'umap';

export interface BeamCandidate {
  id: number;
  distance: number;
  visited: boolean;
  pos: Vector3D | null;
}

export interface BeamStep {
  iteration: number;
  expanded_node: number;
  expanded_distance: number;
  /** Index into hops of the first distance evaluation made in this iteration. */
  eval_start: number;
  eval_count: number;
  expanded_pos: Vector3D | null;
  /** Sorted candidate queue (best first) AFTER this iteration. */
  candidates: BeamCandidate[];
}

export interface ProbeStep {
  rank: number;
  cluster_id: number;
  centroid_distance: number;
  list_len: number;
  best_distance: number | null;
  result_ids: number[];
  centroid_pos: Vector3D | null;
}

export interface SearchResponse {
  results: GpuSearchResult[];
  hops: TraversalHop3D[];
  latency_us: number;
  stats: LatencyStats;
  beam_steps?: BeamStep[];
  probes?: ProbeStep[];
  algorithm?: string;
}

export interface QueryParams {
  top_k: number;
  ef_search: number;
  nprobe: number;
  use_ivf: boolean;
  algorithm?: Algorithm;
  debug?: boolean;
}

export interface Vectors3DResponse {
  points: Vector3D[];
  clusters: number[];
  count: number;
  method?: ProjectionMethod;
}

export interface ProjectionResponse {
  status: string;
  method: ProjectionMethod;
  count: number;
}

export interface PqDistortionResponse {
  cluster_id: number;
  avg_reconstruction_error: number;
  vector_count: number;
}
