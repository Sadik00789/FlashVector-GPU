use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Query, State,
    },
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::info;

use crate::batcher::{BatchKey, BatcherStats, SearchAlgorithm};
use crate::projection::{ProjectionMethod, Projector3D};
use crate::state::AppState;
use engine::{
    gpu_get_memory, BeamStepRecord, GpuSearchResult, GpuVectorIndex, HybridHit, IndexConfig,
    IvfProbeTelemetry, LatencyStats, TraversalHop, Vector3D,
};

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Option<Vec<f32>>,
    pub top_k: Option<usize>,
    pub ef_search: Option<usize>,
    pub nprobe: Option<usize>,
    pub rerank_k: Option<usize>,
    pub use_ivf: Option<bool>,
    pub algorithm: Option<String>,
    pub debug: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct TraversalHop3D {
    pub step: u32,
    pub from_node: u32,
    pub to_node: u32,
    pub distance: f32,
    pub from_pos: Vector3D,
    pub to_pos: Vector3D,
}

#[derive(Debug, Serialize)]
pub struct BeamCandidateResponse {
    pub id: u32,
    pub distance: f32,
    pub visited: bool,
    pub pos: Option<Vector3D>,
}

#[derive(Debug, Serialize)]
pub struct BeamStepResponse {
    pub iteration: u32,
    pub expanded_node: u32,
    pub expanded_distance: f32,
    pub eval_start: u32,
    pub eval_count: u32,
    pub expanded_pos: Option<Vector3D>,
    pub candidates: Vec<BeamCandidateResponse>,
}

#[derive(Debug, Serialize)]
pub struct ProbeStepResponse {
    pub rank: usize,
    pub cluster_id: u32,
    pub centroid_distance: f32,
    pub list_len: usize,
    pub best_distance: Option<f32>,
    pub result_ids: Vec<u32>,
    pub centroid_pos: Option<Vector3D>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<GpuSearchResult>,
    pub hops: Vec<TraversalHop3D>,
    pub latency_us: f64,
    pub stats: LatencyStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beam_steps: Option<Vec<BeamStepResponse>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probes: Option<Vec<ProbeStepResponse>>,
    pub algorithm: String,
}

#[derive(Debug, Deserialize)]
pub struct IndexRequest {
    pub num_vectors: Option<usize>,
    pub dim: Option<usize>,
    pub num_clusters: Option<usize>,
    pub vectors: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct IndexStatsResponse {
    pub num_vectors: usize,
    pub dim: usize,
    pub free_vram_mb: f64,
    pub total_vram_mb: f64,
    pub stats: LatencyStats,
    pub batcher: BatcherStats,
}

#[derive(Debug, Deserialize)]
pub struct Vectors3DQuery {
    pub method: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct Vectors3DResponse {
    pub points: Vec<Vector3D>,
    pub clusters: Vec<u32>,
    pub count: usize,
    pub method: ProjectionMethod,
}

#[derive(Debug, Deserialize)]
pub struct ProjectionRequest {
    pub method: String,
}

#[derive(Debug, Serialize)]
pub struct ProjectionResponse {
    pub status: &'static str,
    pub method: ProjectionMethod,
    pub count: usize,
}

#[derive(Debug, Deserialize)]
pub struct SnapshotRequest {
    pub vectors: Vec<f32>,
    pub dim: usize,
    pub metadata: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Serialize)]
pub struct PqDistortionResponse {
    pub cluster_id: usize,
    pub avg_reconstruction_error: f32,
    pub vector_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    pub dense_query: Vec<f32>,
    pub sparse_query: Vec<(u32, f32)>,
    pub top_k: Option<usize>,
    pub candidate_k: Option<usize>,
    pub ef_search: Option<usize>,
    pub rrf_k: Option<f32>,
    pub dense_weight: Option<f32>,
    pub sparse_weight: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct HybridSearchResponse {
    pub results: Vec<HybridHit>,
    pub latency_us: f64,
}

#[inline]
fn zero_vec3() -> Vector3D {
    Vector3D {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    }
}

fn lookup_pos(
    node_id: u32,
    points_3d: &[Vector3D],
    projector: Option<&Projector3D>,
    index: &GpuVectorIndex,
) -> Option<Vector3D> {
    if node_id == u32::MAX {
        return None;
    }
    if let Some(p) = points_3d.get(node_id as usize) {
        return Some(*p);
    }
    let proj = projector?;
    let vec = index.get_vector(node_id as usize)?;
    Some(proj.project_vector(&vec))
}

fn hops_to_3d(
    hops: Vec<TraversalHop>,
    points_3d: &[Vector3D],
    projector: Option<&Projector3D>,
    index: &GpuVectorIndex,
) -> Vec<TraversalHop3D> {
    hops.into_iter()
        .map(|h| TraversalHop3D {
            step: h.step,
            from_node: h.from_node,
            to_node: h.to_node,
            distance: h.distance,
            from_pos: lookup_pos(h.from_node, points_3d, projector, index)
                .unwrap_or_else(zero_vec3),
            to_pos: lookup_pos(h.to_node, points_3d, projector, index).unwrap_or_else(zero_vec3),
        })
        .collect()
}

fn beam_steps_to_response(
    steps: Vec<BeamStepRecord>,
    points_3d: &[Vector3D],
    projector: Option<&Projector3D>,
    index: &GpuVectorIndex,
) -> Vec<BeamStepResponse> {
    steps
        .into_iter()
        .map(|s| {
            let count = (s.cand_count as usize).min(s.candidates.len());
            let candidates = s.candidates[..count]
                .iter()
                .filter(|c| c.id != u32::MAX)
                .map(|c| BeamCandidateResponse {
                    id: c.id,
                    distance: c.distance,
                    visited: c.visited != 0,
                    pos: lookup_pos(c.id, points_3d, projector, index),
                })
                .collect();
            BeamStepResponse {
                iteration: s.iteration,
                expanded_node: s.expanded_node,
                expanded_distance: s.expanded_distance,
                eval_start: s.eval_start,
                eval_count: s.eval_count,
                expanded_pos: lookup_pos(s.expanded_node, points_3d, projector, index),
                candidates,
            }
        })
        .collect()
}

fn probes_to_response(
    probes: Vec<IvfProbeTelemetry>,
    projector: Option<&Projector3D>,
    index: &GpuVectorIndex,
) -> Vec<ProbeStepResponse> {
    let centroids = index.centroids();
    let dim = index.dim();
    let nlist = centroids.len().checked_div(dim).unwrap_or(0);
    probes
        .into_iter()
        .map(|p| {
            let cid = p.cluster_id as usize;
            let centroid_pos = if cid < nlist {
                projector.map(|proj| proj.project_vector(&centroids[cid * dim..(cid + 1) * dim]))
            } else {
                None
            };
            ProbeStepResponse {
                rank: p.rank,
                cluster_id: p.cluster_id,
                centroid_distance: p.centroid_distance,
                list_len: p.list_len,
                best_distance: p.best_distance,
                result_ids: p.result_ids,
                centroid_pos,
            }
        })
        .collect()
}

pub async fn handle_stats(State(state): State<AppState>) -> Json<IndexStatsResponse> {
    let (num_vectors, dim, stats) = {
        let index = state.index.read();
        (
            index.num_vectors(),
            index.dim(),
            index.metrics().get_stats(),
        )
    };

    let (free_b, total_b) = gpu_get_memory().unwrap_or((0, 0));
    let free_vram_mb = (free_b as f64) / (1024.0 * 1024.0);
    let total_vram_mb = (total_b as f64) / (1024.0 * 1024.0);

    Json(IndexStatsResponse {
        num_vectors,
        dim,
        free_vram_mb,
        total_vram_mb,
        stats,
        batcher: state.batcher.stats(),
    })
}

pub async fn handle_get_3d_vectors(
    State(state): State<AppState>,
    Query(query): Query<Vectors3DQuery>,
) -> Json<Vectors3DResponse> {
    if let Some(req_method) = query.method {
        let target = ProjectionMethod::from_str_lossy(&req_method);
        let cur = *state.projection_method.read();
        if target != cur {
            let s = state.clone();
            let _ = tokio::task::spawn_blocking(move || s.rebuild_projection(target)).await;
        }
    }

    let points = state.projected_points_3d.read().clone();
    let clusters = state.cluster_ids.read().clone();
    let method = state
        .projector
        .read()
        .as_ref()
        .map(Projector3D::method)
        .unwrap_or_else(|| *state.projection_method.read());
    let count = points.len();

    Json(Vectors3DResponse {
        points,
        clusters,
        count,
        method,
    })
}

pub async fn handle_set_projection(
    State(state): State<AppState>,
    Json(payload): Json<ProjectionRequest>,
) -> Json<ProjectionResponse> {
    let method = ProjectionMethod::from_str_lossy(&payload.method);
    let s = state.clone();
    let count = tokio::task::spawn_blocking(move || s.rebuild_projection(method))
        .await
        .unwrap_or(0);

    Json(ProjectionResponse {
        status: "success",
        method,
        count,
    })
}

pub async fn handle_index(
    State(state): State<AppState>,
    Json(payload): Json<IndexRequest>,
) -> Json<serde_json::Value> {
    let num_vectors = payload.num_vectors.unwrap_or(10_000);
    let dim = payload.dim.unwrap_or(128);
    let num_clusters = payload.num_clusters.unwrap_or(16);

    if let Some(vecs) = payload.vectors {
        let method = *state.projection_method.read();
        let actual_count = vecs.len() / dim.max(1);
        {
            let mut idx = state.index.write();
            idx.build(&vecs).expect("Failed to build index");
        }
        state.fit_and_store_projection(&vecs, dim, method);
        *state.dataset_cache.write() = vecs;
        return Json(serde_json::json!({
            "status": "success",
            "message": format!("Index built successfully with {} vectors", actual_count)
        }));
    }

    state.generate_and_index_clustered(num_vectors, dim, num_clusters);
    Json(serde_json::json!({
        "status": "success",
        "message": format!("Index built successfully with {} vectors", num_vectors)
    }))
}

/// Pick the largest PQ partition count that evenly divides the snapshot dimension.
fn choose_m_pq(dim: usize) -> u32 {
    for candidate in [16u32, 8, 4, 2, 1] {
        if dim.is_multiple_of(candidate as usize) {
            return candidate;
        }
    }
    1
}

/// Load an arbitrary externally generated embedding snapshot and replace the
/// live index, 3D projection, and dataset cache.
pub async fn handle_load_snapshot(
    State(state): State<AppState>,
    Json(payload): Json<SnapshotRequest>,
) -> impl IntoResponse {
    let dim = payload.dim;
    if dim == 0 || payload.vectors.is_empty() || payload.vectors.len() % dim != 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "status": "error",
                "message": "dim must be non-zero and vectors.len() must be a positive multiple of dim"
            })),
        );
    }

    let num_vectors = payload.vectors.len() / dim;
    if num_vectors > u32::MAX as usize {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "status": "error",
                "message": "snapshot exceeds the maximum supported vector count"
            })),
        );
    }

    let config = IndexConfig {
        dim: dim as u32,
        max_elements: num_vectors as u32,
        m: 32,
        ef_construction: 128,
        nlist: num_vectors.min(32) as u32,
        m_pq: choose_m_pq(dim),
        metric: 0,
    };

    let mut index = match GpuVectorIndex::new(config) {
        Ok(index) => index,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "status": "error",
                    "message": format!("Failed to allocate GPU index: {:?}", e)
                })),
            );
        }
    };

    if let Err(e) = index.build(&payload.vectors) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to build index from snapshot: {:?}", e)
            })),
        );
    }

    let cluster_ids = payload
        .metadata
        .as_ref()
        .filter(|metadata| metadata.len() == num_vectors)
        .map(|metadata| {
            metadata
                .iter()
                .map(|entry| {
                    entry
                        .get("cluster")
                        .and_then(|value| value.as_u64())
                        .map(|value| value.min(u32::MAX as u64) as u32)
                        .unwrap_or(0)
                })
                .collect::<Vec<u32>>()
        })
        .unwrap_or_default();

    let method = *state.projection_method.read();
    *state.index.write() = index;
    state.fit_and_store_projection(&payload.vectors, dim, method);
    *state.cluster_ids.write() = cluster_ids;
    *state.dataset_cache.write() = payload.vectors;

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "success",
            "message": format!(
                "Snapshot loaded and indexed successfully ({} vectors, dim {})",
                num_vectors, dim
            )
        })),
    )
}

/// Per-cluster PQ reconstruction error for the Studio distortion heatmap.
pub async fn handle_pq_distortion(
    State(state): State<AppState>,
) -> Json<Vec<PqDistortionResponse>> {
    let clusters = {
        let index = state.index.read();
        index.pq_distortion(1_000)
    };

    Json(
        clusters
            .into_iter()
            .map(|cluster| PqDistortionResponse {
                cluster_id: cluster.cluster_id,
                avg_reconstruction_error: cluster.avg_reconstruction_error,
                vector_count: cluster.vector_count,
            })
            .collect(),
    )
}

async fn execute_search(state: &AppState, payload: SearchRequest) -> SearchResponse {
    let top_k = payload.top_k.unwrap_or(10);
    let ef_search = payload.ef_search.unwrap_or(64);
    let nprobe = payload.nprobe.unwrap_or(8);
    let rerank_k = payload.rerank_k.unwrap_or(32);
    let algo = SearchAlgorithm::from_request(payload.algorithm.as_deref(), payload.use_ivf);
    let debug = payload.debug.unwrap_or(false);

    let dim = state.index.read().dim();
    let query = payload.query.unwrap_or_else(|| {
        let mut rng = rand::thread_rng();
        (0..dim)
            .map(|_| rand::Rng::gen_range(&mut rng, -1.0f32..1.0f32))
            .collect()
    });

    if debug {
        let points_3d = state.projected_points_3d.read().clone();
        let proj_guard = state.projector.read();
        let projector = proj_guard.as_ref();
        let index = state.index.read();
        let start = Instant::now();

        let (results, hops_3d, beam_steps, probes) = match algo {
            SearchAlgorithm::Hnsw => {
                if let Ok(trace) = index.search_with_debug(&query, top_k, ef_search, None) {
                    let h = hops_to_3d(trace.hops, &points_3d, projector, &index);
                    let b = beam_steps_to_response(trace.steps, &points_3d, projector, &index);
                    (trace.results, h, Some(b), None)
                } else {
                    (Vec::new(), Vec::new(), Some(Vec::new()), None)
                }
            }
            SearchAlgorithm::Cagra => {
                if let Ok(trace) = index.search_cagra_with_debug(&query, top_k, ef_search, None) {
                    let h = hops_to_3d(trace.hops, &points_3d, projector, &index);
                    let b = beam_steps_to_response(trace.steps, &points_3d, projector, &index);
                    (trace.results, h, Some(b), None)
                } else {
                    (Vec::new(), Vec::new(), Some(Vec::new()), None)
                }
            }
            SearchAlgorithm::IvfPq | SearchAlgorithm::IvfSq8 | SearchAlgorithm::IvfRabitq => {
                let mode = algo.ivf_mode().expect("IVF algorithm must have quant mode");
                if let Ok((res, raw_probes)) =
                    index.search_ivf_with_debug(&query, top_k, nprobe, rerank_k, mode)
                {
                    let p = probes_to_response(raw_probes, projector, &index);
                    (res, Vec::new(), None, Some(p))
                } else {
                    (Vec::new(), Vec::new(), None, Some(Vec::new()))
                }
            }
        };

        let latency_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        let stats = index.metrics().get_stats();
        return SearchResponse {
            results,
            hops: hops_3d,
            latency_us,
            stats,
            beam_steps,
            probes,
            algorithm: algo.as_str().to_string(),
        };
    }

    let ef_or_nprobe = if algo.ivf_mode().is_some() {
        nprobe
    } else {
        ef_search
    };
    let key = BatchKey {
        algorithm: algo,
        top_k,
        ef_or_nprobe,
        rerank_k,
    };

    let (results, latency_us) = state
        .batcher
        .submit(key, query)
        .await
        .unwrap_or_else(|_| (Vec::new(), 0.0));
    let stats = state.index.read().metrics().get_stats();

    SearchResponse {
        results,
        hops: Vec::new(),
        latency_us,
        stats,
        beam_steps: None,
        probes: None,
        algorithm: algo.as_str().to_string(),
    }
}

pub async fn handle_search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Json<SearchResponse> {
    Json(execute_search(&state, payload).await)
}

pub async fn handle_hybrid_search(
    State(state): State<AppState>,
    Json(payload): Json<HybridSearchRequest>,
) -> Json<HybridSearchResponse> {
    let top_k = payload.top_k.unwrap_or(10);
    let candidate_k = payload.candidate_k.unwrap_or(64);
    let ef_search = payload.ef_search.unwrap_or(64);
    let rrf_k = payload.rrf_k.unwrap_or(60.0);
    let dense_weight = payload.dense_weight.unwrap_or(1.0);
    let sparse_weight = payload.sparse_weight.unwrap_or(1.0);

    let index = state.index.read();
    let start = Instant::now();
    let sparse_slice = [payload.sparse_query.as_slice()];
    let hits = index
        .hybrid_search(
            &payload.dense_query,
            &sparse_slice,
            top_k,
            candidate_k,
            ef_search,
            rrf_k,
            dense_weight,
            sparse_weight,
            None,
        )
        .map(|mut rows| rows.pop().unwrap_or_default())
        .unwrap_or_default();
    let latency_us = start.elapsed().as_secs_f64() * 1_000_000.0;

    Json(HybridSearchResponse {
        results: hits,
        latency_us,
    })
}

pub async fn handle_ws_stream(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    info!("New WebSocket client connected to /ws/stream");

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                if let Ok(req) = serde_json::from_str::<SearchRequest>(&text) {
                    let resp = execute_search(&state, req).await;
                    if let Ok(json_str) = serde_json::to_string(&resp) {
                        if sender.send(Message::Text(json_str)).await.is_err() {
                            break;
                        }
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    info!("WebSocket client disconnected");
}
