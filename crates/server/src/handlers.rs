use std::time::Instant;
use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    Json,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use tracing::info;

use engine::{gpu_get_memory, GpuSearchResult, LatencyStats, Vector3D};
use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: Option<Vec<f32>>,
    pub top_k: Option<usize>,
    pub ef_search: Option<usize>,
    pub nprobe: Option<usize>,
    pub use_ivf: Option<bool>,
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
pub struct SearchResponse {
    pub results: Vec<GpuSearchResult>,
    pub hops: Vec<TraversalHop3D>,
    pub latency_us: f64,
    pub stats: LatencyStats,
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
}

#[derive(Debug, Serialize)]
pub struct Vectors3DResponse {
    pub points: Vec<Vector3D>,
    pub clusters: Vec<u32>,
    pub count: usize,
}

pub async fn handle_stats(State(state): State<AppState>) -> Json<IndexStatsResponse> {
    let (num_vectors, dim, stats) = {
        let index = state.index.read();
        (index.num_vectors(), index.dim(), index.metrics().get_stats())
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
    })
}

pub async fn handle_get_3d_vectors(State(state): State<AppState>) -> Json<Vectors3DResponse> {
    let points = state.projected_points_3d.read().clone();
    let clusters = state.cluster_ids.read().clone();
    let count = points.len();

    Json(Vectors3DResponse {
        points,
        clusters,
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
        {
            let mut idx = state.index.write();
            idx.build(&vecs).expect("Failed to build index");
        }
        let projector = crate::projection::PcaProjector3D::fit(&vecs, dim, 2000);
        let points_3d = projector.project_batch(&vecs);
        *state.projector.write() = Some(projector);
        *state.projected_points_3d.write() = points_3d;
        *state.dataset_cache.write() = vecs;
    } else {
        state.generate_and_index_clustered(num_vectors, dim, num_clusters);
    }

    Json(serde_json::json!({
        "status": "success",
        "message": format!("Index built successfully with {} vectors", num_vectors)
    }))
}

pub async fn handle_search(
    State(state): State<AppState>,
    Json(payload): Json<SearchRequest>,
) -> Json<SearchResponse> {
    let top_k = payload.top_k.unwrap_or(10);
    let ef_search = payload.ef_search.unwrap_or(64);
    let nprobe = payload.nprobe.unwrap_or(8);
    let use_ivf = payload.use_ivf.unwrap_or(false);

    let dim = { state.index.read().dim() };
    let query = payload.query.unwrap_or_else(|| {
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rand::Rng::gen_range(&mut rng, -1.0f32..1.0f32)).collect()
    });

    let points_3d = state.projected_points_3d.read().clone();

    let (results, hops, latency_us, stats) = {
        let index = state.index.read();
        let start = Instant::now();
        let (results, hops) = if use_ivf {
            let res = index.search_ivf_pq(&query, top_k, nprobe).unwrap_or_default();
            (res, Vec::new())
        } else {
            index.search_with_trajectory(&query, top_k, ef_search).unwrap_or_default()
        };
        let latency_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        let stats = index.metrics().get_stats();
        (results, hops, latency_us, stats)
    };

    let hops_3d: Vec<TraversalHop3D> = hops
        .into_iter()
        .map(|h| {
            let from_pos = points_3d
                .get(h.from_node as usize)
                .cloned()
                .unwrap_or(Vector3D { x: 0.0, y: 0.0, z: 0.0 });
            let to_pos = points_3d
                .get(h.to_node as usize)
                .cloned()
                .unwrap_or(Vector3D { x: 0.0, y: 0.0, z: 0.0 });

            TraversalHop3D {
                step: h.step,
                from_node: h.from_node,
                to_node: h.to_node,
                distance: h.distance,
                from_pos,
                to_pos,
            }
        })
        .collect();

    Json(SearchResponse {
        results,
        hops: hops_3d,
        latency_us,
        stats,
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

    while let Some(msg) = receiver.next().await {
        if let Ok(msg) = msg {
            match msg {
                Message::Text(text) => {
                    if let Ok(req) = serde_json::from_str::<SearchRequest>(&text) {
                        let top_k = req.top_k.unwrap_or(10);
                        let ef_search = req.ef_search.unwrap_or(64);
                        let dim = { state.index.read().dim() };

                        let query = req.query.unwrap_or_else(|| {
                            let mut rng = rand::thread_rng();
                            (0..dim).map(|_| rand::Rng::gen_range(&mut rng, -1.0f32..1.0f32)).collect()
                        });

                        let points_3d = state.projected_points_3d.read().clone();

                        let (results, hops, latency_us, stats) = {
                            let index = state.index.read();
                            let start = Instant::now();
                            let (results, hops) = index.search_with_trajectory(&query, top_k, ef_search).unwrap_or_default();
                            let latency_us = start.elapsed().as_secs_f64() * 1_000_000.0;
                            let stats = index.metrics().get_stats();
                            (results, hops, latency_us, stats)
                        };

                        let hops_3d: Vec<TraversalHop3D> = hops
                            .into_iter()
                            .map(|h| {
                                let from_pos = points_3d
                                    .get(h.from_node as usize)
                                    .cloned()
                                    .unwrap_or(Vector3D { x: 0.0, y: 0.0, z: 0.0 });
                                let to_pos = points_3d
                                    .get(h.to_node as usize)
                                    .cloned()
                                    .unwrap_or(Vector3D { x: 0.0, y: 0.0, z: 0.0 });

                                TraversalHop3D {
                                    step: h.step,
                                    from_node: h.from_node,
                                    to_node: h.to_node,
                                    distance: h.distance,
                                    from_pos,
                                    to_pos,
                                }
                            })
                            .collect();

                        let resp = SearchResponse {
                            results,
                            hops: hops_3d,
                            latency_us,
                            stats,
                        };

                        if let Ok(serialized) = serde_json::to_string(&resp) {
                            if sender.send(Message::Text(serialized)).await.is_err() {
                                break;
                            }
                        }
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        } else {
            break;
        }
    }
}
