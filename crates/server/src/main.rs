use std::net::SocketAddr;
use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;

mod handlers;
mod projection;
mod state;

use handlers::{handle_get_3d_vectors, handle_index, handle_search, handle_stats, handle_ws_stream};
use state::AppState;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,server=debug,engine=debug".into()),
        )
        .init();

    info!("Initializing FlashVector-GPU Server...");

    // Initialize CUDA GPU context
    if let Err(e) = engine::init_gpu(0) {
        tracing::warn!("Warning during GPU initialization: {:?}", e);
    }

    let state = AppState::new();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/v1/search", post(handle_search))
        .route("/api/v1/index", post(handle_index))
        .route("/api/v1/stats", get(handle_stats))
        .route("/api/v1/vectors/3d", get(handle_get_3d_vectors))
        .route("/ws/stream", get(handle_ws_stream))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("FlashVector-GPU Gateway listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind TCP listener on port 8080");

    axum::serve(listener, app)
        .await
        .expect("Axum server encountered fatal error");
}
