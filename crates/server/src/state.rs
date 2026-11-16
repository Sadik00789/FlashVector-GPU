use parking_lot::RwLock;
use rand_distr::{Distribution, Normal};
use std::sync::Arc;
use tracing::info;

use crate::batcher::SearchBatcher;
use crate::projection::{ProjectionMethod, Projector3D};
use engine::{GpuVectorIndex, IndexConfig, Vector3D};

#[derive(Clone)]
pub struct AppState {
    pub index: Arc<RwLock<GpuVectorIndex>>,
    pub projector: Arc<RwLock<Option<Projector3D>>>,
    pub projection_method: Arc<RwLock<ProjectionMethod>>,
    pub projected_points_3d: Arc<RwLock<Vec<Vector3D>>>,
    pub cluster_ids: Arc<RwLock<Vec<u32>>>,
    pub dataset_cache: Arc<RwLock<Vec<f32>>>,
    pub batcher: Arc<SearchBatcher>,
}

impl AppState {
    pub fn new() -> Self {
        let config = IndexConfig {
            dim: 128,
            max_elements: 100_000,
            m: 32,
            ef_construction: 128,
            nlist: 32,
            m_pq: 16,
            metric: 0,
        };

        let index = Arc::new(RwLock::new(
            GpuVectorIndex::new(config).expect("Failed to initialize GpuVectorIndex"),
        ));
        let batcher = Arc::new(SearchBatcher::for_index(Arc::clone(&index)));

        let state = Self {
            index,
            projector: Arc::new(RwLock::new(None)),
            projection_method: Arc::new(RwLock::new(ProjectionMethod::Pca)),
            projected_points_3d: Arc::new(RwLock::new(Vec::new())),
            cluster_ids: Arc::new(RwLock::new(Vec::new())),
            dataset_cache: Arc::new(RwLock::new(Vec::new())),
            batcher,
        };

        // Initialize with default high-dimensional clustered dataset (10,000 vectors)
        state.generate_and_index_clustered(10_000, 128, 16);
        state
    }

    pub fn fit_and_store_projection(
        &self,
        vecs: &[f32],
        dim: usize,
        method: ProjectionMethod,
    ) -> usize {
        let projector = Projector3D::fit(vecs, dim, method);
        let points_3d = projector.project_batch(vecs);
        let count = points_3d.len();
        *self.projector.write() = Some(projector);
        *self.projection_method.write() = method;
        *self.projected_points_3d.write() = points_3d;
        count
    }

    pub fn rebuild_projection(&self, method: ProjectionMethod) -> usize {
        let dim = self.index.read().dim();
        let data = self.dataset_cache.read().clone();
        if data.is_empty() || dim == 0 {
            *self.projection_method.write() = method;
            return 0;
        }
        self.fit_and_store_projection(&data, dim, method)
    }

    pub fn generate_and_index_clustered(
        &self,
        num_vectors: usize,
        dim: usize,
        num_clusters: usize,
    ) {
        info!(
            "Generating synthetic clustered dataset: {} vectors, {} dim, {} clusters",
            num_vectors, dim, num_clusters
        );
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 0.15).unwrap();

        let mut cluster_centers = Vec::with_capacity(num_clusters * dim);
        for _ in 0..(num_clusters * dim) {
            cluster_centers.push(rand::Rng::gen_range(&mut rng, -1.5f32..1.5f32));
        }

        let mut data = Vec::with_capacity(num_vectors * dim);
        let mut clusters = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let c_id = i % num_clusters;
            clusters.push(c_id as u32);
            let center = &cluster_centers[c_id * dim..(c_id + 1) * dim];

            for d in 0..dim {
                let val: f32 = center[d] + normal.sample(&mut rng);
                data.push(val);
            }
        }

        let method = *self.projection_method.read();
        {
            let mut idx = self.index.write();
            idx.build(&data).expect("Failed to build index on GPU");
        }

        self.fit_and_store_projection(&data, dim, method);
        *self.cluster_ids.write() = clusters;
        *self.dataset_cache.write() = data;

        info!("Dataset generation and GPU index initialization complete");
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
