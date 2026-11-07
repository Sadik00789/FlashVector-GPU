//! HNSW graph construction.
//!
//! Nodes are inserted incrementally: each new node runs a beam search over the
//! graph built so far, then links to the candidates retained by the
//! Malkov-Yashunin diversity heuristic. Edges are made bidirectional, and a
//! reverse edge is only admitted when it is not dominated by an existing
//! neighbor, which keeps the long-range links that greedy descent depends on.

use rayon::prelude::*;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Nodes inserted one at a time before batching restarts. Keeping this small
/// keeps the intra-batch blind spot negligible on large datasets while the
/// sequential path handles datasets small enough to insert outright.
const INSERT_BATCH: usize = 2048;
/// Below this size a fully sequential build is both fast and strictly better,
/// since no node loses access to the nodes inserted alongside it.
const SEQUENTIAL_LIMIT: usize = 50_000;

#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

#[derive(Copy, Clone)]
struct Key(f32);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// Per-thread visited table reused across insertions via epoch stamping.
struct Workspace {
    visited: Vec<u32>,
    epoch: u32,
}

impl Workspace {
    fn new(num_nodes: usize) -> Self {
        Self {
            visited: vec![0; num_nodes],
            epoch: 0,
        }
    }

    fn begin(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visited.iter_mut().for_each(|slot| *slot = 0);
            self.epoch = 1;
        }
        self.epoch
    }
}

struct Graph {
    adjacency: Vec<u32>,
    distances: Vec<f32>,
    degrees: Vec<u32>,
    m_max: usize,
}

impl Graph {
    fn new(num_nodes: usize, m_max: usize) -> Self {
        Self {
            adjacency: vec![u32::MAX; num_nodes * m_max],
            distances: vec![0.0; num_nodes * m_max],
            degrees: vec![0; num_nodes],
            m_max,
        }
    }

    /// Beam search over the inserted prefix of the graph.
    #[allow(clippy::too_many_arguments)]
    fn beam_search(
        &self,
        dataset: &[f32],
        dim: usize,
        prefix: usize,
        entry: u32,
        query: &[f32],
        ef: usize,
        workspace: &mut Workspace,
    ) -> Vec<(u32, f32)> {
        let mut results: BinaryHeap<(Key, u32)> = BinaryHeap::with_capacity(ef + 1);
        let mut frontier: BinaryHeap<Reverse<(Key, u32)>> = BinaryHeap::with_capacity(ef * 2);

        let epoch = workspace.begin();
        let entry_vec = &dataset[entry as usize * dim..(entry as usize + 1) * dim];
        let entry_dist = l2_sq(query, entry_vec);
        workspace.visited[entry as usize] = epoch;
        results.push((Key(entry_dist), entry));
        frontier.push(Reverse((Key(entry_dist), entry)));

        while let Some(Reverse((Key(current_dist), node))) = frontier.pop() {
            if results.len() >= ef && current_dist > results.peek().map(|r| r.0 .0).unwrap_or(f32::INFINITY) {
                break;
            }

            let start = node as usize * self.m_max;
            let degree = self.degrees[node as usize] as usize;
            for slot in 0..degree {
                let neighbor = self.adjacency[start + slot];
                let idx = neighbor as usize;
                // Slots past the inserted prefix still hold the INVALID sentinel.
                if idx >= prefix || workspace.visited[idx] == epoch {
                    continue;
                }
                workspace.visited[idx] = epoch;

                let vector = &dataset[idx * dim..(idx + 1) * dim];
                let dist = l2_sq(query, vector);

                if results.len() < ef || dist < results.peek().map(|r| r.0 .0).unwrap_or(f32::INFINITY) {
                    results.push((Key(dist), neighbor));
                    frontier.push(Reverse((Key(dist), neighbor)));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.into_iter().map(|(Key(d), id)| (id, d)).collect()
    }
}

/// Malkov-Yashunin diverse neighbor selection.
///
/// A candidate is kept only when it is nearer to the base node than to every
/// neighbor accepted so far, which retains the long-range edges that keep
/// greedy search from stalling inside a local cluster. Slots left empty by
/// pruning are filled with the nearest rejected candidates so connectivity does
/// not depend on the pruning outcome.
fn select_neighbors(
    dataset: &[f32],
    dim: usize,
    mut candidates: Vec<(u32, f32)>,
    m: usize,
) -> Vec<(u32, f32)> {
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    let mut selected: Vec<(u32, f32)> = Vec::with_capacity(m);
    let mut rejected: Vec<(u32, f32)> = Vec::new();

    for &(id, dist) in candidates.iter() {
        if selected.len() >= m {
            break;
        }
        let vector = &dataset[id as usize * dim..(id as usize + 1) * dim];
        let mut is_diverse = true;
        for &(kept, _) in &selected {
            let kept_vec = &dataset[kept as usize * dim..(kept as usize + 1) * dim];
            if l2_sq(vector, kept_vec) < dist {
                is_diverse = false;
                break;
            }
        }
        if is_diverse {
            selected.push((id, dist));
        } else {
            rejected.push((id, dist));
        }
    }

    for &(id, dist) in &rejected {
        if selected.len() >= m {
            break;
        }
        selected.push((id, dist));
    }

    selected
}

impl Graph {
    #[allow(clippy::too_many_arguments)]
    fn add_reverse_edge(
        &mut self,
        dataset: &[f32],
        dim: usize,
        target: usize,
        source: u32,
        dist: f32,
    ) {
        let start = target * self.m_max;
        let degree = self.degrees[target] as usize;

        if degree < self.m_max {
            self.adjacency[start + degree] = source;
            self.distances[start + degree] = dist;
            self.degrees[target] = (degree + 1) as u32;
            return;
        }

        // The neighborhood is full, so re-run the diversity heuristic over the
        // existing edges plus the new one. Simply evicting the farthest edge
        // would erode the diversified structure as the node fills up.
        let mut candidates: Vec<(u32, f32)> = (0..degree)
            .map(|slot| (self.adjacency[start + slot], self.distances[start + slot]))
            .collect();
        candidates.push((source, dist));

        let mut selected: Vec<(u32, f32)> = Vec::with_capacity(self.m_max);
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        for &(id, edge_dist) in &candidates {
            if selected.len() >= self.m_max {
                break;
            }
            let vector = &dataset[id as usize * dim..(id as usize + 1) * dim];
            let mut is_diverse = true;
            for &(kept, _) in &selected {
                let kept_vec = &dataset[kept as usize * dim..(kept as usize + 1) * dim];
                if l2_sq(vector, kept_vec) < edge_dist {
                    is_diverse = false;
                    break;
                }
            }
            if is_diverse {
                selected.push((id, edge_dist));
            }
        }

        for slot in 0..self.m_max {
            self.adjacency[start + slot] = u32::MAX;
            self.distances[start + slot] = 0.0;
        }
        for (slot, &(id, edge_dist)) in selected.iter().enumerate() {
            self.adjacency[start + slot] = id;
            self.distances[start + slot] = edge_dist;
        }
        self.degrees[target] = selected.len() as u32;
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_node(
        &mut self,
        dataset: &[f32],
        dim: usize,
        prefix: usize,
        node: usize,
        entry: u32,
        m: usize,
        ef_construction: usize,
        workspace: &mut Workspace,
    ) {
        let query = &dataset[node * dim..(node + 1) * dim];
        let candidates =
            self.beam_search(dataset, dim, prefix, entry, query, ef_construction, workspace);
        let selected = select_neighbors(dataset, dim, candidates, m);

        let start = node * self.m_max;
        for (slot, &(neighbor, dist)) in selected.iter().enumerate() {
            self.adjacency[start + slot] = neighbor;
            self.distances[start + slot] = dist;
        }
        self.degrees[node] = selected.len() as u32;

        for &(neighbor, dist) in &selected {
            self.add_reverse_edge(dataset, dim, neighbor as usize, node as u32, dist);
        }
    }
}

/// Build a single-layer HNSW graph over `dataset`.
///
/// Returns the flattened adjacency matrix (`num_nodes * m_max`), per-node
/// degrees, and the entry point used to seed greedy descent.
pub fn build_hnsw(
    dataset: &[f32],
    num_nodes: usize,
    dim: usize,
    m: usize,
    ef_construction: usize,
) -> (Vec<u32>, Vec<u32>, u32) {
    let m_max = m.max(1);
    let ef = ef_construction.max(m_max + 1).max(1);

    if num_nodes == 0 {
        return (Vec::new(), Vec::new(), 0);
    }
    if num_nodes == 1 {
        return (vec![u32::MAX; m_max], vec![0], 0);
    }

    let mut graph = Graph::new(num_nodes, m_max);
    let mut workspace = Workspace::new(num_nodes);

    // The first node seeds the graph; every later node searches what precedes it.
    let entry = 0u32;
    graph.insert_node(dataset, dim, 1, 1, entry, m_max, ef, &mut workspace);

    let sequential_limit = SEQUENTIAL_LIMIT.min(num_nodes);
    for node in 2..sequential_limit {
        graph.insert_node(dataset, dim, node, node, entry, m_max, ef, &mut workspace);
    }

    let mut start = sequential_limit.max(2);
    while start < num_nodes {
        let end = (start + INSERT_BATCH).min(num_nodes);

        // Every node in the batch searches the same immutable prefix, so the
        // expensive search and selection phase runs in parallel.
        let planned: Vec<Vec<(u32, f32)>> = (start..end)
            .into_par_iter()
            .map_init(
                || Workspace::new(num_nodes),
                |workspace, node| {
                    let query = &dataset[node * dim..(node + 1) * dim];
                    let candidates = graph.beam_search(
                        dataset,
                        dim,
                        start,
                        entry,
                        query,
                        ef,
                        workspace,
                    );
                    select_neighbors(dataset, dim, candidates, m_max)
                },
            )
            .collect();

        // Applying edges mutates the shared graph, so it runs serially.
        for (offset, selected) in planned.into_iter().enumerate() {
            let node = start + offset;
            let base = node * m_max;
            for (slot, &(neighbor, dist)) in selected.iter().enumerate() {
                graph.adjacency[base + slot] = neighbor;
                graph.distances[base + slot] = dist;
            }
            graph.degrees[node] = selected.len() as u32;

            for &(neighbor, dist) in &selected {
                graph.add_reverse_edge(dataset, dim, neighbor as usize, node as u32, dist);
            }
        }

        start = end;
    }

    // Seed greedy descent from the node closest to the dataset centroid: on a
    // single-layer graph a central start point shortens every query's descent.
    let mut centroid = vec![0.0f32; dim];
    for node in 0..num_nodes {
        let vector = &dataset[node * dim..(node + 1) * dim];
        for d in 0..dim {
            centroid[d] += vector[d];
        }
    }
    let inv = 1.0 / num_nodes as f32;
    centroid.iter_mut().for_each(|value| *value *= inv);

    let mut entry_point = 0u32;
    let mut best_dist = f32::INFINITY;
    for node in 0..num_nodes {
        let vector = &dataset[node * dim..(node + 1) * dim];
        let dist = l2_sq(&centroid, vector);
        if dist < best_dist {
            best_dist = dist;
            entry_point = node as u32;
        }
    }

    (graph.adjacency, graph.degrees, entry_point)
}
