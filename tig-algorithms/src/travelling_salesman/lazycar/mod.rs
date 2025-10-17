// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use serde::{Deserialize, Serialize};
use tig_challenges::travelling_salesman::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    // Optionally define hyperparameters here. Example:
    // pub param1: usize,
    // pub param2: f64,
    /// Number of random restarts (independent random start cities).
    #[serde(default = "default_restarts")]
    pub restarts: usize,
    /// Maximum 2-opt passes per local search (-1 = until no improvement).
    #[serde(default = "default_two_opt_passes")]
    pub two_opt_passes: isize,
}

fn default_restarts() -> usize {1}
fn default_two_opt_passes() -> isize { -1 }


fn greedy_from_start(distance: &Vec<Vec<f32>>, start: usize) -> Vec<usize> {
    let n = distance.len();
    let mut route = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    let mut current = start;
    route.push(current);
    visited[current] = true;

    while route.len() < n {
        // pick nearest unvisited
        let mut best_j = None;
        let mut best_d = f32::INFINITY;
        for j in 0..n {
            if !visited[j] {
                let d = distance[current][j];
                if d < best_d {
                    best_d = d;
                    best_j = Some(j);
                }
            }
        }
        let next = best_j.expect("There should be at least one unvisited node");
        route.push(next);
        visited[next] = true;
        current = next;
    }

    route
}

fn route_length(distance: &Vec<Vec<f32>>, route: &[usize]) -> f32 {
    if route.is_empty() { return 0.0; }
    let mut total = 0.0;
    for w in route.windows(2) {
        total += distance[w[0]][w[1]];
    }
    total + distance[*route.last().unwrap()][route[0]]
}

/// Perform a single 2-opt pass (first improvement). Keeps route[0] fixed for determinism.
/// Returns true if an improving move was applied.
fn two_opt_pass(distance: &Vec<Vec<f32>>, route: &mut Vec<usize>) -> bool {
    let n = route.len();
    if n < 4 { return false; }

    // Try all (i, k) with 1 <= i < k <= n-1 (fix the first city at index 0).
    for i in 1..(n - 2) {
        let a = route[i - 1];
        let b = route[i];
        for k in (i + 1)..(n - 0) - 1 {
            let c = route[k];
            let d = route[(k + 1) % n];

            // Delta = (a-b) + (c-d) - (a-c) - (b-d)
            let current = distance[a][b] + distance[c][d];
            let proposal = distance[a][c] + distance[b][d];
            if proposal + f32::EPSILON < current {
                // reverse the segment [i..=k]
                route[i..=k].reverse();
                return true; // first improvement
            }
        }
    }
    false
}

/// Run multiple 2-opt passes until no improvement or we hit a pass cap.
fn two_opt_loop(distance: &Vec<Vec<f32>>, route: &mut Vec<usize>, pass_cap: isize) {
    let mut passes = 0isize;
    loop {
        if pass_cap >= 0 && passes >= pass_cap { break; }
        if !two_opt_pass(distance, route) { break; }
        passes += 1;
    }
}
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // println!("Hello World");
    // If you need random numbers, recommend using SmallRng with challenge.seed:
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    let mut rng = SmallRng::from_seed(challenge.seed);

    // If you need HashMap or HashSet, make sure to use a deterministic hasher for consistent runtime_signature:
    use crate::{seeded_hasher, HashMap, HashSet};
    let hasher = seeded_hasher(&challenge.seed);
    // let map = HashMap::with_hasher(hasher);

    // Support hyperparameters if needed:
    let hyperparameters = match hyperparameters {
        Some(hyperparameters) => {
            serde_json::from_value::<Hyperparameters>(Value::Object(hyperparameters.clone()))
                .map_err(|e| anyhow!("Failed to parse hyperparameters: {}", e))?
        }
        None => Hyperparameters {
            restarts: default_restarts(),
            two_opt_passes: default_two_opt_passes(),
        },
    };

    // use save_solution(&Solution) to save your solution. Overwrites any previous solution

    // return Err(<msg>) if your algorithm encounters an error
    // return Ok(()) if your algorithm is finished

    let n = challenge.distance_matrix.len();
    if n == 0 {
        return Err(anyhow!("Challenge has zero nodes"));
    }


    // Precompute indices 0..n for sampling start nodes
    let indices: Vec<usize> = (0..n).collect();

    let mut best_route = Vec::new();
    let mut best_len = f32::INFINITY;

    for _ in 0..hyperparameters.restarts.max(1) {
        let start = indices[rng.gen_range(0..n)];
        let mut route = greedy_from_start(&challenge.distance_matrix, start);

        // --- Local search: simple 2-opt ---
        two_opt_loop(&challenge.distance_matrix, &mut route, hyperparameters.two_opt_passes);

        let len = route_length(&challenge.distance_matrix, &route);
        if len < best_len {
            best_len = len;
            best_route = route;
        }
    }

    save_solution(&Solution { route: best_route })?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
