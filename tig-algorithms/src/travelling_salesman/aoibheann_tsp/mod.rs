// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::travelling_salesman::*;

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
    if route.is_empty() {
        return 0.0;
    }
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
    if n < 4 {
        return false;
    }

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
        if pass_cap >= 0 && passes >= pass_cap {
            break;
        }
        if !two_opt_pass(distance, route) {
            break;
        }
        passes += 1;
    }
}
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    // println!("Hello World");
    // If you need random numbers, recommend using SmallRng with challenge.seed:
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    let mut rng = SmallRng::from_seed(challenge.seed);

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

    for _ in 0..1 {
        let start = indices[rng.gen_range(0..n)];
        let mut route = greedy_from_start(&challenge.distance_matrix, start);

        // --- Local search: simple 2-opt ---
        two_opt_loop(&challenge.distance_matrix, &mut route, -1);

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
