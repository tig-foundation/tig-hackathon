// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use anyhow::{anyhow, Result};
use std::collections::HashSet;
use tig_challenges::travelling_salesman::*;

/// Entry point — single-threaded farthest insertion + 2-opt solver
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.difficulty.size;
    let dist = &challenge.distance_matrix;

    let mut best_path: Option<Vec<usize>> = None;
    let mut best_length = f32::INFINITY;

    // Select starting points — all for small instances, subset for larger
    let starts: Vec<usize> = if n <= 20 {
        (0..n).collect()
    } else {
        let step = std::cmp::max(1, n / 10);
        (0..n).step_by(step).collect()
    };

    for start in starts {
        if let Some(path) = farthest_insertion(n, dist, start, best_length) {
            let length = tour_length(dist, &path);
            if length < best_length {
                best_length = length;
                best_path = Some(path);
            }
        }
    }

    // Apply 2-opt improvement
    let final_path = if let Some(path) = best_path {
        let improved = two_opt(dist, path.clone());
        let improved_len = tour_length(dist, &improved);
        if improved_len < best_length {
            improved
        } else {
            path
        }
    } else {
        return Err(anyhow!("No valid route found"));
    };

    save_solution(&Solution { route: final_path })?;
    Ok(())
}

/// Farthest insertion heuristic with pruning
fn farthest_insertion(
    n: usize,
    dist: &[Vec<f32>],
    start: usize,
    cutoff: f32,
) -> Option<Vec<usize>> {
    if n < 2 {
        return Some(vec![start]);
    }

    let mut path = vec![start];
    let mut unvisited: HashSet<usize> = (0..n).collect();
    unvisited.remove(&start);

    // Start with the farthest node from start
    let farthest = *unvisited
        .iter()
        .max_by(|&&a, &&b| {
            dist[start][a]
                .partial_cmp(&dist[start][b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    path.push(farthest);
    unvisited.remove(&farthest);

    let mut current_len = dist[start][farthest] + dist[farthest][start];

    while !unvisited.is_empty() {
        // Find the unvisited node farthest from any node in the current tour
        let (node, _) = unvisited
            .iter()
            .map(|&u| {
                let dmin = path.iter().map(|&p| dist[u][p]).fold(f32::INFINITY, f32::min);
                (u, dmin)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();


        let (pos, delta) = best_insertion(&path, dist, node);

        if current_len + delta > cutoff {
            // prune weak branches
            return None;
        }

        path.insert(pos, node);
        unvisited.remove(&node);
        current_len += delta;
    }

    Some(path)
}

/// Compute best insertion position for node minimizing added cost
#[inline]
fn best_insertion(path: &[usize], dist: &[Vec<f32>], node: usize) -> (usize, f32) {
    let mut best_cost = f32::INFINITY;
    let mut best_pos = 0;

    for i in 0..path.len() {
        let a = path[i];
        let b = path[(i + 1) % path.len()];
        let cost = dist[a][node] + dist[node][b] - dist[a][b];
        if cost < best_cost {
            best_cost = cost;
            best_pos = i + 1;
        }
    }

    (best_pos, best_cost)
}

/// 2-opt local search improvement
fn two_opt(dist: &[Vec<f32>], mut path: Vec<usize>) -> Vec<usize> {
    let n = path.len();
    if n < 4 {
        return path;
    }

    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = std::cmp::max(20, n / 2);

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        for i in 0..n - 3 {
            let a = path[i];
            let b = path[i + 1];
            for j in i + 2..n - 1 {
                let c = path[j];
                let d = path[j + 1];

                let delta = (dist[a][c] + dist[b][d]) - (dist[a][b] + dist[c][d]);
                if delta < -1e-6 {
                    path[i + 1..=j].reverse();
                    improved = true;
                }
            }
        }
    }

    path
}

/// Compute total tour length (closed loop)
#[inline]
fn tour_length(dist: &[Vec<f32>], path: &[usize]) -> f32 {
    let mut total = 0.0;
    for i in 0..path.len() - 1 {
        total += dist[path[i]][path[i + 1]];
    }
    total += dist[path[path.len() - 1]][path[0]];
    total
}


// Important! Do not include any tests in this file, it will result in your submission being rejected