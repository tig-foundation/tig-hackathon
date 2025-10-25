// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge

use anyhow::{anyhow, Result};
use tig_challenges::travelling_salesman::*;

/// Entry point — optimized and validated TSP solver
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.difficulty.size;
    let dist = &challenge.distance_matrix;

    let mut best_path: Option<Vec<usize>> = None;
    let mut best_length = f32::INFINITY;

    // Choose a subset of starting points for large n
    let starts: Vec<usize> = if n <= 20 {
        (0..n).collect()
    } else {
        let step = std::cmp::max(1, n / 10);
        (0..n).step_by(step).collect()
    };

    for &start in &starts {
        if let Some(path) = farthest_insertion(n, dist, start, best_length) {
            // Validate path
            if !is_valid_tour(&path, n) {
                continue;
            }
            let length = fast_tour_length(dist, &path);
            if length < best_length {
                best_length = length;
                best_path = Some(path);
            }
        }
    }

    let final_path = if let Some(path) = best_path {
        let improved = two_opt_fast(dist, path.clone());
        let improved_len = fast_tour_length(dist, &improved);
        if improved_len < best_length {
            improved
        } else {
            path
        }
    } else {
        return Err(anyhow!("No valid route found"));
    };

    if !is_valid_tour(&final_path, n) {
        return Err(anyhow!("Generated invalid tour"));
    }

    save_solution(&Solution { route: final_path })?;
    Ok(())
}

/// Farthest insertion with correctness guarantees
fn farthest_insertion(n: usize, dist: &[Vec<f32>], start: usize, cutoff: f32) -> Option<Vec<usize>> {
    if n < 2 {
        return Some(vec![start]);
    }

    let mut path = vec![start];
    let mut unvisited = vec![true; n];
    unvisited[start] = false;

    // pick farthest node from start
    let (farthest, _) = (0..n)
        .filter(|&i| unvisited[i])
        .map(|i| (i, dist[start][i]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    path.push(farthest);
    unvisited[farthest] = false;

    let mut current_len = dist[start][farthest] + dist[farthest][start];

    // precompute minimal distances to path
    let mut dmin = vec![f32::INFINITY; n];
    for i in 0..n {
        if unvisited[i] {
            dmin[i] = dist[i][start].min(dist[i][farthest]);
        }
    }

    loop {
        // find farthest remaining node
        let mut node = None;
        let mut max_d = f32::MIN;
        for i in 0..n {
            if unvisited[i] && dmin[i] > max_d {
                max_d = dmin[i];
                node = Some(i);
            }
        }
        let node = match node {
            Some(v) => v,
            None => break, // all visited
        };

        let (pos, delta) = best_insertion_fast(&path, dist, node);
        if current_len + delta > cutoff {
            // prune this branch
            return None;
        }

        path.insert(pos, node);
        unvisited[node] = false;
        current_len += delta;

        // update dmin only for unvisited nodes
        for i in 0..n {
            if unvisited[i] {
                let new_d = dist[i][node];
                if new_d < dmin[i] {
                    dmin[i] = new_d;
                }
            }
        }
    }

    // final validation
    if path.len() == n {
        Some(path)
    } else {
        None
    }
}

#[inline(always)]
fn best_insertion_fast(path: &[usize], dist: &[Vec<f32>], node: usize) -> (usize, f32) {
    let mut best_cost = f32::INFINITY;
    let mut best_pos = 0;
    let plen = path.len();

    for i in 0..plen {
        let a = path[i];
        let b = path[(i + 1) % plen];
        let cost = dist[a][node] + dist[node][b] - dist[a][b];
        if cost < best_cost {
            best_cost = cost;
            best_pos = i + 1;
        }
    }

    (best_pos, best_cost)
}

fn two_opt_fast(dist: &[Vec<f32>], mut path: Vec<usize>) -> Vec<usize> {
    let n = path.len();
    if n < 4 {
        return path;
    }

    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = std::cmp::max(10, n / 4);

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
                    break; // early exit
                }
            }
            if improved {
                break;
            }
        }
    }

    path
}

#[inline(always)]
fn fast_tour_length(dist: &[Vec<f32>], path: &[usize]) -> f32 {
    let mut total = 0.0;
    let n = path.len();
    for i in 0..n - 1 {
        total += dist[path[i]][path[i + 1]];
    }
    total + dist[path[n - 1]][path[0]]
}

/// simple sanity check for route validity
fn is_valid_tour(path: &[usize], n: usize) -> bool {
    if path.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &p in path {
        if p >= n || seen[p] {
            return false;
        }
        seen[p] = true;
    }
    true
}


// Important! Do not include any tests in this file, it will result in your submission being rejected