// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
// Farthest Insertion + 2-opt for the Travelling Salesman Problem
use anyhow::{anyhow, Result};
use tig_challenges::travelling_salesman::*;
use std::cmp::Ordering;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.difficulty.size;
    let dm = &challenge.distance_matrix;

    if n == 0 {
        return Err(anyhow!("Empty instance"));
    }

    let mut best_path: Option<Vec<usize>> = None;
    let mut best_length = f32::INFINITY;

    // Try several deterministic starting anchors:
    // - all nodes for small n
    // - evenly spaced subset for larger n
    let strategies: Vec<usize> = if n <= 20 {
        (0..n).collect()
    } else {
        let step = (n / 12).max(1);
        (0..n).step_by(step).collect()
    };

    for start in strategies {
        if let Some((tour, mut length)) = farthest_insertion(n, dm, start, best_length) {
            // 2-opt first-improvement on the cycle
            let (tour2, len2) = two_opt_improvement(dm, tour, length, 10 * n.max(10));
            length = len2;

            if length < best_length {
                best_length = length;
                best_path = Some(tour2);
            }
        }
    }

    let final_solution = best_path.ok_or_else(|| anyhow!("No valid solution found"))?;
    save_solution(&Solution { route: final_solution })?;
    Ok(())
}

/// Proper Farthest Insertion:
/// 1) Build an initial 3-node tour (farthest pair + best third).
/// 2) Repeatedly pick the unvisited node farthest from the *tour* (min distance to any tour node),
///    and insert it in the position of minimum length increase.
/// Maintains and returns the *closed* tour length. Prunes if it already exceeds `best_so_far`.
fn farthest_insertion(
    n: usize,
    dm: &[Vec<f32>],
    start: usize,
    best_so_far: f32,
) -> Option<(Vec<usize>, f32)> {
    match n {
        1 => return Some((vec![start], 0.0)),
        2 => {
            let other = if start == 0 { 1 } else { 0 };
            let len = dm[start][other] + dm[other][start];
            return Some((vec![start, other], len));
        }
        _ => {}
    }

    // Step 1: initial 3-node cycle
    let a = start;
    let b = (0..n).filter(|&k| k != a)
        .max_by(|&x, &y| fcmp(dm[a][x], dm[a][y]))
        ?;
    let c = (0..n).filter(|&k| k != a && k != b)
        .max_by(|&x, &y| fcmp(dm[a][x] + dm[b][x], dm[a][y] + dm[b][y]))
        ?;

    let mut tour = vec![a, b, c];
    let mut in_tour = vec![false; n];
    for &v in &tour { in_tour[v] = true; }

    // Closed tour length
    let mut length = cycle_length(dm, &tour);

    // Distance from each node to current tour (min over tour nodes)
    let mut dist_to_tour = vec![f32::INFINITY; n];
    for v in 0..n {
        if !in_tour[v] {
            dist_to_tour[v] = tour.iter().map(|&t| dm[v][t]).fold(f32::INFINITY, f32::min);
        }
    }

    // Step 2: insert remaining nodes
    for _ in 0..(n - tour.len()) {
        // Pick the node farthest from the current tour
        let u = (0..n).filter(|&v| !in_tour[v])
            .max_by(|&x, &y| fcmp(dist_to_tour[x], dist_to_tour[y]))?;

        // Find cheapest insertion position (between i and i+1, wrap around)
        let (best_pos, best_delta) = {
            let mut best_pos = 0usize;
            let mut best_delta = f32::INFINITY;
            for i in 0..tour.len() {
                let j = (i + 1) % tour.len();
                let a = tour[i];
                let b = tour[j];
                let delta = dm[a][u] + dm[u][b] - dm[a][b];
                if delta < best_delta {
                    best_delta = delta;
                    best_pos = j;
                }
            }
            (best_pos, best_delta)
        };

        // Apply insertion
        tour.insert(best_pos, u);
        in_tour[u] = true;
        length += best_delta;

        // Early prune: we already exceed best known closed length
        if length > best_so_far {
            return None;
        }

        // Update distances to tour incrementally
        for v in 0..n {
            if !in_tour[v] {
                let dv = dm[v][u];
                if dv < dist_to_tour[v] {
                    dist_to_tour[v] = dv;
                }
            }
        }
    }

    Some((tour, length))
}

/// First-improvement 2-opt on a *cycle* with wrap-around.
/// Returns (improved_tour, improved_length). `max_passes` caps work.
fn two_opt_improvement(
    dm: &[Vec<f32>],
    mut tour: Vec<usize>,
    mut length: f32,
    max_passes: usize,
) -> (Vec<usize>, f32) {
    let n = tour.len();
    if n < 4 { return (tour, length); }

    let mut improved = true;
    let mut passes = 0usize;

    while improved && passes < max_passes {
        improved = false;
        passes += 1;

        'outer: for i in 0..n {
            let i_next = (i + 1) % n;
            let a = tour[i];
            let b = tour[i_next];

            // j must be at least i+2, and we skip the case that reconnects the same edge (i=0, j=n-1)
            let mut j = (i + 2) % n;
            while j != i {
                let j_next = (j + 1) % n;
                if !(i == 0 && j == n - 1) {
                    let c = tour[j];
                    let d = tour[j_next];

                    let delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d];
                    if delta < -1e-6 {
                        // reverse segment (i_next..=j) in linear index space
                        reverse_segment_cyclic(&mut tour, i_next, j);
                        length += delta;
                        improved = true;
                        break 'outer;
                    }
                }
                j = (j + 1) % n;
            }
        }
    }

    (tour, length)
}

/// Reverse tour segment on a cycle between indices [l..r] inclusive (cyclic-safe).
fn reverse_segment_cyclic(tour: &mut [usize], l: usize, r: usize) {
    let n = tour.len();
    let mut i = l;
    let mut j = r;
    let steps = ((r + n - l) % n + 1) / 2;
    for _ in 0..steps {
        tour.swap(i, j);
        i = (i + 1) % n;
        j = (j + n - 1) % n;
    }
}

/// Closed tour length (includes edge last->first).
fn cycle_length(dm: &[Vec<f32>], path: &[usize]) -> f32 {
    match path.len() {
        0 | 1 => 0.0,
        2 => dm[path[0]][path[1]] + dm[path[1]][path[0]],
        _ => {
            let mut total = 0.0;
            for i in 0..path.len() {
                let j = (i + 1) % path.len();
                total += dm[path[i]][path[j]];
            }
            total
        }
    }
}

#[inline]
fn fcmp(a: f32, b: f32) -> Ordering {
    // Be defensive with NaNs: treat NaN as "worse"
    a.partial_cmp(&b).unwrap_or_else(|| {
        if a.is_nan() && b.is_nan() { Ordering::Equal }
        else if a.is_nan() { Ordering::Less } else { Ordering::Greater }
    })
}