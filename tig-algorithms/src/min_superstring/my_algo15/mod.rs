use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::Instant;
use tig_challenges::min_superstring::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let start_time = Instant::now();
    let n = challenge.strings.len();
    if n == 0 {
        return Ok(());
    }

    let mut rng = SmallRng::from_seed(challenge.seed);

    // Generate permutations only when needed and cache them
    let all_perms: Vec<Vec<Vec<u8>>> = challenge
        .strings
        .iter()
        .map(|s| {
            let mut perms = generate_unique_perms_bytes(s.as_bytes());
            if perms.len() > 20 {
                // Limit permutations to most promising ones
                perms.truncate(20);
            }
            perms
        })
        .collect();

    // Work with byte arrays for faster comparison
    let mut current_perms: Vec<Vec<u8>> = challenge
        .strings
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();

    // Initial greedy order
    let mut overlap_matrix = build_overlap_matrix_bytes(&current_perms, n);
    let mut current_order = greedy_order_bytes(&overlap_matrix, n);
    let mut current_len = calculate_len_fast(&overlap_matrix, &current_order);

    let mut best_perms = current_perms.clone();
    let mut best_order = current_order.clone();
    let mut best_len = current_len;

    save_solution(&build_solution_bytes(&best_perms, &best_order, n)?)?;

    // Aggressive simulated annealing parameters
    let mut temperature = (best_len as f64) * 0.2;
    let cooling_rate = 0.9998;
    let time_limit_ms = 19000;
    
    let mut iterations = 0u64;
    let mut last_improvement = 0u64;
    let max_no_improvement = 100000;

    // Position lookup for O(1) access
    let mut pos_in_order = vec![0; n];
    for (pos, &idx) in current_order.iter().enumerate() {
        pos_in_order[idx] = pos;
    }

    while start_time.elapsed().as_millis() < time_limit_ms {
        iterations += 1;

        // Restart if stuck for too long
        if iterations - last_improvement > max_no_improvement {
            temperature = (best_len as f64) * 0.15;
            last_improvement = iterations;
        }

        // Favor reordering (90% vs 10%) - it's much faster
        if rng.gen::<f32>() < 0.9 && n > 2 {
            // 2-opt move
            let i = rng.gen_range(0..n - 1);
            let j = rng.gen_range(i + 1..n.min(i + 6)); // Limit range for faster computation

            let delta = calculate_2opt_delta_optimized(&overlap_matrix, &current_order, i, j);

            if delta < 0 || (temperature > 1e-9 && rng.gen::<f64>() < (-(delta as f64) / temperature).exp()) {
                // Apply move
                for k in i..=j {
                    pos_in_order[current_order[k]] = j - (k - i);
                }
                current_order[i..=j].reverse();
                current_len = (current_len as isize + delta) as usize;

                if current_len < best_len {
                    best_len = current_len;
                    best_perms = current_perms.clone();
                    best_order = current_order.clone();
                    last_improvement = iterations;
                    save_solution(&build_solution_bytes(&best_perms, &best_order, n)?)?;
                }
            }
        } else {
            // Permutation move
            let str_idx = rng.gen_range(0..n);
            if all_perms[str_idx].len() <= 1 {
                continue;
            }

            let perm_idx = rng.gen_range(0..all_perms[str_idx].len());
            let new_perm = &all_perms[str_idx][perm_idx];

            if *new_perm == current_perms[str_idx] {
                continue;
            }

            let pos = pos_in_order[str_idx];
            let delta = calculate_permute_delta_optimized(
                &overlap_matrix,
                &current_order,
                &current_perms,
                str_idx,
                new_perm,
                pos,
            );

            if delta < 0 || (temperature > 1e-9 && rng.gen::<f64>() < (-(delta as f64) / temperature).exp()) {
                update_overlap_matrix_bytes(&mut overlap_matrix, &current_perms, str_idx, new_perm, n);
                current_perms[str_idx] = new_perm.clone();
                current_len = (current_len as isize + delta) as usize;

                if current_len < best_len {
                    best_len = current_len;
                    best_perms = current_perms.clone();
                    best_order = current_order.clone();
                    last_improvement = iterations;
                    save_solution(&build_solution_bytes(&best_perms, &best_order, n)?)?;
                }
            }
        }

        temperature *= cooling_rate;
    }

    Ok(())
}

#[inline(always)]
fn calculate_overlap_bytes(s1: &[u8], s2: &[u8]) -> usize {
    for k in (1..=4.min(s1.len().min(s2.len()))).rev() {
        if s1[s1.len() - k..] == s2[..k] {
            return k;
        }
    }
    0
}

fn build_overlap_matrix_bytes(perms: &[Vec<u8>], n: usize) -> Vec<Vec<u8>> {
    let mut matrix = vec![vec![0u8; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                matrix[i][j] = calculate_overlap_bytes(&perms[i], &perms[j]) as u8;
            }
        }
    }
    matrix
}

#[inline(always)]
fn calculate_len_fast(overlap_matrix: &[Vec<u8>], order: &[usize]) -> usize {
    if order.is_empty() {
        return 0;
    }
    let mut len = 5;
    for i in 0..order.len() - 1 {
        len += 5 - overlap_matrix[order[i]][order[i + 1]] as usize;
    }
    len
}

#[inline(always)]
fn calculate_2opt_delta_optimized(
    overlap_matrix: &[Vec<u8>],
    order: &[usize],
    i: usize,
    j: usize,
) -> isize {
    let n = order.len();
    let mut delta = 0isize;

    // Boundary edges
    if i > 0 {
        delta += overlap_matrix[order[i - 1]][order[i]] as isize;
        delta -= overlap_matrix[order[i - 1]][order[j]] as isize;
    }

    if j + 1 < n {
        delta += overlap_matrix[order[j]][order[j + 1]] as isize;
        delta -= overlap_matrix[order[i]][order[j + 1]] as isize;
    }

    // Internal edges
    for k in i..j {
        delta += overlap_matrix[order[k]][order[k + 1]] as isize;
        delta -= overlap_matrix[order[k + 1]][order[k]] as isize;
    }

    delta
}

#[inline(always)]
fn calculate_permute_delta_optimized(
    overlap_matrix: &[Vec<u8>],
    order: &[usize],
    current_perms: &[Vec<u8>],
    str_idx: usize,
    new_perm: &[u8],
    pos: usize,
) -> isize {
    let n = order.len();
    let mut delta = 0isize;

    if pos > 0 {
        let pred = order[pos - 1];
        delta += overlap_matrix[pred][str_idx] as isize;
        delta -= calculate_overlap_bytes(&current_perms[pred], new_perm) as isize;
    }

    if pos + 1 < n {
        let succ = order[pos + 1];
        delta += overlap_matrix[str_idx][succ] as isize;
        delta -= calculate_overlap_bytes(new_perm, &current_perms[succ]) as isize;
    }

    delta
}

#[inline(always)]
fn update_overlap_matrix_bytes(
    overlap_matrix: &mut Vec<Vec<u8>>,
    current_perms: &[Vec<u8>],
    changed_idx: usize,
    new_perm: &[u8],
    n: usize,
) {
    for i in 0..n {
        if i != changed_idx {
            overlap_matrix[changed_idx][i] = calculate_overlap_bytes(new_perm, &current_perms[i]) as u8;
            overlap_matrix[i][changed_idx] = calculate_overlap_bytes(&current_perms[i], new_perm) as u8;
        }
    }
}

fn greedy_order_bytes(overlaps: &[Vec<u8>], n: usize) -> Vec<usize> {
    if n <= 1 {
        return (0..n).collect();
    }

    let (start_node, end_node) = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .max_by_key(|&(i, j)| overlaps[i][j])
        .unwrap_or((0, 1));

    let mut path = vec![start_node, end_node];
    let mut used = vec![false; n];
    used[start_node] = true;
    used[end_node] = true;

    while path.len() < n {
        let first = path[0];
        let last = path[path.len() - 1];

        let best_prepend = (0..n)
            .filter(|&k| !used[k])
            .max_by_key(|&k| overlaps[k][first]);

        let best_append = (0..n)
            .filter(|&k| !used[k])
            .max_by_key(|&k| overlaps[last][k]);

        match (best_prepend, best_append) {
            (Some(prepend_node), Some(append_node)) => {
                if overlaps[prepend_node][first] > overlaps[last][append_node] {
                    path.insert(0, prepend_node);
                    used[prepend_node] = true;
                } else {
                    path.push(append_node);
                    used[append_node] = true;
                }
            }
            (Some(prepend_node), None) => {
                path.insert(0, prepend_node);
                used[prepend_node] = true;
            }
            (None, Some(append_node)) => {
                path.push(append_node);
                used[append_node] = true;
            }
            (None, None) => break,
        }
    }

    path
}

fn build_solution_bytes(perms: &[Vec<u8>], order: &[usize], n: usize) -> Result<Solution> {
    let mut superstring_idxs = vec![0; n];
    if order.is_empty() {
        return Ok(Solution {
            permuted_strings: perms.iter().map(|p| String::from_utf8_lossy(p).to_string()).collect(),
            superstring_idxs,
        });
    }

    let mut current_pos = 0;
    superstring_idxs[order[0]] = 0;

    for i in 0..order.len() - 1 {
        let prev_idx = order[i];
        let curr_idx = order[i + 1];
        let overlap = calculate_overlap_bytes(&perms[prev_idx], &perms[curr_idx]);
        current_pos += 5 - overlap;
        superstring_idxs[curr_idx] = current_pos;
    }

    Ok(Solution {
        permuted_strings: perms.iter().map(|p| String::from_utf8_lossy(p).to_string()).collect(),
        superstring_idxs,
    })
}

fn generate_unique_perms_bytes(s: &[u8]) -> Vec<Vec<u8>> {
    let mut chars = s.to_vec();
    chars.sort_unstable();
    
    let mut results = Vec::new();
    let mut current = Vec::with_capacity(5);
    let mut used = [false; 5];

    fn backtrack(
        chars: &[u8],
        current: &mut Vec<u8>,
        used: &mut [bool; 5],
        results: &mut Vec<Vec<u8>>,
    ) {
        if current.len() == 5 {
            results.push(current.clone());
            return;
        }
        for i in 0..5 {
            if used[i] || (i > 0 && chars[i] == chars[i - 1] && !used[i - 1]) {
                continue;
            }
            used[i] = true;
            current.push(chars[i]);
            backtrack(chars, current, used, results);
            current.pop();
            used[i] = false;
        }
    }

    backtrack(&chars, &mut current, &mut used, &mut results);
    
    // Remove duplicates efficiently
    results.sort_unstable();
    results.dedup();
    results
}
