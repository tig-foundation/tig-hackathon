use anyhow::{Result};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use std::collections::HashSet;
use std::time::Instant;
use tig_challenges::min_superstring::*;

// --- Main Solver Function ---

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let start_time = Instant::now();
    let n = challenge.strings.len();
    if n == 0 {
        return Ok(());
    }

    // 1. Pre-computation: Generate all unique permutations for each input string.
    let all_perms: Vec<Vec<String>> = challenge
        .strings
        .iter()
        .map(|s| generate_unique_perms(s))
        .collect();

    let mut rng = SmallRng::from_seed(challenge.seed);

    // 2. Initial State Generation using a greedy algorithm.
    let mut current_perms: Vec<String> = challenge.strings.clone();
    let mut current_order: Vec<usize> = greedy_order(&current_perms, n);
    let mut current_len = calculate_len(&current_perms, &current_order);

    let mut best_perms = current_perms.clone();
    let mut best_order = current_order.clone();
    let mut best_len = current_len;
    save_solution(&build_solution(&best_perms, &best_order, n)?)?;

    // 3. Simulated Annealing Loop
    let mut temperature = (best_len as f64) * 0.1; // Heuristic for initial temp
    let cooling_rate = 0.99999; // Slow cooling for more exploration
    let time_limit_ms = 19800;

    while start_time.elapsed().as_millis() < time_limit_ms {
        // Choose a move: 80% reorder, 20% re-permute
        if rng.gen::<f32>() < 0.8 && n > 2 {
            // --- Reorder Move (2-opt) ---
            let i = rng.gen_range(0..n - 1);
            let j = rng.gen_range(i + 1..n);
            
            let mut next_order = current_order.clone();
            next_order[i..=j].reverse();
            let next_len = calculate_len(&current_perms, &next_order);
            let delta_len = next_len as isize - current_len as isize;

            if accept_move(delta_len, temperature, &mut rng) {
                current_order = next_order;
                current_len = next_len;
            }
        } else {
            // --- Re-permute Move ---
            let str_idx_to_change = rng.gen_range(0..n);
            if all_perms[str_idx_to_change].len() <= 1 { continue; }
            
            let new_perm_str = all_perms[str_idx_to_change].choose(&mut rng).unwrap();
            if *new_perm_str == current_perms[str_idx_to_change] { continue; }

            let mut next_perms = current_perms.clone();
            next_perms[str_idx_to_change] = new_perm_str.clone();
            let next_len = calculate_len(&next_perms, &current_order);
            let delta_len = next_len as isize - current_len as isize;

            if accept_move(delta_len, temperature, &mut rng) {
                current_perms = next_perms;
                current_len = next_len;
            }
        }

        // Check if we found a new best solution
        if current_len < best_len {
            best_len = current_len;
            best_perms = current_perms.clone();
            best_order = current_order.clone();
            save_solution(&build_solution(&best_perms, &best_order, n)?)?;
        }

        temperature *= cooling_rate;
    }

    Ok(())
}

/// Simulated Annealing acceptance probability function.
fn accept_move(delta: isize, temp: f64, rng: &mut impl Rng) -> bool {
    delta < 0 || (temp > 1e-9 && rng.gen::<f64>() < (-(delta as f64) / temp).exp())
}

// --- Helper Functions ---

/// Calculates the overlap between the end of s1 and the start of s2.
fn calculate_overlap(s1: &str, s2: &str) -> usize {
    (1..=4)
        .rev()
        .find(|&k| s1.ends_with(&s2[..k]))
        .unwrap_or(0)
}

/// Calculates the total length of the superstring for a given configuration.
fn calculate_len(perms: &[String], order: &[usize]) -> usize {
    if order.is_empty() { return 0; }
    order.windows(2).fold(5, |acc, w| {
        let s1 = &perms[w[0]];
        let s2 = &perms[w[1]];
        acc + 5 - calculate_overlap(s1, s2)
    })
}

/// Constructs a greedy path (order) by always choosing the best next overlap.
fn greedy_order(perms: &[String], n: usize) -> Vec<usize> {
    if n <= 1 { return (0..n).collect(); }
    
    let mut overlaps = vec![vec![0; n]; n];
    let (mut start_node, mut end_node, _) = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .map(|(i, j)| {
            let overlap = calculate_overlap(&perms[i], &perms[j]);
            overlaps[i][j] = overlap;
            (i, j, overlap)
        })
        .max_by_key(|&(_, _, overlap)| overlap)
        .unwrap_or((0, 1, 0));
    
    let mut path = vec![start_node, end_node];
    let mut used = vec![false; n];
    used[start_node] = true;
    used[end_node] = true;

    while path.len() < n {
        let first = path[0];
        let last = *path.last().unwrap();
        
        let best_prepend = (0..n)
            .filter(|&k| !used[k])
            .map(|k| (overlaps[k][first], k))
            .max_by_key(|&(score, _)| score);

        let best_append = (0..n)
            .filter(|&k| !used[k])
            .map(|k| (overlaps[last][k], k))
            .max_by_key(|&(score, _)| score);

        match (best_prepend, best_append) {
            (Some((prepend_score, prepend_node)), Some((append_score, append_node))) => {
                if prepend_score > append_score {
                    path.insert(0, prepend_node);
                    used[prepend_node] = true;
                } else {
                    path.push(append_node);
                    used[append_node] = true;
                }
            },
            (Some((_, prepend_node)), None) => {
                path.insert(0, prepend_node);
                used[prepend_node] = true;
            },
            (None, Some((_, append_node))) => {
                path.push(append_node);
                used[append_node] = true;
            },
            (None, None) => break, // Should not happen if path.len() < n
        }
    }
    path
}

/// Constructs the final Solution struct from the best found state.
fn build_solution(perms: &[String], order: &[usize], n: usize) -> Result<Solution> {
    let mut superstring_idxs = vec![0; n];
    if order.is_empty() {
        return Ok(Solution { permuted_strings: perms.to_vec(), superstring_idxs });
    }
    
    let mut current_pos = 0;
    superstring_idxs[order[0]] = 0;

    for i in 0..order.len() - 1 {
        let prev_idx = order[i];
        let curr_idx = order[i + 1];
        let overlap = calculate_overlap(&perms[prev_idx], &perms[curr_idx]);
        current_pos += 5 - overlap;
        superstring_idxs[curr_idx] = current_pos;
    }

    Ok(Solution {
        permuted_strings: perms.to_vec(),
        superstring_idxs,
    })
}

/// Generates all unique permutations of a string.
fn generate_unique_perms(s: &str) -> Vec<String> {
    let mut chars: Vec<char> = s.chars().collect();
    chars.sort_unstable();
    let mut results = HashSet::new();
    let mut current = String::with_capacity(5);
    let mut used = vec![false; 5];
    
    fn backtrack(
        chars: &[char],
        current: &mut String,
        used: &mut [bool],
        results: &mut HashSet<String>,
    ) {
        if current.len() == 5 {
            results.insert(current.clone());
            return;
        }
        for i in 0..5 {
            if used[i] || (i > 0 && chars[i] == chars[i-1] && !used[i-1]) {
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
    results.into_iter().collect()
}