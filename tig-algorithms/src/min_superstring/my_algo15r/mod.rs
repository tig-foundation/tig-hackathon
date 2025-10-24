use anyhow::Result;
use rand::{rngs::SmallRng, SeedableRng};
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

    // Build permutations (limit per string for speed)
    let all_perms: Vec<Vec<Vec<u8>>> = challenge
        .strings
        .iter()
        .map(|s| {
            let mut perms = generate_unique_perms_bytes(s.as_bytes());
            if perms.len() > 20 {
                perms.truncate(20);
            }
            perms
        })
        .collect();

    // This block is the single biggest improvement over all simulated approaches:
    // For each string, greedily pick its permutation (from all_perms[i]) that,
    // when plugged into the global greedy merge, yields the shortest superstring.
    // Do this iteratively, one string at a time.

    // Start with original permutations
    let mut best_perms = challenge
        .strings
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect::<Vec<_>>();

    // Greedily optimize permutations string by string
    let mut improved = true;
    let max_passes = 2; // Avoids diminishing returns
    let mut passes = 0;
    while improved && passes < max_passes {
        passes += 1;
        improved = false;
        for i in 0..n {
            let mut best_local = best_perms[i].clone();
            let mut best_len = {
                let overlap_matrix = build_overlap_matrix_bytes(&best_perms, n);
                let order = greedy_order_bytes(&overlap_matrix, n);
                calculate_len_fast(&overlap_matrix, &order)
            };
            for perm in &all_perms[i] {
                if *perm == best_perms[i] { continue; }
                let mut candidate = best_perms.clone();
                candidate[i] = perm.clone();
                let overlap_matrix = build_overlap_matrix_bytes(&candidate, n);
                let order = greedy_order_bytes(&overlap_matrix, n);
                let len = calculate_len_fast(&overlap_matrix, &order);
                if len < best_len {
                    best_len = len;
                    best_local = perm.clone();
                    improved = true;
                }
            }
            best_perms[i] = best_local;
        }
    }
    // After local greedy permutation improvement, re-run greedy merge for final answer
    let overlap_matrix = build_overlap_matrix_bytes(&best_perms, n);
    let order = greedy_order_bytes(&overlap_matrix, n);

    save_solution(&build_solution_bytes(&best_perms, &order, n)?)?;
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
    if order.is_empty() { return 0; }
    let mut len = 5;
    for i in 0..order.len() - 1 {
        len += 5 - overlap_matrix[order[i]][order[i + 1]] as usize;
    }
    len
}

// Same greedy merge implementation as before
fn greedy_order_bytes(overlaps: &[Vec<u8>], n: usize) -> Vec<usize> {
    if n <= 1 { return (0..n).collect(); }
    let (start, end) = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .max_by_key(|&(i, j)| overlaps[i][j])
        .unwrap_or((0, 1));
    let mut path = vec![start, end];
    let mut used = vec![false; n]; used[start] = true; used[end] = true;
    while path.len() < n {
        let first = path[0];
        let last = *path.last().unwrap();
        let best_prepend = (0..n)
            .filter(|&k| !used[k])
            .max_by_key(|&k| overlaps[k][first]);
        let best_append = (0..n)
            .filter(|&k| !used[k])
            .max_by_key(|&k| overlaps[last][k]);
        match (best_prepend, best_append) {
            (Some(pre_node), Some(ap_node)) => {
                if overlaps[pre_node][first] > overlaps[last][ap_node] {
                    path.insert(0, pre_node); used[pre_node] = true;
                } else {
                    path.push(ap_node); used[ap_node] = true;
                }
            }
            (Some(pre_node), None) => { path.insert(0, pre_node); used[pre_node] = true; }
            (None, Some(ap_node)) => { path.push(ap_node); used[ap_node] = true; }
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
    let mut cur_pos = 0;
    superstring_idxs[order[0]] = 0;
    for i in 0..order.len() - 1 {
        let prev = order[i]; let curr = order[i + 1];
        let overlap = calculate_overlap_bytes(&perms[prev], &perms[curr]);
        cur_pos += 5 - overlap;
        superstring_idxs[curr] = cur_pos;
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
            used[i] = true; current.push(chars[i]);
            backtrack(chars, current, used, results);
            current.pop(); used[i] = false;
        }
    }
    backtrack(&chars, &mut current, &mut used, &mut results);
    // Remove dups
    results.sort_unstable();
    results.dedup();
    results
}
