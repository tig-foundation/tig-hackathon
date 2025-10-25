// Fast Min-Superstring with intra-string permutations allowed.
// Strategy:
// 1) Build asymmetric cost matrix cost[i][j] = 5 - min(4, |multiset_intersection(i,j)|).
// 2) Build a *cycle* with Farthest Insertion + 2-opt (first improvement).
// 3) Break the heaviest edge -> path order.
// 4) Assign concrete overlap letters greedily and build concrete permutations & indices.

use anyhow::{anyhow, Result};
use std::cmp::Ordering;
use tig_challenges::min_superstring::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.strings.len();
    if n == 0 {
        return Ok(());
    }
    // We assume ASCII (typical for TIG). Use 128-size histograms for speed.
    const A: usize = 128;
    const BLOCK: usize = 5;
    const MAX_OVERLAP: u8 = 4;

    // 1) Precompute letter counts and overlap-based costs
    let mut counts: Vec<[u8; A]> = vec![[0u8; A]; n];
    for (i, s) in challenge.strings.iter().enumerate() {
        let mut k = 0usize;
        for ch in s.chars() {
            let b = ch as usize;
            if b < A {
                counts[i][b] = counts[i][b].saturating_add(1);
            }
            k += 1;
        }
        // Defensive: ensure exactly 5 chars; otherwise we still work, but BLOCK should match.
        debug_assert!(k == BLOCK, "expected 5-char strings, got {}", k);
    }

    // overlap_ij[i][j] = max achievable overlap (0..=4)
    let mut overlap_ij: Vec<Vec<u8>> = vec![vec![0u8; n]; n];
    let mut cost: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                overlap_ij[i][j] = 0;
                cost[i][j] = 5.0;
            } else {
                let mut inter = 0u8;
                // sum of min(counts) across ASCII
                for c in 0..A {
                    inter = inter.saturating_add(counts[i][c].min(counts[j][c]));
                    if inter >= MAX_OVERLAP { break; }
                }
                let ov = inter.min(MAX_OVERLAP);
                overlap_ij[i][j] = ov;
                cost[i][j] = (BLOCK as i32 - ov as i32) as f32;
            }
        }
    }

    // 2) Build a *cycle* quickly, try a few deterministic anchors (like your fast TSP)
    let mut best_cycle: Option<Vec<usize>> = None;
    let mut best_len = f32::INFINITY;

    let strategies: Vec<usize> = if n <= 20 { (0..n).collect() } else {
        let step = (n / 12).max(1);
        (0..n).step_by(step).collect()
    };

    for start in strategies {
        if let Some((tour, mut len)) = farthest_insertion_cycle(n, &cost, start, best_len) {
            let (tour2, len2) = two_opt_cycle_first_improvement(&cost, tour, len, 10 * n.max(10));
            len = len2;
            if len < best_len {
                best_len = len;
                best_cycle = Some(tour2);
            }
        }
    }
    let cycle = best_cycle.ok_or_else(|| anyhow!("Failed to build cycle"))?;

    // 3) Break the heaviest edge to get a *path* order
    let path = break_cycle_to_path(&cost, &cycle);

    // 4) Allocate actual overlap letters and build concrete permutations + indices
    let (permuted, idxs) = realize_permutations_and_indices(&challenge.strings, &counts, &overlap_ij, &path);

    save_solution(&Solution {
        permuted_strings: permuted,
        superstring_idxs: idxs,
    })?;
    Ok(())
}

// ---------- Cycle construction (Farthest Insertion + 2-opt) ----------

fn farthest_insertion_cycle(
    n: usize,
    dm: &[Vec<f32>],
    start: usize,
    prune_if_len_exceeds: f32,
) -> Option<(Vec<usize>, f32)> {
    match n {
        0 => return None,
        1 => return Some((vec![start], 0.0)),
        2 => {
            let other = if start == 0 { 1 } else { 0 };
            let len = dm[start][other] + dm[other][start];
            return Some((vec![start, other], len));
        }
        _ => {}
    }

    // Initial 3-node cycle: farthest pair from start, then best third
    let a = start;
    let b = (0..n).filter(|&k| k != a)
        .max_by(|&x, &y| fcmp(dm[a][x], dm[a][y]))?;
    let c = (0..n).filter(|&k| k != a && k != b)
        .max_by(|&x, &y| fcmp(dm[a][x] + dm[b][x], dm[a][y] + dm[b][y]))?;

    let mut tour = vec![a, b, c];
    let mut in_tour = vec![false; n];
    for &v in &tour { in_tour[v] = true; }

    let mut length = cycle_length(dm, &tour);

    // Dist to tour = min over tour nodes (use undirected-ish signal via min(dm[v][t], dm[t][v]))
    let mut dist_to_tour = vec![f32::INFINITY; n];
    for v in 0..n {
        if !in_tour[v] {
            let mut best = f32::INFINITY;
            for &t in &tour {
                let d = dm[v][t].min(dm[t][v]);
                if d < best { best = d; }
            }
            dist_to_tour[v] = best;
        }
    }

    for _ in 0..(n - tour.len()) {
        // farthest from tour
        let u = (0..n).filter(|&v| !in_tour[v])
            .max_by(|&x, &y| fcmp(dist_to_tour[x], dist_to_tour[y]))?;

        // best insertion position (between i and i+1, wrap)
        let (best_pos, best_delta) = {
            let mut best_pos = 0usize;
            let mut best_delta = f32::INFINITY;
            for i in 0..tour.len() {
                let j = (i + 1) % tour.len();
                let a = tour[i];
                let b = tour[j];
                // oriented cycle: edge a->b is replaced by a->u + u->b
                let delta = dm[a][u] + dm[u][b] - dm[a][b];
                if delta < best_delta {
                    best_delta = delta;
                    best_pos = j;
                }
            }
            (best_pos, best_delta)
        };

        tour.insert(best_pos, u);
        in_tour[u] = true;
        length += best_delta;

        if length > prune_if_len_exceeds {
            return None;
        }

        // update dist_to_tour
        for v in 0..n {
            if !in_tour[v] {
                let mut best = dist_to_tour[v];
                for &t in &[u] {
                    let d = dm[v][t].min(dm[t][v]);
                    if d < best { best = d; }
                }
                dist_to_tour[v] = best;
            }
        }
    }

    Some((tour, length))
}

fn two_opt_cycle_first_improvement(
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

            let mut j = (i + 2) % n;
            while j != i {
                let j_next = (j + 1) % n;
                if !(i == 0 && j == n - 1) {
                    let c = tour[j];
                    let d = tour[j_next];
                    let delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d];
                    if delta < -1e-6 {
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
    a.partial_cmp(&b).unwrap_or_else(|| {
        if a.is_nan() && b.is_nan() { Ordering::Equal }
        else if a.is_nan() { Ordering::Less } else { Ordering::Greater }
    })
}

fn break_cycle_to_path(dm: &[Vec<f32>], cycle: &[usize]) -> Vec<usize> {
    let n = cycle.len();
    if n <= 2 { return cycle.to_vec(); }

    // Remove the *heaviest* edge in the cycle
    let mut worst_edge = 0usize;
    let mut worst_cost = f32::NEG_INFINITY;
    for i in 0..n {
        let j = (i + 1) % n;
        let c = dm[cycle[i]][cycle[j]];
        if c > worst_cost {
            worst_cost = c;
            worst_edge = j; // start path at "j"
        }
    }
    // Rotate so path starts at worst_edge and ends at index worst_edge-1
    let mut path = Vec::with_capacity(n);
    for k in 0..n {
        path.push(cycle[(worst_edge + k) % n]);
    }
    path
}

// ---------- Realizing concrete permutations + indices ----------

fn realize_permutations_and_indices(
    strings: &[String],
    counts: &[[u8; 128]],
    overlap_ij: &[Vec<u8>],
    order: &[usize],
) -> (Vec<String>, Vec<usize>) {
    const A: usize = 128;
    const BLOCK: usize = 5;

    let n = order.len();
    let mut rem: Vec<[i8; A]> = counts.iter().map(|h| {
        let mut r = [0i8; A];
        for i in 0..A { r[i] = h[i] as i8; }
        r
    }).collect();

    // Sequences for left (prefix) and right (suffix) overlaps per node in path
    let mut left_seq: Vec<Vec<u8>> = vec![Vec::new(); n];
    let mut right_seq: Vec<Vec<u8>> = vec![Vec::new(); n];

    // Allocate overlaps along the path greedily (deterministic, a..z..ASCII order)
    for t in 0..n.saturating_sub(1) {
        let i = order[t];
        let j = order[t + 1];
        let target = overlap_ij[i][j] as usize;

        // pick letters common to rem[i] & rem[j], up to target
        let mut seq = Vec::with_capacity(target);
        if target > 0 {
            for c in 0..A {
                if seq.len() == target { break; }
                let take = rem[i][c].min(rem[j][c]).max(0);
                if take > 0 {
                    let take_u = take as usize;
                    let need = target - seq.len();
                    let use_k = take_u.min(need);
                    for _ in 0..use_k {
                        seq.push(c as u8);
                    }
                    rem[i][c] -= use_k as i8;
                    rem[j][c] -= use_k as i8;
                }
            }
        }
        right_seq[t] = seq.clone();
        left_seq[t + 1] = seq;
    }

    // Build actual permuted strings: [left_seq] + middle leftovers + [right_seq]
    let mut permuted = vec![String::new(); strings.len()];

    // Map from node index -> position in path
    let mut pos_in_path = vec![usize::MAX; strings.len()];
    for (p, &v) in order.iter().enumerate() { pos_in_path[v] = p; }

    for &v in order {
        let p = pos_in_path[v];
        let mut s = String::with_capacity(BLOCK);

        // prefix (left)
        for &b in &left_seq[p] {
            s.push((b as char));
        }
        // middle (remaining)
        let mut remaining = 0usize;
        for c in 0..A { remaining += rem[v][c].max(0) as usize; }
        if remaining > 0 {
            for c in 0..A {
                let k = rem[v][c].max(0) as usize;
                for _ in 0..k { s.push(c as u8 as char); }
                rem[v][c] -= k as i8;
            }
        }
        // suffix (right)
        for &b in &right_seq[p] {
            s.push((b as char));
        }

        // If due to earlier reductions we got shorter/longer than 5, pad/trim deterministically.
        // (Should be rare; mostly exactly 5.)
        if s.chars().count() < BLOCK {
            // Pad with original letters (won't normally happen because rem hits zero)
            let orig = &strings[v];
            for ch in orig.chars() {
                if s.chars().count() >= BLOCK { break; }
                s.push(ch);
            }
        } else if s.chars().count() > BLOCK {
            // Trim middle
            let mut out = String::with_capacity(BLOCK);
            // Keep prefix, then fill until BLOCK, then ensure suffix matches
            for ch in s.chars().take(BLOCK) { out.push(ch); }
            s = out;
        }

        permuted[v] = s;
    }

    // Compute superstring indices from realized overlaps
    let mut idxs = vec![0usize; strings.len()];
    if !order.is_empty() {
        idxs[order[0]] = 0;
        let mut cur = 0usize;
        for t in 0..n.saturating_sub(1) {
            let i = order[t];
            let j = order[t + 1];

            // actual achieved overlap = min( len(suffix(i)), len(prefix(j)) )
            let achieved = right_seq[t].len();
            cur += 5 - achieved;
            idxs[j] = cur;
        }
    }

    (permuted, idxs)
}