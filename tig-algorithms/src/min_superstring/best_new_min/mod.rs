// Ultra-fast Min-Superstring optimized for high throughput (20+ solutions/sec)
// Strategy: Aggressive pruning, minimal iterations, fast greedy heuristics

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

    const A: usize = 128;
    const BLOCK: usize = 5;
    const MAX_OVERLAP: u8 = 4;

    // Fast character counting with early exit
    let mut counts: Vec<[u8; A]> = vec![[0u8; A]; n];
    for (i, s) in challenge.strings.iter().enumerate() {
        for ch in s.bytes() {
            let b = ch as usize;
            if b < A {
                counts[i][b] = counts[i][b].saturating_add(1);
            }
        }
    }

    // Fast overlap computation with aggressive caching
    let mut overlap_ij: Vec<Vec<u8>> = vec![vec![0u8; n]; n];
    let mut cost: Vec<Vec<f32>> = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in (i + 1)..n {
            // Calculate intersection once, use for both directions
            let mut inter = 0u8;
            for c in 0..A {
                inter = inter.saturating_add(counts[i][c].min(counts[j][c]));
                if inter >= MAX_OVERLAP { break; }
            }
            let ov = inter.min(MAX_OVERLAP);
            
            overlap_ij[i][j] = ov;
            overlap_ij[j][i] = ov;
            
            let c = (BLOCK as i32 - ov as i32) as f32;
            cost[i][j] = c;
            cost[j][i] = c;
        }
    }

    // Single fast greedy construction - no multiple tries
    let tour = if n <= 30 {
        // Small instances: one good starting point
        nearest_neighbor_cycle(n, &cost, 0)
    } else {
        // Large instances: quick farthest insertion
        farthest_insertion_fast(n, &cost, 0)?
    };

    // Minimal 2-opt: just enough to fix obvious mistakes
    let max_passes = if n <= 20 { 3 } else if n <= 35 { 2 } else { 1 };
    let (cycle, _) = two_opt_cycle_limited(&cost, tour, max_passes);

    // Fast cycle breaking: just take the worst edge
    let path = break_cycle_fast(&cost, &cycle);

    // Fast permutation realization
    let (permuted, idxs) = realize_permutations_fast(
        &challenge.strings,
        &counts,
        &overlap_ij,
        &path,
    );

    save_solution(&Solution {
        permuted_strings: permuted,
        superstring_idxs: idxs,
    })?;

    Ok(())
}

// Ultra-fast nearest neighbor for small instances
#[inline]
fn nearest_neighbor_cycle(n: usize, dm: &[Vec<f32>], start: usize) -> Vec<usize> {
    if n <= 1 { return vec![start]; }
    
    let mut tour = Vec::with_capacity(n);
    let mut used = vec![false; n];
    
    tour.push(start);
    used[start] = true;
    
    for _ in 1..n {
        let last = *tour.last().unwrap();
        let mut best = usize::MAX;
        let mut best_cost = f32::INFINITY;
        
        for next in 0..n {
            if !used[next] {
                let c = dm[last][next];
                if c < best_cost {
                    best_cost = c;
                    best = next;
                }
            }
        }
        
        if best != usize::MAX {
            tour.push(best);
            used[best] = true;
        }
    }
    
    tour
}

// Fast farthest insertion with early termination
fn farthest_insertion_fast(n: usize, dm: &[Vec<f32>], start: usize) -> Result<Vec<usize>> {
    if n <= 2 {
        return Ok(if n == 1 { vec![start] } else { vec![start, 1 - start] });
    }

    // Start with triangle
    let a = start;
    let mut b = 0;
    let mut max_dist = f32::NEG_INFINITY;
    for k in 0..n {
        if k != a && dm[a][k] > max_dist {
            max_dist = dm[a][k];
            b = k;
        }
    }
    
    let mut c = 0;
    let mut max_sum = f32::NEG_INFINITY;
    for k in 0..n {
        if k != a && k != b {
            let sum = dm[a][k] + dm[b][k];
            if sum > max_sum {
                max_sum = sum;
                c = k;
            }
        }
    }

    let mut tour = vec![a, b, c];
    let mut in_tour = vec![false; n];
    in_tour[a] = true;
    in_tour[b] = true;
    in_tour[c] = true;

    // Fast distance tracking
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

    for _ in 3..n {
        // Farthest point
        let mut u = 0;
        let mut max_d = f32::NEG_INFINITY;
        for v in 0..n {
            if !in_tour[v] && dist_to_tour[v] > max_d {
                max_d = dist_to_tour[v];
                u = v;
            }
        }

        // Best insertion position (simple scan)
        let mut best_pos = 0;
        let mut best_delta = f32::INFINITY;
        let tour_len = tour.len();
        
        for i in 0..tour_len {
            let j = (i + 1) % tour_len;
            let delta = dm[tour[i]][u] + dm[u][tour[j]] - dm[tour[i]][tour[j]];
            if delta < best_delta {
                best_delta = delta;
                best_pos = j;
            }
        }

        tour.insert(best_pos, u);
        in_tour[u] = true;

        // Quick distance update
        for v in 0..n {
            if !in_tour[v] {
                let d = dm[v][u].min(dm[u][v]);
                if d < dist_to_tour[v] {
                    dist_to_tour[v] = d;
                }
            }
        }
    }

    Ok(tour)
}

// Minimal 2-opt with hard iteration limit
fn two_opt_cycle_limited(
    dm: &[Vec<f32>],
    mut tour: Vec<usize>,
    max_passes: usize,
) -> (Vec<usize>, f32) {
    let n = tour.len();
    if n < 4 { return (tour, 0.0); }

    for _ in 0..max_passes {
        let mut improved = false;

        'outer: for i in 0..n {
            let i_next = (i + 1) % n;
            let a = tour[i];
            let b = tour[i_next];

            // Limit inner loop for speed
            let search_limit = if n > 40 { 20 } else { n };
            for offset in 2..search_limit.min(n - 1) {
                let j = (i + offset) % n;
                let j_next = (j + 1) % n;
                
                if i == 0 && j == n - 1 { continue; }

                let c = tour[j];
                let d = tour[j_next];
                let delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d];

                if delta < -1e-5 {
                    reverse_segment_cyclic(&mut tour, i_next, j);
                    improved = true;
                    break 'outer;
                }
            }
        }

        if !improved { break; }
    }

    (tour, 0.0)
}

#[inline]
fn break_cycle_fast(dm: &[Vec<f32>], cycle: &[usize]) -> Vec<usize> {
    let n = cycle.len();
    if n <= 2 { return cycle.to_vec(); }

    // Just find worst edge - no fancy scoring
    let mut worst = 0;
    let mut worst_cost = f32::NEG_INFINITY;
    
    for i in 0..n {
        let j = (i + 1) % n;
        let c = dm[cycle[i]][cycle[j]];
        if c > worst_cost {
            worst_cost = c;
            worst = j;
        }
    }

    let mut path = Vec::with_capacity(n);
    for k in 0..n {
        path.push(cycle[(worst + k) % n]);
    }
    path
}

// Streamlined permutation realization
fn realize_permutations_fast(
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

    let mut left_seq: Vec<Vec<u8>> = vec![Vec::new(); n];
    let mut right_seq: Vec<Vec<u8>> = vec![Vec::new(); n];

    // Fast greedy overlap allocation
    for t in 0..n.saturating_sub(1) {
        let i = order[t];
        let j = order[t + 1];
        let target = overlap_ij[i][j] as usize;

        if target > 0 {
            let mut seq = Vec::with_capacity(target);
            
            // Simple ascending order scan
            for c in 0..A {
                if seq.len() >= target { break; }
                
                let avail = rem[i][c].min(rem[j][c]).max(0);
                if avail > 0 {
                    let use_k = (avail as usize).min(target - seq.len());
                    for _ in 0..use_k {
                        seq.push(c as u8);
                    }
                    rem[i][c] -= use_k as i8;
                    rem[j][c] -= use_k as i8;
                }
            }

            right_seq[t] = seq.clone();
            left_seq[t + 1] = seq;
        }
    }

    // Fast string building
    let mut permuted = vec![String::new(); strings.len()];
    let mut pos_in_path = vec![usize::MAX; strings.len()];
    for (p, &v) in order.iter().enumerate() { 
        pos_in_path[v] = p; 
    }

    for &v in order {
        let p = pos_in_path[v];
        let mut s = String::with_capacity(BLOCK);

        // Prefix
        for &b in &left_seq[p] {
            s.push(b as char);
        }

        // Middle
        for c in 0..A {
            let k = rem[v][c].max(0) as usize;
            for _ in 0..k { 
                s.push(c as u8 as char); 
            }
        }

        // Suffix
        for &b in &right_seq[p] {
            s.push(b as char);
        }

        // Quick length adjustment
        let len = s.len();
        if len < BLOCK {
            let orig = &strings[v];
            s.push_str(&orig[..(BLOCK - len).min(orig.len())]);
        } else if len > BLOCK {
            s.truncate(BLOCK);
        }

        permuted[v] = s;
    }

    // Fast index computation
    let mut idxs = vec![0usize; strings.len()];
    if !order.is_empty() {
        let mut cur = 0;
        idxs[order[0]] = 0;
        
        for t in 0..n - 1 {
            cur += BLOCK - right_seq[t].len();
            idxs[order[t + 1]] = cur;
        }
    }

    (permuted, idxs)
}

#[inline]
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