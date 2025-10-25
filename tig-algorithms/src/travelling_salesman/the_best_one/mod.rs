// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
// Ultra-Fast TSP Solver - Optimized for Maximum Speed
use anyhow::Result;
use tig_challenges::travelling_salesman::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.difficulty.size;
    
    if n == 0 { return Ok(()); }
    if n == 1 {
        save_solution(&Solution { route: vec![0] })?;
        return Ok(());
    }

    let dm = &challenge.distance_matrix;
    
    // Ultra-fast RNG from seed
    let mut rng = u32::from_le_bytes([
        challenge.seed[0],
        challenge.seed[1],
        challenge.seed[2],
        challenge.seed[3],
    ]);

    // Quick greedy construction - just 2 attempts
    let tour1 = nearest_neighbor_fast(n, dm, 0);
    let len1 = tour_dist(dm, &tour1);
    
    let start2 = ((rng as usize) % n).min(n - 1);
    rng = lcg(rng);
    let tour2 = nearest_neighbor_fast(n, dm, start2);
    let len2 = tour_dist(dm, &tour2);
    
    let (mut best, mut best_len) = if len1 < len2 {
        (tour1, len1)
    } else {
        (tour2, len2)
    };

    // Single quick 2-opt pass
    two_opt_once(&mut best, dm, &mut best_len);
    save_solution(&Solution { route: best.clone() })?;

    // Fast simulated annealing - very short
    let mut curr = best.clone();
    let mut curr_len = best_len;
    let mut temp = best_len * 0.1;
    
    for _ in 0..5000 {
        let mut next = curr.clone();
        
        let mv = rng % 100;
        rng = lcg(rng);
        
        if mv < 70 {
            // Fast 2-opt
            let i = (rng as usize) % (n - 2);
            rng = lcg(rng);
            let j = ((rng as usize) % (n - i - 2)) + i + 2;
            rng = lcg(rng);
            next[i + 1..j].reverse();
        } else {
            // Fast swap
            let i = (rng as usize) % n;
            rng = lcg(rng);
            let j = (rng as usize) % n;
            next.swap(i, j);
        }
        
        let next_len = tour_dist(dm, &next);
        let delta = next_len - curr_len;
        
        if delta < 0.0 || rand_f32(&mut rng) < (-delta / temp).exp() {
            curr = next;
            curr_len = next_len;
            
            if curr_len < best_len {
                best = curr.clone();
                best_len = curr_len;
                two_opt_once(&mut best, dm, &mut best_len);
                curr = best.clone();
                curr_len = best_len;
                save_solution(&Solution { route: best.clone() })?;
            }
        }
        
        temp *= 0.998;
    }

    save_solution(&Solution { route: best })?;
    Ok(())
}

#[inline]
fn nearest_neighbor_fast(n: usize, dm: &[Vec<f32>], start: usize) -> Vec<usize> {
    let mut tour = Vec::with_capacity(n);
    tour.push(start);
    let mut visited = vec![false; n];
    visited[start] = true;
    let mut curr = start;

    for _ in 1..n {
        let mut best = 0;
        let mut best_dist = f32::INFINITY;

        for j in 0..n {
            if !visited[j] {
                let d = dm[curr][j];
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
        }

        tour.push(best);
        visited[best] = true;
        curr = best;
    }

    tour
}

#[inline]
fn two_opt_once(tour: &mut [usize], dm: &[Vec<f32>], length: &mut f32) {
    let n = tour.len();
    if n < 4 { return; }

    let mut improved = true;
    
    while improved {
        improved = false;
        
        for i in 0..n - 1 {
            let a = tour[i];
            let b = tour[i + 1];
            
            for j in i + 2..n {
                if i == 0 && j == n - 1 { continue; }
                
                let c = tour[j];
                let d = tour[(j + 1) % n];
                
                let old = dm[a][b] + dm[c][d];
                let new = dm[a][c] + dm[b][d];
                
                if new < old - 1e-9 {
                    tour[i + 1..=j].reverse();
                    *length += new - old;
                    improved = true;
                    break;
                }
            }
            if improved { break; }
        }
    }
}

#[inline]
fn tour_dist(dm: &[Vec<f32>], tour: &[usize]) -> f32 {
    let n = tour.len();
    if n < 2 { return 0.0; }
    
    let mut sum = 0.0;
    for i in 0..n {
        sum += dm[tour[i]][tour[(i + 1) % n]];
    }
    sum
}

#[inline]
fn lcg(x: u32) -> u32 {
    x.wrapping_mul(1664525).wrapping_add(1013904223)
}

#[inline]
fn rand_f32(rng: &mut u32) -> f32 {
    *rng = lcg(*rng);
    (*rng as f32) / (u32::MAX as f32)
}