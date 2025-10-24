// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::balanced_square::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let n = (challenge.numbers.len() as f64).sqrt() as usize;
    
    if n * n != challenge.numbers.len() {
        return Err(anyhow!("Invalid challenge: numbers length is not a perfect square"));
    }
    
    // Initialize with greedy arrangement
    let mut current = greedy_init(&challenge.numbers, n, &mut rng);
    let mut current_var = calc_variance(&current, &challenge.numbers);
    let mut best = current.clone();
    let mut best_var = current_var;
    
    // Simulated annealing parameters
    let mut temp = 2500.0;
    let cooling = 0.99996;
    let min_temp = 0.01;
    let max_iter = 10_000_000;
    let mut no_improve = 0;
    
    for iter in 0..max_iter {
        if temp < min_temp {
            break;
        }
        
        // Generate neighbor by swapping positions
        let mut neighbor = current.clone();
        let swaps = if temp > 100.0 && rng.gen::<f64>() < 0.15 { 2 } else { 1 };
        for _ in 0..swaps {
            swap_random(&mut neighbor, n, &mut rng);
        }
        
        let neighbor_var = calc_variance(&neighbor, &challenge.numbers);
        let delta = neighbor_var - current_var;
        
        // Accept or reject
        if delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
            current = neighbor;
            current_var = neighbor_var;
            
            if current_var < best_var {
                best = current.clone();
                best_var = current_var;
                no_improve = 0;
            } else {
                no_improve += 1;
            }
        } else {
            no_improve += 1;
        }
        
        // Reheat if stuck
        if no_improve > 80000 {
            temp = temp.max(300.0);
            no_improve = 0;
        }
        
        temp *= cooling;
        
        // Save periodically
        if iter % 50000 == 0 {
            save_solution(&Solution { arrangement: best.clone() })?;
            if best_var < 1.0 {
                return Ok(());
            }
        }
    }
    
    // Final save
    save_solution(&Solution { arrangement: best })?;
    Ok(())
}

fn calc_variance(arr: &[Vec<usize>], nums: &[i32]) -> f64 {
    let n = arr.len();
    let mut sums = Vec::with_capacity(2 * n + 2);
    
    // Row sums
    for row in arr {
        sums.push(row.iter().map(|&i| nums[i]).sum::<i32>());
    }
    
    // Column sums
    for c in 0..n {
        sums.push((0..n).map(|r| nums[arr[r][c]]).sum::<i32>());
    }
    
    // Diagonal sums
    sums.push((0..n).map(|i| nums[arr[i][i]]).sum::<i32>());
    sums.push((0..n).map(|i| nums[arr[i][n - 1 - i]]).sum::<i32>());
    
    // Calculate variance
    let mean = sums.iter().sum::<i32>() as f64 / sums.len() as f64;
    sums.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / sums.len() as f64
}

fn greedy_init(nums: &[i32], n: usize, rng: &mut SmallRng) -> Vec<Vec<usize>> {
    let mut indexed: Vec<(usize, i32)> = nums.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by_key(|&(_, v)| v);
    
    // Interleave high and low values
    let mut indices = Vec::with_capacity(n * n);
    let mut lo = 0;
    let mut hi = indexed.len() - 1;
    while lo <= hi {
        indices.push(indexed[hi].0);
        if lo < hi {
            indices.push(indexed[lo].0);
            lo += 1;
        }
        if hi > 0 {
            hi -= 1;
        } else {
            break;
        }
    }
    
    // Add controlled randomness
    for i in 0..indices.len() {
        if rng.gen::<f64>() < 0.3 {
            let j = (i + rng.gen_range(1..=n.min(3))).min(indices.len() - 1);
            indices.swap(i, j);
        }
    }
    
    // Convert to 2D
    (0..n).map(|i| indices[i * n..(i + 1) * n].to_vec()).collect()
}

fn swap_random(arr: &mut [Vec<usize>], n: usize, rng: &mut SmallRng) {
    let (r1, c1) = (rng.gen_range(0..n), rng.gen_range(0..n));
    let (r2, c2) = (rng.gen_range(0..n), rng.gen_range(0..n));
    let temp = arr[r1][c1];
    arr[r1][c1] = arr[r2][c2];
    arr[r2][c2] = temp;
}

// Important! Do not include any tests in this file, it will result in your submission being rejected