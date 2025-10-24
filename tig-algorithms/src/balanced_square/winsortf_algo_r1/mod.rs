// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use tig_challenges::balanced_square::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};

fn calculate_variance(numbers: &[i32], arrangement: &[Vec<usize>]) -> f64 {
    let n = arrangement.len();
    let mut sums = Vec::with_capacity(2 * n + 2);
    
    // Row sums
    for row in arrangement {
        sums.push(row.iter().map(|&idx| numbers[idx] as i64).sum::<i64>());
    }
    
    // Column sums
    for col in 0..n {
        sums.push(arrangement.iter().map(|row| numbers[row[col]] as i64).sum::<i64>());
    }
    
    // Main diagonal
    sums.push((0..n).map(|i| numbers[arrangement[i][i]] as i64).sum::<i64>());
    
    // Anti-diagonal
    sums.push((0..n).map(|i| numbers[arrangement[i][n - 1 - i]] as i64).sum::<i64>());
    
    let mean = sums.iter().sum::<i64>() as f64 / sums.len() as f64;
    sums.iter().map(|&s| {
        let diff = s as f64 - mean;
        diff * diff
    }).sum::<f64>() / sums.len() as f64
}

fn create_greedy_arrangement(numbers: &[i32], n: usize, rng: &mut SmallRng) -> Vec<Vec<usize>> {
    let mut sorted_indices: Vec<usize> = (0..numbers.len()).collect();
    sorted_indices.sort_by_key(|&i| std::cmp::Reverse(numbers[i]));
    
    let mut arrangement = vec![vec![0; n]; n];
    let mut used = vec![false; numbers.len()];
    
    // Distribute high values across different rows/cols/diagonals
    let mut positions: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            positions.push((i, j));
        }
    }
    
    // Shuffle positions for randomness
    for i in (1..positions.len()).rev() {
        let j = rng.gen_range(0..=i);
        positions.swap(i, j);
    }
    
    // Place numbers in a balanced way
    for (pos_idx, &num_idx) in sorted_indices.iter().enumerate() {
        let (row, col) = positions[pos_idx];
        arrangement[row][col] = num_idx;
        used[num_idx] = true;
    }
    
    arrangement
}

fn swap_positions(arrangement: &mut Vec<Vec<usize>>, rng: &mut SmallRng) {
    let n = arrangement.len();
    let r1 = rng.gen_range(0..n);
    let c1 = rng.gen_range(0..n);
    let r2 = rng.gen_range(0..n);
    let c2 = rng.gen_range(0..n);
    
    let temp = arrangement[r1][c1];
    arrangement[r1][c1] = arrangement[r2][c2];
    arrangement[r2][c2] = temp;
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let n = (challenge.numbers.len() as f64).sqrt() as usize;
    
    // Initialize with greedy approach
    let mut best_arrangement = create_greedy_arrangement(&challenge.numbers, n, &mut rng);
    let mut best_variance = calculate_variance(&challenge.numbers, &best_arrangement);
    
    save_solution(&Solution {
        arrangement: best_arrangement.clone(),
    })?;
    
    // Simulated annealing parameters
    let mut temperature = 1000.0;
    let cooling_rate = 0.9995;
    let min_temp = 0.01;
    let mut iterations = 0;
    let max_iterations = 1_000_000;
    
    let mut current_arrangement = best_arrangement.clone();
    let mut current_variance = best_variance;
    
    while temperature > min_temp && iterations < max_iterations {
        // Create neighbor solution
        let mut new_arrangement = current_arrangement.clone();
        
        // Try multiple swaps for larger moves
        let num_swaps = if rng.gen::<f64>() < 0.1 { 2 } else { 1 };
        for _ in 0..num_swaps {
            swap_positions(&mut new_arrangement, &mut rng);
        }
        
        let new_variance = calculate_variance(&challenge.numbers, &new_arrangement);
        
        // Accept or reject
        let delta = new_variance - current_variance;
        if delta < 0.0 || rng.gen::<f64>() < (-delta / temperature).exp() {
            current_arrangement = new_arrangement;
            current_variance = new_variance;
            
            // Update best solution
            if current_variance < best_variance {
                best_variance = current_variance;
                best_arrangement = current_arrangement.clone();
                
                save_solution(&Solution {
                    arrangement: best_arrangement.clone(),
                })?;
            }
        }
        
        temperature *= cooling_rate;
        iterations += 1;
        
        // Periodic restarts with perturbation
        if iterations % 50000 == 0 {
            if rng.gen::<f64>() < 0.3 {
                current_arrangement = best_arrangement.clone();
                for _ in 0..n {
                    swap_positions(&mut current_arrangement, &mut rng);
                }
                current_variance = calculate_variance(&challenge.numbers, &current_arrangement);
                temperature = 100.0;
            }
        }
    }
    
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
