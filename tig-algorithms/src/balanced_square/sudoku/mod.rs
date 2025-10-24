// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use tig_challenges::balanced_square::*;
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let size = challenge.difficulty.size;
    let n_squared = size * size;
    
    // Create indices vector
    let mut indices: Vec<usize> = (0..n_squared).collect();
    
    // Baseline does 'size' shuffles, we do 20x more
    let num_shuffles = size * 20;
    
    // Store all candidates with their variance
    let mut candidates: Vec<(f32, Vec<Vec<usize>>)> = Vec::with_capacity(num_shuffles);
    
    // Phase 1: Generate 20x more random starts than baseline
    for _ in 0..num_shuffles {
        // Shuffle indices just like baseline
        indices.shuffle(&mut rng);
        
        // Place in order just like baseline
        let mut arrangement = vec![vec![0usize; size]; size];
        for i in 0..size {
            for j in 0..size {
                arrangement[i][j] = indices[i * size + j];
            }
        }
        
        let variance = calculate_variance(&challenge.numbers, &arrangement, size)?;
        candidates.push((variance, arrangement));
    }
    
    // Sort candidates by variance (best first)
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // Phase 2: Apply local search to best 5%
    let num_to_optimize = std::cmp::max(1, num_shuffles / 20); // 5% of candidates
    
    let mut best_variance = f32::MAX;
    
    for i in 0..num_to_optimize {
        let (mut current_variance, mut arrangement) = candidates[i].clone();
        
        // Apply local search with greedy swaps
        let local_search_iterations = size * 100;
        
        for _ in 0..local_search_iterations {
            let pos1 = rng.gen_range(0..n_squared);
            let pos2 = rng.gen_range(0..n_squared);
            
            if pos1 == pos2 {
                continue;
            }
            
            let (i1, j1) = (pos1 / size, pos1 % size);
            let (i2, j2) = (pos2 / size, pos2 % size);
            
            // Perform swap
            let temp = arrangement[i1][j1];
            arrangement[i1][j1] = arrangement[i2][j2];
            arrangement[i2][j2] = temp;
            
            let new_variance = calculate_variance(&challenge.numbers, &arrangement, size)?;
            
            if new_variance < current_variance {
                // Keep the improvement
                current_variance = new_variance;
                
                if current_variance < best_variance {
                    best_variance = current_variance;
                    save_solution(&Solution { arrangement: arrangement.clone() })?;
                }
            } else {
                // Revert the swap
                let temp = arrangement[i1][j1];
                arrangement[i1][j1] = arrangement[i2][j2];
                arrangement[i2][j2] = temp;
            }
        }
    }
    
    Ok(())
}

fn calculate_variance(numbers: &[i32], arrangement: &Vec<Vec<usize>>, size: usize) -> Result<f32> {
    let mut sums = Vec::new();
    
    // Row sums
    for i in 0..size {
        let row_sum: i32 = (0..size)
            .map(|j| numbers[arrangement[i][j]])
            .sum();
        sums.push(row_sum);
    }
    
    // Column sums
    for j in 0..size {
        let col_sum: i32 = (0..size)
            .map(|i| numbers[arrangement[i][j]])
            .sum();
        sums.push(col_sum);
    }
    
    // Main diagonal sum
    let diag1_sum: i32 = (0..size)
        .map(|i| numbers[arrangement[i][i]])
        .sum();
    sums.push(diag1_sum);
    
    // Anti-diagonal sum
    let diag2_sum: i32 = (0..size)
        .map(|i| numbers[arrangement[i][size - 1 - i]])
        .sum();
    sums.push(diag2_sum);
    
    // Calculate variance
    let mean = sums.iter().sum::<i32>() as f32 / sums.len() as f32;
    let variance = sums.iter()
        .map(|&x| (x as f32 - mean).powi(2))
        .sum::<f32>() / sums.len() as f32;
    
    Ok(variance)
}