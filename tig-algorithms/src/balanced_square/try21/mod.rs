// my_algo.rs

use anyhow::Result;
use rand::Rng;
use std::time::{Duration, Instant};
use tig_challenges::balanced_square::{Challenge, Solution};

/// Represents the current state of the square arrangement and its associated metrics.
#[derive(Clone)]
struct State {
    arrangement: Vec<Vec<usize>>,
    values: Vec<Vec<i32>>,
    sums: Vec<i64>,
    variance: f64,
    n: usize,
}

impl State {
    /// Creates a new State from an initial arrangement.
    fn new(arrangement: Vec<Vec<usize>>, challenge: &Challenge) -> Self {
        let n = arrangement.len();
        let values = (0..n)
            .map(|r| {
                (0..n)
                    .map(|c| challenge.numbers[arrangement[r][c]])
                    .collect()
            })
            .collect();
        let sums = calculate_all_sums(&values, n);
        let variance = calculate_variance(&sums);
        Self {
            arrangement,
            values,
            sums,
            variance,
            n,
        }
    }

    /// Performs a swap of two cells and efficiently updates the sums and variance.
    fn swap_and_update(&mut self, r1: usize, c1: usize, r2: usize, c2: usize) {
        let val1 = self.values[r1][c1];
        let val2 = self.values[r2][c2];

        self.arrangement.swap(r1, c1, r2, c2);
        self.values.swap(r1, c1, r2, c2);

        update_sums(&mut self.sums, self.n, r1, c1, r2, c2, val1, val2);
        self.variance = calculate_variance(&self.sums);
    }
}

/// The main entry point for solving the challenge.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let start_time = Instant::now();
    let n = (challenge.numbers.len() as f64).sqrt() as usize;
    if n == 0 {
        return Ok(());
    }

    let mut current_state = State::new(generate_greedy_solution(challenge, n), challenge);
    let mut best_state = current_state.clone();
    
    save_solution(&Solution { arrangement: best_state.arrangement.clone() })?;

    let mut rng = rand::thread_rng();
    // Slightly increased initial temperature for more exploration
    let mut temperature = best_state.variance * 0.2; 
    let cooling_rate = 0.99995;
    let time_limit = Duration::from_millis(19800);

    // --- Main Simulated Annealing Loop ---
    while start_time.elapsed() < time_limit {
        let (r1, c1, r2, c2) = if rng.gen::<f32>() < 0.3 {
            // **TARGETED SWAP**: 30% chance to fix the worst-offending line
            get_targeted_swap_coords(&current_state, &mut rng)
        } else {
            // **RANDOM SWAP**: 70% chance for broad exploration
            get_random_swap_coords(n, &mut rng)
        };

        let old_variance = current_state.variance;
        current_state.swap_and_update(r1, c1, r2, c2);
        
        let new_variance = current_state.variance;
        let delta_variance = new_variance - old_variance;

        // Acceptance condition: always accept improvements, sometimes accept bad moves
        if delta_variance < 0.0 || rng.gen::<f64>() < (-delta_variance / temperature).exp() {
            if new_variance < best_state.variance {
                best_state = current_state.clone();
                save_solution(&Solution { arrangement: best_state.arrangement.clone() })?;
            }
        } else {
            // Revert the swap
            current_state.swap_and_update(r1, c1, r2, c2);
        }

        temperature *= cooling_rate;
    }

    Ok(())
}

/// Generates two distinct random cell coordinates for a swap.
fn get_random_swap_coords(n: usize, rng: &mut impl Rng) -> (usize, usize, usize, usize) {
    let (r1, c1) = (rng.gen_range(0..n), rng.gen_range(0..n));
    let (mut r2, mut c2);
    loop {
        r2 = rng.gen_range(0..n);
        c2 = rng.gen_range(0..n);
        if r1 != r2 || c1 != c2 {
            break;
        }
    }
    (r1, c1, r2, c2)
}

/// Finds the line (row/col/diag) with the most extreme sum and picks a cell from it to swap.
fn get_targeted_swap_coords(state: &State, rng: &mut impl Rng) -> (usize, usize, usize, usize) {
    let n = state.n;
    let mean = state.sums.iter().sum::<i64>() as f64 / state.sums.len() as f64;
    
    // Find the index of the sum with the largest deviation from the mean
    let (outlier_idx, _) = state.sums
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, (s as f64 - mean).abs()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap_or((0, 0.0));

    // Determine the coordinates of the first cell from the outlier line
    let (r1, c1) = match outlier_idx {
        i if i < n => (i, rng.gen_range(0..n)), // It's a row
        i if i < 2 * n => (rng.gen_range(0..n), i - n), // It's a column
        i if i == 2 * n => { let k = rng.gen_range(0..n); (k, k) }, // Main diagonal
        _ => { let k = rng.gen_range(0..n); (k, n - 1 - k) }, // Anti-diagonal
    };
    
    // Pick a second cell randomly, ensuring it's different
    let (mut r2, mut c2);
    loop {
        r2 = rng.gen_range(0..n);
        c2 = rng.gen_range(0..n);
        if r1 != r2 || c1 != c2 {
            break;
        }
    }
    (r1, c1, r2, c2)
}

/// Generates an initial solution using a greedy spiral pattern.
fn generate_greedy_solution(challenge: &Challenge, n: usize) -> Vec<Vec<usize>> {
    let mut indexed_numbers: Vec<(usize, i32)> =
        challenge.numbers.iter().copied().enumerate().collect();
    indexed_numbers.sort_unstable_by_key(|k| k.1);

    let mut arrangement = vec![vec![0; n]; n];
    let (mut top, mut bottom, mut left, mut right) = (0, n - 1, 0, n - 1);
    let mut head = 0;
    let mut tail = n * n - 1;
    let mut use_head = true;

    while top <= bottom && left <= right {
        for i in left..=right {
            let idx = if use_head { head } else { tail };
            arrangement[top][i] = indexed_numbers[idx].0;
            if use_head { head += 1; } else { tail -= 1; }
            use_head = !use_head;
        }
        top += 1;
        if top > bottom { break; }

        for i in top..=bottom {
            let idx = if use_head { head } else { tail };
            arrangement[i][right] = indexed_numbers[idx].0;
            if use_head { head += 1; } else { tail -= 1; }
            use_head = !use_head;
        }
        right -= 1;
        if left > right { break; }

        for i in (left..=right).rev() {
            let idx = if use_head { head } else { tail };
            arrangement[bottom][i] = indexed_numbers[idx].0;
            if use_head { head += 1; } else { tail -= 1; }
            use_head = !use_head;
        }
        bottom -= 1;
        
        for i in (top..=bottom).rev() {
            let idx = if use_head { head } else { tail };
            arrangement[i][left] = indexed_numbers[idx].0;
            if use_head { head += 1; } else { tail -= 1; }
            use_head = !use_head;
        }
        left += 1;
    }
    arrangement
}

/// Calculates all 2n+2 sums from scratch.
fn calculate_all_sums(values: &Vec<Vec<i32>>, n: usize) -> Vec<i64> {
    let mut sums = vec![0i64; 2 * n + 2];
    for r in 0..n { sums[r] = values[r].iter().map(|&x| x as i64).sum(); }
    for c in 0..n { sums[n + c] = (0..n).map(|r| values[r][c] as i64).sum(); }
    sums[2 * n] = (0..n).map(|i| values[i][i] as i64).sum();
    sums[2 * n + 1] = (0..n).map(|i| values[i][n - 1 - i] as i64).sum();
    sums
}

/// Calculates variance from a slice of sums.
fn calculate_variance(sums: &[i64]) -> f64 {
    let count = sums.len() as f64;
    if count == 0.0 { return 0.0; }
    let mean: f64 = sums.iter().sum::<i64>() as f64 / count;
    sums.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / count
}

/// **CRITICAL OPTIMIZATION**: Updates sums incrementally after a swap.
fn update_sums(
    sums: &mut [i64], n: usize,
    r1: usize, c1: usize, r2: usize, c2: usize,
    val1: i32, val2: i32,
) {
    let diff = val1 as i64 - val2 as i64;
    sums[r1] -= diff;
    sums[r2] += diff;
    sums[n + c1] -= diff;
    sums[n + c2] += diff;
    if r1 == c1 { sums[2 * n] -= diff; }
    if r2 == c2 { sums[2 * n] += diff; }
    if r1 + c1 == n - 1 { sums[2 * n + 1] -= diff; }
    if r2 + c2 == n - 1 { sums[2 * n + 1] += diff; }
}

/// Extension trait to allow swapping elements between two different nested Vecs.
trait Swap2D<T> {
    fn swap(&mut self, r1: usize, c1: usize, r2: usize, c2: usize);
}

impl<T> Swap2D<T> for Vec<Vec<T>> {
    fn swap(&mut self, r1: usize, c1: usize, r2: usize, c2: usize) {
        if r1 == r2 {
            self[r1].swap(c1, c2);
        } else {
            let ptr1 = &mut self[r1][c1] as *mut T;
            let ptr2 = &mut self[r2][c2] as *mut T;
            unsafe { std::ptr::swap(ptr1, ptr2); }
        }
    }
}