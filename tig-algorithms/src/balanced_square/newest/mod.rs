// TIG challenge detection pattern: tig_challenges::<challenge_name>
use anyhow::Result;
use tig_challenges::balanced_square::*;

/// Balanced Square — Greedy single-pass O(n²) heuristic minimizing sum variance.
/// Strategy:
/// 1. Sort all values in descending order.
/// 2. Maintain two pseudo-random bijective orderings (A and B).
/// 3. For each number, try placing it in the better of two candidate cells
///    (computed in O(1) each) based on the expected variance delta.
/// 4. Save once and exit for optimal wall-time efficiency.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let values = &challenge.numbers;
    let total_cells = values.len();
    let matrix_size = (total_cells as f64).sqrt() as usize;
    debug_assert_eq!(matrix_size * matrix_size, total_cells);

    // Indices sorted by descending numeric value (unstable sort = faster).
    let mut sorted_indices: Vec<usize> = (0..total_cells).collect();
    sorted_indices.sort_unstable_by(|&i, &j| values[j].cmp(&values[i]));

    // Index mapping:
    // rows [0..n), cols [n..2n), diag_main: 2n, diag_anti: 2n+1
    let score_count = 2 * matrix_size + 2;

    // Deterministic bijective index generators (A and B sequences).
    // These generate traversal orders that cover all cells exactly once.
    #[inline(always)]
    fn seq_a(t: usize, n: usize, a: usize, s: usize) -> usize {
        let r = t % n;
        let q = t / n;
        let c = (a.wrapping_mul(q) + r + s) % n;
        r * n + c
    }

    #[inline(always)]
    fn seq_b(t: usize, n: usize, a: usize, s: usize) -> usize {
        let c = t % n;
        let q = t / n;
        let r = (a.wrapping_mul(q) + c + s) % n;
        r * n + c
    }

    // Parameters ensuring full coverage:
    // For even n, n−1 is coprime with n; for odd n, 1 is.
    let a1 = 1usize;
    let a2 = if matrix_size % 2 == 0 { matrix_size - 1 } else { 1 };
    let shift_a = 0usize;
    let shift_b = (matrix_size / 2) % matrix_size; // helps with diagonals for even n

    // Objective: minimize variance → proportional to m*Σ(s²) - (Σs)².
    let mut score_sums = vec![0i64; score_count];
    let mut sum_all: i128 = 0;
    let mut sum_squares: i128 = 0;
    let m128: i128 = score_count as i128;

    /// Estimate how much variance changes if we place a value at (r, c)
    #[inline(always)]
    fn delta_variance(
        sums: &[i64],
        sum_total: i128,
        m128: i128,
        n: usize,
        r: usize,
        c: usize,
        val: i32,
    ) -> i128 {
        let v = val as i128;
        let mut delta_sum = v;
        let mut delta_sum_sq = 0i128;

        // row
        let row_sum = sums[r] as i128;
        delta_sum_sq += 2 * row_sum * v + v * v;
        // col
        let col_sum = sums[n + c] as i128;
        delta_sum += v;
        delta_sum_sq += 2 * col_sum * v + v * v;
        // main diag
        if r == c {
            let diag_sum = sums[2 * n] as i128;
            delta_sum += v;
            delta_sum_sq += 2 * diag_sum * v + v * v;
        }
        // anti diag
        if r + c + 1 == n {
            let diag_sum = sums[2 * n + 1] as i128;
            delta_sum += v;
            delta_sum_sq += 2 * diag_sum * v + v * v;
        }

        m128 * delta_sum_sq - (2 * sum_total * delta_sum + delta_sum * delta_sum)
    }

    /// Apply placement: update running sums efficiently
    #[inline(always)]
    fn apply_placement(
        sums: &mut [i64],
        total_sum: &mut i128,
        total_sq: &mut i128,
        n: usize,
        r: usize,
        c: usize,
        val: i32,
    ) {
        let v = val as i128;
        let update = |index: usize, sums: &mut [i64], total_sum: &mut i128, total_sq: &mut i128| {
            let prev = sums[index] as i128;
            *total_sum += v;
            *total_sq += 2 * prev * v + v * v;
            sums[index] = (prev + v) as i64;
        };

        update(r, sums, total_sum, total_sq); // row
        update(n + c, sums, total_sum, total_sq); // col
        if r == c {
            update(2 * n, sums, total_sum, total_sq); // main diag
        }
        if r + c + 1 == n {
            update(2 * n + 1, sums, total_sum, total_sq); // anti diag
        }
    }

    // Fill the grid: pick the better cell from A and B for each value.
    let mut used = vec![0u8; total_cells];
    let mut pos_a = 0usize;
    let mut pos_b = 0usize;
    let mut flat_grid = vec![0usize; total_cells];

    for &idx in &sorted_indices {
        let val = values[idx];

        // next free cell in A
        let mut cell_a = 0usize;
        let mut has_a = false;
        while pos_a < total_cells {
            let candidate = seq_a(pos_a, matrix_size, a1, shift_a);
            if used[candidate] == 0 {
                cell_a = candidate;
                has_a = true;
                break;
            }
            pos_a += 1;
        }

        // next free cell in B
        let mut cell_b = 0usize;
        let mut has_b = false;
        while pos_b < total_cells {
            let candidate = seq_b(pos_b, matrix_size, a2, shift_b);
            if used[candidate] == 0 {
                cell_b = candidate;
                has_b = true;
                break;
            }
            pos_b += 1;
        }

        // Decide which position minimizes the variance increment
        let choose_a = if !has_b {
            true
        } else if !has_a {
            false
        } else {
            let (ra, ca) = (cell_a / matrix_size, cell_a % matrix_size);
            let (rb, cb) = (cell_b / matrix_size, cell_b % matrix_size);
            let delta_a = delta_variance(&score_sums, sum_all, m128, matrix_size, ra, ca, val);
            let delta_b = delta_variance(&score_sums, sum_all, m128, matrix_size, rb, cb, val);
            delta_a <= delta_b
        };

        let chosen_cell = if choose_a { cell_a } else { cell_b };
        used[chosen_cell] = 1;
        if choose_a {
            pos_a += 1;
        } else {
            pos_b += 1;
        }

        let (r, c) = (chosen_cell / matrix_size, chosen_cell % matrix_size);
        apply_placement(&mut score_sums, &mut sum_all, &mut sum_squares, matrix_size, r, c, val);
        flat_grid[chosen_cell] = idx;
    }

    // Build 2D matrix layout
    let mut arrangement = Vec::with_capacity(matrix_size);
    for r in 0..matrix_size {
        let mut row = Vec::with_capacity(matrix_size);
        for c in 0..matrix_size {
            row.push(flat_grid[r * matrix_size + c]);
        }
        arrangement.push(row);
    }

    // Single save for efficiency
    save_solution(&Solution { arrangement })?;
    Ok(())
}
