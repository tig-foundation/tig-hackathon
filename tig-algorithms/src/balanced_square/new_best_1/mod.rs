// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::{Duration, Instant};
use tig_challenges::balanced_square::*;

/* ========= Tunables (safe; no private fields) ========= */
// If best variance doesn't improve by at least MIN_IMPROVE for PLATEAU_MS, exit early.
const PLATEAU_MS: u64 = 350;          // early-stop if no improvement for this long
const MIN_IMPROVE: f64 = 1e-7;        // "meaningful" variance delta (post-plateau)
// If we ever get below GOAL_VAR, exit immediately (small, challenge-agnostic).
const GOAL_VAR: f64 = 1.0;            // works well for n=20 in practice
// Absolute ceiling — we rarely need this; keeps us from running the full ~20s.
const TIME_BUDGET_MS: u64 = 6000;     // 6s safety cap (tune as you like)

// Save throttling (reshape + I/O has cost)
const SAVE_THROTTLE_MS: u64 = 700;
// Maximize raw iteration throughput (random swaps are cheap)
const RANDOM_SWAPS_ONLY: bool = true;

/* ========= Flat layout helpers (hot) ========= */
#[inline(always)] fn rc_to_idx(r: usize, c: usize, n: usize) -> usize { r * n + c }
#[inline(always)] fn idx_to_rc(idx: usize, n: usize) -> (usize, usize) { (idx / n, idx % n) }
#[inline(always)] fn row_line(r: usize) -> usize { r }
#[inline(always)] fn col_line(n: usize, c: usize) -> usize { n + c }
#[inline(always)] fn main_diag_line(n: usize) -> usize { 2 * n }
#[inline(always)] fn anti_diag_line(n: usize) -> usize { 2 * n + 1 }
#[inline(always)] fn is_on_main_diag(r: usize, c: usize) -> bool { r == c }
#[inline(always)] fn is_on_anti_diag(r: usize, c: usize, n: usize) -> bool { r + c == n - 1 }
#[inline(always)]
fn variance_from(sum_lines: i64, sumsq_lines: i64, inv_m: f64) -> f64 {
    let mean = (sum_lines as f64) * inv_m;
    (sumsq_lines as f64) * inv_m - mean * mean
}

/* ========= Seeding & scoring ========= */

/// Seed by placing larger numbers in a row-snake (cheap, good spread).
fn seed_arrangement(numbers: &[i32], n: usize) -> Vec<usize> {
    let mut idxs: Vec<usize> = (0..numbers.len()).collect();
    idxs.sort_unstable_by_key(|&i| std::cmp::Reverse(numbers[i]));
    let mut arr = vec![0usize; n * n];
    let mut k = 0;
    for r in 0..n {
        if r % 2 == 0 {
            for c in 0..n { arr[rc_to_idx(r, c, n)] = idxs[k]; k += 1; }
        } else {
            for c in (0..n).rev() { arr[rc_to_idx(r, c, n)] = idxs[k]; k += 1; }
        }
    }
    arr
}

/// Build all line sums once (O(n²)).
fn init_line_sums(numbers: &[i32], n: usize, arr: &[usize]) -> (Vec<i64>, i64, i64) {
    let m = 2 * n + 2;
    let mut line_sums = vec![0i64; m];
    for r in 0..n {
        for c in 0..n {
            let v = numbers[arr[rc_to_idx(r, c, n)]] as i64;
            line_sums[row_line(r)] += v;
            line_sums[col_line(n, c)] += v;
            if is_on_main_diag(r, c) { line_sums[main_diag_line(n)] += v; }
            if is_on_anti_diag(r, c, n) { line_sums[anti_diag_line(n)] += v; }
        }
    }
    let sum_lines: i64 = line_sums.iter().sum();
    let sumsq_lines: i64 = line_sums.iter().map(|&x| x * x).sum();
    (line_sums, sum_lines, sumsq_lines)
}

/// Lines affected by swapping cells p and q; write into buf (≤8), return count.
fn affected_lines(n: usize, p: usize, q: usize, buf: &mut [usize; 8]) -> usize {
    let (pr, pc) = idx_to_rc(p, n);
    let (qr, qc) = idx_to_rc(q, n);
    let mut k = 0usize;
    let mut push = |x: usize| { for i in 0..k { if buf[i] == x { return; } } buf[k] = x; k += 1; };
    // p
    push(row_line(pr));
    push(col_line(n, pc));
    if is_on_main_diag(pr, pc) { push(main_diag_line(n)); }
    if is_on_anti_diag(pr, pc, n) { push(anti_diag_line(n)); }
    // q
    push(row_line(qr));
    push(col_line(n, qc));
    if is_on_main_diag(qr, qc) { push(main_diag_line(n)); }
    if is_on_anti_diag(qr, qc, n) { push(anti_diag_line(n)); }
    k
}

/// Score swap p<->q by touching only affected lines (O(1) wrt n²).
/// Returns (new_var, new_sum, new_sumsq, k, old_line_vals[8], new_line_vals[8]).
fn score_swap(
    numbers: &[i32], n: usize, arr: &[usize], p: usize, q: usize,
    line_sums: &[i64], sum_lines: i64, sumsq_lines: i64, inv_m: f64, tmp_lines: &mut [usize; 8],
) -> (f64, i64, i64, usize, [i64; 8], [i64; 8]) {
    let vp = numbers[arr[p]] as i64;
    let vq = numbers[arr[q]] as i64;
    let dv = vq - vp;

    let k = affected_lines(n, p, q, tmp_lines);
    let (pr, pc) = idx_to_rc(p, n);
    let (qr, qc) = idx_to_rc(q, n);

    let mut new_sum = sum_lines;
    let mut new_sumsq = sumsq_lines;
    let mut oldv = [0i64; 8];
    let mut newv = [0i64; 8];

    for i in 0..k {
        let line = tmp_lines[i];

        let p_in = match line {
            l if l < n      => l == pr,
            l if l < 2 * n  => (l - n) == pc,
            l if l == 2 * n => pr == pc,
            _               => pr + pc == n - 1,
        };
        let q_in = match line {
            l if l < n      => l == qr,
            l if l < 2 * n  => (l - n) == qc,
            l if l == 2 * n => qr == qc,
            _               => qr + qc == n - 1,
        };

        let delta = if p_in && !q_in { dv } else if !p_in && q_in { -dv } else { 0 };
        let old = line_sums[line];
        let newl = old + delta;
        oldv[i] = old;
        newv[i] = newl;

        if delta != 0 {
            new_sum += delta;
            new_sumsq += newl * newl - old * old;
        }
    }

    (variance_from(new_sum, new_sumsq, inv_m), new_sum, new_sumsq, k, oldv, newv)
}

#[inline(always)]
fn apply_swap_and_update(
    arr: &mut [usize], p: usize, q: usize,
    line_sums: &mut [i64], tmp_lines: &[usize; 8], k: usize,
    old_vals: &[i64; 8], new_vals: &[i64; 8],
) {
    arr.swap(p, q);
    for i in 0..k {
        let line = tmp_lines[i];
        if old_vals[i] != new_vals[i] {
            line_sums[line] = new_vals[i];
        }
    }
}

/* ========= Neighbors ========= */
#[inline(always)]
fn pick_random_distinct_pair(rng: &mut SmallRng, len: usize) -> (usize, usize) {
    let a = rng.gen_range(0..len);
    let mut b = rng.gen_range(0..len);
    if a == b { b = (b + 1) % len; }
    (a, b)
}

/* ========= Utilities ========= */
fn to_2d(arr: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut out = vec![vec![0usize; n]; n];
    for r in 0..n { for c in 0..n { out[r][c] = arr[rc_to_idx(r, c, n)]; } }
    out
}

/* ========= Solver ========= */

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let n = (challenge.numbers.len() as f64).sqrt() as usize;
    let m = 2 * n + 2;
    let inv_m = 1.0 / (m as f64);

    // ---- Seed & initial score ----
    let mut arr = seed_arrangement(&challenge.numbers, n);
    let (mut line_sums, mut sum_lines, mut sumsq_lines) = init_line_sums(&challenge.numbers, n, &arr);
    let mut cur_var = variance_from(sum_lines, sumsq_lines, inv_m);

    // Initial save
    save_solution(&Solution { arrangement: to_2d(&arr, n) })?;
    if cur_var <= GOAL_VAR {
        // We already did great — exit early without needing any hidden threshold.
        return Ok(());
    }

    // Best tracking + throttled saves
    let mut best_arr = arr.clone();
    let mut best_var = cur_var;
    let mut last_save = Instant::now();
    let save_throttle = Duration::from_millis(SAVE_THROTTLE_MS);

    // Early-stopping: plateau detector
    let mut last_best_time = Instant::now();

    // Annealing with a short safety cap (not the full ~20s)
    let t_start = Instant::now();
    let time_budget = Duration::from_millis(TIME_BUDGET_MS);
    let mut temperature = 1200.0f64;
    let cooling = 0.99975f64;

    let mut tmp_lines: [usize; 8] = [0; 8];

    while t_start.elapsed() < time_budget {
        let (p, q) = if RANDOM_SWAPS_ONLY {
            pick_random_distinct_pair(&mut rng, n * n)
        } else {
            pick_random_distinct_pair(&mut rng, n * n) // (keep it simple & fast)
        };

        let (new_var, new_sum, new_sumsq, k, oldv, newv) =
            score_swap(&challenge.numbers, n, &arr, p, q, &line_sums, sum_lines, sumsq_lines, inv_m, &mut tmp_lines);

        let delta = new_var - cur_var;
        let accept = delta <= 0.0 || rng.gen::<f64>().ln() < -delta / temperature.max(1e-9);

        if accept {
            apply_swap_and_update(&mut arr, p, q, &mut line_sums, &tmp_lines, k, &oldv, &newv);
            sum_lines = new_sum;
            sumsq_lines = new_sumsq;
            cur_var = new_var;

            if cur_var + MIN_IMPROVE < best_var {
                best_var = cur_var;
                best_arr.copy_from_slice(&arr);
                last_best_time = Instant::now();

                // If we’ve reached a very small variance, we’re done.
                if best_var <= GOAL_VAR {
                    let _ = save_solution(&Solution { arrangement: to_2d(&best_arr, n) });
                    return Ok(());
                }

                // Save occasionally (reshape+I/O)
                if last_save.elapsed() >= save_throttle {
                    last_save = Instant::now();
                    let _ = save_solution(&Solution { arrangement: to_2d(&best_arr, n) });
                }
            }
        }

        temperature *= cooling;
        if temperature < 0.05 { temperature = 15.0; } // tiny reheat to avoid freeze

        // Plateau early-stop: no meaningful improvement for PLATEAU_MS
        if last_best_time.elapsed() >= Duration::from_millis(PLATEAU_MS) {
            // Save the best and exit; we’ve flatlined.
            let _ = save_solution(&Solution { arrangement: to_2d(&best_arr, n) });
            return Ok(());
        }
    }

    // Safety cap reached — save best and exit
    let _ = save_solution(&Solution { arrangement: to_2d(&best_arr, n) });
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
