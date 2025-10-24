// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::time::{Duration, Instant};
use tig_challenges::balanced_square::*;

/// O(1) helper: update a single aggregate sum and the global stats.
#[inline]
fn bump(sum_all: &mut f64, sum_sq: &mut f64, target: &mut i64, delta: i64) {
    if delta == 0 {
        return;
    }
    let old = *target as f64;
    let new = (old as i64 + delta) as f64;
    *sum_all += new - old;
    *sum_sq += new * new - old * old;
    *target += delta;
}

/// Compute variance of an arrangement (for seed selection).
fn variance_of_arrangement(numbers: &[i32], arr: &[Vec<usize>]) -> f64 {
    let n = arr.len();
    let m = (2 * n + 2) as f64;

    let mut sum_all = 0.0;
    let mut sum_sq = 0.0;

    // rows & cols
    for i in 0..n {
        let mut rsum = 0i64;
        let mut csum = 0i64;
        for j in 0..n {
            rsum += numbers[arr[i][j]] as i64;
            csum += numbers[arr[j][i]] as i64;
        }
        let r = rsum as f64;
        let c = csum as f64;
        sum_all += r + c;
        sum_sq += r * r + c * c;
    }
    // diags
    let mut d0 = 0i64;
    let mut d1 = 0i64;
    for i in 0..n {
        d0 += numbers[arr[i][i]] as i64;
        d1 += numbers[arr[i][n - 1 - i]] as i64;
    }
    let d0f = d0 as f64;
    let d1f = d1 as f64;
    sum_all += d0f + d1f;
    sum_sq += d0f * d0f + d1f * d1f;

    let mean = sum_all / m;
    (sum_sq / m) - mean * mean
}

/// Latin-style seed builder variants over sorted groups.
fn best_seed_from_variants(numbers: &[i32]) -> Vec<usize> {
    let n2 = numbers.len();
    let n = (n2 as f64).sqrt() as usize;

    // Sorted indices by value (ascending)
    let mut idxs: Vec<usize> = (0..n2).collect();
    idxs.sort_unstable_by_key(|&i| numbers[i]);

    // Split into n groups of size n
    let mut groups: Vec<Vec<usize>> = (0..n)
        .map(|g| idxs[g * n..g * n + n].to_vec())
        .collect();

    // Within-group orderings
    fn interleave_low_high(v: &[usize]) -> Vec<usize> {
        let mut out = Vec::with_capacity(v.len());
        let (mut l, mut r) = (0usize, v.len().saturating_sub(1));
        let mut take_low = true;
        while l <= r {
            if take_low {
                out.push(v[l]);
                l += 1;
            } else {
                out.push(v[r]);
                if r == 0 {
                    break;
                }
                r -= 1;
            }
            take_low = !take_low;
        }
        out
    }

    // Build one arrangement variant
    // Latin mapping with slope s: column c = (g - s*r) mod n
    // Choose element idx inside group: (a*r + b*g) mod n
    fn build(
        groups: &[Vec<usize>],
        n: usize,
        s: usize,
        a: usize,
        b: usize,
        order_kind: u8, // 0 asc, 1 desc, 2 interleave
    ) -> Vec<Vec<usize>> {
        let mut arr = vec![vec![0usize; n]; n];
        for g in 0..n {
            let mut order = match order_kind {
                0 => groups[g].clone(),
                1 => {
                    let mut t = groups[g].clone();
                    t.reverse();
                    t
                }
                _ => interleave_low_high(&groups[g]),
            };
            for r in 0..n {
                let c = (g + n - ((s * r) % n)) % n;
                let pick = order[(a * r + b * g) % n];
                arr[r][c] = pick;
            }
        }
        arr
    }

    // Try a tiny grid of variants; pick the best.
    let slopes: Vec<usize> = if n <= 1 { vec![0] } else { vec![1, n - 1] };
    let mut strides: Vec<usize> = vec![1];
    if n > 1 {
        strides.push(n - 1);
    }
    if n % 2 == 1 && n > 1 {
        strides.push(2);
    }
    let shifts = [0usize, 1usize];
    let order_kinds = [0u8, 1u8, 2u8];

    let mut best_arr: Option<Vec<Vec<usize>>> = None;
    let mut best_var = f64::INFINITY;

    for &s in &slopes {
        for &a in &strides {
            for &b in &shifts {
                for &ok in &order_kinds {
                    let arr = build(&groups, n, s, a, b, ok);
                    let v = variance_of_arrangement(numbers, &arr);
                    if v < best_var {
                        best_var = v;
                        best_arr = Some(arr);
                    }
                }
            }
        }
    }

    // Flatten to row-major 1D index list
    let arr = best_arr.unwrap_or_else(|| {
        // Fallback: simple snake fill of ascending
        let mut simple = vec![vec![0usize; n]; n];
        let mut k = 0;
        for r in 0..n {
            if r % 2 == 0 {
                for c in 0..n {
                    simple[r][c] = idxs[k];
                    k += 1;
                }
            } else {
                for c in (0..n).rev() {
                    simple[r][c] = idxs[k];
                    k += 1;
                }
            }
        }
        simple
    });

    let mut flat = Vec::with_capacity(n2);
    for r in 0..n {
        for c in 0..n {
            flat.push(arr[r][c]);
        }
    }
    flat
}

/// Keeps all sums and the variance in O(1) per accepted swap.
struct SumTracker {
    rows: Vec<i64>,
    cols: Vec<i64>,
    d0: i64,
    d1: i64,
    sum_all: f64,
    sum_sq: f64,
    m: f64,
}

impl SumTracker {
    fn new(numbers: &[i32], arr: &[usize], n: usize) -> Self {
        let mut rows = vec![0i64; n];
        let mut cols = vec![0i64; n];
        let mut d0 = 0i64;
        let mut d1 = 0i64;

        for r in 0..n {
            for c in 0..n {
                let v = numbers[arr[r * n + c]] as i64;
                rows[r] += v;
                cols[c] += v;
                if r == c {
                    d0 += v;
                }
                if r + c == n - 1 {
                    d1 += v;
                }
            }
        }

        let mut sum_all = 0.0;
        let mut sum_sq = 0.0;
        for &s in &rows {
            sum_all += s as f64;
            sum_sq += (s as f64) * (s as f64);
        }
        for &s in &cols {
            sum_all += s as f64;
            sum_sq += (s as f64) * (s as f64);
        }
        sum_all += d0 as f64;
        sum_sq += (d0 as f64) * (d0 as f64);
        sum_all += d1 as f64;
        sum_sq += (d1 as f64) * (d1 as f64);

        let m = (2 * n + 2) as f64;
        Self {
            rows,
            cols,
            d0,
            d1,
            sum_all,
            sum_sq,
            m,
        }
    }

    #[inline]
    fn variance(&self) -> f64 {
        let mean = self.sum_all / self.m;
        (self.sum_sq / self.m) - mean * mean
    }

    /// Apply the swap (r1,c1,v1) <-> (r2,c2,v2) to the maintained sums.
    fn apply_swap(
        &mut self,
        n: usize,
        r1: usize,
        c1: usize,
        v1: i64,
        r2: usize,
        c2: usize,
        v2: i64,
    ) {
        // Rows
        bump(&mut self.sum_all, &mut self.sum_sq, &mut self.rows[r1], -v1 + v2);
        if r2 != r1 {
            bump(&mut self.sum_all, &mut self.sum_sq, &mut self.rows[r2], -v2 + v1);
        }
        // Cols
        bump(&mut self.sum_all, &mut self.sum_sq, &mut self.cols[c1], -v1 + v2);
        if c2 != c1 {
            bump(&mut self.sum_all, &mut self.sum_sq, &mut self.cols[c2], -v2 + v1);
        }
        // Main diag
        let on_d0_1 = r1 == c1;
        let on_d0_2 = r2 == c2;
        match (on_d0_1, on_d0_2) {
            (true, true) => {}
            (true, false) => bump(&mut self.sum_all, &mut self.sum_sq, &mut self.d0, -v1 + v2),
            (false, true) => bump(&mut self.sum_all, &mut self.sum_sq, &mut self.d0, -v2 + v1),
            (false, false) => {}
        }
        // Anti diag
        let on_d1_1 = r1 + c1 == n - 1;
        let on_d1_2 = r2 + c2 == n - 1;
        match (on_d1_1, on_d1_2) {
            (true, true) => {}
            (true, false) => bump(&mut self.sum_all, &mut self.sum_sq, &mut self.d1, -v1 + v2),
            (false, true) => bump(&mut self.sum_all, &mut self.sum_sq, &mut self.d1, -v2 + v1),
            (false, false) => {}
        }
    }

    /// Delta-variance if we swap (no commit).
    fn delta_variance(
        &self,
        n: usize,
        r1: usize,
        c1: usize,
        v1: i64,
        r2: usize,
        c2: usize,
        v2: i64,
    ) -> f64 {
        let mut sum_all = self.sum_all;
        let mut sum_sq = self.sum_sq;

        #[inline]
        fn bump_local(sum_all: &mut f64, sum_sq: &mut f64, s_old: i64, delta: i64) {
            if delta == 0 {
                return;
            }
            let old = s_old as f64;
            let new = (s_old + delta) as f64;
            *sum_all += new - old;
            *sum_sq += new * new - old * old;
        }

        // Rows
        bump_local(&mut sum_all, &mut sum_sq, self.rows[r1], -v1 + v2);
        if r2 != r1 {
            bump_local(&mut sum_all, &mut sum_sq, self.rows[r2], -v2 + v1);
        }
        // Cols
        bump_local(&mut sum_all, &mut sum_sq, self.cols[c1], -v1 + v2);
        if c2 != c1 {
            bump_local(&mut sum_all, &mut sum_sq, self.cols[c2], -v2 + v1);
        }
        // Diags
        let on_d0_1 = r1 == c1;
        let on_d0_2 = r2 == c2;
        match (on_d0_1, on_d0_2) {
            (true, true) => {}
            (true, false) => bump_local(&mut sum_all, &mut sum_sq, self.d0, -v1 + v2),
            (false, true) => bump_local(&mut sum_all, &mut sum_sq, self.d0, -v2 + v1),
            (false, false) => {}
        }
        let on_d1_1 = r1 + c1 == n - 1;
        let on_d1_2 = r2 + c2 == n - 1;
        match (on_d1_1, on_d1_2) {
            (true, true) => {}
            (true, false) => bump_local(&mut sum_all, &mut sum_sq, self.d1, -v1 + v2),
            (false, true) => bump_local(&mut sum_all, &mut sum_sq, self.d1, -v2 + v1),
            (false, false) => {}
        }

        let m = self.m;
        let new_var = (sum_sq / m) - (sum_all / m).powi(2);
        new_var - self.variance()
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let numbers = &challenge.numbers;
    let n2 = numbers.len();
    if n2 == 0 {
        return Err(anyhow!("empty instance"));
    }
    let n = (n2 as f64).sqrt() as usize;
    if n * n != n2 {
        return Err(anyhow!("numbers length must be a perfect square"));
    }

    // --- Time budget (harness kills at ~20s)
    let start = Instant::now();
    let time_budget = Duration::from_millis(19_500);
    let mut rng = SmallRng::from_seed(challenge.seed);

    // --- Construct a strong seed deterministically
    let mut arr = best_seed_from_variants(numbers); // flat vec of length n^2
    let mut tracker = SumTracker::new(numbers, &arr, n);
    let mut best_arr = arr.clone();
    let mut best_var = tracker.variance();

    // Save initial best
    save_solution(&Solution {
        arrangement: (0..n)
            .map(|r| best_arr[r * n..r * n + n].to_vec())
            .collect(),
    })?;

    if best_var <= 1e-12 {
        return Ok(());
    }

    // --- SA parameters
    let t0 = 150.0_f64;
    let t1 = 0.01_f64;

    let mut no_improve_steps: u64 = 0;
    let mut since_restart_steps: u64 = 0;

    // Main search loop (with soft restarts)
    while start.elapsed() < time_budget {
        let elapsed = start.elapsed().as_secs_f64() / time_budget.as_secs_f64();
        let frac = elapsed.clamp(0.0, 1.0);
        let temp = t0 * (1.0 - frac) + t1 * frac;

        // Neighborhood mix: 0=global swap, 1=row swap, 2=col swap (favor targeted early)
        let choice = if rng.gen::<f64>() < 0.4 { 1 } else if rng.gen::<f64>() < 0.4 { 2 } else { 0 };

        let (p, q) = match choice {
            1 => {
                // same row
                let r = rng.gen_range(0..n);
                let c1 = rng.gen_range(0..n);
                let mut c2 = rng.gen_range(0..n);
                if c2 == c1 {
                    c2 = (c2 + 1) % n;
                }
                (r * n + c1, r * n + c2)
            }
            2 => {
                // same column
                let c = rng.gen_range(0..n);
                let r1 = rng.gen_range(0..n);
                let mut r2 = rng.gen_range(0..n);
                if r2 == r1 {
                    r2 = (r2 + 1) % n;
                }
                (r1 * n + c, r2 * n + c)
            }
            _ => {
                // global
                let p = rng.gen_range(0..n2);
                let mut q = rng.gen_range(0..n2);
                if q == p {
                    q = (q + 1) % n2;
                }
                (p, q)
            }
        };

        let (r1, c1) = (p / n, p % n);
        let (r2, c2) = (q / n, q % n);
        let v1 = numbers[arr[p]] as i64;
        let v2 = numbers[arr[q]] as i64;

        let dvar = tracker.delta_variance(n, r1, c1, v1, r2, c2, v2);
        let accept = if dvar <= 0.0 {
            true
        } else {
            let prob = (-dvar / temp.max(1e-9)).exp();
            rng.gen::<f64>() < prob
        };

        if accept {
            tracker.apply_swap(n, r1, c1, v1, r2, c2, v2);
            arr.swap(p, q);

            let cur_var = tracker.variance();
            if cur_var + 1e-12 < best_var {
                best_var = cur_var;
                best_arr.clone_from(&arr);
                // Save improvement
                save_solution(&Solution {
                    arrangement: (0..n)
                        .map(|r| best_arr[r * n..r * n + n].to_vec())
                        .collect(),
                })?;
                no_improve_steps = 0;
                since_restart_steps = 0;

                if best_var <= 1e-12 {
                    return Ok(());
                }
            } else {
                no_improve_steps += 1;
                since_restart_steps += 1;
            }
        } else {
            no_improve_steps += 1;
            since_restart_steps += 1;
        }

        // Stagnation handling: kick or soft restart
        if no_improve_steps > (5000u64).saturating_add((n as u64) * 200) {
            // small kick: shuffle a random row/col a few times (annealed acceptance)
            if rng.gen::<bool>() {
                // row kick
                let r = rng.gen_range(0..n);
                for _ in 0..3 {
                    let c_a = rng.gen_range(0..n);
                    let c_b = rng.gen_range(0..n);
                    if c_a == c_b {
                        continue;
                    }
                    let p = r * n + c_a;
                    let q = r * n + c_b;
                    let (r1, c1) = (r, c_a);
                    let (r2, c2) = (r, c_b);
                    let v1 = numbers[arr[p]] as i64;
                    let v2 = numbers[arr[q]] as i64;
                    let dvar = tracker.delta_variance(n, r1, c1, v1, r2, c2, v2);
                    if dvar <= 0.0 || rng.gen::<f64>() < (-dvar / temp.max(1e-9)).exp() {
                        tracker.apply_swap(n, r1, c1, v1, r2, c2, v2);
                        arr.swap(p, q);
                    }
                }
            } else {
                // col kick
                let c = rng.gen_range(0..n);
                for _ in 0..3 {
                    let r_a = rng.gen_range(0..n);
                    let r_b = rng.gen_range(0..n);
                    if r_a == r_b {
                        continue;
                    }
                    let p = r_a * n + c;
                    let q = r_b * n + c;
                    let (r1, c1) = (r_a, c);
                    let (r2, c2) = (r_b, c);
                    let v1 = numbers[arr[p]] as i64;
                    let v2 = numbers[arr[q]] as i64;
                    let dvar = tracker.delta_variance(n, r1, c1, v1, r2, c2, v2);
                    if dvar <= 0.0 || rng.gen::<f64>() < (-dvar / temp.max(1e-9)).exp() {
                        tracker.apply_swap(n, r1, c1, v1, r2, c2, v2);
                        arr.swap(p, q);
                    }
                }
            }
            no_improve_steps = 0;

            // Occasionally soft-restart from a fresh seed (keep best found so far)
            if since_restart_steps > 30_000 && start.elapsed() < time_budget.saturating_sub(Duration::from_millis(500)) {
                arr = best_seed_from_variants(numbers);
                tracker = SumTracker::new(numbers, &arr, n);
                since_restart_steps = 0;
                // opportunistic save current state
                save_solution(&Solution {
                    arrangement: (0..n)
                        .map(|r| arr[r * n..r * n + n].to_vec())
                        .collect(),
                })?;
            }
        }
    }

    // Final save of the best candidate and exit
    save_solution(&Solution {
        arrangement: (0..n)
            .map(|r| best_arr[r * n..r * n + n].to_vec())
            .collect(),
    })?;

    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
