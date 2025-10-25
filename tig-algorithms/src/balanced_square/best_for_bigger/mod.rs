use anyhow::Result;
use tig_challenges::balanced_square::*;

/// Multi-start deterministic O(n²) greedy:
/// - Counting-sort indices by value (values are 1..=100)
/// - For 6 parameter sets, precompute A/B (+ parity C) scan orders
/// - For each value (desc), place to the best of 2–3 next-free candidates (ΔJ in O(1))
/// - Keep the best arrangement across runs; single save, then return
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nums = &challenge.numbers;
    let total = nums.len();
    let n = (total as f64).sqrt() as usize;
    debug_assert_eq!(n * n, total);
    let m = 2 * n + 2;
    let m_i64 = m as i64;

    /* ---------- counting-sort indices by value (1..=100) ---------- */
    const K: usize = 100;
    let mut buckets: [Vec<usize>; K + 1] = std::array::from_fn(|_| Vec::new());
    for i in 0..total { buckets[nums[i] as usize].push(i); }
    let mut order = Vec::with_capacity(total);
    for v in (1..=K).rev() { order.extend(buckets[v].iter().copied()); }

    /* ---------- helpers for objective ---------- */
    #[inline(always)]
    fn delta_obj_place_i64(sums: &[i64], sum_s: i64, m_i64: i64, n: usize, r: usize, c: usize, v: i32) -> i64 {
        // ΔJ for placing v at (r,c); J = m*Σ s² - (Σ s)²
        let dd = v as i64;

        let s_row = sums[r];
        let s_col = sums[n + c];

        let mut d_sum  = dd + dd;
        let mut d_sum2 = (2 * s_row + dd) * dd + (2 * s_col + dd) * dd;

        if r == c {
            let s = sums[2 * n];
            d_sum  += dd;
            d_sum2 += (2 * s + dd) * dd;
        }
        if r + c + 1 == n {
            let s = sums[2 * n + 1];
            d_sum  += dd;
            d_sum2 += (2 * s + dd) * dd;
        }

        m_i64 * d_sum2 - (2 * sum_s * d_sum + d_sum * d_sum)
    }

    #[inline(always)]
    fn apply_place_i64(sums: &mut [i64], sum_s: &mut i64, sum_s2: &mut i64, n: usize, r: usize, c: usize, v: i32) {
        let dd = v as i64;

        let s = sums[r];
        *sum_s  += dd;
        *sum_s2 += (2 * s + dd) * dd;
        sums[r] = s + dd;

        let s = sums[n + c];
        *sum_s  += dd;
        *sum_s2 += (2 * s + dd) * dd;
        sums[n + c] = s + dd;

        if r == c {
            let s = sums[2 * n];
            *sum_s  += dd;
            *sum_s2 += (2 * s + dd) * dd;
            sums[2 * n] = s + dd;
        }
        if r + c + 1 == n {
            let s = sums[2 * n + 1];
            *sum_s  += dd;
            *sum_s2 += (2 * s + dd) * dd;
            sums[2 * n + 1] = s + dd;
        }
    }

    #[inline(always)]
    fn build_seq_a(total: usize, n: usize, a: usize, s: usize) -> Vec<usize> {
        // A: r = t % n; q = t / n; c = (a*q + r + s) % n
        let mut seq = vec![0usize; total];
        for t in 0..total {
            let r = t % n;
            let q = t / n;
            let c = (a.wrapping_mul(q) + r + s) % n;
            seq[t] = r * n + c;
        }
        seq
    }
    #[inline(always)]
    fn build_seq_b(total: usize, n: usize, a: usize, s: usize) -> Vec<usize> {
        // B: c = t % n; q = t / n; r = (a*q + c + s) % n
        let mut seq = vec![0usize; total];
        for t in 0..total {
            let c = t % n;
            let q = t / n;
            let r = (a.wrapping_mul(q) + c + s) % n;
            seq[t] = r * n + c;
        }
        seq
    }
    #[inline(always)]
    fn build_seq_parity(total: usize, n: usize) -> Vec<usize> {
        // C: even (r+c) first, then odd
        let mut seq = Vec::with_capacity(total);
        seq.extend((0..total).filter(|&id| ((id / n) + (id % n)) & 1 == 0));
        seq.extend((0..total).filter(|&id| ((id / n) + (id % n)) & 1 == 1));
        seq
    }
    #[inline(always)]
    fn next_free(ptr: &mut usize, seq: &[usize], used: &[u8]) -> Option<usize> {
        while *ptr < seq.len() {
            let id = seq[*ptr];
            if used[id] == 0 { return Some(id); }
            *ptr += 1;
        }
        None
    }

    // parameter sets (a, s) for A/B; choose coprime steps and varied shifts
    let candidates: &[(usize, usize, usize, usize, bool)] = &[
        // (aA, sA, aB, sB, use_parity_C)
        (1, 0, if n % 2 == 0 { n - 1 } else { 1 }, (n / 2) % n, true),
        (1, n / 3, if n % 2 == 0 { n - 1 } else { 1 }, (n / 4) % n, true),
        (1, n / 5, 1, n / 2, true),
        (if n % 2 == 0 { n - 1 } else { 1 }, 0, 1, (n / 2) % n, true),
        (1, 0, 1, 0, false), // plain Latin pairs
        (if n % 2 == 0 { n - 1 } else { 1 }, n / 2, if n % 2 == 0 { n - 1 } else { 1 }, 0, false),
    ];

    let mut best_J = i64::MAX;
    let mut best_flat = vec![0usize; total];

    for &(aA, sA, aB, sB, use_c) in candidates {
        // Precompute sequences for this run
        let seq_a = build_seq_a(total, n, aA, sA);
        let seq_b = build_seq_b(total, n, aB, sB);
        let seq_c = if use_c { build_seq_parity(total, n) } else { Vec::new() };

        // State
        let mut used = vec![0u8; total];
        let mut pa = 0usize;
        let mut pb = 0usize;
        let mut pc = 0usize;
        let mut flat = vec![0usize; total];

        // Objective accumulators
        let mut sums = vec![0i64; m];
        let mut sum_s: i64 = 0;
        let mut sum_s2: i64 = 0;

        // Greedy fill
        for &idx in &order {
            let v = nums[idx];

            // up to 3 candidates
            let mut cand_id = [usize::MAX; 3];
            let mut have = [false; 3];

            if let Some(id) = next_free(&mut pa, &seq_a, &used) { cand_id[0] = id; have[0] = true; }
            if let Some(id) = next_free(&mut pb, &seq_b, &used) { cand_id[1] = id; have[1] = true; }
            if use_c {
                if let Some(id) = next_free(&mut pc, &seq_c, &used) { cand_id[2] = id; have[2] = true; }
            }

            // choose best ΔJ (there will always be at least one)
            let mut best_k = 0usize;
            let mut best_d = i64::MAX;
            for k in 0..have.len() {
                if !have[k] { continue; }
                let id = cand_id[k];
                let r = id / n;
                let c = id % n;
                let d = delta_obj_place_i64(&sums, sum_s, m_i64, n, r, c, v);
                if d < best_d {
                    best_d = d;
                    best_k = k;
                }
            }

            let id = cand_id[best_k];
            match best_k {
                0 => pa += 1,
                1 => pb += 1,
                _ => pc += 1,
            }
            used[id] = 1;

            let r = id / n;
            let c = id % n;
            apply_place_i64(&mut sums, &mut sum_s, &mut sum_s2, n, r, c, v);
            flat[id] = idx;
        }

        // Objective value for this run (proportional to variance)
        let J = (m_i64 * sum_s2) - (sum_s * sum_s);
        if J < best_J {
            best_J = J;
            best_flat.copy_from_slice(&flat);
        }
    }

    // Materialize the best matrix and save once
    let mut arrangement = Vec::with_capacity(n);
    for r in 0..n {
        let base = r * n;
        let mut row = Vec::with_capacity(n);
        for c in 0..n { row.push(best_flat[base + c]); }
        arrangement.push(row);
    }

    save_solution(&Solution { arrangement })?;
    Ok(())
}
