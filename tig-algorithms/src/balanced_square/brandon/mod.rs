// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::Result;
use tig_challenges::balanced_square::*;

/// Modo TURBO: una pasada O(n²), sin RNG, sin búsqueda local, un save y terminar.
/// - Ordena valores desc.
/// - Dos órdenes biyectivos (A y B) generados al vuelo.
/// - Para cada número, coloca en la mejor de dos celdas (delta O(1) del objetivo entero).
/// - Guarda una vez y return Ok(()) para minimizar wall-time.
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nums = &challenge.numbers;
    let total = nums.len();
    let n = (total as f64).sqrt() as usize;
    debug_assert_eq!(n * n, total);

    // Índices ordenados por valor descendente (inestable = más rápido).
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_unstable_by(|&i, &j| nums[j].cmp(&nums[i]));

    // Filas [0..n), Cols [n..2n), diag0: 2n, diag1: 2n+1
    let m = 2 * n + 2;

    // Órdenes biyectivos "on the fly".
    // A: r = t % n; q = t / n; c = (a*q + r + s) % n
    // B: c = t % n; q = t / n; r = (a*q + c + s) % n
    #[inline(always)]
    fn id_seq_a(t: usize, n: usize, a: usize, s: usize) -> usize {
        let r = t % n;
        let q = t / n;
        let c = (a.wrapping_mul(q) + r + s) % n;
        r * n + c
    }
    #[inline(always)]
    fn id_seq_b(t: usize, n: usize, a: usize, s: usize) -> usize {
        let c = t % n;
        let q = t / n;
        let r = (a.wrapping_mul(q) + c + s) % n;
        r * n + c
    }

    // Pasos/shift: para n par, n-1 es coprimo con n; para impar, 1 lo es.
    let a1 = 1usize;
    let a2 = if n % 2 == 0 { n - 1 } else { 1 };
    let s1 = 0usize;
    let s2 = (n / 2) % n; // ayuda con diagonales en n par

    // Libro mayor del objetivo: J = m*Σ s² - (Σ s)² (proporcional a la varianza).
    let mut sums = vec![0i64; m];
    let mut sum_s: i128 = 0;
    let mut sum_s2: i128 = 0;
    let m128: i128 = m as i128;

    #[inline(always)]
    fn delta_obj_place(sums: &[i64], sum_s: i128, m128: i128, n: usize, r: usize, c: usize, v: i32) -> i128 {
        let dd = v as i128;
        // fila
        let s0r = sums[r] as i128;
        let mut d_sum  = dd;
        let mut d_sum2 = 2 * s0r * dd + dd * dd;
        // col
        let s0c = sums[n + c] as i128;
        d_sum  += dd;
        d_sum2 += 2 * s0c * dd + dd * dd;
        // diag principal
        if r == c {
            let s0 = sums[2 * n] as i128;
            d_sum  += dd;
            d_sum2 += 2 * s0 * dd + dd * dd;
        }
        // diag secundaria
        if r + c + 1 == n {
            let s0 = sums[2 * n + 1] as i128;
            d_sum  += dd;
            d_sum2 += 2 * s0 * dd + dd * dd;
        }
        m128 * d_sum2 - (2 * sum_s * d_sum + d_sum * d_sum)
    }

    #[inline(always)]
    fn apply_place(
        sums: &mut [i64], sum_s: &mut i128, sum_s2: &mut i128,
        n: usize, r: usize, c: usize, v: i32
    ) {
        let dd = v as i128;
        // fila
        let s0r = sums[r] as i128;
        *sum_s  += dd; *sum_s2 += 2 * s0r * dd + dd * dd; sums[r] = (s0r + dd) as i64;
        // col
        let s0c = sums[n + c] as i128;
        *sum_s  += dd; *sum_s2 += 2 * s0c * dd + dd * dd; sums[n + c] = (s0c + dd) as i64;
        // diag principal
        if r == c {
            let s0 = sums[2 * n] as i128;
            *sum_s  += dd; *sum_s2 += 2 * s0 * dd + dd * dd; sums[2 * n] = (s0 + dd) as i64;
        }
        // diag secundaria
        if r + c + 1 == n {
            let s0 = sums[2 * n + 1] as i128;
            *sum_s  += dd; *sum_s2 += 2 * s0 * dd + dd * dd; sums[2 * n + 1] = (s0 + dd) as i64;
        }
    }

    // Relleno: elegimos entre la próxima celda libre de A y de B.
    let mut used = vec![0u8; total]; // 0=libre,1=usada (más rápido que Vec<bool>)
    let mut pa = 0usize;
    let mut pb = 0usize;
    let mut flat = vec![0usize; total];

    for &idx in &order {
        let v = nums[idx];

        // siguiente libre de A
        let mut ida = 0usize;
        let mut has_a = false;
        while pa < total {
            let cand = id_seq_a(pa, n, a1, s1);
            if used[cand] == 0 { ida = cand; has_a = true; break; }
            pa += 1;
        }
        // siguiente libre de B
        let mut idb = 0usize;
        let mut has_b = false;
        while pb < total {
            let cand = id_seq_b(pb, n, a2, s2);
            if used[cand] == 0 { idb = cand; has_b = true; break; }
            pb += 1;
        }

        // elegir (fallback seguro si uno se agota)
        let pick_a = if !has_b {
            true
        } else if !has_a {
            false
        } else {
            let (ra, ca) = (ida / n, ida % n);
            let (rb, cb) = (idb / n, idb % n);
            let da = delta_obj_place(&sums, sum_s, m128, n, ra, ca, v);
            let db = delta_obj_place(&sums, sum_s, m128, n, rb, cb, v);
            da <= db
        };

        let id = if pick_a { ida } else { idb };
        used[id] = 1;
        if pick_a { pa += 1; } else { pb += 1; }

        let (r, c) = (id / n, id % n);
        apply_place(&mut sums, &mut sum_s, &mut sum_s2, n, r, c, v);
        flat[id] = idx;
    }

    // a matriz
    let mut arrangement = Vec::with_capacity(n);
    for r in 0..n {
        let base = r * n;
        let mut row = Vec::with_capacity(n);
        for c in 0..n { row.push(flat[base + c]); }
        arrangement.push(row);
    }

    // ÚNICO save y terminamos para maximizar throughput.
    save_solution(&Solution { arrangement })?;
    return Ok(());
}