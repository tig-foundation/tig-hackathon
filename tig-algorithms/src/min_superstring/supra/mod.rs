use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::min_superstring::{Challenge, Solution};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.strings.len();
    if n == 0 {
        return Err(anyhow!("Empty challenge"));
    }

    fn counts_of(s: &str) -> [u8; 26] {
        let mut cnt = [0u8; 26];
        for &b in s.as_bytes() {
            let idx = (b - b'a') as usize;
            cnt[idx] += 1;
        }
        cnt
    }

    #[inline]
    fn intersect_sum(a: &[u8; 26], b: &[u8; 26]) -> u8 {
        let mut s = 0u8;
        for i in 0..26 {
            s = s.saturating_add(a[i].min(b[i]));
        }
        s
    }

    let counts: Vec<[u8; 26]> = challenge.strings.iter().map(|s| counts_of(s)).collect();

    fn score_order(
        order: &[usize],
        counts: &[[u8; 26]],
        keep_edges: bool,
    ) -> (usize, Option<Vec<[u8; 26]>>) {
        let n = order.len();
        let mut avail: Vec<[u8; 26]> = counts.to_vec();
        let mut saved: usize = 0;
        let mut edges: Vec<[u8; 26]> = Vec::new();
        if keep_edges {
            edges.resize(n.saturating_sub(1), [0u8; 26]);
        }

        for j in 0..n.saturating_sub(1) {
            let a = order[j];
            let b = order[j + 1];
            let mut x = [0u8; 26];
            let mut step_saved = 0usize;

            for c in 0..26 {
                let v = avail[a][c].min(avail[b][c]);
                if v > 0 {
                    x[c] = v;
                    avail[a][c] -= v;
                    avail[b][c] -= v;
                    step_saved += v as usize;
                }
            }
            saved += step_saved;
            if keep_edges {
                edges[j] = x;
            }
        }
        (saved, if keep_edges { Some(edges) } else { None })
    }

    fn greedy_order_from(start: usize, counts: &[[u8; 26]]) -> Vec<usize> {
        let n = counts.len();
        let mut order = Vec::with_capacity(n);
        let mut used = vec![false; n];
        order.push(start);
        used[start] = true;

        for _ in 1..n {
            let last = *order.last().unwrap();
            let mut best = None::<(u8, usize)>;
            for j in 0..n {
                if used[j] {
                    continue;
                }
                let w = intersect_sum(&counts[last], &counts[j]);
                match best {
                    None => best = Some((w, j)),
                    Some((bw, bj)) => {
                        if w > bw || (w == bw && j < bj) {
                            best = Some((w, j));
                        }
                    }
                }
            }
            let next = best.map(|t| t.1).unwrap();
            used[next] = true;
            order.push(next);
        }
        order
    }

    let mut totals: Vec<(u32, usize)> = (0..n)
        .map(|i| {
            let mut s = 0u32;
            for j in 0..n {
                if i != j {
                    s += intersect_sum(&counts[i], &counts[j]) as u32;
                }
            }
            (s, i)
        })
        .collect();
    totals.sort_by(|a, b| b.cmp(a)); // descending

    let mut rng = SmallRng::from_seed(challenge.seed);
    let k = n.min(8);
    let mut candidates: Vec<Vec<usize>> = Vec::new();

    for t in totals.iter().take(k) {
        candidates.push(greedy_order_from(t.1, &counts));
    }
    let extra = (k / 2).max(1);
    for _ in 0..extra {
        let s = rng.gen_range(0..n);
        candidates.push(greedy_order_from(s, &counts));
    }

    let mut best_order = {
        let mut best = None::<(isize, Vec<usize>)>;
        for o in candidates {
            let (saved, _) = score_order(&o, &counts, false);
            let score = saved as isize;
            if best.as_ref().map_or(true, |(bs, _)| score > *bs) {
                best = Some((score, o));
            }
        }
        best.unwrap().1
    };

    let mut best_saved = score_order(&best_order, &counts, false).0 as isize;

    let n_usize = n;
    let two_opt_tries = (n_usize.saturating_mul(20)).clamp(500, 20_000); // scale with n
    for _ in 0..two_opt_tries {
        if n_usize < 4 {
            break;
        }
        let i = rng.gen_range(0..n_usize - 2);
        let j = rng.gen_range(i + 1..n_usize - 1);
        best_order[i + 1..=j].reverse();
        let new_saved = score_order(&best_order, &counts, false).0 as isize;
        if new_saved > best_saved {
            best_saved = new_saved;
        } else {
            best_order[i + 1..=j].reverse();
        }
    }

    let (saved_final, Some(edge_vecs)) = score_order(&best_order, &counts, true) else {
        unreachable!();
    };

    let mut prefix_seq: Vec<String> = vec![String::new(); n];
    let mut suffix_seq: Vec<String> = vec![String::new(); n];

    for j in 0..best_order.len().saturating_sub(1) {
        let a = best_order[j];
        let b = best_order[j + 1];
        let x = &edge_vecs[j];

        let mut seq = String::with_capacity(x.iter().map(|&v| v as usize).sum());
        for c in 0..26 {
            for _ in 0..x[c] {
                seq.push((b'a' as u8 + c as u8) as char);
            }
        }
        suffix_seq[a] = seq.clone();
        prefix_seq[b] = seq;
    }

    let mut permuted: Vec<String> = vec![String::new(); n];
    for idx in 0..n {
        let mut used = [0u8; 26];

        let mut s = String::with_capacity(5);
        for &ch in prefix_seq[idx].as_bytes() {
            let c = (ch - b'a') as usize;
            used[c] += 1;
            s.push(ch as char);
        }

        let mut need = [0u8; 26];
        for c in 0..26 {
            let suf_use = suffix_seq[idx].as_bytes().iter().filter(|&&x| (x - b'a') as usize == c).count() as u8;
            let total = counts[idx][c];
            let pre_use = used[c];
            let rem = total.saturating_sub(pre_use).saturating_sub(suf_use);
            need[c] = rem;
        }
        for c in 0..26 {
            for _ in 0..need[c] {
                used[c] += 1;
                s.push((b'a' + c as u8) as char);
            }
        }

        for &ch in suffix_seq[idx].as_bytes() {
            let c = (ch - b'a') as usize;
            used[c] += 1;
            s.push(ch as char);
        }

        debug_assert_eq!(s.len(), 5, "permuted string must be length 5");
        #[cfg(debug_assertions)]
        {
            let mut chk = [0u8; 26];
            for &b in s.as_bytes() {
                chk[(b - b'a') as usize] += 1;
            }
            for c in 0..26 {
                debug_assert_eq!(
                    chk[c], counts[idx][c],
                    "letter counts mismatch for string {idx}"
                );
            }
        }

        permuted[idx] = s;
    }

    // 4) Build the superstring implicitly and compute start indices
    let mut super_len = 0usize;
    let mut starts: Vec<usize> = vec![0; n];

    let first = best_order[0];
    starts[first] = 0;
    super_len = 5; // start with the first permuted string laid at index 0

    for j in 1..best_order.len() {
        let curr = best_order[j];
        let ov = prefix_seq[curr].len();
        starts[curr] = super_len.saturating_sub(ov);
        super_len += 5 - ov;
    }

    let solution = Solution {
        permuted_strings: permuted,
        superstring_idxs: starts,
    };
    save_solution(&solution)?;

    let saved_chars = saved_final;
    let final_len = super_len;
    println!(
        "Placed {} strings. Saved {} chars via overlaps. Final superstring length: {}",
        n, saved_chars, final_len
    );

    Ok(())
}

