use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::travelling_salesman::{Challenge, Solution};

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.node_positions.len();
    if n == 0 {
        return Err(anyhow!("Empty challenge"));
    }
    if n == 1 {
        let solution = Solution { route: vec![0] };
        save_solution(&solution)?;
        println!("Final route length: 0.0");
        return Ok(());
    }

    let dm: &Vec<Vec<f32>> = &challenge.distance_matrix;
    let mut rng = SmallRng::from_seed(challenge.seed);

    #[inline]
    fn tour_length(route: &[usize], dm: &[Vec<f32>]) -> f32 {
        let n = route.len();
        let mut d = 0.0_f32;
        for i in 0..n {
            let a = route[i];
            let b = route[(i + 1) % n];
            d += dm[a][b];
        }
        d
    }

    // Nearest Neighbor tour from a fixed start
    fn nn_tour(start: usize, dm: &[Vec<f32>]) -> Vec<usize> {
        let n = dm.len();
        let mut route = Vec::with_capacity(n);
        let mut unvisited: Vec<usize> = (0..n).collect();

        route.push(start);
        let pos = unvisited.iter().position(|&x| x == start).unwrap();
        unvisited.swap_remove(pos);

        let mut cur = start;
        while !unvisited.is_empty() {
            let (best_i, _) = unvisited
                .iter()
                .enumerate()
                .map(|(i, &cand)| (i, dm[cur][cand]))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            cur = unvisited.swap_remove(best_i);
            route.push(cur);
        }
        route
    }

    // Classic non-wrapping 2-Opt (first improvement). Safe and fast.
    fn two_opt_first_improvement(route: &mut [usize], dm: &[Vec<f32>]) -> f32 {
        let n = route.len();
        if n < 4 {
            return tour_length(route, dm);
        }
        loop {
            let mut improved = false;

            for i in 0..(n - 1) {
                let a = route[i];
                let b = route[i + 1];

                for j in (i + 2)..n {
                    // Skip the move that would break the (n-1,0) edge adjacency
                    if i == 0 && j == n - 1 {
                        continue;
                    }
                    let c = route[j];
                    let d = route[(j + 1) % n]; // j+1==n -> wraps to 0, but we skip that case above

                    let delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d];
                    if delta < -1e-6 {
                        route[i + 1..=j].reverse();
                        improved = true;
                        break; // first-improvement
                    }
                }
                if improved {
                    break; // restart outer loop
                }
            }

            if !improved {
                break;
            }
        }
        tour_length(route, dm)
    }

    // Double-bridge large perturbation (4-break)
    fn double_bridge(rng: &mut SmallRng, route: &mut Vec<usize>) {
        let n = route.len();
        if n < 8 {
            return; // too small to benefit
        }
        // four increasing cut points roughly in quartiles
        let mut cuts = [
            rng.gen_range(1..(n / 4)),
            rng.gen_range((n / 4)..(n / 2)),
            rng.gen_range((n / 2)..(3 * n / 4)),
            rng.gen_range((3 * n / 4)..(n - 1)),
        ];
        cuts.sort_unstable();
        let (a, b, c, d) = (cuts[0], cuts[1], cuts[2], cuts[3]);

        // segments: [0..a) [a..b) [b..c) [c..d) [d..n)
        // recombine: [0..a) [c..d) [b..c) [a..b) [d..n)
        let mut new_route = Vec::with_capacity(n);
        new_route.extend_from_slice(&route[..a]);
        new_route.extend_from_slice(&route[c..d]);
        new_route.extend_from_slice(&route[b..c]);
        new_route.extend_from_slice(&route[a..b]);
        new_route.extend_from_slice(&route[d..]);
        *route = new_route;
    }

    // ---- 1) Multi-start NN --------------------------------------------------
    let starts = n.min(8);
    let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(starts + 1);

    // evenly spaced starts
    let stride = (n / starts.max(1)).max(1);
    for k in 0..starts {
        candidates.push(nn_tour((k * stride) % n, dm));
    }
    // plus one random start for variety
    candidates.push(nn_tour(rng.gen_range(0..n), dm));

    let (mut best_route, _) = candidates
        .into_iter()
        .map(|r| {
            let l = tour_length(&r, dm);
            (r, l)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    let mut best_len = two_opt_first_improvement(&mut best_route, dm);

    // ---- 2) Iterated Local Search (double-bridge + 2-Opt) ------------------
    // Mild SA-like acceptance helps escape shallow basins.
    let mut temperature: f32 = 100.0;
    let cooling: f32 = 0.995;
    let min_temp: f32 = 1e-3;
    let ils_iters = 100 + 2 * n;

    for _ in 0..ils_iters {
        let mut cand = best_route.clone();
        double_bridge(&mut rng, &mut cand);
        let cand_len = two_opt_first_improvement(&mut cand, dm);

        let delta = cand_len - best_len;
        if delta < 0.0 || rng.gen::<f32>() < (-delta / temperature).exp() {
            best_route = cand;
            best_len = cand_len;
        }
        temperature = (temperature * cooling).max(min_temp);
    }

    // ---- 3) Save ------------------------------------------------------------
    let solution = Solution { route: best_route };
    save_solution(&solution)?;
    println!("Final route length: {:.6}", best_len);
    Ok(())
}
