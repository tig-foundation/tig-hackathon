// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::travelling_salesman::*;


fn greedy_from_start(distance: &Vec<Vec<f32>>, start: usize) -> Vec<usize> {
    let n = distance.len();
    let mut route = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    let mut current = start;
    route.push(current);
    visited[current] = true;

    while route.len() < n {
        // pick nearest unvisited
        let mut best_j = None;
        let mut best_d = f32::INFINITY;
        for j in 0..n {
            if !visited[j] {
                let d = distance[current][j];
                if d < best_d {
                    best_d = d;
                    best_j = Some(j);
                }
            }
        }
        let next = best_j.expect("There should be at least one unvisited node");
        route.push(next);
        visited[next] = true;
        current = next;
    }

    route
}

fn route_length(distance: &Vec<Vec<f32>>, route: &[usize]) -> f32 {
    if route.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for w in route.windows(2) {
        total += distance[w[0]][w[1]];
    }
    total + distance[*route.last().unwrap()][route[0]]
}
pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    let mut rng = SmallRng::from_seed(challenge.seed);


    let n = challenge.distance_matrix.len();
    let start = rng.gen_range(0..n);
    let route = greedy_from_start(&challenge.distance_matrix, start);

    // Save and return
    save_solution(&Solution { route })?;
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
