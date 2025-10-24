// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::travelling_salesman::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.node_positions.len();
    if n == 0 {
        return Err(anyhow!("Empty challenge"));
    }

    // Initialize deterministic RNG using the provided seed
    let mut rng = SmallRng::from_seed(challenge.seed);

    // Helper closure to compute total distance of a tour
    let total_distance = |route: &Vec<usize>| -> f32 {
        let mut d: f32 = 0.0;
        for i in 0..n {
            let a = route[i];
            let b = route[(i + 1) % n];
            d += challenge.distance_matrix[a][b];
        }
        d
    };

    // 1. Construct an initial tour using Nearest Neighbor heuristic
    let mut unvisited: Vec<usize> = (0..n).collect();
    let mut route: Vec<usize> = Vec::with_capacity(n);

    // Start at a random node for variability
    let mut current_index = rng.gen_range(0..n);
    route.push(current_index);
    unvisited.swap_remove(current_index);

    while !unvisited.is_empty() {
        // Find the nearest unvisited neighbor to the current node
        let mut nearest_pos = 0;
        let mut nearest_dist = f32::INFINITY;
        for (idx, &candidate) in unvisited.iter().enumerate() {
            let dist = challenge.distance_matrix[current_index][candidate];
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_pos = idx;
            }
        }
        current_index = unvisited.remove(nearest_pos);
        route.push(current_index);
    }

    // Keep track of the best tour found
    let mut best_route = route.clone();
    let mut best_dist = total_distance(&best_route);

    // 2. Improve route using 2-Opt local optimization
    {
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n - 2 {
                for j in i + 2..n {
                    if j - i == 1 {
                        continue;
                    }
                    let a = route[i];
                    let b = route[(i + 1) % n];
                    let c = route[j % n];
                    let d = route[(j + 1) % n];
                    let delta = challenge.distance_matrix[a][c] + challenge.distance_matrix[b][d]
                        - challenge.distance_matrix[a][b]
                        - challenge.distance_matrix[c][d];
                    if delta < -1e-6 {
                        route[(i + 1)..=j].reverse();
                        improved = true;
                    }
                }
            }
        }
    }

    best_route = route.clone();
    best_dist = total_distance(&best_route);

    // 3. Apply Simulated Annealing to escape local minima and refine the tour
    let mut temperature: f32 = 1000.0;
    let cooling_rate: f32 = 0.995;
    let min_temperature: f32 = 1e-3;

    while temperature > min_temperature {
        // Create a new candidate route by swapping two random positions
        let mut new_route = best_route.clone();
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        new_route.swap(i, j);
        let new_dist = total_distance(&new_route);
        let delta = new_dist - best_dist;
        // Accept new route if it's better, or with a probability based on temperature
        if delta < 0.0 || rng.gen::<f32>() < (-delta / temperature).exp() {
            best_route = new_route;
            best_dist = new_dist;
        }
        temperature *= cooling_rate;
    }

    // Save the computed solution
    let solution = Solution { route: best_route };
    save_solution(&solution)?;

    // Print final route length for debugging or analysis
    println!("Final route length: {:.3}", best_dist);

    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected