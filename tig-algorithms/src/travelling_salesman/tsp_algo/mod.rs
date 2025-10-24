use anyhow::Result;
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use tig_challenges::travelling_salesman::*;
use std::time::{Duration, Instant};

// Compute total distance of a tour
fn tour_length(tour: &[usize], challenge: &Challenge) -> f64 {
    let mut sum = 0.0_f64; // Explicitly ensure sum is f64
    let n = tour.len();
    for i in 0..n {
        // Cast the distance to f64 for consistent floating-point arithmetic
        sum += challenge.distance_matrix[tour[i]][tour[(i + 1) % n]] as f64;
    }
    sum
}

// Nearest neighbor heuristic for initial tour
fn nearest_neighbor(challenge: &Challenge) -> Vec<usize> {
    let n = challenge.distance_matrix.len();
    let mut visited = vec![false; n];
    let mut tour = vec![0];
    visited[0] = true;

    for _ in 1..n {
        let last = *tour.last().unwrap();
        let mut nearest = None;
        let mut min_dist = f64::INFINITY;

        for i in 0..n {
            if !visited[i] {
                // Cast the distance to f64 before comparison/assignment
                let d = challenge.distance_matrix[last][i] as f64;
                if d < min_dist {
                    min_dist = d;
                    nearest = Some(i);
                }
            }
        }
        let next_city = nearest.unwrap();
        visited[next_city] = true;
        tour.push(next_city);
    }

    tour
}

// 2-opt swap utility: reverses the segment between indices i and k (inclusive).
fn two_opt_swap(tour: &mut Vec<usize>, i: usize, k: usize) {
    tour[i..=k].reverse();
}

// 2-opt local search (optimized for performance using incremental delta calculation)
fn two_opt(tour: &mut Vec<usize>, challenge: &Challenge) {
    let n = tour.len();
    let mut improved = true;
    let mut _current_len = tour_length(tour, challenge); // Internal state tracking

    let dist = |a: usize, b: usize| challenge.distance_matrix[a][b] as f64;

    while improved {
        improved = false;
        'outer: for i in 0..n - 1 {
            // i is the first node of the first edge: (tour[i], tour[i_next])
            let i_next = (i + 1) % n;
            let c1 = tour[i];

            for k in i + 1..n {
                // k is the first node of the second edge: (tour[k], tour[k_next])
                let k_next = (k + 1) % n;

                // Edges involved in the swap: (c1, c2) and (c3, c4) -> (c1, c3) and (c2, c4)
                let c2 = tour[i_next];
                let c3 = tour[k];
                let c4 = tour[k_next];

                // Calculate change in length
                let old_dist = dist(c1, c2) + dist(c3, c4);
                let new_dist = dist(c1, c3) + dist(c2, c4);
                
                let delta = new_dist - old_dist;

                // Check for improvement
                if delta < -1e-9 { // Use a small epsilon for floating point comparison
                    // The segment to reverse is between c2 (index i_next) and c3 (index k)
                    two_opt_swap(tour, i_next, k);
                    _current_len += delta;
                    improved = true;
                    // Restart search from the beginning after improvement
                    break 'outer;
                }
            }
        }
    }
}

// Simulated Annealing for one tour (now running for more steps and without internal 2-opt)
fn simulated_annealing(
    tour: &mut Vec<usize>,
    challenge: &Challenge,
    rng: &mut SmallRng,
    max_steps: usize,
) {
    let n = tour.len();
    let mut temp = 5000.0; // Increased starting temperature for better exploration
    let cooling_rate = 0.9995; // Slightly slower cooling rate to maximize steps
    let dist = |a: usize, b: usize| challenge.distance_matrix[a][b] as f64;

    for _step in 0..max_steps {
        if temp < 1e-9 { break; } // Early exit if temperature is too low
        
        // Choose two random non-adjacent indices i and k for a 2-opt move
        let mut i = rng.gen_range(0..n);
        let mut k = rng.gen_range(0..n);
        
        // Ensure i and k are different and ordered (i < k)
        if i == k { continue; }
        if i > k { std::mem::swap(&mut i, &mut k); }
        
        // i is the start of the segment reversal, k is the end.
        
        // Use 2-opt for perturbation, which swaps two non-adjacent edges
        let i_prev = (i + n - 1) % n;
        let k_next = (k + 1) % n;

        let c1 = tour[i_prev]; // City before the segment start
        let c2 = tour[i];      // Segment start city
        let c3 = tour[k];      // Segment end city
        let c4 = tour[k_next]; // City after the segment end

        // Old edges: (c1, c2) and (c3, c4)
        // New edges: (c1, c3) and (c2, c4)
        
        let old_dist = dist(c1, c2) + dist(c3, c4);
        let new_dist = dist(c1, c3) + dist(c2, c4);
        let delta = new_dist - old_dist;
        
        // Acceptance criteria (Metropolis criterion)
        if delta < 0.0 || rng.gen::<f64>() < (-delta / temp).exp() {
            two_opt_swap(tour, i, k); // Apply the swap
        }

        // Apply cooling
        temp *= cooling_rate;
    }
}

// Crossover: Order Crossover (OX)
fn crossover(parent1: &[usize], parent2: &[usize], rng: &mut SmallRng) -> Vec<usize> {
    let n = parent1.len();
    let mut child = vec![usize::MAX; n];

    let (i, k) = {
        let mut i = rng.gen_range(0..n);
        let mut k = rng.gen_range(0..n);
        if i > k { std::mem::swap(&mut i, &mut k); }
        (i, k)
    };

    // Copy slice from parent1
    for j in i..=k {
        child[j] = parent1[j];
    }

    // Fill remaining positions from parent2
    let mut idx = (k + 1) % n;
    let start_idx_parent2 = (k + 1) % n;

    // Use a proper way to iterate through parent2 cyclically starting from (k+1)%n
    let mut parent2_cyclical_iterator = parent2.iter().cycle().skip(start_idx_parent2);

    let mut count = 0;
    while count < n - (k - i + 1) {
        let &city = parent2_cyclical_iterator.next().unwrap();
        
        // Check if city is already present in the copied segment
        let mut is_present = false;
        // Optimized check: since we only copied parent1[i..=k], we only check that segment in child
        for j in i..=k {
            if child[j] == city {
                is_present = true;
                break;
            }
        }
        
        if !is_present {
            // Place the city in the child tour
            child[idx] = city;
            idx = (idx + 1) % n;
            count += 1;
            
            // If we have wrapped back to the start of the copied segment (index i), we stop
            if idx == i { break; } 
        }
    }

    child
}

// Mutation: swap two cities
fn mutate(tour: &mut Vec<usize>, rng: &mut SmallRng, rate: f64) {
    if rng.gen::<f64>() < rate {
        let n = tour.len();
        let i = rng.gen_range(0..n);
        let j = rng.gen_range(0..n);
        tour.swap(i, j);
    }
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let n = challenge.distance_matrix.len();
    let pop_size = 150; // Increased population size for more diversity and more generations
    let mut population: Vec<Vec<usize>> = Vec::new();

    // 1. Initialize population: 1 nearest neighbor + random permutations
    let nn = nearest_neighbor(challenge);
    population.push(nn.clone());
    
    for _ in 1..pop_size {
        let mut tour: Vec<usize> = (0..n).collect();
        tour.shuffle(&mut rng);
        population.push(tour);
    }

    let total_time = Duration::from_secs(19); // Time limit
    let start_time = Instant::now();
    let mut best = population[0].clone();
    let mut best_len = tour_length(&best, challenge);
    
    // Save initial best solution
    save_solution(&Solution { route: best.clone() })?;

    let sa_steps = 400; // Adjusted SA steps per tour per generation
    let mutation_rate = 0.05; // Slightly increased mutation rate for more exploration

    while start_time.elapsed() < total_time {
        /* // We are skipping the full 2-opt on the entire population here to save time for more generations.
        // The local search (intensification) is now focused on new children and the SA phase.
        for tour in &mut population {
            if start_time.elapsed() >= total_time { break; }
            two_opt(tour, challenge); 
        }
        
        if start_time.elapsed() >= total_time { break; }
        */

        // Apply SA to population (Diversification/Perturbation)
        for tour in &mut population {
             if start_time.elapsed() >= total_time { break; }
             simulated_annealing(tour, challenge, &mut rng, sa_steps);
        }

        // Evaluate and save best
        let mut current_best_in_pop = best_len;
        let mut current_best_tour = None;

        for tour in &population {
            let len = tour_length(tour, challenge);
            if len < current_best_in_pop {
                current_best_in_pop = len;
                current_best_tour = Some(tour.clone());
            }
        }
        
        if let Some(new_best_tour) = current_best_tour {
            best_len = current_best_in_pop;
            best = new_best_tour;
            save_solution(&Solution { route: best.clone() })?;
        }

        // Selection: keep top half (Elitism)
        population.sort_by(|a, b| tour_length(a, challenge).partial_cmp(&tour_length(b, challenge)).unwrap());
        
        let elite_count = pop_size / 2;
        let elite_count = elite_count.max(1); 
        // Keep the best half of the population
        population.truncate(elite_count);
        
        if start_time.elapsed() >= total_time { break; }

        // Generate offspring with crossover + mutation (Diversification)
        while population.len() < pop_size {
            if start_time.elapsed() >= total_time { break; }
            
            // Choose parents randomly from the elite population
            let p1 = population.choose(&mut rng).unwrap();
            let p2 = population.choose(&mut rng).unwrap();
            
            let mut child = crossover(p1, p2, &mut rng);
            mutate(&mut child, &mut rng, mutation_rate);
            
            // Apply 2-opt immediately to the child before adding (Intensification)
            two_opt(&mut child, challenge); 
            
            population.push(child);
        }
    }

    Ok(())
}
