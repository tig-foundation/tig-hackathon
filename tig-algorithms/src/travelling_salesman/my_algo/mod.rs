// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
// Furthest Insertion algorithm for the Travelling Salesman Problem
use anyhow::{anyhow, Result};
use tig_challenges::travelling_salesman::*;
use std::collections::HashSet;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.difficulty.size;
    let distance_matrix = &challenge.distance_matrix;
    
    let mut best_path: Option<Vec<usize>> = None;
    let mut best_length = f32::INFINITY;
    
    // Try furthest insertion with different starting strategies
    let strategies = if n <= 20 {
        // For small instances, try all starting nodes
        (0..n).collect::<Vec<_>>()
    } else {
        // For larger instances, try a subset of strategic starting nodes
        let step = std::cmp::max(1, n / 10);
        (0..n).step_by(step).collect::<Vec<_>>()
    };
    
    for start_node in strategies {
        // Try farthest insertion heuristic
        if let Some(path) = farthest_insertion_heuristic(n, distance_matrix, start_node, best_length) {
            let length = calculate_tour_length(distance_matrix, &path);
            if length < best_length {
                best_path = Some(path);
                best_length = length;
            }
        }
    }
    
    // Apply 2-opt improvement to the best solution found
    let final_solution = if let Some(path) = best_path {
        let improved_path = two_opt_improvement(distance_matrix, path.clone());
        let final_length = calculate_tour_length(distance_matrix, &improved_path);
        if final_length < best_length {
            improved_path
        } else {
            path
        }
    } else {
        return Err(anyhow!("No valid solution found"));
    };
    
    // Create and save the best solution found
    let solution = Solution { 
        route: final_solution
    };
    save_solution(&solution)?;
    
    Ok(())
}


fn farthest_insertion_heuristic(
    n: usize, 
    distance_matrix: &[Vec<f32>], 
    start_node: usize, 
    best_length: f32
) -> Option<Vec<usize>> {
    if n < 2 {
        return Some(vec![start_node]);
    }
    
    if n == 2 {
        let other_node = if start_node == 0 { 1 } else { 0 };
        return Some(vec![start_node, other_node]);
    }
    
    let mut path = vec![start_node];
    let mut unvisited: HashSet<usize> = (0..n).collect();
    unvisited.remove(&start_node);
    let mut current_length = 0.0;
    
    // Start with the nearest node to create an initial tour
    let nearest = *unvisited
        .iter()
        .min_by(|&&a, &&b| {
            distance_matrix[start_node][a]
                .partial_cmp(&distance_matrix[start_node][b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    
    path.push(nearest);
    unvisited.remove(&nearest);
    current_length += distance_matrix[start_node][nearest];
    
    // Complete initial tour
    current_length += distance_matrix[nearest][start_node];
    path.push(start_node);
    
    // Remove the duplicate start node for insertion process
    path.pop();
    
    while !unvisited.is_empty() {
        let mut best_insertion = None;
        let mut best_cost_increase = f32::INFINITY;
        
        // Find the best node to insert
        for &node in &unvisited {
            let mut min_cost_increase = f32::INFINITY;
            let mut best_position = 0;
            
            // Find best position to insert this node
            for i in 0..path.len() {
                let prev = path[i];
                let next = path[(i + 1) % path.len()];
                let cost_increase = distance_matrix[prev][node] + distance_matrix[node][next] - distance_matrix[prev][next];
                
                if cost_increase < min_cost_increase {
                    min_cost_increase = cost_increase;
                    best_position = i + 1;
                }
            }
            
            if min_cost_increase < best_cost_increase {
                best_cost_increase = min_cost_increase;
                best_insertion = Some((node, best_position));
            }
        }
        
        if let Some((node, position)) = best_insertion {
            path.insert(position, node);
            unvisited.remove(&node);
            current_length += best_cost_increase;
            
            // Early termination
            if current_length > best_length {
                return None;
            }
        } else {
            break;
        }
    }
    
    Some(path)
}

fn two_opt_improvement(distance_matrix: &[Vec<f32>], mut path: Vec<usize>) -> Vec<usize> {
    let n = path.len();
    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = 100; // Prevent infinite loops
    
    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;
        
        for i in 1..n - 1 {
            for j in i + 1..n {
                if j - i == 1 { continue; } // Skip adjacent edges
                
                // Calculate current cost of edges to be removed
                let current_cost = distance_matrix[path[i-1]][path[i]] + distance_matrix[path[j-1]][path[j]];
                
                // Calculate new cost if we swap
                let new_cost = distance_matrix[path[i-1]][path[j-1]] + distance_matrix[path[i]][path[j]];
                
                if new_cost < current_cost {
                    // Perform the 2-opt swap
                    path[i..j].reverse();
                    improved = true;
                }
            }
        }
    }
    
    path
}

fn calculate_tour_length(distance_matrix: &[Vec<f32>], path: &[usize]) -> f32 {
    if path.len() < 2 {
        return 0.0;
    }
    
    let mut total = 0.0;
    for i in 0..path.len() - 1 {
        total += distance_matrix[path[i]][path[i + 1]];
    }
    total
}

// Important! Do not include any tests in this file, it will result in your submission being rejected

