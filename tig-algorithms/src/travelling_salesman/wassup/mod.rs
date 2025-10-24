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
    
    // Adjust strategies for larger inputs
    let strategies = if n <= 15 {
        (0..n).collect::<Vec<_>>()
    } else {
        let step = std::cmp::max(1, n / 5); // Fewer starting points for n ≥ 25
        (0..n).step_by(step).take(5).collect::<Vec<_>>()
    };
    
    for start_node in strategies {
        if let Some(path) = farthest_insertion_heuristic(n, distance_matrix, start_node, best_length) {
            let length = calculate_tour_length(distance_matrix, &path);
            if length < best_length {
                best_path = Some(path.clone());
                best_length = length;
                let solution = Solution { route: path };
                save_solution(&solution)?;
            }
        }
    }
    
    // Apply local search improvement
    let final_solution = if let Some(path) = best_path {
        let improved_path = local_search_improvement(distance_matrix, path.clone());
        let final_length = calculate_tour_length(distance_matrix, &improved_path);
        if final_length < best_length {
            let solution = Solution { route: improved_path.clone() };
            save_solution(&solution)?;
            improved_path
        } else {
            path
        }
    } else {
        return Err(anyhow!("No valid solution found"));
    };
    
    let solution = Solution { route: final_solution };
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
        let length = distance_matrix[start_node][other_node] + distance_matrix[other_node][start_node];
        if length > best_length {
            return None;
        }
        return Some(vec![start_node, other_node]);
    }
    
    let mut path = vec![start_node];
    let mut unvisited: HashSet<usize> = (0..n).collect();
    unvisited.remove(&start_node);
    let mut current_length = 0.0;
    
    // Select initial node based on max average distance to reduce clustering
    let farthest = *unvisited
        .iter()
        .max_by(|&&a, &&b| {
            let avg_a = (0..n).map(|i| distance_matrix[a][i]).sum::<f32>() / n as f32;
            let avg_b = (0..n).map(|i| distance_matrix[b][i]).sum::<f32>() / n as f32;
            avg_a.partial_cmp(&avg_b).unwrap_or(std::cmp::Ordering::Equal)
        })?;
    
    path.push(farthest);
    unvisited.remove(&farthest);
    current_length += distance_matrix[start_node][farthest];
    
    while !unvisited.is_empty() {
        // Select node with maximum min distance to current tour
        let (next_node, _) = unvisited
            .iter()
            .map(|&node| {
                let min_dist = path
                    .iter()
                    .map(|&p| distance_matrix[p][node])
                    .fold(f32::INFINITY, |a, b| a.min(b));
                (node, min_dist)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        
        // Find best insertion position
        let mut best_cost_increase = f32::INFINITY;
        let mut best_position = 0;
        
        for i in 0..path.len() {
            let prev = path[i];
            let next = path[(i + 1) % path.len()];
            let cost_increase = distance_matrix[prev][next_node] + 
                               distance_matrix[next_node][next] - 
                               distance_matrix[prev][next];
            
            if cost_increase < best_cost_increase {
                best_cost_increase = cost_increase;
                best_position = i + 1;
            }
        }
        
        path.insert(best_position, next_node);
        unvisited.remove(&next_node);
        current_length += best_cost_increase;
        
        if current_length > best_length {
            return None;
        }
    }
    
    // Check final tour length including return to start
    let final_length = calculate_tour_length(distance_matrix, &path);
    if final_length > best_length {
        return None;
    }
    
    Some(path)
}

fn local_search_improvement(distance_matrix: &[Vec<f32>], mut path: Vec<usize>) -> Vec<usize> {
    let n = path.len();
    let max_iterations = if n > 50 { 30 } else { 50 }; // Adjust for large inputs
    let mut improved = true;
    
    for _ in 0..max_iterations {
        improved = false;
        let current_length = calculate_tour_length(distance_matrix, &path);
        
        // 2-opt moves
        for i in 1..n - 1 {
            for j in i + 2..n {
                let prev_i = path[i - 1];
                let curr_i = path[i];
                let prev_j = path[j - 1];
                let curr_j = path[j % n];
                
                let old_cost = distance_matrix[prev_i][curr_i] + distance_matrix[prev_j][curr_j];
                let new_cost = distance_matrix[prev_i][prev_j] + distance_matrix[curr_i][curr_j];
                
                if new_cost < old_cost - 1e-6 {
                    path[i..j].reverse();
                    improved = true;
                }
            }
        }
        
        // Selective 3-opt moves (limited to avoid excessive computation)
        if n <= 50 {
            for i in 1..n - 2 {
                for j in i + 2..n - 1 {
                    for k in j + 2..n {
                        // Try one 3-opt case: reverse two segments
                        let a = path[i - 1];
                        let b = path[i];
                        let c = path[j - 1];
                        let d = path[j];
                        let e = path[k - 1];
                        let f = path[k % n];
                        
                        let old_cost = distance_matrix[a][b] + distance_matrix[c][d] + distance_matrix[e][f];
                        let new_cost = distance_matrix[a][c] + distance_matrix[b][e] + distance_matrix[d][f];
                        
                        if new_cost < old_cost - 1e-6 {
                            let mut new_path = path.clone();
                            new_path[i..j].reverse();
                            new_path[j..k].reverse();
                            path = new_path;
                            improved = true;
                        }
                    }
                }
            }
        }
        
        if !improved || calculate_tour_length(distance_matrix, &path) >= current_length {
            break;
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
    total += distance_matrix[path[path.len() - 1]][path[0]];
    total
}