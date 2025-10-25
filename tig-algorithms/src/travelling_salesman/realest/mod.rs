use anyhow::{anyhow, Result};
use std::collections::HashSet;
use tig_challenges::travelling_salesman::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.difficulty.size;
    let dist = &challenge.distance_matrix;

    if n == 0 {
        return Err(anyhow!("Empty problem"));
    }
    if n == 1 {
        return save_solution(&Solution { route: vec![0] });
    }
    if n == 2 {
        return save_solution(&Solution { route: vec![0, 1] });
    }

    let mut best_route = None;
    let mut best_distance = f32::INFINITY;

    // Smart starting points
    let key_starts = [0, n / 4, n / 2, 3 * n / 4, n - 1];
    
    for &start in &key_starts {
        let route = nearest_neighbor(n, dist, start);
        let d = tour_distance(dist, &route);
        if d < best_distance {
            best_distance = d;
            best_route = Some(route);
        }
    }

    for i in 1..std::cmp::min(10, n / 5) {
        let start = (i * 13) % n;
        let route = nearest_neighbor(n, dist, start);
        let d = tour_distance(dist, &route);
        if d < best_distance {
            best_distance = d;
            best_route = Some(route);
        }
    }

    // CRITICAL: Improved Christofides with better matching
    let christofides_route = christofides_tour(n, dist);
    let christofides_dist = tour_distance(dist, &christofides_route);
    if christofides_dist < best_distance {
        best_distance = christofides_dist;
        best_route = Some(christofides_route);
    }

    let mut route = best_route.unwrap();

    // Optimized pipeline
    route = two_opt_fast(dist, route, n, 60); // Slightly increased from 50
    route = or_opt_single_pass(dist, route, n);
    route = two_opt_fast(dist, route, n, 60);
    route = three_opt_targeted(dist, route, n);
    route = two_opt_fast(dist, route, n, 40); // Final polish

    if !is_valid_route(&route, n) {
        return Err(anyhow!("Generated invalid route"));
    }

    save_solution(&Solution { route })?;
    Ok(())
}

fn nearest_neighbor(n: usize, dist: &[Vec<f32>], start: usize) -> Vec<usize> {
    let mut route = Vec::with_capacity(n);
    let mut unvisited: HashSet<usize> = (0..n).collect();
    
    let mut current = start;
    route.push(current);
    unvisited.remove(&current);

    while !unvisited.is_empty() {
        let mut nearest = None;
        let mut min_dist = f32::INFINITY;

        for &node in &unvisited {
            if dist[current][node] < min_dist {
                min_dist = dist[current][node];
                nearest = Some(node);
            }
        }

        if let Some(next) = nearest {
            route.push(next);
            unvisited.remove(&next);
            current = next;
        }
    }

    route
}

fn christofides_tour(n: usize, dist: &[Vec<f32>]) -> Vec<usize> {
    let mst = prim_mst(n, dist);
    let odd_nodes = find_odd_degree_nodes(n, &mst);
    
    // IMPROVED: Better greedy matching strategy
    let matching = improved_greedy_matching(&odd_nodes, dist);
    
    let multigraph = combine_graphs(n, &mst, &matching);
    let eulerian_tour = find_eulerian_tour(&multigraph, 0);
    shortcut_tour(&eulerian_tour)
}

fn prim_mst(n: usize, dist: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let mut in_mst = vec![false; n];
    let mut min_edge = vec![f32::INFINITY; n];
    let mut parent = vec![None; n];
    min_edge[0] = 0.0;

    for _ in 0..n {
        let mut u = None;
        let mut best = f32::INFINITY;
        for v in 0..n {
            if !in_mst[v] && min_edge[v] < best {
                best = min_edge[v];
                u = Some(v);
            }
        }
        let u = u.unwrap();
        in_mst[u] = true;
        for v in 0..n {
            if !in_mst[v] && dist[u][v] < min_edge[v] {
                min_edge[v] = dist[u][v];
                parent[v] = Some(u);
            }
        }
    }

    let mut edges = Vec::new();
    for v in 1..n {
        if let Some(u) = parent[v] {
            edges.push((u, v));
        }
    }
    edges
}

fn find_odd_degree_nodes(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut degree = vec![0; n];
    for &(u, v) in edges {
        degree[u] += 1;
        degree[v] += 1;
    }
    degree.iter().enumerate().filter_map(|(i, &d)| if d % 2 == 1 { Some(i) } else { None }).collect()
}

/// IMPROVED: Better greedy matching - find globally best pairs iteratively
fn improved_greedy_matching(nodes: &[usize], dist: &[Vec<f32>]) -> Vec<(usize, usize)> {
    if nodes.is_empty() {
        return Vec::new();
    }

    let mut unmatched: Vec<usize> = nodes.to_vec();
    let mut matches = Vec::new();

    // Key improvement: Always pick the globally best pair among remaining nodes
    while unmatched.len() > 1 {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_cost = dist[unmatched[0]][unmatched[1]];

        // Find the best pair globally
        for i in 0..unmatched.len() {
            for j in i + 1..unmatched.len() {
                let cost = dist[unmatched[i]][unmatched[j]];
                if cost < best_cost {
                    best_cost = cost;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Match this best pair
        let u = unmatched[best_i];
        let v = unmatched[best_j];
        matches.push((u, v));

        // Remove matched nodes (remove larger index first to maintain indices)
        if best_j > best_i {
            unmatched.remove(best_j);
            unmatched.remove(best_i);
        } else {
            unmatched.remove(best_i);
            unmatched.remove(best_j);
        }
    }

    matches
}

fn combine_graphs(n: usize, mst: &[(usize, usize)], matching: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut graph = vec![Vec::new(); n];
    for &(u, v) in mst.iter().chain(matching.iter()) {
        graph[u].push(v);
        graph[v].push(u);
    }
    graph
}

fn find_eulerian_tour(graph: &[Vec<usize>], start: usize) -> Vec<usize> {
    let mut adj = graph.to_vec();
    let mut stack = vec![start];
    let mut circuit = Vec::new();

    while let Some(&u) = stack.last() {
        if adj[u].is_empty() {
            circuit.push(u);
            stack.pop();
        } else {
            let v = adj[u].pop().unwrap();
            if let Some(pos) = adj[v].iter().position(|&x| x == u) {
                adj[v].remove(pos);
            }
            stack.push(v);
        }
    }
    circuit.reverse();
    circuit
}

fn shortcut_tour(tour: &[usize]) -> Vec<usize> {
    let mut visited = HashSet::new();
    let mut path = Vec::new();
    for &node in tour {
        if visited.insert(node) {
            path.push(node);
        }
    }
    path
}

fn two_opt_fast(dist: &[Vec<f32>], mut route: Vec<usize>, n: usize, max_passes: usize) -> Vec<usize> {
    if n < 4 {
        return route;
    }

    for _ in 0..max_passes {
        let mut improved = false;

        for i in 0..n - 1 {
            for j in i + 2..n {
                if j == i + 1 {
                    continue;
                }

                let a = route[i];
                let b = route[i + 1];
                let c = route[j];
                let d = if j + 1 < n { route[j + 1] } else { route[0] };

                let current = dist[a][b] + dist[c][d];
                let swapped = dist[a][c] + dist[b][d];

                if swapped < current - 1e-6 {
                    route[i + 1..=j].reverse();
                    improved = true;
                    break;
                }
            }
            if improved {
                break;
            }
        }

        if !improved {
            break;
        }
    }

    route
}

fn or_opt_single_pass(dist: &[Vec<f32>], mut route: Vec<usize>, n: usize) -> Vec<usize> {
    if n < 5 {
        return route;
    }

    for len in 1..=2 {
        for i in 0..n - len {
            let mut best_j = None;
            let mut best_gain = 0.0;

            for j in 0..n {
                if j >= i && j <= i + len {
                    continue;
                }

                let gain = calculate_or_opt_gain(dist, &route, n, i, len, j);
                if gain < best_gain {
                    best_gain = gain;
                    best_j = Some(j);
                }
            }

            if let Some(j) = best_j {
                if best_gain < -1e-6 {
                    route = apply_or_opt_move(&route, i, len, j);
                }
            }
        }
    }

    route
}

fn calculate_or_opt_gain(
    dist: &[Vec<f32>],
    route: &[usize],
    n: usize,
    start: usize,
    len: usize,
    insert_pos: usize,
) -> f32 {
    let prev_start = if start == 0 { n - 1 } else { start - 1 };
    let after_end = (start + len) % n;
    
    let seg_first = route[start];
    let seg_last = route[start + len - 1];
    
    let remove_cost = -(dist[route[prev_start]][seg_first]
        + dist[seg_last][route[after_end]]
        - dist[route[prev_start]][route[after_end]]);
    
    let insert_before = if insert_pos == 0 { n - 1 } else { insert_pos - 1 };
    let insert_cost = dist[route[insert_before]][seg_first]
        + dist[seg_last][route[insert_pos]]
        - dist[route[insert_before]][route[insert_pos]];
    
    remove_cost + insert_cost
}

fn apply_or_opt_move(route: &[usize], start: usize, len: usize, insert_pos: usize) -> Vec<usize> {
    let mut new_route = Vec::new();
    let segment: Vec<usize> = route[start..start + len].to_vec();
    
    if insert_pos < start {
        new_route.extend(&route[..insert_pos]);
        new_route.extend(&segment);
        new_route.extend(&route[insert_pos..start]);
        new_route.extend(&route[start + len..]);
    } else {
        new_route.extend(&route[..start]);
        new_route.extend(&route[start + len..insert_pos]);
        new_route.extend(&segment);
        new_route.extend(&route[insert_pos..]);
    }
    
    new_route
}

fn three_opt_targeted(dist: &[Vec<f32>], mut route: Vec<usize>, n: usize) -> Vec<usize> {
    if n < 6 {
        return route;
    }

    for _ in 0..5 {
        let mut improved = false;
        let step = std::cmp::max(1, n / 20);
        
        for i in (0..n.saturating_sub(5)).step_by(step) {
            for j in (i + 2..n.saturating_sub(3)).step_by(step) {
                for k in (j + 2..n.saturating_sub(1)).step_by(step) {
                    let a = route[i];
                    let b = route[i + 1];
                    let c = route[j];
                    let d = route[j + 1];
                    let e = route[k];
                    let f = route[k + 1];

                    let current = dist[a][b] + dist[c][d] + dist[e][f];

                    let cost1 = dist[a][c] + dist[b][d] + dist[e][f];
                    if cost1 < current - 1e-6 {
                        route[i + 1..=j].reverse();
                        improved = true;
                        break;
                    }

                    let cost2 = dist[a][b] + dist[c][e] + dist[d][f];
                    if cost2 < current - 1e-6 {
                        route[j + 1..=k].reverse();
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
            if improved {
                break;
            }
        }

        if !improved {
            break;
        }
    }

    route
}

fn is_valid_route(route: &[usize], n: usize) -> bool {
    if route.len() != n {
        return false;
    }
    let mut seen = vec![false; n];
    for &node in route {
        if node >= n || seen[node] {
            return false;
        }
        seen[node] = true;
    }
    true
}

#[inline]
fn tour_distance(dist: &[Vec<f32>], route: &[usize]) -> f32 {
    let n = route.len();
    let mut total = 0.0;
    for i in 0..n {
        total += dist[route[i]][route[(i + 1) % n]];
    }
    total
}
