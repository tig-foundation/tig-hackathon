// Hyper-optimized TSP for [50,200]: Fast multi-start + 3-opt + aggressive local search
use anyhow::{anyhow, Result};
use tig_challenges::travelling_salesman::*;
use std::cmp::Ordering;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let n = challenge.difficulty.size;
    let dm = &challenge.distance_matrix;
    if n == 0 { return Err(anyhow!("Empty instance")); }
    if n == 1 { return save_solution(&Solution { route: vec![0] }); }

    // Large k-NN for quality
    let k_cand = 40.min(n - 1);
    let knn = build_knn(dm, k_cand);

    let mut best_tour: Option<Vec<usize>> = None;
    let mut best_len = f32::INFINITY;

    // Strategy 1: Christofides (strongest baseline)
    if let Some(tour) = christofides_build(n, dm) {
        let len = cycle_length(dm, &tour);
        if len < best_len {
            best_len = len;
            best_tour = Some(tour);
        }
    }

    // Strategy 2: Aggressive Farthest Insertion from many starts
    let fi_count = if n <= 50 { 40 } else { 30 };
    let step = (n / fi_count).max(1);
    for start in (0..n).step_by(step) {
        if let Some((tour, len)) = farthest_insertion_with_prune(n, dm, start, best_len) {
            if len < best_len {
                best_len = len;
                best_tour = Some(tour);
            }
        }
    }

    let mut tour = best_tour.ok_or_else(|| anyhow!("No tour found"))?;
    let mut len = best_len;

    // Build position array once
    let mut pos = vec![0usize; n];
    for (i, &v) in tour.iter().enumerate() { pos[v] = i; }

    // Ultra-aggressive optimization sequence
    // Round 1: Heavy 2-opt + 3-opt
    (tour, len, pos) = two_opt_fast(&knn, dm, tour, len, pos, 20 * n);
    (tour, len, pos) = three_opt_fast(&knn, dm, tour, len, pos, 4 * n);
    
    // Round 2: Or-opt variants
    (tour, len, pos) = or_opt_1_fast(&knn, dm, tour, len, pos, 4 * n);
    (tour, len, pos) = or_opt_2_fast(&knn, dm, tour, len, pos, 2 * n);
    
    // Round 3: More 2-opt + 3-opt
    (tour, len, pos) = two_opt_fast(&knn, dm, tour, len, pos, 12 * n);
    (tour, len, pos) = three_opt_fast(&knn, dm, tour, len, pos, 2 * n);
    
    // Round 4: Final Or-opt polish
    (tour, len, pos) = or_opt_1_fast(&knn, dm, tour, len, pos, 2 * n);
    
    // Round 5: Last 2-opt
    (tour, len, pos) = two_opt_fast(&knn, dm, tour, len, pos, 6 * n);

    // Ensure valid route starting at 0
    let mut route = tour;
    if let Some(p0) = route.iter().position(|&v| v == 0) {
        if p0 != 0 { route = rotate_to_start(route, p0); }
    } else {
        route = (0..n).collect();
    }
    if !validate_route(n, &route) { route = (0..n).collect(); }

    save_solution(&Solution { route })?;
    Ok(())
}

/* ============== Christofides ============== */

fn christofides_build(n: usize, dm: &[Vec<f32>]) -> Option<Vec<usize>> {
    if n == 1 { return Some(vec![0]); }
    if n == 2 { return Some(vec![0, 1]); }

    let (parent, mut deg) = prim_mst(n, dm)?;
    let odd: Vec<usize> = (0..n).filter(|&v| deg[v] % 2 == 1).collect();
    let matching = greedy_matching(&odd, dm);
    for (u, v) in &matching { deg[*u] += 1; deg[*v] += 1; }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for v in 1..n { adj[parent[v]].push(v); adj[v].push(parent[v]); }
    for (u, v) in matching { adj[u].push(v); adj[v].push(u); }

    let eul = eulerian_circuit(&adj, 0)?;
    let mut seen = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    for &v in &eul {
        if !seen[v] { seen[v] = true; tour.push(v); if tour.len() == n { break; } }
    }
    Some(tour)
}

fn prim_mst(n: usize, dm: &[Vec<f32>]) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut in_mst = vec![false; n];
    let mut key = vec![f32::INFINITY; n];
    let mut parent = vec![0usize; n];
    key[0] = 0.0;

    for _ in 0..n {
        let mut u = n;
        let mut best = f32::INFINITY;
        for v in 0..n {
            if !in_mst[v] && key[v] < best { best = key[v]; u = v; }
        }
        if u == n { return None; }
        in_mst[u] = true;
        for v in 0..n {
            if !in_mst[v] && dm[u][v] < key[v] { key[v] = dm[u][v]; parent[v] = u; }
        }
    }
    let mut deg = vec![0usize; n];
    for v in 1..n { deg[parent[v]] += 1; deg[v] += 1; }
    Some((parent, deg))
}

fn greedy_matching(odd: &[usize], dm: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n = dm.len();
    let mut used = vec![false; n];
    let mut rem: Vec<usize> = odd.to_vec();
    let mut edges = Vec::new();
    while rem.len() > 1 {
        let u = rem.pop().unwrap();
        if used[u] { continue; }
        let mut best = (n, f32::INFINITY);
        for &v in &rem {
            if !used[v] && dm[u][v] < best.1 { best = (v, dm[u][v]); }
        }
        let v = best.0;
        if v < n {
            used[u] = true; used[v] = true;
            rem.retain(|&x| x != v);
            edges.push((u, v));
        }
    }
    edges
}

fn eulerian_circuit(adj: &[Vec<usize>], start: usize) -> Option<Vec<usize>> {
    let mut g = adj.to_vec();
    let mut st = Vec::new();
    let mut cyc = Vec::new();
    let mut cur = start;
    loop {
        if let Some(&v) = g[cur].last() {
            g[cur].pop();
            if let Some(i) = g[v].iter().position(|&x| x == cur) { g[v].swap_remove(i); }
            st.push(cur);
            cur = v;
        } else {
            cyc.push(cur);
            if let Some(prev) = st.pop() { cur = prev; } else { break; }
        }
    }
    cyc.reverse();
    Some(cyc)
}

/* ============== Farthest Insertion ============== */

fn farthest_insertion_with_prune(n: usize, dm: &[Vec<f32>], start: usize, thresh: f32) -> Option<(Vec<usize>, f32)> {
    if n == 1 { return Some((vec![start], 0.0)); }
    if n == 2 {
        let other = 1 - start;
        return Some((vec![start, other], dm[start][other] + dm[other][start]));
    }

    let a = start;
    let b = (0..n).filter(|&k| k != a).max_by(|&x, &y| fcmp(dm[a][x], dm[a][y]))?;
    let c = (0..n).filter(|&k| k != a && k != b).max_by(|&x, &y| fcmp(dm[a][x] + dm[b][x], dm[a][y] + dm[b][y]))?;

    let mut tour = vec![a, b, c];
    let mut in_tour = vec![false; n];
    for &v in &tour { in_tour[v] = true; }
    let mut length = cycle_length(dm, &tour);
    
    let mut dist = vec![f32::INFINITY; n];
    for v in 0..n {
        if !in_tour[v] {
            for &t in &tour {
                let d = dm[v][t].min(dm[t][v]);
                if d < dist[v] { dist[v] = d; }
            }
        }
    }

    for _ in 0..(n - 3) {
        let u = (0..n).filter(|&v| !in_tour[v]).max_by(|&x, &y| fcmp(dist[x], dist[y]))?;
        
        let mut best_pos = 0;
        let mut best_delta = f32::INFINITY;
        for i in 0..tour.len() {
            let j = (i + 1) % tour.len();
            let delta = dm[tour[i]][u] + dm[u][tour[j]] - dm[tour[i]][tour[j]];
            if delta < best_delta { best_delta = delta; best_pos = j; }
        }
        
        tour.insert(best_pos, u);
        in_tour[u] = true;
        length += best_delta;
        if length > thresh { return None; }

        for v in 0..n {
            if !in_tour[v] {
                let d = dm[v][u].min(dm[u][v]);
                if d < dist[v] { dist[v] = d; }
            }
        }
    }
    Some((tour, length))
}

/* ============== Local Search ============== */

fn build_knn(dm: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    let n = dm.len();
    (0..n).map(|i| {
        let mut idx: Vec<usize> = (0..n).filter(|&v| v != i).collect();
        idx.sort_by(|&a, &b| fcmp(dm[i][a], dm[i][b]));
        idx.truncate(k);
        idx
    }).collect()
}

fn two_opt_fast(knn: &[Vec<usize>], dm: &[Vec<f32>], mut tour: Vec<usize>, 
                mut len: f32, mut pos: Vec<usize>, max_pass: usize) -> (Vec<usize>, f32, Vec<usize>) {
    let n = tour.len();
    if n < 4 { return (tour, len, pos); }

    let mut pass = 0;
    let mut imp = true;
    while imp && pass < max_pass {
        imp = false;
        pass += 1;
        
        'outer: for i in 0..n {
            let a = tour[i];
            let b = tour[(i + 1) % n];
            
            for &c in &knn[a] {
                let j = pos[c];
                if j <= i || j == (i + 1) % n || (i == 0 && j == n - 1) { continue; }
                
                let d = tour[(j + 1) % n];
                let delta = dm[a][c] + dm[b][d] - dm[a][b] - dm[c][d];
                if delta < -1e-6 {
                    reverse_and_update(&mut tour, &mut pos, (i + 1) % n, j);
                    len += delta;
                    imp = true;
                    break 'outer;
                }
            }
        }
    }
    (tour, len, pos)
}

fn three_opt_fast(knn: &[Vec<usize>], dm: &[Vec<f32>], mut tour: Vec<usize>,
                  mut len: f32, mut pos: Vec<usize>, max_pass: usize) -> (Vec<usize>, f32, Vec<usize>) {
    let n = tour.len();
    if n < 6 { return (tour, len, pos); }

    let mut pass = 0;
    let mut imp = true;
    while imp && pass < max_pass {
        imp = false;
        pass += 1;

        'outer: for i in 0..n {
            let a = tour[i];
            let b = tour[(i + 1) % n];
            
            for &c in knn[a].iter().take(15) {
                let j = pos[c];
                if j <= i + 1 || j >= i + n - 2 { continue; }
                let d = tour[(j + 1) % n];
                
                for &e in knn[d].iter().take(10) {
                    let k = pos[e];
                    if k <= j + 1 || k >= i + n - 1 { continue; }
                    let f = tour[(k + 1) % n];
                    
                    // Try reconnection: a-c, b-e, d-f
                    let old_cost = dm[a][b] + dm[c][d] + dm[e][f];
                    let new_cost = dm[a][c] + dm[b][e] + dm[d][f];
                    let delta = new_cost - old_cost;
                    
                    if delta < -1e-6 {
                        // Apply 3-opt move (reverse segments)
                        reverse_and_update(&mut tour, &mut pos, (i + 1) % n, j);
                        reverse_and_update(&mut tour, &mut pos, (j + 1) % n, k);
                        len += delta;
                        imp = true;
                        break 'outer;
                    }
                }
            }
        }
    }
    (tour, len, pos)
}

fn or_opt_1_fast(knn: &[Vec<usize>], dm: &[Vec<f32>], mut tour: Vec<usize>,
                 mut len: f32, mut pos: Vec<usize>, max_pass: usize) -> (Vec<usize>, f32, Vec<usize>) {
    let n = tour.len();
    if n < 4 { return (tour, len, pos); }

    let mut pass = 0;
    let mut imp = true;
    while imp && pass < max_pass {
        imp = false;
        pass += 1;

        for i in 0..n {
            let v = tour[i];
            let prev = tour[(i + n - 1) % n];
            let next = tour[(i + 1) % n];
            let rem_cost = -dm[prev][v] - dm[v][next] + dm[prev][next];

            let mut best_gain = 0.0;
            let mut best_after = None;

            for &a in &knn[v] {
                let p = pos[a];
                let b = tour[(p + 1) % n];
                if a == v || b == v || a == prev || b == next { continue; }

                let add_cost = -dm[a][b] + dm[a][v] + dm[v][b];
                let delta = rem_cost + add_cost;
                if delta < best_gain { best_gain = delta; best_after = Some(p); }
            }

            if let Some(p) = best_after {
                remove_insert(&mut tour, &mut pos, i, p);
                len += best_gain;
                imp = true;
                break;
            }
        }
    }
    (tour, len, pos)
}

fn or_opt_2_fast(knn: &[Vec<usize>], dm: &[Vec<f32>], mut tour: Vec<usize>,
                 mut len: f32, mut pos: Vec<usize>, max_pass: usize) -> (Vec<usize>, f32, Vec<usize>) {
    let n = tour.len();
    if n < 5 { return (tour, len, pos); }

    let mut pass = 0;
    let mut imp = true;
    while imp && pass < max_pass {
        imp = false;
        pass += 1;

        for i in 0..n {
            let v1 = tour[i];
            let v2 = tour[(i + 1) % n];
            let prev = tour[(i + n - 1) % n];
            let next = tour[(i + 2) % n];

            let rem_cost = -dm[prev][v1] - dm[v1][v2] - dm[v2][next] + dm[prev][next];

            let mut best_gain = 0.0;
            let mut best_after = None;

            for &a in knn[v1].iter().chain(knn[v2].iter()) {
                let p = pos[a];
                let b = tour[(p + 1) % n];
                if a == v1 || a == v2 || b == v1 || b == v2 || a == prev || b == next { continue; }

                let add_cost = -dm[a][b] + dm[a][v1] + dm[v1][v2] + dm[v2][b];
                let delta = rem_cost + add_cost;
                if delta < best_gain { best_gain = delta; best_after = Some(p); }
            }

            if let Some(p) = best_after {
                remove_seg_insert(&mut tour, &mut pos, i, 2, p);
                len += best_gain;
                imp = true;
                break;
            }
        }
    }
    (tour, len, pos)
}

/* ============== Utilities ============== */

fn reverse_and_update(t: &mut [usize], pos: &mut [usize], l: usize, r: usize) {
    let n = t.len();
    let mut i = l % n;
    let mut j = r % n;
    let steps = ((r + n - l) % n + 1) / 2;
    for _ in 0..steps {
        t.swap(i, j);
        pos[t[i]] = i;
        pos[t[j]] = j;
        i = (i + 1) % n;
        j = (j + n - 1) % n;
    }
}

fn remove_insert(t: &mut Vec<usize>, pos: &mut [usize], idx: usize, after: usize) {
    let n = t.len();
    if idx == (after + 1) % n { return; }
    let v = t.remove(idx);
    for i in idx..t.len() { pos[t[i]] = i; }
    let ins = if after < idx { after + 1 } else { after }.min(t.len());
    t.insert(ins, v);
    for i in ins..t.len() { pos[t[i]] = i; }
}

fn remove_seg_insert(t: &mut Vec<usize>, pos: &mut [usize], start: usize, len: usize, after: usize) {
    if len == 0 { return; }
    let mut seg = Vec::with_capacity(len);
    for _ in 0..len { seg.push(t.remove(start)); }
    for i in start..t.len() { pos[t[i]] = i; }
    let ins = (if after < start { after + 1 } else { after - len + 1 }).min(t.len());
    for (k, &v) in seg.iter().enumerate() { t.insert(ins + k, v); }
    for i in ins..t.len() { pos[t[i]] = i; }
}

fn rotate_to_start(v: Vec<usize>, p: usize) -> Vec<usize> {
    if p == 0 { return v; }
    [&v[p..], &v[..p]].concat()
}

fn cycle_length(dm: &[Vec<f32>], path: &[usize]) -> f32 {
    let n = path.len();
    if n <= 1 { return 0.0; }
    (0..n).map(|i| dm[path[i]][path[(i + 1) % n]]).sum()
}

fn validate_route(n: usize, route: &[usize]) -> bool {
    if route.len() != n { return false; }
    let mut seen = vec![false; n];
    for &v in route {
        if v >= n || seen[v] { return false; }
        seen[v] = true;
    }
    true
}

#[inline]
fn fcmp(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(if a.is_nan() { Ordering::Less } else { Ordering::Greater })
}