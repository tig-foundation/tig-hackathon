use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::min_superstring::*;
use std::time::{Duration, Instant};
use std::collections::HashSet;

// --- Data Structures and Helpers ---

// State representation for our Simultaneous Simulated Annealing
#[derive(Clone)]
struct State {
    /// The order of string indices, e.g., [2, 0, 1].
    order: Vec<usize>,
    /// `perm_indices[i]` = the index of the permutation to use for string `i`.
    perm_indices: Vec<usize>,
}

// Global precomputed data structure for fast lookups
struct OverlapCache {
    /// all_perms[i][j] is the j-th unique anagram of challenge.strings[i]
    all_perms: Vec<Vec<String>>,
    /// overlap_matrix[i][p1_idx][j][p2_idx] = overlap between all_perms[i][p1_idx] and all_perms[j][p2_idx]
    overlap_matrix: Vec<Vec<Vec<Vec<usize>>>>,
}

/// Calculates the max overlap of a suffix of s1 matching a prefix of s2.
/// Uses direct string slicing for robustness and speed.
fn calculate_overlap_raw(s1: &str, s2: &str) -> usize {
    let len = s1.len(); // Should be 5

    // Iterate k from 4 (max possible overlap) down to 1.
    for k in (1..len).rev() { 
        // s1 suffix of length k starts at index (len - k)
        let s1_suffix = &s1[len - k..];
        // s2 prefix of length k is &s2[..k]
        let s2_prefix = &s2[..k];

        // The verifier expects the overlap used here to be EXACTLY correct.
        if s1_suffix == s2_prefix {
            return k;
        }
    }
    0
}

/// Helper (Heap's Algorithm) to find all unique permutations
fn heap_permutation(chars: &mut Vec<char>, k: usize, results: &mut HashSet<String>) {
    if k == 1 {
        // Base case: store the current permutation
        results.insert(chars.iter().collect());
    } else {
        // Recurse without swapping for the first k-1 elements
        heap_permutation(chars, k - 1, results);

        // Swapping the (k-1)-th element with previous elements
        for i in 0..(k - 1) {
            // Swap based on parity of k
            if k % 2 == 0 {
                chars.swap(i, k - 1);
            } else {
                chars.swap(0, k - 1);
            }
            // Recurse again after the swap
            heap_permutation(chars, k - 1, results);
        }
    }
}

/// Generates all unique permutations for a single string
fn get_all_perms(s: &str) -> Vec<String> {
    let mut chars: Vec<char> = s.chars().collect();
    let mut results = HashSet::new();
    
    // Pass the length as a simple value to avoid the Rust E0502 borrowing error
    let len = chars.len(); 
    heap_permutation(&mut chars, len, &mut results);
    
    results.into_iter().collect()
}


// Precomputes all necessary data for fast lookup
fn build_overlap_cache(challenge: &Challenge) -> OverlapCache {
    let n = challenge.strings.len();
    
    // 1. Generate all unique permutations for all strings
    let all_perms: Vec<Vec<String>> = challenge
        .strings
        .iter()
        .map(|s| get_all_perms(s))
        .collect();

    // 2. Precompute all overlaps
    let mut overlap_matrix = vec![];
    for i in 0..n { // String 1 index
        let num_p1 = all_perms[i].len();
        let mut row_i = vec![];
        
        // Initialize the cache for String i (We use s1_idx as the outer index)
        for _ in 0..num_p1 { 
            // We use s2_idx as the inner index (n) and then p2_idx (num_p2)
            // This is a placeholder initialization; the size is fixed dynamically below.
            row_i.push(vec![vec![0; 1]; n]); 
        }

        for p1_idx in 0..num_p1 { // Permutation 1 index
            let s1 = &all_perms[i][p1_idx];
            for j in 0..n { // String 2 index
                let num_p2 = all_perms[j].len();
                let mut col_j = vec![0; num_p2];
                for p2_idx in 0..num_p2 { // Permutation 2 index
                    let s2 = &all_perms[j][p2_idx];
                    col_j[p2_idx] = calculate_overlap_raw(s1, s2);
                }
                row_i[p1_idx][j] = col_j;
            }
        }
        overlap_matrix.push(row_i);
    }
    
    OverlapCache { all_perms, overlap_matrix }
}

/// Fast calculation of the total superstring length using the cache
fn calculate_total_length_fast(
    state: &State,
    cache: &OverlapCache,
) -> usize {
    let n = state.order.len();
    if n == 0 {
        return 0;
    }

    // String length is determined from the first string. Assumed constant.
    let string_len = cache.all_perms[state.order[0]][state.perm_indices[state.order[0]]].len();
    let mut total_len = string_len;

    for i in 1..n {
        // Get the string indices from the order (s1 -> s2)
        let s1_idx = state.order[i - 1];
        let s2_idx = state.order[i];

        // Get the chosen permutation indices
        let p1_idx = state.perm_indices[s1_idx];
        let p2_idx = state.perm_indices[s2_idx];

        // Lookup the precomputed overlap (O(1))
        let overlap = cache.overlap_matrix[s1_idx][p1_idx][s2_idx][p2_idx];
        
        // Cost added = string_len - overlap
        total_len += string_len - overlap;
    }

    total_len
}


// --- Main Solver Function ---

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let start_time = Instant::now();
    // Set time limit slightly earlier than 20s to ensure we save a solution before timeout.
    let time_limit = Duration::from_secs(19); 
    let mut rng = SmallRng::from_seed(challenge.seed);

    let n = challenge.strings.len();
    if n == 0 { return Ok(()); }
    
    // 1. Precompute all data structures (Must finish quickly)
    let cache = build_overlap_cache(challenge);
    if start_time.elapsed() > time_limit { return Ok(()); }

    // 2. Initialize State
    let initial_order: Vec<usize> = (0..n).collect();
    let initial_perms: Vec<usize> = vec![0; n]; // Use the first perm (lexicographical order)

    let mut current_state = State {
        order: initial_order,
        perm_indices: initial_perms,
    };
    let mut best_state = current_state.clone();

    // 3. Initial evaluation
    let mut current_len = calculate_total_length_fast(&current_state, &cache);
    let mut best_len = current_len;

    // Helper to save the current best solution
    let mut save_best = |state: &State, len: usize| -> Result<()> {
        if len < best_len {
            best_len = len;
            best_state = state.clone();
            
            // --- CRITICAL FIX: Calculate the correct superstring_idxs (start positions) ---
            
            // The length of each string (all are 5)
            let string_len = cache.all_perms[state.order[0]][state.perm_indices[state.order[0]]].len();

            // This map holds the start index for each ORIGINAL string index (0..N-1)
            let mut start_indices_map = vec![0; n]; 

            let first_string_idx = state.order[0];
            // The first string in the optimal order always starts at 0
            start_indices_map[first_string_idx] = 0;
            
            // current_pos tracks the end position (exclusive) of the superstring so far.
            let mut current_pos = string_len; 

            for i in 1..n {
                let s1_idx = state.order[i - 1]; // Previous string (A) in the final order
                let s2_idx = state.order[i];     // Current string (B) to append

                let p1_idx = state.perm_indices[s1_idx];
                let p2_idx = state.perm_indices[s2_idx];

                // Look up the overlap (Overlap(A, B))
                let overlap = cache.overlap_matrix[s1_idx][p1_idx][s2_idx][p2_idx];
                
                // The new string (B) starts at the end of the previous superstring, minus the overlap.
                let start_of_s2 = current_pos - overlap;
                
                // Store the start index for the ORIGINAL string index (s2_idx)
                start_indices_map[s2_idx] = start_of_s2;

                // Update the end position of the superstring
                current_pos = start_of_s2 + string_len;
            }
            
            // --- End CRITICAL FIX ---

            // Construct the permuted_strings list based on the best state
            let solution_strings = (0..n)
                .map(|i| cache.all_perms[i][best_state.perm_indices[i]].clone())
                .collect();
            
            save_solution(&Solution {
                permuted_strings: solution_strings,
                superstring_idxs: start_indices_map, // Now correctly contains the start positions
            })?;
        }
        Ok(())
    };
    
    save_best(&current_state, current_len)?;


    // 4. Simultaneous Simulated Annealing
    let mut temp = 10.0; // Lower starting temperature to focus on optimization earlier
    let cooling_rate = 0.9999; // Slightly faster cooling rate for quicker convergence
    
    while start_time.elapsed() < time_limit {
        let mut new_state = current_state.clone();

        // 50/50 chance for an Order change or a Permutation change
        if rng.gen_bool(0.5) {
            // Move 1: Swap two elements in the ORDER (ATSP move)
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            new_state.order.swap(i, j);
        } else {
            // Move 2: Change the PERMUTATION of one string (Anagram move)
            let str_idx = rng.gen_range(0..n);
            let num_perms = cache.all_perms[str_idx].len();
            if num_perms > 1 {
                // Pick a new permutation index randomly, ensuring it's different from the current one
                let mut new_perm_idx = rng.gen_range(0..num_perms);
                while new_perm_idx == current_state.perm_indices[str_idx] {
                    new_perm_idx = rng.gen_range(0..num_perms);
                }
                new_state.perm_indices[str_idx] = new_perm_idx;
            }
        }

        // Calculate cost of the new state (O(N) due to precomputation)
        let new_len = calculate_total_length_fast(&new_state, &cache);
        let delta = new_len as f64 - current_len as f64;

        // Metropolis-Hastings acceptance criterion
        if delta < 0.0 || (temp > 1e-9 && rng.gen::<f64>() < (-delta / temp).exp()) {
            current_state = new_state;
            current_len = new_len;

            // Save the best solution found so far
            save_best(&current_state, current_len)?;
        }

        // Cool the temperature
        temp *= cooling_rate;
    }

    Ok(())
}
