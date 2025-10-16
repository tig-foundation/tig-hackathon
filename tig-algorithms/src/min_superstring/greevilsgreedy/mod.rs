// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
use tig_challenges::min_superstring::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let n = challenge.strings.len();

    if n == 0 {
        return Err(anyhow!("No strings provided"));
    }

    // Initialize available strings (indices)
    let mut available: Vec<usize> = (0..n).collect();

    // Pick a random starting string
    let start_idx = rng.gen_range(0..n);
    available.retain(|&x| x != start_idx);

    // Track the construction order and permutations
    let mut construction_order = vec![start_idx];
    let mut permuted = vec![String::new(); n];
    permuted[start_idx] = challenge.strings[start_idx].clone();

    // Current superstring state (built incrementally)
    let mut current_superstring = challenge.strings[start_idx].clone();

    // Build the superstring by greedily selecting strings with maximum right overlap
    while !available.is_empty() {
        let mut best_overlap = 0;
        let mut ties = Vec::new();

        // Search for the string with maximum right_overlap
        // Stop early if we find one with overlap >= 3
        for &idx in &available {
            let (overlap, _) =
                find_best_right_overlap(&current_superstring, &challenge.strings[idx]);

            if overlap > best_overlap {
                best_overlap = overlap;
                ties.clear();
                ties.push(idx);

                // Early exit if we found an overlap of 3 or more
                if best_overlap >= 3 {
                    break;
                }
            } else if overlap == best_overlap {
                ties.push(idx);
            }
        }

        // Break ties randomly
        let chosen_idx = if ties.is_empty() {
            return Err(anyhow!("No string found"));
        } else if ties.len() > 1 {
            *ties.choose(&mut rng).unwrap()
        } else {
            ties[0]
        };

        // Compute the best permutation for the chosen string
        let (overlap, chosen_permutation) =
            find_best_right_overlap(&current_superstring, &challenge.strings[chosen_idx]);

        // Add the permuted string to construction order
        construction_order.push(chosen_idx);
        permuted[chosen_idx] = chosen_permutation.clone();

        // Update current superstring by appending the non-overlapping part
        current_superstring.push_str(&chosen_permutation[overlap..]);

        // Remove from available
        available.retain(|&x| x != chosen_idx);
    }

    // Build the solution
    // We need to convert construction order to superstring_idxs
    let superstring_idxs = build_superstring_indices(&permuted, &construction_order);

    let solution = Solution {
        permuted_strings: permuted,
        superstring_idxs,
    };

    save_solution(&solution)?;
    Ok(())
}

/// Find the best right overlap between string1 (fixed) and string2 (permutable)
/// Returns (overlap_length, best_permutation_of_string2)
fn find_best_right_overlap(string1: &str, string2: &str) -> (usize, String) {
    let s1_chars: Vec<char> = string1.chars().collect();
    let s2_chars: Vec<char> = string2.chars().collect();
    let n1 = s1_chars.len();
    let n2 = s2_chars.len();

    let mut best_overlap = 0;
    let mut best_permutation = string2.to_string();

    // Try all permutations of string2
    let mut permutation = s2_chars.clone();
    generate_permutations(&mut permutation, 0, &mut |perm| {
        // Check right overlap with this permutation
        let max_overlap = n1.min(n2);
        for overlap_len in (1..=max_overlap).rev() {
            // Check if the last overlap_len characters of string1 match the first overlap_len of perm
            let mut matches = true;
            for i in 0..overlap_len {
                if s1_chars[n1 - overlap_len + i] != perm[i] {
                    matches = false;
                    break;
                }
            }

            if matches && overlap_len > best_overlap {
                best_overlap = overlap_len;
                best_permutation = perm.iter().collect();
                break; // Found max overlap for this permutation
            }
        }
    });

    (best_overlap, best_permutation)
}

/// Generate all permutations using Heap's algorithm
fn generate_permutations<F>(arr: &mut Vec<char>, start: usize, callback: &mut F)
where
    F: FnMut(&Vec<char>),
{
    if start == arr.len() {
        callback(arr);
        return;
    }

    for i in start..arr.len() {
        arr.swap(start, i);
        generate_permutations(arr, start + 1, callback);
        arr.swap(start, i);
    }
}

/// Build superstring indices from permuted strings and construction order
fn build_superstring_indices(permuted: &[String], construction_order: &[usize]) -> Vec<usize> {
    let n = permuted.len();
    let mut indices = vec![0; n];

    if n == 0 {
        return indices;
    }

    // First string starts at position 0
    indices[construction_order[0]] = 0;
    let mut current_pos = permuted[construction_order[0]].len();

    // Place subsequent strings
    for i in 1..construction_order.len() {
        let prev_idx = construction_order[i - 1];
        let curr_idx = construction_order[i];

        // Find overlap between previous string and current string
        let overlap = find_overlap(&permuted[prev_idx], &permuted[curr_idx]);

        // Current string starts at: previous position + previous length - overlap
        indices[curr_idx] = current_pos - overlap;
        current_pos += permuted[curr_idx].len() - overlap;
    }

    indices
}

/// Find the overlap when string2 follows string1 (both are already fixed)
fn find_overlap(string1: &str, string2: &str) -> usize {
    let s1_chars: Vec<char> = string1.chars().collect();
    let s2_chars: Vec<char> = string2.chars().collect();
    let n1 = s1_chars.len();
    let n2 = s2_chars.len();

    let max_overlap = n1.min(n2);

    for overlap_len in (1..=max_overlap).rev() {
        let mut matches = true;
        for i in 0..overlap_len {
            if s1_chars[n1 - overlap_len + i] != s2_chars[i] {
                matches = false;
                break;
            }
        }
        if matches {
            return overlap_len;
        }
    }

    0
}

// Important! Do not include any tests in this file, it will result in your submission being rejected