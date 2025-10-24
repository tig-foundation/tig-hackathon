// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use std::collections::HashSet;
use tig_challenges::min_superstring::*;

fn find_best_overlap_with_permutation(s1: &str, s2: &str) -> (usize, bool, String) {
    // Returns (overlap_length, is_left_direction, best_permutation)
    // is_left_direction: true means s2 goes before s1, false means s2 goes after s1
    let mut max_overlap = 0;
    let mut is_left = false;
    let mut best_permutation = s2.to_string();
    
    // Generate all permutations of s2
    let chars: Vec<char> = s2.chars().collect();
    let mut permutations = Vec::new();
    generate_permutations(&chars, &mut vec![], &mut permutations);
    
    for perm in permutations {
        let perm_str: String = perm.iter().collect();
        
        // Check if perm_str can be appended to the right of s1
        for i in 1..=s1.len().min(perm_str.len()) {
            if s1.len() >= i && perm_str.len() >= i && s1[s1.len()-i..] == perm_str[..i] {
                if i > max_overlap {
                    max_overlap = i;
                    is_left = false;
                    best_permutation = perm_str.clone();
                }
            }
        }
        
        // Check if perm_str can be appended to the left of s1
        for i in 1..=s1.len().min(perm_str.len()) {
            if s1.len() >= i && perm_str.len() >= i && s1[..i] == perm_str[perm_str.len()-i..] {
                if i > max_overlap {
                    max_overlap = i;
                    is_left = true;
                    best_permutation = perm_str.clone();
                }
            }
        }
    }
    
    (max_overlap, is_left, best_permutation)
}

fn generate_permutations(chars: &[char], current: &mut Vec<char>, result: &mut Vec<Vec<char>>) {
    if current.len() == chars.len() {
        result.push(current.clone());
        return;
    }
    
    for &ch in chars {
        if current.iter().filter(|&&c| c == ch).count() < chars.iter().filter(|&&c| c == ch).count() {
            current.push(ch);
            generate_permutations(chars, current, result);
            current.pop();
        }
    }
}

fn min_superstring(strings: &[String]) -> (Vec<String>, Vec<usize>) {
    if strings.is_empty() {
        return (Vec::new(), Vec::new());
    }
    
    // Start with the first string
    let mut superstring = strings[0].clone();
    let mut starting_indices = vec![0; strings.len()];
    let mut used_strings = HashSet::new();
    used_strings.insert(0);
    let mut permuted_strings = vec![String::new(); strings.len()]; // Track the actual permuted strings used
    permuted_strings[0] = strings[0].clone(); // First string is used as-is
    
    while used_strings.len() < strings.len() {
        let mut best_overlap = 0;
        let mut best_string_idx = 0;
        let mut best_direction = false;
        let mut best_permutation = String::new();
        
        // Find the string with the best overlap (allowing permutation)
        for (i, string) in strings.iter().enumerate() {
            if used_strings.contains(&i) {
                continue;
            }
            
            let (overlap, is_left, permutation) = find_best_overlap_with_permutation(&superstring, string);
            if overlap > best_overlap {
                best_overlap = overlap;
                best_string_idx = i;
                best_direction = is_left;
                best_permutation = permutation;
            }
        }
        
        // If no overlap found, just append the first unused string
        if best_overlap == 0 {
            for (i, string) in strings.iter().enumerate() {
                if !used_strings.contains(&i) {
                    best_string_idx = i;
                    best_direction = false;
                    best_permutation = string.clone();
                    break;
                }
            }
        }
        
        // Store the permuted string at the correct index
        permuted_strings[best_string_idx] = best_permutation.clone();
        
        // Append the string
        if !best_direction {
            // Append to the right
            starting_indices[best_string_idx] = superstring.len() - best_overlap;
            superstring = superstring + &best_permutation[best_overlap..];
        } else {
            // Append to the left
            starting_indices[best_string_idx] = 0;
            // Update all existing indices
            for i in 0..strings.len() {
                if used_strings.contains(&i) {
                    starting_indices[i] += best_permutation.len() - best_overlap;
                }
            }
            superstring = best_permutation[..best_permutation.len() - best_overlap].to_string() + &superstring;
        }
        
        used_strings.insert(best_string_idx);
    }
    
    // Return the permuted strings and starting indices
    (permuted_strings, starting_indices)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    // Run the greedy algorithm on the challenge strings
    let (permuted_strings, superstring_idxs) = min_superstring(&challenge.strings);
    
    // Create the solution
    let solution = Solution {
        permuted_strings,
        superstring_idxs,
    };
    
    // Save the solution
    save_solution(&solution)?;
    
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected


