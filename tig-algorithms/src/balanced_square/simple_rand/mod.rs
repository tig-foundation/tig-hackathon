// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use anyhow::{anyhow, Result};
use tig_challenges::balanced_square::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    use rand::{rngs::SmallRng, seq::SliceRandom, Rng, SeedableRng};
    let mut rng = SmallRng::from_seed(challenge.seed.clone());
    let mut numbers = challenge.numbers.clone();
    let size = challenge.difficulty.size;

    let mut best_variance = f32::MAX;
    for _ in 0..10000 {
        let solution = Solution {
            square: (0..size)
                .map(|i| {
                    (0..size)
                        .map(|j| numbers[(i * size + j) % numbers.len()])
                        .collect::<Vec<i32>>()
                })
                .collect::<Vec<Vec<i32>>>(),
        };
        let v = challenge.calc_variance(&solution)?;
        if v < best_variance {
            best_variance = v;
            let _ = save_solution(&solution);
        }
        numbers.shuffle(&mut rng);
    }
    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
