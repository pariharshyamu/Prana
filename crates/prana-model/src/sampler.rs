//! Token sampling: greedy argmax or temperature sampling with a small
//! deterministic xorshift RNG (seedable, dependency-free, reproducible runs).

use prana_kernels::softmax;

pub enum Sampler {
    Greedy,
    Temperature { temp: f32, rng: XorShift },
}

impl Sampler {
    /// Greedy if `temp == 0.0`, otherwise temperature sampling with `seed`.
    pub fn new(temp: f32, seed: u64) -> Self {
        if temp <= 0.0 {
            Sampler::Greedy
        } else {
            Sampler::Temperature { temp, rng: XorShift::new(seed) }
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        match self {
            Sampler::Greedy => argmax(logits),
            Sampler::Temperature { temp, rng } => {
                let scaled: Vec<f32> = logits.iter().map(|l| l / *temp).collect();
                let probs = softmax(&scaled);
                let mut r = rng.next_f32();
                for (i, p) in probs.iter().enumerate() {
                    r -= p;
                    if r <= 0.0 {
                        return i as u32;
                    }
                }
                (probs.len() - 1) as u32 // float round-off: last bucket
            }
        }
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// xorshift64* — tiny, deterministic, good enough for sampling.
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_the_max() {
        let mut s = Sampler::new(0.0, 42);
        assert_eq!(s.sample(&[0.1, 3.0, -1.0, 2.9]), 1);
    }

    #[test]
    fn temperature_sampling_is_seeded_and_in_range() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut a = Sampler::new(0.8, 7);
        let mut b = Sampler::new(0.8, 7);
        for _ in 0..50 {
            let (x, y) = (a.sample(&logits), b.sample(&logits));
            assert_eq!(x, y, "same seed must give same stream");
            assert!((x as usize) < logits.len());
        }
    }

    #[test]
    fn rng_is_roughly_uniform() {
        let mut rng = XorShift::new(123);
        let n = 10_000;
        let mean: f32 = (0..n).map(|_| rng.next_f32()).sum::<f32>() / n as f32;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
    }
}
