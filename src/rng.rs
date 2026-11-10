//! Seeded pseudo-random numbers from a fixed, documented algorithm.
//!
//! The crate does not use an external random-number crate, so the random
//! choices for a given seed cannot change when a dependency changes. Each
//! use of randomness reads its own [`Stream`], derived from the user's seed
//! with SplitMix64. Each stream is a xoshiro256++ generator
//! (Blackman and Vigna, "Scrambled linear pseudorandom number generators",
//! ACM TOMS 2021).

/// The increment of SplitMix64 (the golden ratio in 64-bit fixed point).
const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// An odd constant that spreads stream ids over the SplitMix64 state space.
const STREAM_SPREAD: u64 = 0xD1B5_4A32_D192_ED03;

/// Advances a SplitMix64 state and returns its next output.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX_GAMMA);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One independent use of randomness in a run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stream {
    /// Choice of the initial centroids.
    Initialization,
    /// Choice of donor clusters for dead-cluster revival.
    Revival,
    /// Row order of one epoch (0-based).
    Epoch(u64),
}

impl Stream {
    fn id(self) -> u64 {
        match self {
            Self::Initialization => 0,
            Self::Revival => 1,
            Self::Epoch(epoch) => epoch.wrapping_add(2),
        }
    }
}

/// A xoshiro256++ generator.
#[derive(Clone, Debug)]
pub(crate) struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Creates the generator of `stream` for the user seed `seed`.
    pub(crate) fn new(seed: u64, stream: Stream) -> Self {
        let mut seeder = seed;
        let mut state = splitmix64(&mut seeder) ^ stream.id().wrapping_mul(STREAM_SPREAD);
        // SplitMix64 is a bijection of its state, so four consecutive outputs
        // are never all zero, which xoshiro256++ requires.
        let state = [
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
        ];
        Self { state }
    }

    /// Returns the next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Returns a uniform sample from the open interval (0, 1).
    ///
    /// The result is `(m + 0.5) / 2^53` for a uniform 53-bit integer `m`, so
    /// it is never 0 or 1, and `ln(-ln(u))` is always finite.
    pub(crate) fn open_unit(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64 + 0.5) * SCALE
    }

    /// Returns a uniform integer in `0..bound`, without modulo bias.
    ///
    /// Uses the multiply-and-reject method of Lemire, "Fast random integer
    /// generation in an interval", ACM TOMACS 2019.
    ///
    /// # Panics
    ///
    /// Panics if `bound` is zero.
    pub(crate) fn below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "Rng::below needs a positive bound");
        let mut product = u128::from(self.next_u64()) * u128::from(bound);
        if (product as u64) < bound {
            let threshold = bound.wrapping_neg() % bound;
            while (product as u64) < threshold {
                product = u128::from(self.next_u64()) * u128::from(bound);
            }
        }
        (product >> 64) as u64
    }

    /// Shuffles `items` uniformly with the Fisher–Yates method.
    pub(crate) fn shuffle<V>(&mut self, items: &mut [V]) {
        for i in (1..items.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xoshiro256plusplus_matches_reference_outputs() {
        // Outputs of the reference C implementation from the state {1, 2, 3, 4}.
        let mut rng = Rng {
            state: [1, 2, 3, 4],
        };
        let expected: [u64; 6] = [
            41_943_041,
            58_720_359,
            3_588_806_011_781_223,
            3_591_011_842_654_386,
            9_228_616_714_210_784_205,
            9_973_669_472_204_895_162,
        ];
        for value in expected {
            assert_eq!(rng.next_u64(), value);
        }
    }

    #[test]
    fn splitmix64_matches_reference_outputs() {
        // Outputs of the reference C implementation from the state 1234567.
        let mut state = 1_234_567;
        let expected: [u64; 3] = [
            6_457_827_717_110_365_317,
            3_203_168_211_198_807_973,
            9_817_491_932_198_370_423,
        ];
        for value in expected {
            assert_eq!(splitmix64(&mut state), value);
        }
    }

    #[test]
    fn streams_are_reproducible_and_distinct() {
        let draw = |stream| {
            let mut rng = Rng::new(42, stream);
            [rng.next_u64(), rng.next_u64()]
        };
        assert_eq!(draw(Stream::Revival), draw(Stream::Revival));
        assert_ne!(draw(Stream::Initialization), draw(Stream::Revival));
        assert_ne!(draw(Stream::Epoch(0)), draw(Stream::Epoch(1)));
        assert_ne!(
            Rng::new(1, Stream::Revival).next_u64(),
            Rng::new(2, Stream::Revival).next_u64()
        );
    }

    #[test]
    fn open_unit_stays_inside_the_interval() {
        let mut rng = Rng::new(7, Stream::Revival);
        for _ in 0..10_000 {
            let u = rng.open_unit();
            assert!(u > 0.0 && u < 1.0);
            assert!((-u.ln()).ln().is_finite());
        }
    }

    #[test]
    fn below_is_in_range_and_covers_it() {
        let mut rng = Rng::new(3, Stream::Initialization);
        let mut seen = [0u32; 7];
        for _ in 0..7_000 {
            seen[rng.below(7) as usize] += 1;
        }
        // Each bucket expects 1000 draws; 800 is more than six standard deviations below.
        assert!(seen.iter().all(|&count| count > 800), "{seen:?}");
        assert_eq!(rng.below(1), 0);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut items: Vec<usize> = (0..1000).collect();
        Rng::new(9, Stream::Epoch(0)).shuffle(&mut items);
        assert_ne!(items, (0..1000).collect::<Vec<_>>());
        items.sort_unstable();
        assert_eq!(items, (0..1000).collect::<Vec<_>>());
    }
}
