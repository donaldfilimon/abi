//! Zig-compatible Wyhash.
//!
//! A faithful port of `std.hash.Wyhash` from the pinned Zig toolchain, because
//! **no Rust crate matches it**. The `wyhash` 0.5 crate implements a different
//! version of the algorithm and returns entirely different values — for seed `3`
//! over `"hel"`, Zig gives `10846395113768030678` and the crate gives
//! `490820195397404894`.
//!
//! That distinction is load-bearing rather than academic. `textEmbedding` derives
//! each of its 32 vector buckets *and each bucket's sign* from this hash, and the
//! resulting vectors are persisted into the user's WDBX store alongside vectors
//! Zig already wrote. A different hash would produce embeddings that do not match
//! the existing ones, so cosine and semantic search across old and new records
//! would be silently wrong — no error, just degraded ranking.
//!
//! ## Verification
//!
//! `tests/wyhash_zig_refs.txt` holds 188 `(seed, len, hash)` triples emitted by
//! the pinned Zig toolchain, covering every branch of the algorithm: the
//! small-key paths (`len < 4`, `4 <= len <= 16`), the `final1`-only range
//! (`17..=48`), and the 48-byte round loop including its strict `i + 48 < len`
//! boundary at exactly 48 and 96 bytes. To regenerate:
//!
//! ```zig
//! // zig test this against the pinned toolchain, then strip the "REF " prefix.
//! const lens = [_]usize{ 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32,
//!                        33, 47, 48, 49, 63, 64, 95, 96, 97, 128, 200 };
//! const seeds = [_]u64{ 0, 1, 2, 3, 0xA11CE, 0xdeadbeef, 0xffffffffffffffff };
//! var buf: [256]u8 = undefined;
//! for (buf[0..], 0..) |*b, i| b.* = @truncate(i *% 31 +% 7);
//! for (seeds) |seed| for (lens) |len|
//!     std.debug.print("REF {d} {d} {d}\n", .{ seed, len, std.hash.Wyhash.hash(seed, buf[0..len]) });
//! ```

/// The four Wyhash secret constants, as Zig declares them.
const SECRET: [u64; 4] = [
    0xa076_1d64_78bd_642f,
    0xe703_7ed1_a0b4_28db,
    0x8ebc_6af0_9c88_c6e3,
    0x5899_65cc_7537_4cc3,
];

/// 64x64 -> 128 multiply, returning `(low, high)`.
///
/// Zig's `mum` mutates two pointers; returning a tuple is the same operation.
/// The `u128` product of two `u64`s cannot overflow, so no wrapping is needed
/// despite Zig's `*%`. The low/high split is the intentional 128→64 truncation
/// that defines the algorithm.
#[inline]
#[allow(clippy::cast_possible_truncation)]
const fn mum(a: u64, b: u64) -> (u64, u64) {
    let wide = (a as u128) * (b as u128);
    (wide as u64, (wide >> 64) as u64)
}

/// `mum` folded into a single value by XOR.
#[inline]
const fn mix(a: u64, b: u64) -> u64 {
    let (low, high) = mum(a, b);
    low ^ high
}

/// Little-endian read of `N` bytes from the front of `data`, zero-extended.
///
/// # Panics
///
/// Panics if `data` is shorter than `N`. Every call site slices with a length it
/// has already bounds-checked, matching Zig's `assert`-guarded `read`.
#[inline]
fn read<const N: usize>(data: &[u8]) -> u64 {
    let mut bytes = [0_u8; 8];
    bytes[..N].copy_from_slice(&data[..N]);
    u64::from_le_bytes(bytes)
}

/// Zig-compatible Wyhash state, for incremental hashing.
///
/// Most callers want [`hash`]. This exists because Zig's `update`/`final` pair is
/// part of the same module, and the streaming form has a subtlety worth keeping
/// in one place: an input that is exactly 48-byte aligned must run its last full
/// block through `final1` rather than `round`.
#[derive(Debug, Clone, Copy)]
struct Wyhash {
    a: u64,
    b: u64,
    state: [u64; 3],
    total_len: u64,
}

impl Wyhash {
    /// Seed the state, matching Zig's `init`.
    #[inline]
    const fn new(seed: u64) -> Self {
        let seeded = seed ^ mix(seed ^ SECRET[0], SECRET[1]);
        Self {
            a: 0,
            b: 0,
            state: [seeded; 3],
            total_len: 0,
        }
    }

    /// Handle an input of at most 16 bytes.
    #[inline]
    fn small_key(&mut self, input: &[u8]) {
        debug_assert!(input.len() <= 16);

        if input.len() >= 4 {
            let end = input.len() - 4;
            // `quarter` is 0 for len 4..=7 and 4 for len 8..=15, which is what
            // makes the two reads overlap for short inputs.
            let quarter = (input.len() >> 3) << 2;
            self.a = (read::<4>(input) << 32) | read::<4>(&input[quarter..]);
            self.b = (read::<4>(&input[end..]) << 32) | read::<4>(&input[end - quarter..]);
        } else if !input.is_empty() {
            self.a = (u64::from(input[0]) << 16)
                | (u64::from(input[input.len() >> 1]) << 8)
                | u64::from(input[input.len() - 1]);
            self.b = 0;
        } else {
            self.a = 0;
            self.b = 0;
        }
    }

    /// Consume one full 48-byte block.
    #[inline]
    fn round(&mut self, block: &[u8]) {
        debug_assert!(block.len() >= 48);
        for i in 0..3 {
            let a = read::<8>(&block[16 * i..]);
            let b = read::<8>(&block[16 * i + 8..]);
            self.state[i] = mix(a ^ SECRET[i + 1], b ^ self.state[i]);
        }
    }

    /// Collapse the three lanes into lane 0.
    #[inline]
    const fn final0(&mut self) {
        self.state[0] ^= self.state[1] ^ self.state[2];
    }

    /// Absorb the tail. `input` must be at least 16 bytes long.
    #[inline]
    fn final1(&mut self, input: &[u8], start: usize) {
        debug_assert!(input.len() >= 16);
        debug_assert!(input.len() - start <= 48);
        let tail = &input[start..];

        // Strictly `<`: a tail of exactly 16 bytes is handled by the two reads
        // below rather than by this loop.
        let mut i = 0;
        while i + 16 < tail.len() {
            self.state[0] = mix(
                read::<8>(&tail[i..]) ^ SECRET[1],
                read::<8>(&tail[i + 8..]) ^ self.state[0],
            );
            i += 16;
        }

        // Both reads index the *whole* input, not the tail, so they can reach
        // back before `start`.
        self.a = read::<8>(&input[input.len() - 16..]);
        self.b = read::<8>(&input[input.len() - 8..]);
    }

    /// Produce the digest.
    #[inline]
    fn final2(&mut self) -> u64 {
        self.a ^= SECRET[1];
        self.b ^= self.state[0];
        let (low, high) = mum(self.a, self.b);
        self.a = low;
        self.b = high;
        mix(self.a ^ SECRET[0] ^ self.total_len, self.b ^ SECRET[1])
    }
}

/// Hash `input` with `seed`, bit-identically to Zig's `std.hash.Wyhash.hash`.
///
/// Argument order follows Zig's (`seed` first), deliberately: the Rust `wyhash`
/// crate takes `(bytes, seed)`, and matching Zig here makes a mistaken swap
/// between the two a type-level impossibility rather than a silent bug.
#[must_use]
pub fn hash(seed: u64, input: &[u8]) -> u64 {
    let mut state = Wyhash::new(seed);

    if input.len() <= 16 {
        state.small_key(input);
    } else {
        let mut i = 0;
        if input.len() >= 48 {
            // Strictly `<`, so an exactly-48-byte-aligned input leaves its last
            // full block to `final1`. This is the edge the reference vectors at
            // len 48 and 96 pin down.
            while i + 48 < input.len() {
                state.round(&input[i..]);
                i += 48;
            }
            state.final0();
        }
        state.final1(input, i);
    }

    state.total_len = input.len() as u64;
    state.final2()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator used for the reference vectors: `buf[i] = (i * 31 + 7) % 256`.
    #[allow(clippy::cast_possible_truncation)] // deliberate low-byte truncation (= Zig `@truncate`)
    fn reference_input(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i.wrapping_mul(31).wrapping_add(7)) as u8)
            .collect()
    }

    #[test]
    fn matches_every_zig_reference_vector() {
        let refs = include_str!("../tests/wyhash_zig_refs.txt");
        let mut checked = 0;
        for line in refs.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut fields = line.split_whitespace();
            let seed: u64 = fields.next().expect("seed").parse().expect("seed parses");
            let len: usize = fields.next().expect("len").parse().expect("len parses");
            let expected: u64 = fields.next().expect("hash").parse().expect("hash parses");

            let input = reference_input(len);
            assert_eq!(
                hash(seed, &input),
                expected,
                "seed={seed} len={len} diverges from Zig"
            );
            checked += 1;
        }
        assert!(
            checked >= 180,
            "expected the full reference set, checked only {checked}"
        );
    }

    #[test]
    fn matches_zig_on_the_empty_input() {
        // Emitted separately from the table because the test-runner banner
        // swallowed the first generated line.
        assert_eq!(hash(0, b""), 290_873_116_282_709_081);
    }

    #[test]
    fn matches_zig_on_the_embedding_gram_shapes() {
        // The exact call shape textEmbedding uses: seed = n-gram length, input =
        // the lowercased gram. These four are the values a wrong Wyhash would
        // silently change, taking every persisted vector with them.
        assert_eq!(hash(1, b"h"), 8_151_176_929_567_807_605);
        assert_eq!(hash(2, b"he"), 5_976_454_099_606_738_610);
        assert_eq!(hash(3, b"hel"), 10_846_395_113_768_030_678);
        assert_eq!(hash(0xA11CE, b"hello world"), 15_812_251_943_945_396_959);
    }

    #[test]
    fn differs_from_the_rust_wyhash_crate() {
        // Guards the whole reason this module exists: if a future dependency bump
        // ever made the crate agree, this test failing is the signal to
        // re-evaluate — not a reason to swap back silently.
        assert_ne!(hash(3, b"hel"), 490_820_195_397_404_894);
    }

    #[test]
    fn seed_changes_the_digest() {
        assert_ne!(hash(0, b"abi"), hash(1, b"abi"));
    }

    #[test]
    fn length_boundaries_do_not_panic() {
        // Exercises every branch boundary directly, including the debug_asserts.
        for len in 0..=200 {
            let input = reference_input(len);
            let _ = hash(0xA11CE, &input);
        }
    }
}
