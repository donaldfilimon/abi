//! Portable hashing utilities (feature-flag surface).
//!
//! Ported from `src/features/hash/mod.zig`. Wyhash is the Zig-compatible
//! implementation in [`crate::wyhash`]; FNV-1a is a stable portable fallback.

use crate::wyhash;

/// 128-bit hash as two `u64` halves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hash128 {
    /// High 64 bits.
    pub hi: u64,
    /// Low 64 bits.
    pub lo: u64,
}

/// Feature always enabled in the Rust workspace.
#[must_use]
pub const fn is_enabled() -> bool {
    true
}

/// Zig-compatible Wyhash (`seed` first).
#[must_use]
pub fn wyhash64(data: &[u8], seed: u64) -> u64 {
    wyhash::hash(seed, data)
}

/// Convenience: Wyhash with seed `0`.
#[must_use]
pub fn hash64(data: &[u8]) -> u64 {
    wyhash64(data, 0)
}

/// Portable FNV-1a 64-bit hash.
#[must_use]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const PRIME: u64 = 0x1000_0000_01b3;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// 128-bit hash from two Wyhash passes with fixed seeds.
#[must_use]
pub fn wyhash128(data: &[u8]) -> Hash128 {
    Hash128 {
        hi: wyhash::hash(0x1234_5678_9abc_def0, data),
        lo: wyhash::hash(0xfedc_ba98_7654_3210, data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wyhash_is_seed_sensitive() {
        let data = b"hello feature scaffolding";
        assert_eq!(wyhash64(data, 0), wyhash64(data, 0));
        assert_ne!(wyhash64(data, 0), wyhash64(data, 1));
        assert_ne!(wyhash64(data, 0), 0);
    }

    #[test]
    fn fnv1a_is_stable() {
        assert_eq!(fnv1a_64(b"abi"), fnv1a_64(b"abi"));
        assert_ne!(fnv1a_64(b"abi"), 0);
    }

    #[test]
    fn wyhash128_halves_differ() {
        let h = wyhash128(b"distinct");
        assert_ne!(h.hi, h.lo);
        assert_ne!(h.hi, 0);
        assert_ne!(h.lo, 0);
    }
}
