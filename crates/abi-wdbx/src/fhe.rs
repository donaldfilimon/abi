//! DGHV-style reference somewhat-homomorphic encryption for encrypted bits.
//!
//! The parameters are correctness-test parameters, not cryptographically secure
//! sizes. The scheme is not bootstrapped or security-audited. The optional
//! `experimental-dghv-bootstrap` feature adds a deliberately named
//! secret-key-assisted educational refresh. That experiment is **not**
//! cryptographic bootstrapping and provides no protection for real data.

#[cfg(feature = "full-fhe")]
#[path = "tfhe_demo.rs"]
pub mod tfhe_demo;

use num_bigint::BigUint;
use num_traits::{One, ToPrimitive};

/// Approximate secret-modulus bit length.
pub const REF_P_BITS: u16 = 126;
/// Fresh encryption noise bit count.
pub const REF_NOISE_BITS: u8 = 20;
/// Multiplicative depth covered by tests.
pub const VERIFIED_MUL_DEPTH: u8 = 3;

/// Arbitrary-precision ciphertext integer.
pub type DghvCipher = BigUint;

/// Symmetric key plus public cofactor and modulus.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DghvKeypair {
    /// Large odd secret modulus.
    pub p: BigUint,
    /// Public cofactor bound.
    pub q0: BigUint,
    /// Public modulus `p * q0`.
    pub x0: BigUint,
}

/// Seeded deterministic generator used by the reference tests and demos.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DghvRng(u64);

impl DghvRng {
    /// Construct a deterministic generator.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_u128(&mut self) -> u128 {
        u128::from(self.next_u64()) << 64 | u128::from(self.next_u64())
    }
}

/// Generate a deterministic reference keypair.
pub fn dghv_keygen(random: &mut DghvRng) -> DghvKeypair {
    let required_bit = 1_u128 << (REF_P_BITS - 1);
    let p = BigUint::from(random.next_u128() | required_bit | 1);
    let q0 = BigUint::from(random.next_u128() | required_bit);
    let x0 = &p * &q0;
    DghvKeypair { p, q0, x0 }
}

/// Encrypt one bit with fresh reference noise.
pub fn dghv_encrypt(keypair: &DghvKeypair, random: &mut DghvRng, bit: bool) -> DghvCipher {
    let noise_bound = 1_u64 << REF_NOISE_BITS;
    let noise = BigUint::from(random.next_u64() % noise_bound);
    let q0 = keypair
        .q0
        .to_u128()
        .expect("reference q0 is at most 128 bits");
    let quotient = BigUint::from(random.next_u128() % q0);
    let message = BigUint::from(u8::from(bit));
    (&keypair.p * quotient + (noise << 1) + message) % &keypair.x0
}

/// Decrypt a ciphertext to one bit.
#[must_use]
pub fn dghv_decrypt(keypair: &DghvKeypair, cipher: &DghvCipher) -> bool {
    let residue = (cipher % &keypair.x0) % &keypair.p;
    (&residue % BigUint::from(2_u8)) == BigUint::one()
}

/// Homomorphic addition, which evaluates XOR on plaintext bits.
#[must_use]
pub fn dghv_add(keypair: &DghvKeypair, left: &DghvCipher, right: &DghvCipher) -> DghvCipher {
    (left + right) % &keypair.x0
}

/// Homomorphic multiplication, which evaluates AND and consumes one depth.
#[must_use]
pub fn dghv_mul(keypair: &DghvKeypair, left: &DghvCipher, right: &DghvCipher) -> DghvCipher {
    (left * right) % &keypair.x0
}

/// Re-encrypt a bit after decrypting it with the secret key.
///
/// This function exists only for the explicitly enabled educational DGHV
/// experiment. It is **not cryptographic bootstrapping**: it requires the
/// secret key, is not security-audited, and must not be used to protect data.
#[cfg(feature = "experimental-dghv-bootstrap")]
pub fn dghv_secret_key_assisted_refresh(
    keypair: &DghvKeypair,
    random: &mut DghvRng,
    cipher: &DghvCipher,
) -> DghvCipher {
    dghv_encrypt(keypair, random, dghv_decrypt(keypair, cipher))
}

/// Evaluate NAND and re-encrypt its plaintext result with fresh noise.
///
/// This is a secret-key-assisted educational refresh, **not cryptographic
/// bootstrapping**. It is unaudited and provides no security guarantee.
#[cfg(feature = "experimental-dghv-bootstrap")]
pub fn dghv_secret_key_assisted_nand(
    keypair: &DghvKeypair,
    random: &mut DghvRng,
    left: &DghvCipher,
    right: &DghvCipher,
) -> DghvCipher {
    let encrypted_and = dghv_mul(keypair, left, right);
    dghv_encrypt(keypair, random, !dghv_decrypt(keypair, &encrypted_and))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition_and_multiplication_hold_over_seeded_keys_and_bits() {
        let mut random = DghvRng::new(0xF0E1_D2C3_B4A5_9687);
        for _ in 0..128 {
            let keypair = dghv_keygen(&mut random);
            let left = random.next_u64() & 1 == 1;
            let right = random.next_u64() & 1 == 1;
            let encrypted_left = dghv_encrypt(&keypair, &mut random, left);
            let encrypted_right = dghv_encrypt(&keypair, &mut random, right);
            assert_eq!(dghv_decrypt(&keypair, &encrypted_left), left);
            assert_eq!(dghv_decrypt(&keypair, &encrypted_right), right);
            assert_eq!(
                dghv_decrypt(
                    &keypair,
                    &dghv_add(&keypair, &encrypted_left, &encrypted_right)
                ),
                left ^ right
            );
            assert_eq!(
                dghv_decrypt(
                    &keypair,
                    &dghv_mul(&keypair, &encrypted_left, &encrypted_right)
                ),
                left & right
            );
        }
    }

    #[test]
    fn depth_two_circuit_evaluates_correctly() {
        let mut random = DghvRng::new(0x1122_3344_5566_7788);
        for _ in 0..64 {
            let keypair = dghv_keygen(&mut random);
            let bits = [
                random.next_u64() & 1 == 1,
                random.next_u64() & 1 == 1,
                random.next_u64() & 1 == 1,
                random.next_u64() & 1 == 1,
            ];
            let encrypted: Vec<_> = bits
                .iter()
                .map(|&bit| dghv_encrypt(&keypair, &mut random, bit))
                .collect();
            let left = dghv_mul(&keypair, &encrypted[0], &encrypted[1]);
            let right = dghv_mul(&keypair, &encrypted[2], &encrypted[3]);
            assert_eq!(
                dghv_decrypt(&keypair, &dghv_add(&keypair, &left, &right)),
                (bits[0] & bits[1]) ^ (bits[2] & bits[3])
            );
        }
    }

    #[test]
    fn chained_multiplication_is_verified_through_depth_three() {
        let mut random = DghvRng::new(0x9988_7766_5544_3322);
        for _ in 0..48 {
            let keypair = dghv_keygen(&mut random);
            let bits = [
                random.next_u64() & 1 == 1,
                random.next_u64() & 1 == 1,
                random.next_u64() & 1 == 1,
                random.next_u64() & 1 == 1,
            ];
            let encrypted: Vec<_> = bits
                .iter()
                .map(|&bit| dghv_encrypt(&keypair, &mut random, bit))
                .collect();
            let depth_one = dghv_mul(&keypair, &encrypted[0], &encrypted[1]);
            let depth_two = dghv_mul(&keypair, &depth_one, &encrypted[2]);
            let depth_three = dghv_mul(&keypair, &depth_two, &encrypted[3]);
            assert_eq!(
                dghv_decrypt(&keypair, &depth_three),
                bits.into_iter().all(|bit| bit)
            );
        }
    }

    #[test]
    fn parameter_constants_and_fresh_noise_bound_are_explicit() {
        assert_eq!(REF_P_BITS, 126);
        assert_eq!(REF_NOISE_BITS, 20);
        assert_eq!(VERIFIED_MUL_DEPTH, 3);
        let mut random = DghvRng::new(0x00A1_1CE5);
        let bound = BigUint::one() << (REF_NOISE_BITS + 1);
        for _ in 0..64 {
            let keypair = dghv_keygen(&mut random);
            let bit = random.next_u64() & 1 == 1;
            let cipher = dghv_encrypt(&keypair, &mut random, bit);
            assert_eq!(dghv_decrypt(&keypair, &cipher), bit);
            let residue = (&cipher % &keypair.x0) % &keypair.p;
            assert!(residue < bound || residue > keypair.p.clone() - &bound);
        }
    }

    #[test]
    fn identical_seed_reproduces_keys_and_ciphertexts() {
        let mut left = DghvRng::new(7);
        let mut right = DghvRng::new(7);
        let left_key = dghv_keygen(&mut left);
        let right_key = dghv_keygen(&mut right);
        assert_eq!(left_key, right_key);
        assert_eq!(
            dghv_encrypt(&left_key, &mut left, true),
            dghv_encrypt(&right_key, &mut right, true)
        );
    }

    #[cfg(feature = "experimental-dghv-bootstrap")]
    #[test]
    fn educational_refresh_has_complete_truth_tables_and_fresh_ciphertexts() {
        let mut random = DghvRng::new(0xD6_48_52_45_46_52_45_53);
        let keypair = dghv_keygen(&mut random);
        for left in [false, true] {
            for right in [false, true] {
                let encrypted_left = dghv_encrypt(&keypair, &mut random, left);
                let encrypted_right = dghv_encrypt(&keypair, &mut random, right);
                let sum_cipher = dghv_add(&keypair, &encrypted_left, &encrypted_right);
                let product_cipher = dghv_mul(&keypair, &encrypted_left, &encrypted_right);
                let refreshed_negation = dghv_secret_key_assisted_nand(
                    &keypair,
                    &mut random,
                    &encrypted_left,
                    &encrypted_right,
                );
                assert_eq!(dghv_decrypt(&keypair, &sum_cipher), left ^ right);
                assert_eq!(dghv_decrypt(&keypair, &product_cipher), left & right);
                assert_eq!(dghv_decrypt(&keypair, &refreshed_negation), !(left & right));
            }
        }

        let encrypted = dghv_encrypt(&keypair, &mut random, true);
        let refreshed_once = dghv_secret_key_assisted_refresh(&keypair, &mut random, &encrypted);
        let refreshed_twice = dghv_secret_key_assisted_refresh(&keypair, &mut random, &encrypted);
        assert!(dghv_decrypt(&keypair, &refreshed_once));
        assert!(dghv_decrypt(&keypair, &refreshed_twice));
        assert_ne!(encrypted, refreshed_once);
        assert_ne!(refreshed_once, refreshed_twice);
    }
}
