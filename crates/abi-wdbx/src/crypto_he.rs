//! Reference single-key additive masking over a prime field.
//!
//! This uses non-cryptographic `WyHash` as its mask function and caller-supplied
//! nonces. It demonstrates additive homomorphism; it is not production
//! encryption, multi-key HE, or FHE.

/// 61-bit prime-field modulus used by the reference scheme.
pub const HE_MODULUS: u64 = (1 << 61) - 1;

/// Additive HE operation failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeError {
    /// Plaintext is outside the field.
    PlaintextTooLarge,
    /// A ciphertext without nonce metadata cannot be decrypted.
    EmptyCiphertext,
}

impl std::fmt::Display for HeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlaintextTooLarge => formatter.write_str("plaintext is outside the HE field"),
            Self::EmptyCiphertext => formatter.write_str("ciphertext has no nonce metadata"),
        }
    }
}

impl std::error::Error for HeError {}

fn mask(secret: u64, nonce: u64) -> u64 {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&secret.to_le_bytes());
    bytes[8..].copy_from_slice(&nonce.to_le_bytes());
    wyhash::wyhash(&bytes, 0xA11CE) % HE_MODULUS
}

fn add_mod(left: u64, right: u64) -> u64 {
    (left + right) % HE_MODULUS
}

fn sub_mod(left: u64, right: u64) -> u64 {
    (left + (HE_MODULUS - right % HE_MODULUS)) % HE_MODULUS
}

/// Owned masked value and the nonces whose masks it contains.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HeCipher {
    /// Masked field value.
    pub value: u64,
    /// Nonces accumulated through homomorphic addition.
    pub nonces: Vec<u64>,
}

/// Single trusted secret key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeKey {
    secret: u64,
}

impl HeKey {
    /// Construct a key from a caller-supplied reference secret.
    #[must_use]
    pub const fn new(secret: u64) -> Self {
        Self { secret }
    }

    /// Encrypt one field element under a caller-managed unique nonce.
    pub fn encrypt(self, plaintext: u64, nonce: u64) -> Result<HeCipher, HeError> {
        if plaintext >= HE_MODULUS {
            return Err(HeError::PlaintextTooLarge);
        }
        Ok(HeCipher {
            value: add_mod(plaintext, mask(self.secret, nonce)),
            nonces: vec![nonce],
        })
    }

    /// Decrypt a ciphertext created or aggregated under this key.
    pub fn decrypt(self, cipher: &HeCipher) -> Result<u64, HeError> {
        if cipher.nonces.is_empty() {
            return Err(HeError::EmptyCiphertext);
        }
        let accumulated_mask = cipher
            .nonces
            .iter()
            .fold(0, |total, &nonce| add_mod(total, mask(self.secret, nonce)));
        Ok(sub_mod(cipher.value, accumulated_mask))
    }
}

/// Homomorphically add two ciphertexts without decrypting their operands.
#[must_use]
pub fn he_add(left: &HeCipher, right: &HeCipher) -> HeCipher {
    let mut nonces = Vec::with_capacity(left.nonces.len() + right.nonces.len());
    nonces.extend_from_slice(&left.nonces);
    nonces.extend_from_slice(&right.nonces);
    HeCipher {
        value: add_mod(left.value, right.value),
        nonces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encryption_round_trips_and_rejects_invalid_inputs() {
        let key = HeKey::new(0xDEAD_BEEF);
        let cipher = key.encrypt(42, 1).expect("encrypt");
        assert_eq!(key.decrypt(&cipher), Ok(42));
        assert_eq!(key.encrypt(HE_MODULUS, 2), Err(HeError::PlaintextTooLarge));
        assert_eq!(
            key.decrypt(&HeCipher {
                value: 0,
                nonces: Vec::new()
            }),
            Err(HeError::EmptyCiphertext)
        );
    }

    #[test]
    fn ciphertext_sum_decrypts_to_plaintext_sum() {
        let key = HeKey::new(0x1234_5678);
        let left = key.encrypt(1000, 11).expect("left");
        let right = key.encrypt(337, 22).expect("right");
        assert_eq!(key.decrypt(&he_add(&left, &right)), Ok(1337));
    }

    #[test]
    fn many_values_aggregate_without_operand_decryption() {
        let key = HeKey::new(99);
        let mut accumulated = key.encrypt(0, 0).expect("zero");
        let mut expected = 0;
        for value in 1..=50 {
            accumulated = he_add(
                &accumulated,
                &key.encrypt(value, 1000 + value).expect("value"),
            );
            expected += value;
        }
        assert_eq!(key.decrypt(&accumulated), Ok(expected));
    }

    #[test]
    fn wrong_key_does_not_recover_plaintext() {
        let cipher = HeKey::new(0xAAAA).encrypt(7, 5).expect("encrypt");
        assert_ne!(HeKey::new(0xBBBB).decrypt(&cipher), Ok(7));
    }
}
