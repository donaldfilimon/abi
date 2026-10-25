//! Versioned HPKE ciphertext boundary. Payloads never enter authority persistence.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
/// Bounded opaque HPKE packet, with X25519 encapsulated key and authenticated ciphertext.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedPayload {
    /// X25519 encapsulated key.
    pub encapsulated_key: Vec<u8>,
    /// ChaCha20-Poly1305 ciphertext including tag.
    pub ciphertext: Vec<u8>,
}
use hpke::{Deserializable as _, Kem as _, Serializable as _};
type Kem = hpke::kem::X25519HkdfSha256;
type Aead = hpke::aead::ChaCha20Poly1305;
type Kdf = hpke::kdf::HkdfSha256;
/// Node/client secret HPKE key. Keep exclusively on user infrastructure.
pub struct RelayKey {
    private: <Kem as hpke::Kem>::PrivateKey,
    public: <Kem as hpke::Kem>::PublicKey,
}
impl RelayKey {
    /// Generate a fresh X25519 keypair from OS-backed entropy.
    pub fn generate() -> Result<Self> {
        Self::from_seed(&crate::authority::secret_bytes())
    }
    /// Restore from a user-protected 32-byte random seed, never a password.
    pub fn from_seed(seed: &[u8; 32]) -> Result<Self> {
        let (private, public) = Kem::derive_keypair(seed);
        Ok(Self { private, public })
    }
    /// Public encryption key, bound by exact local enrollment approval.
    #[must_use]
    pub fn public_key(&self) -> [u8; 32] {
        self.public
            .to_bytes()
            .as_slice()
            .try_into()
            .expect("fixed X25519 public key")
    }
    /// Open with exact protocol, tenant, job and direction context.
    pub fn open(&self, context: &[u8], packet: &SealedPayload) -> Result<Vec<u8>> {
        packet.validate()?;
        let enc = <Kem as hpke::Kem>::EncappedKey::from_bytes(&packet.encapsulated_key)
            .map_err(|_| Error::Denied)?;
        hpke::single_shot_open::<Aead, Kdf, Kem>(
            &hpke::OpModeR::Base,
            &self.private,
            &enc,
            b"abi.relay.hpke.v1",
            &packet.ciphertext,
            context,
        )
        .map_err(|_| Error::Denied)
    }
}
impl SealedPayload {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.encapsulated_key.len() != 32 || !(16..=65552).contains(&self.ciphertext.len()) {
            return Err(Error::Invalid);
        }
        Ok(())
    }
    pub(crate) fn bytes(&self) -> usize {
        self.encapsulated_key.len() + self.ciphertext.len()
    }
}
/// RFC 9180 base-mode X25519/HKDF-SHA256/ChaCha20-Poly1305 encryption.
/// Authentication of node requests and dispatches is a separate Ed25519 layer.
pub fn seal(public_key: &[u8; 32], context: &[u8], plaintext: &[u8]) -> Result<SealedPayload> {
    if plaintext.len() > 65536 || context.len() > 4096 {
        return Err(Error::Invalid);
    }
    let key = <Kem as hpke::Kem>::PublicKey::from_bytes(public_key).map_err(|_| Error::Denied)?;
    let (enc, ciphertext) = hpke::single_shot_seal::<Aead, Kdf, Kem, _>(
        &hpke::OpModeS::Base,
        &key,
        b"abi.relay.hpke.v1",
        plaintext,
        context,
        &mut rand::rng(),
    )
    .map_err(|_| Error::Denied)?;
    Ok(SealedPayload {
        encapsulated_key: enc.to_bytes().to_vec(),
        ciphertext,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hpke_roundtrip_rejects_other_keys_context_and_ciphertext_mutation() {
        let node = RelayKey::generate().unwrap();
        let other = RelayKey::generate().unwrap();
        let packet = seal(
            &node.public_key(),
            b"tenant-a/job-1/request",
            b"private-canary",
        )
        .unwrap();
        assert_eq!(
            node.open(b"tenant-a/job-1/request", &packet).unwrap(),
            b"private-canary"
        );
        assert!(other.open(b"tenant-a/job-1/request", &packet).is_err());
        assert!(node.open(b"tenant-b/job-1/request", &packet).is_err());
        let mut changed = packet.clone();
        changed.ciphertext[0] ^= 1;
        assert!(node.open(b"tenant-a/job-1/request", &changed).is_err());
        assert!(
            !packet
                .ciphertext
                .windows(b"private-canary".len())
                .any(|b| b == b"private-canary")
        );
    }
}
