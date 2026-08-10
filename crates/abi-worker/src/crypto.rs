//! Explicit coordinator-key trust and strict Ed25519 verification.

use crate::{CancellationRequest, JobEnvelope, WorkerError};
use ed25519_dalek::{Signature, VerifyingKey};
use std::collections::BTreeMap;

/// Startup-configured coordinator signing keys keyed by stable identity.
#[derive(Clone, Debug, Default)]
pub struct CoordinatorTrustStore {
    keys: BTreeMap<String, TrustedCoordinator>,
}

#[derive(Clone, Debug)]
struct TrustedCoordinator {
    audience: String,
    key: VerifyingKey,
}

impl CoordinatorTrustStore {
    /// Empty trust store. No coordinator is implicitly trusted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one unique coordinator key.
    pub fn insert(
        &mut self,
        key_id: impl Into<String>,
        audience: impl Into<String>,
        key_bytes: &[u8],
    ) -> Result<(), WorkerError> {
        let key_id = key_id.into();
        let audience = audience.into();
        crate::types::validate_id("coordinator key id", &key_id)?;
        crate::types::validate_id("coordinator audience", &audience)?;
        if self.keys.contains_key(&key_id) {
            return Err(WorkerError::Invalid(format!(
                "duplicate coordinator key id '{key_id}'"
            )));
        }
        let bytes: [u8; 32] = key_bytes.try_into().map_err(|_| {
            WorkerError::Invalid("coordinator public key must contain 32 bytes".into())
        })?;
        let key = VerifyingKey::from_bytes(&bytes)
            .map_err(|_| WorkerError::Invalid("coordinator public key is invalid".into()))?;
        self.keys
            .insert(key_id, TrustedCoordinator { audience, key });
        Ok(())
    }

    pub(crate) fn verify_envelope(&self, envelope: &JobEnvelope) -> Result<(), WorkerError> {
        self.verify(
            &envelope.signer_key_id,
            &envelope.coordinator_audience,
            &crate::canonical::envelope_bytes(envelope)?,
            &envelope.signature,
        )
    }

    pub(crate) fn verify_cancellation(
        &self,
        request: &CancellationRequest,
    ) -> Result<(), WorkerError> {
        self.verify(
            &request.signer_key_id,
            &request.coordinator_audience,
            &crate::canonical::cancellation_bytes(request)?,
            &request.signature,
        )
    }

    fn verify(
        &self,
        key_id: &str,
        audience: &str,
        payload: &[u8],
        signature: &[u8],
    ) -> Result<(), WorkerError> {
        let coordinator = self.keys.get(key_id).ok_or_else(|| {
            WorkerError::Authentication(format!("unknown coordinator key '{key_id}'"))
        })?;
        if coordinator.audience != audience {
            return Err(WorkerError::Authentication(
                "coordinator key is bound to a different audience".into(),
            ));
        }
        let signature_bytes: [u8; 64] = signature
            .try_into()
            .map_err(|_| WorkerError::Authentication("signature must contain 64 bytes".into()))?;
        coordinator
            .key
            .verify_strict(payload, &Signature::from_bytes(&signature_bytes))
            .map_err(|_| WorkerError::Authentication("signature verification failed".into()))
    }
}
