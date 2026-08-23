use crate::{CapabilityPackage, Digest, Request};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;

/// Content-free record of a simulated effect.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RecordedEffect {
    /// Request commitment.
    pub request_digest: Digest,
    /// Package commitment.
    pub package_digest: Digest,
    /// Scope commitment.
    pub scope_digest: Digest,
    /// Parameter commitment.
    pub parameter_digest: Digest,
    /// Deterministic simulated result commitment.
    pub result_digest: Digest,
}

/// Local recording-only actuator with injected postcondition facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordingActuator {
    satisfied_postconditions: BTreeSet<String>,
    records: Vec<RecordedEffect>,
}

impl RecordingActuator {
    /// Construct from a frozen synthetic platform-state fixture.
    #[must_use]
    pub fn new<I, S>(satisfied_postconditions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            satisfied_postconditions: satisfied_postconditions
                .into_iter()
                .map(Into::into)
                .collect(),
            records: Vec::new(),
        }
    }

    /// Preview typed postconditions without mutating recording state.
    #[must_use]
    pub fn postconditions_hold(&self, package: &CapabilityPackage) -> bool {
        package
            .postconditions
            .is_subset(&self.satisfied_postconditions)
    }

    /// Record a digest-only simulated effect.
    pub(crate) fn record(&mut self, request: &Request, package: &CapabilityPackage) -> Digest {
        let scope_digest = digest_json(&request.scope);
        let result_digest = digest_parts(&[
            request.request_digest.as_bytes(),
            package.digest.as_bytes(),
            scope_digest.as_bytes(),
        ]);
        self.records.push(RecordedEffect {
            request_digest: request.request_digest,
            package_digest: package.digest,
            scope_digest,
            parameter_digest: request.parameter_digest,
            result_digest,
        });
        result_digest
    }

    /// Observe content-free recorded calls.
    #[must_use]
    pub fn records(&self) -> &[RecordedEffect] {
        &self.records
    }
}

pub(crate) fn digest_json<T: Serialize>(value: &T) -> Digest {
    let bytes = serde_json::to_vec(value).expect("closed serializable type");
    let mut output = [0_u8; 32];
    output.copy_from_slice(&Sha256::digest(bytes));
    Digest::from_bytes(output)
}

fn digest_parts(parts: &[&[u8]]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(b"abi-capability-recording-v1\0");
    for part in parts {
        hasher.update(part);
    }
    let mut output = [0_u8; 32];
    output.copy_from_slice(&hasher.finalize());
    Digest::from_bytes(output)
}
