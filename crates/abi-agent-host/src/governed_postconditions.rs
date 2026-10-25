//! Bounded, request-bound observations for every compiled postcondition.
use crate::Postconditions;
use abi_capability::Digest;
use serde::Serialize;
use std::collections::BTreeSet;

/// Provenance of an observation supplied by the startup-owned trusted fact provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ObservationSource {
    /// Injected deterministic recording fixture, not a live platform assertion.
    RecordingFixture,
    /// A startup-owned local adapter observed the actual resulting resource.
    TrustedLocalAdapter,
}

/// One individually named, fresh, request-bound postcondition observation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PostconditionObservation {
    /// Exact compiled predicate identifier; bounded to64 bytes.
    pub predicate: String,
    /// The request whose resulting state was observed.
    pub request_digest: Digest,
    /// Trusted observation time.
    pub observed_at_ms: u64,
    /// Exclusive trusted freshness expiry.
    pub expires_at_ms: u64,
    /// Content-free evidence commitment, required for verified predicates.
    pub evidence_digest: Option<Digest>,
    /// Closed supported provenance class.
    pub source: ObservationSource,
    /// Individually observed status.
    pub status: Postconditions,
}

pub(crate) fn verify(
    required: &BTreeSet<String>,
    observations: &[PostconditionObservation],
    request: Digest,
    now: u64,
) -> Postconditions {
    if observations.len() != required.len() || observations.len() > 32 {
        return Postconditions::Unverifiable;
    }
    let mut seen = BTreeSet::new();
    for observation in observations {
        if observation.predicate.len() > 64
            || !required.contains(&observation.predicate)
            || !seen.insert(&observation.predicate)
            || observation.request_digest != request
            || observation.observed_at_ms > now
            || observation.expires_at_ms <= now
            || (observation.status == Postconditions::Verified
                && observation
                    .evidence_digest
                    .is_none_or(|d| d == Digest::default()))
        {
            return Postconditions::Unverifiable;
        }
    }
    if observations
        .iter()
        .any(|o| o.status == Postconditions::Failed)
    {
        Postconditions::Failed
    } else if observations
        .iter()
        .any(|o| o.status == Postconditions::Unverifiable)
    {
        Postconditions::Unverifiable
    } else {
        Postconditions::Verified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observation() -> PostconditionObservation {
        PostconditionObservation {
            predicate: "observed".into(),
            request_digest: Digest::from_bytes([1; 32]),
            observed_at_ms: 90,
            expires_at_ms: 110,
            evidence_digest: Some(Digest::from_bytes([2; 32])),
            source: ObservationSource::RecordingFixture,
            status: Postconditions::Verified,
        }
    }
    #[test]
    fn every_predicate_requires_exact_fresh_request_bound_evidence() {
        let required = BTreeSet::from(["observed".into()]);
        let good = observation();
        assert_eq!(
            verify(
                &required,
                std::slice::from_ref(&good),
                good.request_digest,
                100
            ),
            Postconditions::Verified
        );
        for observations in [
            vec![],
            vec![good.clone(), good.clone()],
            vec![PostconditionObservation {
                predicate: "other".into(),
                ..good.clone()
            }],
            vec![PostconditionObservation {
                observed_at_ms: 101,
                ..good.clone()
            }],
            vec![PostconditionObservation {
                expires_at_ms: 100,
                ..good.clone()
            }],
            vec![PostconditionObservation {
                request_digest: Digest::default(),
                ..good.clone()
            }],
            vec![PostconditionObservation {
                evidence_digest: None,
                ..good.clone()
            }],
        ] {
            assert_eq!(
                verify(&required, &observations, good.request_digest, 100),
                Postconditions::Unverifiable
            );
        }
    }
}
