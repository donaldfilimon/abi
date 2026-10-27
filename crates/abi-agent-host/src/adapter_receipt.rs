//! Single-use adapter receipts as the only postcondition evidence.
//!
//! A live executor records a receipt only after its own check of the
//! effect passed (the desktop adapter saw the approved bundle frontmost; the
//! message adapter got a well-formed provider id). [`ReceiptFacts`] wraps the
//! caller's upstream [`GovernedFactProvider`] and answers the owned
//! postcondition from that receipt alone: an upstream claim about it is never
//! consulted, so a permissive or misconfigured provider cannot mark the run
//! verified. The receipt is consumed by the first postcondition read, whatever
//! its result, and is fresh for one second after the executor recorded it.
use crate::{
    GovernedFactProvider, GovernedFacts, ObservationSource, PostconditionObservation,
    Postconditions,
};
use abi_capability::{Clock, Digest, Request, lifecycle::LifecycleError, source::PackageSource};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
};

const FRESH_MS: u64 = 1000;

/// Shared between one executor and its fact decorator.
pub(crate) struct Receipts {
    predicate: &'static str,
    source: PackageSource,
    package: Digest,
    parameters: Digest,
    clock: Arc<dyn Clock>,
    receipt: Mutex<Option<Receipt>>,
}
struct Receipt {
    request: Digest,
    evidence: Digest,
    completed_at: u64,
}

impl Receipts {
    pub(crate) fn new(
        predicate: &'static str,
        source: PackageSource,
        package: Digest,
        parameters: Digest,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        Arc::new(Self {
            predicate,
            source,
            package,
            parameters,
            clock,
            receipt: Mutex::new(None),
        })
    }
    /// Drop any earlier receipt before a new attempt.
    pub(crate) fn clear(&self) {
        if let Ok(mut slot) = self.receipt.lock() {
            *slot = None;
        }
    }
    /// Record the executor's own evidence for `request`. Without a recorded
    /// receipt the postcondition stays unverifiable, which is the honest result.
    pub(crate) fn record(&self, request: Digest, evidence: Digest) {
        let completed_at = self.clock.now_ms();
        if completed_at.checked_add(FRESH_MS).is_none() {
            return;
        }
        if let Ok(mut slot) = self.receipt.lock() {
            *slot = Some(Receipt {
                request,
                evidence,
                completed_at,
            });
        }
    }
    pub(crate) fn handle(self: &Arc<Self>) -> ReceiptObservationHandle {
        ReceiptObservationHandle {
            receipts: Arc::clone(self),
        }
    }
    fn matches(&self, request: &Request) -> bool {
        request.computed_digest() == request.request_digest
            && request.package_digest == self.package
            && request.scope == self.source.resource_scope
            && request.capability_id == self.source.id
            && request.capability_version == self.source.version
            && request.parameter_digest == self.parameters
    }
}

/// Opaque handle to one executor's receipts.
pub struct ReceiptObservationHandle {
    receipts: Arc<Receipts>,
}
impl ReceiptObservationHandle {
    /// Keep `upstream` authority, but answer the owned postcondition from the
    /// executor's receipt only.
    pub fn decorate<F: GovernedFactProvider>(self, upstream: F) -> ReceiptFacts<F> {
        ReceiptFacts {
            upstream,
            receipts: self.receipts,
        }
    }
}

/// Upstream facts plus receipt-backed postcondition evidence.
pub struct ReceiptFacts<F> {
    upstream: F,
    receipts: Arc<Receipts>,
}
impl<F: GovernedFactProvider> GovernedFactProvider for ReceiptFacts<F> {
    fn observe(&mut self, request: &Request) -> Result<GovernedFacts, LifecycleError> {
        if !self.receipts.matches(request) {
            return Err(LifecycleError::InvalidState);
        }
        self.upstream.observe(request)
    }
    fn postconditions(
        &mut self,
        request: &Request,
        required: &BTreeSet<String>,
    ) -> Result<Vec<PostconditionObservation>, LifecycleError> {
        // Every attempt consumes the receipt, including malformed requests.
        let receipt = self
            .receipts
            .receipt
            .lock()
            .map_err(|_| LifecycleError::Unavailable)?
            .take()
            .ok_or(LifecycleError::Unavailable)?;
        let now = self.receipts.clock.now_ms();
        let expires = receipt
            .completed_at
            .checked_add(FRESH_MS)
            .ok_or(LifecycleError::Unavailable)?;
        if !self.receipts.matches(request)
            || receipt.request != request.request_digest
            || required.len() != 1
            || !required.contains(self.receipts.predicate)
            || receipt.completed_at > now
            || now >= expires
            || now >= self.receipts.source.expiration_criteria.expires_at_ms
        {
            return Err(LifecycleError::Unavailable);
        }
        Ok(vec![PostconditionObservation {
            predicate: self.receipts.predicate.into(),
            request_digest: receipt.request,
            observed_at_ms: receipt.completed_at,
            expires_at_ms: expires,
            evidence_digest: Some(receipt.evidence),
            source: ObservationSource::TrustedLocalAdapter,
            status: Postconditions::Verified,
        }])
    }
}
