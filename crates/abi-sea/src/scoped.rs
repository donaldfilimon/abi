//! Trusted-host scope binding over an exclusively owned WDBX store.
//!
//! This is NOT an ACL on shared WDBX. The startup owner must supply a dedicated
//! store containing only records readable by this principal and scope, including
//! its audit history and router state. Construction does not verify that premise.
//! Never construct this from request metadata, or reuse a shared-store top-k view.
//! Each read independently evaluates the canonical memory.read package and fresh
//! host-supplied authority facts, then commits required redacted audit before any
//! store access. No authorization cache is retained.

use crate::evidence::{EvidenceContext, gather_evidence};
use abi_capability::{
    AuditRecord, AuditSink, AuthorizationContext, Decision, Digest, Kernel, Principal, Request,
    Scope,
    source::{CompiledSource, SourceFacts},
};
use abi_wdbx::VersionedStore;

/// Closed failure; never reveals store existence, dimensions, counts or contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallDenied {
    /// The request or fresh canonical authority did not permit this read.
    Authorization,
    /// Required audit could not be committed.
    Audit,
}

/// Startup-owned identity of the one approved memory-read source.
#[derive(Clone)]
pub struct ReadBinding {
    /// Trusted authenticated principal, never taken from retrieved content.
    pub principal: Principal,
    /// Exact dedicated-store scope.
    pub scope: Scope,
    /// Startup approved source version.
    pub version: String,
    /// Startup compiled package identity.
    pub package_digest: Digest,
}

/// An owned store whose complete contents the owner has already scope-isolated.
pub struct TrustedScopeStore {
    store: VersionedStore,
    binding: ReadBinding,
}
impl TrustedScopeStore {
    /// Trusted startup assertion, NOT ownership verification. See module contract.
    #[must_use]
    pub const fn from_owner_authorized_store(store: VersionedStore, binding: ReadBinding) -> Self {
        Self { store, binding }
    }

    /// Authorize and audit before touching data, stats, scoring or prompt packing.
    ///
    /// The trusted caller must refresh revocation/policy facts for this operation;
    /// observation timestamps are checked but are not proof of fact acquisition.
    #[allow(clippy::too_many_arguments)]
    pub fn recall(
        &self,
        request: &Request,
        context: &AuthorizationContext<'_>,
        source: &CompiledSource,
        facts: &SourceFacts<'_>,
        observed_at_ms: u64,
        expires_at_ms: u64,
        audit: &mut dyn AuditSink,
        input: &str,
        limit: usize,
    ) -> Result<EvidenceContext, RecallDenied> {
        let now = context.now_ms.now_ms();
        if request.principal != self.binding.principal
            || request.scope != self.binding.scope
            || request.capability_id != "memory.read"
            || request.capability_version != self.binding.version
            || request.package_digest != self.binding.package_digest
            || observed_at_ms > now
            || now >= expires_at_ms
            || facts.now_ms != now
            || facts.input != &serde_json::json!({"query": input})
            || request.parameter_digest != digest_json(facts.input)
            || source.check_prerequisites(request, facts).is_err()
        {
            return Err(RecallDenied::Authorization);
        }
        let registry = abi_capability::CapabilityRegistry::compile([source.package().clone()])
            .map_err(|_| RecallDenied::Authorization)?;
        let result = Kernel::new(&registry).evaluate_source_bound(
            request,
            context,
            source,
            Digest::default(),
            None,
        );
        audit
            .record(&AuditRecord {
                attempt_id: request.attempt_id.clone(),
                request_digest: request.request_digest,
                principal_digest: digest_json(&request.principal),
                scope_digest: digest_json(&request.scope),
                package_digest: request.package_digest,
                grant_digest: result.grant_digest,
                stage: 0,
                decision: result.decision,
                reason: result.reason,
                parameter_digest: request.parameter_digest,
                approval_required: result.required_approval,
                cancelled: context.cancellation.is_cancelled(),
                redacted: true,
            })
            .map_err(|_| RecallDenied::Audit)?;
        if result.decision != Decision::Allow || context.cancellation.is_cancelled() {
            return Err(RecallDenied::Authorization);
        }
        Ok(gather_evidence(&self.store, input, limit))
    }
}

fn digest_json(value: &impl serde::Serialize) -> Digest {
    use sha2::{Digest as _, Sha256};
    Digest::from_bytes(
        Sha256::digest(serde_json::to_vec(value).expect("typed authority serializes")).into(),
    )
}
