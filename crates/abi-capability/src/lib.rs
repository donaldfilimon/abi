//! Deny-by-default capability authorization for local recording adapters.
//!
//! This crate deliberately has no production actuator. Inputs are request scoped,
//! decisions and reasons are closed, audit persistence is fallible and bounded,
//! and all observable records contain digests rather than user material.

mod audit;
mod change;
mod credential;
mod kernel;
mod recording;
mod registry;
mod types;

pub use audit::{AuditError, AuditRecord, AuditSink, BoundedMemoryAuditSink};
pub use change::{ChangeApproval, ChangeSet, CompensationClass};
pub use credential::TenantCredentialResolver;
pub use kernel::{AuthorizationContext, Kernel, KernelError, KernelOutcome};
pub use recording::{RecordedEffect, RecordingActuator};
pub use registry::{CapabilityRegistry, RegistryError};
pub use types::{
    Approval, ApprovalLevel, ApprovalState, CapabilityPackage, Clock, CredentialRef, Decision,
    Digest, FixedClock, Grant, PlatformFacts, Principal, PrincipalKind, ReasonCode, Receipt,
    Request, Reversibility, RevocationState, RiskClass, Scope, SideEffect,
};
