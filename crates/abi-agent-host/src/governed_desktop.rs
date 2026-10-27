//! Governed, single-shot desktop application activation (macOS).
//!
//! The approval binds one exact reverse-DNS bundle identifier: the compiled
//! source's resource digest must commit to it, and the executed call input must
//! equal `{"bundle_id": <approved>}` after JSON parsing. The executor first
//! reads the frontmost application (a backend that is unsupported or denied
//! refuses here, before any effect), then calls [`DesktopBackend::activate`] at
//! most once and verifies the postcondition `desktop.frontmost_matches` by
//! reading the frontmost application again (at most 20 polls, 100 ms apart).
//! An unverified activation is reported as a failure and never retried.
//!
//! This adapter activates applications only. Screen capture, pointer and text
//! input are not part of it, and a local pass is not signed-distribution or
//! other-platform evidence.
use crate::{GovernedLiveExecutor, ToolExecutionContext, ToolExecutionError, ToolOutput};
use abi_agent_runtime::ToolCall;
use abi_capability::{
    ApprovalLevel, Clock, Digest, Request, SideEffect,
    lifecycle::PreparedExecution,
    source::{AdapterBinding, CompiledSource, PackageSource, ResidencyPolicy},
};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest as _, Sha256};
use std::{sync::Arc, thread, time::Duration};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "macos")]
pub use macos::MacDesktop;
#[cfg(not(target_os = "macos"))]
pub use unsupported::MacDesktop;

const TOOL: &str = "desktop_activate";
const SOURCE_ID: &str = "abi.desktop.activate";
const SOURCE_VERSION: &str = "1.0.0";
const POSTCONDITION: &str = "desktop.frontmost_matches";
const MAX_POLLS: u32 = 20;
const POLL_INTERVAL: Duration = Duration::from_millis(100);

/// Why a desktop backend could not observe or act.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DesktopError {
    /// This platform has no desktop backend.
    Unsupported,
    /// The operating system refused the observation or action.
    PermissionDenied,
    /// The backend ran but did not produce a usable result.
    Failed,
}

/// Observes and activates desktop applications.
pub trait DesktopBackend {
    /// Bundle identifier of the frontmost application.
    ///
    /// # Errors
    ///
    /// A [`DesktopError`] when the frontmost application cannot be read.
    fn frontmost_bundle_id(&mut self) -> Result<String, DesktopError>;
    /// Ask the system to activate `bundle_id`. Never retries internally.
    ///
    /// # Errors
    ///
    /// A [`DesktopError`] when the activation request fails.
    fn activate(&mut self, bundle_id: &str) -> Result<(), DesktopError>;
}

/// True for a reverse-DNS bundle identifier: 3..=255 bytes of `[A-Za-z0-9.-]`,
/// at least one dot, every dot-separated label non-empty and starting with an
/// ASCII alphanumeric (so it can never be read as a command-line option).
#[must_use]
pub fn valid_bundle_id(value: &str) -> bool {
    (3..=255).contains(&value.len())
        && value.contains('.')
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
        && value.split('.').all(|label| {
            label
                .bytes()
                .next()
                .is_some_and(|b| b.is_ascii_alphanumeric())
        })
}

/// Versioned resource commitment the compiled source must carry for `bundle_id`.
#[must_use]
pub fn resource_digest(bundle_id: &str) -> Digest {
    commitment(&("abi.desktop.approved.v1", input_value(bundle_id)))
}

/// Sealed activation executor bound to one approved bundle and compiled source.
pub struct GovernedDesktopExecutor<B: DesktopBackend> {
    approved: String,
    backend: B,
    source: PackageSource,
    package: Digest,
    schema: Digest,
    clock: Arc<dyn Clock>,
    attempted: bool,
    activations: u32,
}

impl<B: DesktopBackend> GovernedDesktopExecutor<B> {
    /// Bind `approved_bundle_id` to one exact compiled source and trusted clock.
    ///
    /// # Errors
    ///
    /// `desktop_source_mismatch` when the bundle identifier is malformed, or the
    /// source does not declare this capability exactly or does not commit to it.
    pub fn bind(
        source: &CompiledSource,
        approved_bundle_id: String,
        backend: B,
        clock: Arc<dyn Clock>,
    ) -> Result<Self, ToolExecutionError> {
        let src = source.source();
        let output = output(&approved_bundle_id);
        if !valid_bundle_id(&approved_bundle_id)
            || src.id != SOURCE_ID
            || src.version != SOURCE_VERSION
            || src.resource_scope.resource_digest != resource_digest(&approved_bundle_id)
            || src.adapter_bindings != [adapter()]
            || src.input_schema != input_schema()
            || src.output_schema != output_schema()
            || src.side_effect != SideEffect::PlatformWrite
            || src.residency != ResidencyPolicy::LocalOnly
            || src.approval_policy < ApprovalLevel::A1Actor
            || src.confirmation_policy < ApprovalLevel::A1Actor
            || src.budgets.max_calls != 1
            || (src.budgets.max_input_bytes as usize)
                < input_value(&approved_bundle_id).to_string().len()
            || (src.budgets.max_output_bytes as usize) < output.len()
            || src.postconditions.len() != 1
            || !src.postconditions.contains(POSTCONDITION)
            || !src.receipt_requirements.require_postcondition_evidence
        {
            return Err(error("desktop_source_mismatch"));
        }
        Ok(Self {
            approved: approved_bundle_id,
            backend,
            source: src.clone(),
            package: source.package().digest,
            schema: source.schema_digest(),
            clock,
            attempted: false,
            activations: 0,
        })
    }

    /// How many times the backend has been asked to activate (0 or 1).
    #[must_use]
    pub const fn activations(&self) -> u32 {
        self.activations
    }

    fn bound(
        &self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
    ) -> bool {
        let input = input_value(&self.approved);
        let id = commitment(&(
            "abi.host.execution.v2",
            &request.attempt_id,
            &request.principal,
            &request.scope,
            &request.capability_id,
            &request.capability_version,
            &call.id,
            &call.name,
        ));
        let binding = prepared.binding();
        request.computed_digest() == request.request_digest
            && request.package_digest == self.package
            && request.scope == self.source.resource_scope
            && request.capability_id == self.source.id
            && request.capability_version == self.source.version
            && request.parameter_digest == commitment(&input)
            && source.source() == &self.source
            && source.package().digest == self.package
            && source.schema_digest() == self.schema
            && call.name == TOOL
            && !call.id.is_empty()
            && call.id.len() <= 128
            && call
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._:-".contains(&b))
            && call.input.len() <= self.source.budgets.max_input_bytes as usize
            && serde_json::from_str::<serde_json::Value>(&call.input).ok() == Some(input)
            && prepared.execution_id() == id
            && request.call_id == id.to_string().trim_start_matches("sha256:")
            && binding.request_digest == request.request_digest
            && binding.package_digest == self.package
            && binding.schema_digest == self.schema
            && binding.principal_digest == commitment(&request.principal)
            && binding.scope_digest == commitment(&request.scope)
            && binding.compensation_origin.is_none()
            && binding.required_approval >= ApprovalLevel::A1Actor
            && binding.required_approval
                >= self
                    .source
                    .approval_policy
                    .max(self.source.confirmation_policy)
            && binding.policy_digest
                == commitment(&(
                    "abi.policy.v1",
                    &request.policy_version,
                    &request.guild_constitution_version,
                    &request.safety_policy_version,
                    request.revocation_epoch,
                ))
    }

    fn live(&self, context: ToolExecutionContext<'_>) -> bool {
        self.clock.now_ms() < self.source.expiration_criteria.expires_at_ms
            && !context.should_stop()
    }

    /// Poll the frontmost application until it is the approved bundle.
    fn verify_frontmost(&mut self, context: ToolExecutionContext<'_>) -> bool {
        for poll in 0..MAX_POLLS {
            if poll > 0 {
                if context.should_stop() {
                    return false;
                }
                thread::sleep(POLL_INTERVAL);
            }
            if self
                .backend
                .frontmost_bundle_id()
                .is_ok_and(|front| front == self.approved)
            {
                return true;
            }
        }
        false
    }
}

impl<B: DesktopBackend> GovernedLiveExecutor for GovernedDesktopExecutor<B> {
    fn binding(&self) -> AdapterBinding {
        adapter()
    }
    fn execute(
        &mut self,
        call: &ToolCall,
        request: &Request,
        source: &CompiledSource,
        prepared: &PreparedExecution,
        context: ToolExecutionContext<'_>,
    ) -> Result<ToolOutput, ToolExecutionError> {
        if self.attempted {
            return Err(error("desktop_already_attempted"));
        }
        if !self.bound(call, request, source, prepared) || context.should_stop() {
            return Err(error("desktop_binding_refused"));
        }
        if !self.live(context) {
            return Err(error("desktop_stopped"));
        }
        // Probe before the effect: an unusable backend refuses with no activation.
        self.backend.frontmost_bundle_id().map_err(backend_error)?;
        if !self.live(context) {
            return Err(error("desktop_stopped"));
        }
        // Consume before the effect: whatever happens next, never again.
        self.attempted = true;
        self.activations += 1;
        self.backend.activate(&self.approved).map_err(|failure| {
            if failure == DesktopError::Failed {
                error("desktop_activation_uncertain")
            } else {
                backend_error(failure)
            }
        })?;
        if !self.verify_frontmost(context) {
            return Err(error("desktop_frontmost_unverified"));
        }
        let payload = output(&self.approved);
        if payload.len() > self.source.budgets.max_output_bytes as usize {
            return Err(error("desktop_output_limit"));
        }
        Ok(ToolOutput::ok(payload))
    }
}

fn backend_error(failure: DesktopError) -> ToolExecutionError {
    error(match failure {
        DesktopError::Unsupported => "desktop_unsupported",
        DesktopError::PermissionDenied => "desktop_permission_denied",
        DesktopError::Failed => "desktop_observation_failed",
    })
}
fn input_value(bundle_id: &str) -> serde_json::Value {
    json!({ "bundle_id": bundle_id })
}
fn output(bundle_id: &str) -> String {
    json!({ "bundle_id": bundle_id, "frontmost": true }).to_string()
}
fn adapter() -> AdapterBinding {
    AdapterBinding {
        adapter_id: "abi.desktop".into(),
        api_version: "1".into(),
    }
}
fn input_schema() -> serde_json::Value {
    json!({
        "type":"object",
        "properties":{"bundle_id":{"type":"string"}},
        "required":["bundle_id"],
        "additionalProperties":false
    })
}
fn output_schema() -> serde_json::Value {
    json!({
        "type":"object",
        "properties":{"bundle_id":{"type":"string"},"frontmost":{"const":true}},
        "required":["bundle_id","frontmost"],
        "additionalProperties":false
    })
}
fn error(code: &'static str) -> ToolExecutionError {
    ToolExecutionError::new(code)
}
fn commitment<T: Serialize>(value: &T) -> Digest {
    Digest::from_bytes(
        Sha256::digest(serde_json::to_vec(value).expect("closed desktop commitment")).into(),
    )
}

#[cfg(not(target_os = "macos"))]
mod unsupported {
    use super::{DesktopBackend, DesktopError};

    /// Desktop backend stub for platforms without a supported activation path.
    #[derive(Debug, Default)]
    pub struct MacDesktop;
    impl MacDesktop {
        /// A backend that refuses every observation and action as unsupported.
        #[must_use]
        pub const fn new() -> Self {
            Self
        }
    }
    impl DesktopBackend for MacDesktop {
        fn frontmost_bundle_id(&mut self) -> Result<String, DesktopError> {
            Err(DesktopError::Unsupported)
        }
        fn activate(&mut self, _bundle_id: &str) -> Result<(), DesktopError> {
            Err(DesktopError::Unsupported)
        }
    }
}
