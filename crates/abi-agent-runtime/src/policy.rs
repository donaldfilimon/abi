//! Authorization decisions and audit recording.
//!
//! A policy decides; it never acts. Nothing in this module runs a tool,
//! touches the filesystem, resolves a path, or opens a file — including the
//! audit sinks, which record in memory only. A host that wants a durable audit
//! log implements [`AuditSink`] itself, outside this crate.

use crate::tool::{ToolCall, ToolEffect, ToolSpec};
use std::sync::Mutex;

/// Verdict returned for one [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyDecision {
    /// The call may proceed without further checks.
    Allow,
    /// The call may proceed only after an explicit human confirmation.
    RequireConfirmation {
        /// Why confirmation is being asked for.
        reason: String,
    },
    /// The call is refused.
    Deny {
        /// Why the call was refused.
        reason: String,
    },
}

impl PolicyDecision {
    /// Refuse with `reason`.
    #[must_use]
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Ask for confirmation with `reason`.
    #[must_use]
    pub fn confirm(reason: impl Into<String>) -> Self {
        Self::RequireConfirmation {
            reason: reason.into(),
        }
    }

    /// Whether the call may proceed with no further interaction.
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Stable lowercase label, suitable for logs and audit records.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::RequireConfirmation { .. } => "require_confirmation",
            Self::Deny { .. } => "deny",
        }
    }
}

/// Authorization contract consulted before a host runs a tool.
///
/// The trait is object-safe so a host can swap policies at runtime. An
/// implementation must be a pure function of its inputs: it may not execute
/// anything, and it may not consult mutable external state through this call.
pub trait ExecutionPolicy: Send + Sync {
    /// Stable policy name, recorded in audit entries.
    fn name(&self) -> &str;

    /// Decide whether `call` may proceed.
    ///
    /// `spec` is `None` when the registry does not describe the named tool. A
    /// policy should treat an undescribed tool as unauthorized.
    fn authorize(&self, call: &ToolCall, spec: Option<&ToolSpec>) -> PolicyDecision;
}

/// A policy that refuses every call.
///
/// This is the safe default: a host that has not chosen a policy must not be
/// able to run tools by accident.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenyAllPolicy;

impl ExecutionPolicy for DenyAllPolicy {
    fn name(&self) -> &'static str {
        "deny-all"
    }

    fn authorize(&self, _call: &ToolCall, _spec: Option<&ToolSpec>) -> PolicyDecision {
        PolicyDecision::deny("deny-all policy refuses every tool call")
    }
}

/// A policy keyed on the tool's declared [`ToolEffect`].
///
/// Read-only tools are allowed, mutating tools require confirmation, and
/// destructive or undescribed tools are refused. The declared effect is the
/// tool author's claim; this policy trusts it and does not verify it.
#[derive(Clone, Copy, Debug, Default)]
pub struct EffectScopedPolicy;

impl ExecutionPolicy for EffectScopedPolicy {
    fn name(&self) -> &'static str {
        "effect-scoped"
    }

    fn authorize(&self, _call: &ToolCall, spec: Option<&ToolSpec>) -> PolicyDecision {
        match spec {
            None => PolicyDecision::deny("tool is not described by the registry"),
            Some(spec) => match spec.effect {
                ToolEffect::ReadOnly => PolicyDecision::Allow,
                ToolEffect::Mutating => PolicyDecision::confirm("tool declares a mutating effect"),
                ToolEffect::Destructive => {
                    PolicyDecision::deny("tool declares a destructive effect")
                }
            },
        }
    }
}

/// One authorization event, as handed to an [`AuditSink`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditEntry {
    /// Name of the policy that decided.
    pub policy: String,
    /// Correlation identifier of the call.
    pub call_id: String,
    /// Name of the tool the model asked for.
    pub tool: String,
    /// The verdict.
    pub decision: PolicyDecision,
}

impl AuditEntry {
    /// Pair a call with the verdict a named policy returned for it.
    #[must_use]
    pub fn new(policy: &str, call: &ToolCall, decision: PolicyDecision) -> Self {
        Self {
            policy: policy.to_owned(),
            call_id: call.id.clone(),
            tool: call.name.clone(),
            decision,
        }
    }
}

/// Destination for authorization events.
///
/// The trait is object-safe and takes `&self`, so a sink can be shared. This
/// crate ships only in-memory implementations; persistence is a host concern.
pub trait AuditSink: Send + Sync {
    /// Record one authorization event.
    fn record(&self, entry: &AuditEntry);
}

/// An [`AuditSink`] that discards every entry.
#[derive(Clone, Copy, Debug, Default)]
pub struct NullAuditSink;

impl AuditSink for NullAuditSink {
    fn record(&self, _entry: &AuditEntry) {}
}

/// An [`AuditSink`] that retains entries in memory, in arrival order.
///
/// Intended for tests and for hosts that forward the batch elsewhere. It has
/// no ceiling of its own, so a host that keeps one for a long-lived process is
/// responsible for draining it.
#[derive(Debug, Default)]
pub struct MemoryAuditSink {
    entries: Mutex<Vec<AuditEntry>>,
}

impl MemoryAuditSink {
    /// An empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot of the entries recorded so far.
    #[must_use]
    pub fn entries(&self) -> Vec<AuditEntry> {
        self.entries
            .lock()
            .map(|entries| entries.clone())
            .unwrap_or_default()
    }

    /// Number of entries recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.lock().map_or(0, |entries| entries.len())
    }

    /// Whether nothing has been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl AuditSink for MemoryAuditSink {
    fn record(&self, entry: &AuditEntry) {
        if let Ok(mut entries) = self.entries.lock() {
            entries.push(entry.clone());
        }
    }
}

/// Authorize `call` against `policy`, recording the verdict in `audit`.
///
/// This is the only place the two contracts are joined, and it still performs
/// no execution: the caller decides what to do with the returned verdict.
pub fn authorize_and_audit(
    policy: &dyn ExecutionPolicy,
    audit: &dyn AuditSink,
    call: &ToolCall,
    spec: Option<&ToolSpec>,
) -> PolicyDecision {
    let decision = policy.authorize(call, spec);
    audit.record(&AuditEntry::new(policy.name(), call, decision.clone()));
    decision
}

#[cfg(test)]
mod tests {
    use super::{
        AuditEntry, AuditSink, DenyAllPolicy, EffectScopedPolicy, ExecutionPolicy, MemoryAuditSink,
        NullAuditSink, PolicyDecision, authorize_and_audit,
    };
    use crate::tool::{ToolCall, ToolEffect, ToolSpec};

    #[test]
    fn deny_all_refuses_even_a_read_only_tool() {
        let policy = DenyAllPolicy;
        let call = ToolCall::new("c1", "read_file", "{}");
        let spec = ToolSpec::new("read_file");
        let decision = policy.authorize(&call, Some(&spec));
        assert!(!decision.is_allowed());
        assert_eq!(decision.as_str(), "deny");
        assert_eq!(policy.name(), "deny-all");
    }

    #[test]
    fn effect_scoped_policy_keys_on_the_declared_effect() {
        let policy = EffectScopedPolicy;
        let call = ToolCall::new("c1", "tool", "{}");

        let read_only = ToolSpec::new("tool").with_effect(ToolEffect::ReadOnly);
        assert_eq!(
            policy.authorize(&call, Some(&read_only)),
            PolicyDecision::Allow
        );

        let mutating = ToolSpec::new("tool").with_effect(ToolEffect::Mutating);
        assert_eq!(
            policy.authorize(&call, Some(&mutating)).as_str(),
            "require_confirmation"
        );

        let destructive = ToolSpec::new("tool").with_effect(ToolEffect::Destructive);
        assert!(!policy.authorize(&call, Some(&destructive)).is_allowed());

        // An undescribed tool is refused, never allowed by default.
        assert_eq!(
            policy.authorize(&call, None),
            PolicyDecision::deny("tool is not described by the registry")
        );
    }

    #[test]
    fn audit_records_arrive_in_order_with_the_policy_name() {
        let policy = EffectScopedPolicy;
        let audit = MemoryAuditSink::new();
        assert!(audit.is_empty());

        let read = ToolCall::new("c1", "read_file", "{}");
        let write = ToolCall::new("c2", "write_file", "{}");
        let read_spec = ToolSpec::new("read_file");
        let write_spec = ToolSpec::new("write_file").with_effect(ToolEffect::Mutating);

        assert!(authorize_and_audit(&policy, &audit, &read, Some(&read_spec)).is_allowed());
        assert!(!authorize_and_audit(&policy, &audit, &write, Some(&write_spec)).is_allowed());

        let entries = audit.entries();
        assert_eq!(audit.len(), 2);
        assert_eq!(entries[0].call_id, "c1");
        assert_eq!(entries[0].tool, "read_file");
        assert_eq!(entries[0].policy, "effect-scoped");
        assert_eq!(entries[0].decision, PolicyDecision::Allow);
        assert_eq!(entries[1].decision.as_str(), "require_confirmation");
    }

    #[test]
    fn null_sink_discards() {
        let sink = NullAuditSink;
        let call = ToolCall::new("c1", "read_file", "{}");
        sink.record(&AuditEntry::new("deny-all", &call, PolicyDecision::Allow));
    }

    #[test]
    fn decision_constructors_carry_their_reason() {
        assert_eq!(
            PolicyDecision::confirm("why"),
            PolicyDecision::RequireConfirmation {
                reason: "why".to_owned()
            }
        );
        assert_eq!(PolicyDecision::Allow.as_str(), "allow");
    }
}
