//! Tool description, invocation, and result vocabulary.
//!
//! This module describes tools; it never runs one. There is no shell, no
//! filesystem access, and no process spawning anywhere in this crate — a host
//! that wants tool execution implements it outside and reports the outcome
//! back as a [`ToolResult`].

use crate::capture::{CapturedText, MAX_CAPTURED_BYTES};

/// Declared blast radius of a tool, used by an [`ExecutionPolicy`] to decide
/// what a call is allowed to do.
///
/// The declaration is the tool author's claim about the tool, not a property
/// this crate can verify.
///
/// [`ExecutionPolicy`]: crate::ExecutionPolicy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ToolEffect {
    /// Observes state without changing it.
    #[default]
    ReadOnly,
    /// Changes state in a way the caller could undo.
    Mutating,
    /// Changes state irreversibly.
    Destructive,
}

impl ToolEffect {
    /// Stable lowercase label, suitable for logs and audit records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read_only",
            Self::Mutating => "mutating",
            Self::Destructive => "destructive",
        }
    }
}

/// Description of one tool offered to a model.
///
/// `input_schema` is an opaque string. This crate does not parse, validate, or
/// depend on any `JSON` library — the host owns schema semantics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolSpec {
    /// Unique tool name as the model will call it.
    pub name: String,
    /// Human-readable description passed through to the provider.
    pub description: String,
    /// Opaque input-schema text, carried verbatim.
    pub input_schema: String,
    /// Declared effect used for authorization decisions.
    pub effect: ToolEffect,
}

impl ToolSpec {
    /// Describe a read-only tool with an empty schema.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            input_schema: String::new(),
            effect: ToolEffect::ReadOnly,
        }
    }

    /// Attach a description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Attach opaque schema text.
    #[must_use]
    pub fn with_input_schema(mut self, schema: impl Into<String>) -> Self {
        self.input_schema = schema.into();
        self
    }

    /// Declare the tool's effect.
    #[must_use]
    pub fn with_effect(mut self, effect: ToolEffect) -> Self {
        self.effect = effect;
        self
    }
}

/// A structured tool invocation requested by a model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolCall {
    /// Correlation identifier chosen by the provider.
    pub id: String,
    /// Name of the tool being invoked.
    pub name: String,
    /// Opaque argument text, carried verbatim.
    pub input: String,
}

impl ToolCall {
    /// Build a call for `name` with opaque `input`.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            input: input.into(),
        }
    }
}

/// Terminal state of a tool invocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    /// The host ran the tool and it succeeded.
    Ok,
    /// The host ran the tool and it failed.
    Error,
    /// An [`ExecutionPolicy`] refused the call, so it never ran.
    ///
    /// [`ExecutionPolicy`]: crate::ExecutionPolicy
    Denied,
}

impl ToolStatus {
    /// Stable lowercase label, suitable for logs and audit records.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Error => "error",
            Self::Denied => "denied",
        }
    }
}

/// Outcome of a tool invocation, with its payload bounded on capture.
///
/// The payload uses the same [`MAX_CAPTURED_BYTES`] ceiling as streamed text,
/// so a tool that returns a very large blob cannot grow the transcript without
/// bound. Truncation is disclosed through [`CapturedText::truncated`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolResult {
    /// Identifier of the [`ToolCall`] this answers.
    pub call_id: String,
    /// Terminal state of the invocation.
    pub status: ToolStatus,
    /// Bounded payload text.
    pub payload: CapturedText,
}

impl ToolResult {
    /// Record an outcome, capturing `payload` under [`MAX_CAPTURED_BYTES`].
    #[must_use]
    pub fn new(call_id: impl Into<String>, status: ToolStatus, payload: &str) -> Self {
        Self::with_limit(call_id, status, payload, MAX_CAPTURED_BYTES)
    }

    /// Record an outcome, capturing `payload` under an explicit ceiling.
    #[must_use]
    pub fn with_limit(
        call_id: impl Into<String>,
        status: ToolStatus,
        payload: &str,
        limit: usize,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            status,
            payload: CapturedText::capture_with_limit(payload, limit),
        }
    }

    /// Record a successful invocation.
    #[must_use]
    pub fn ok(call_id: impl Into<String>, payload: &str) -> Self {
        Self::new(call_id, ToolStatus::Ok, payload)
    }

    /// Record a failed invocation.
    #[must_use]
    pub fn error(call_id: impl Into<String>, payload: &str) -> Self {
        Self::new(call_id, ToolStatus::Error, payload)
    }

    /// Record a refusal, carrying the policy's stated reason.
    #[must_use]
    pub fn denied(call_id: impl Into<String>, reason: &str) -> Self {
        Self::new(call_id, ToolStatus::Denied, reason)
    }
}

/// Read-only catalogue of the tools a host offers.
///
/// The trait is object-safe. It deliberately exposes no `invoke` method:
/// looking a tool up and running one are different responsibilities, and only
/// the first belongs here.
pub trait ToolRegistry: Send + Sync {
    /// Every tool in the catalogue, in a stable order.
    fn specs(&self) -> Vec<ToolSpec>;

    /// Look one tool up by name.
    fn spec(&self, name: &str) -> Option<ToolSpec> {
        self.specs().into_iter().find(|spec| spec.name == name)
    }

    /// Number of tools in the catalogue.
    fn len(&self) -> usize {
        self.specs().len()
    }

    /// Whether the catalogue is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A fixed, in-memory [`ToolRegistry`].
///
/// It stores descriptions and returns them in insertion order. It holds no
/// handlers, so nothing here can execute.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StaticToolRegistry {
    specs: Vec<ToolSpec>,
}

impl StaticToolRegistry {
    /// An empty catalogue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one tool description, preserving insertion order.
    #[must_use]
    pub fn with_spec(mut self, spec: ToolSpec) -> Self {
        self.specs.push(spec);
        self
    }
}

impl FromIterator<ToolSpec> for StaticToolRegistry {
    fn from_iter<I: IntoIterator<Item = ToolSpec>>(iter: I) -> Self {
        Self {
            specs: iter.into_iter().collect(),
        }
    }
}

impl ToolRegistry for StaticToolRegistry {
    fn specs(&self) -> Vec<ToolSpec> {
        self.specs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        StaticToolRegistry, ToolCall, ToolEffect, ToolRegistry, ToolResult, ToolSpec, ToolStatus,
    };

    fn registry() -> StaticToolRegistry {
        StaticToolRegistry::new()
            .with_spec(ToolSpec::new("read_file").with_description("read a file"))
            .with_spec(ToolSpec::new("write_file").with_effect(ToolEffect::Mutating))
    }

    #[test]
    fn registry_preserves_insertion_order() {
        let names: Vec<String> = registry()
            .specs()
            .into_iter()
            .map(|spec| spec.name)
            .collect();
        assert_eq!(names, vec!["read_file".to_owned(), "write_file".to_owned()]);
    }

    #[test]
    fn registry_lookup_matches_by_name() {
        let registry = registry();
        assert_eq!(
            registry.spec("write_file").map(|spec| spec.effect),
            Some(ToolEffect::Mutating)
        );
        assert_eq!(registry.spec("shell"), None);
        assert_eq!(registry.len(), 2);
        assert!(!registry.is_empty());
        assert!(StaticToolRegistry::new().is_empty());
    }

    #[test]
    fn registry_collects_from_an_iterator() {
        let registry: StaticToolRegistry = vec![ToolSpec::new("a"), ToolSpec::new("b")]
            .into_iter()
            .collect();
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn spec_builders_set_every_field() {
        let spec = ToolSpec::new("grep")
            .with_description("search")
            .with_input_schema("{\"type\":\"object\"}")
            .with_effect(ToolEffect::ReadOnly);
        assert_eq!(spec.name, "grep");
        assert_eq!(spec.description, "search");
        assert_eq!(spec.input_schema, "{\"type\":\"object\"}");
        assert_eq!(spec.effect, ToolEffect::ReadOnly);
        assert_eq!(ToolEffect::default(), ToolEffect::ReadOnly);
    }

    #[test]
    fn call_carries_opaque_input() {
        let call = ToolCall::new("c1", "grep", "{\"pattern\":\"x\"}");
        assert_eq!(call.id, "c1");
        assert_eq!(call.name, "grep");
        assert_eq!(call.input, "{\"pattern\":\"x\"}");
    }

    #[test]
    fn results_bound_their_payload() {
        let long = "x".repeat(64);
        let result = ToolResult::with_limit("c1", ToolStatus::Ok, &long, 8);
        assert_eq!(result.payload.text(), "xxxxxxxx");
        assert!(result.payload.truncated());
        assert_eq!(result.payload.dropped_bytes(), 56);

        assert_eq!(ToolResult::ok("c1", "fine").status, ToolStatus::Ok);
        assert_eq!(ToolResult::error("c1", "boom").status, ToolStatus::Error);
        let denied = ToolResult::denied("c1", "policy denies destructive tools");
        assert_eq!(denied.status, ToolStatus::Denied);
        assert_eq!(denied.payload.text(), "policy denies destructive tools");
    }

    #[test]
    fn labels_are_stable() {
        assert_eq!(ToolEffect::ReadOnly.as_str(), "read_only");
        assert_eq!(ToolEffect::Mutating.as_str(), "mutating");
        assert_eq!(ToolEffect::Destructive.as_str(), "destructive");
        assert_eq!(ToolStatus::Ok.as_str(), "ok");
        assert_eq!(ToolStatus::Error.as_str(), "error");
        assert_eq!(ToolStatus::Denied.as_str(), "denied");
    }
}
