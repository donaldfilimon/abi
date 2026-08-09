//! The model-provider contract, plus the object-safety assertions that keep
//! every contract in this crate usable as `dyn Trait`.

use crate::{error::RuntimeError, request::ModelRequest, run::RunContext};

/// A source of model output.
///
/// An implementation translates a [`ModelRequest`] into a stream of events
/// written through [`RunContext::emit`]. It must honour the [`Flow`] value that
/// `emit` returns: once told to stop, it returns `Ok(())` promptly rather than
/// continuing to generate.
///
/// This crate contains no adapter that talks to a network. The two providers
/// it ships ([`EchoProvider`] and [`ScriptedProvider`]) are deterministic
/// fixtures so downstream code can be tested without one.
///
/// [`Flow`]: crate::Flow
/// [`EchoProvider`]: crate::EchoProvider
/// [`ScriptedProvider`]: crate::ScriptedProvider
pub trait ModelProvider: Send + Sync {
    /// Stable provider identifier, used in errors and logs.
    fn id(&self) -> &str;

    /// Produce output for `request`, writing events through `run`.
    ///
    /// Returning `Ok(())` means the provider finished or was told to stop;
    /// [`RunContext`] decides which of those it was. Returning `Err` means no
    /// usable answer was produced — a provider must never fabricate one.
    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError>;
}

/// Compile-time proof that the crate's contracts are `dyn`-compatible.
///
/// Each line fails to compile if the corresponding trait ever gains a
/// generic method, a `Self`-returning method, or another object-safety
/// violation — the failure lands on the trait definition, not on some distant
/// caller.
mod dyn_compatible {
    use crate::{
        event::EventSink,
        policy::{AuditSink, ExecutionPolicy},
        provider::ModelProvider,
        tool::ToolRegistry,
    };

    const _: Option<&dyn ModelProvider> = None;
    const _: Option<&dyn EventSink> = None;
    const _: Option<&dyn ToolRegistry> = None;
    const _: Option<&dyn ExecutionPolicy> = None;
    const _: Option<&dyn AuditSink> = None;
}

#[cfg(test)]
mod tests {
    use crate::{
        event::{CollectingSink, EventSink},
        policy::{AuditSink, DenyAllPolicy, ExecutionPolicy, MemoryAuditSink},
        provider::ModelProvider,
        testing::{EchoProvider, ScriptedProvider},
        tool::{StaticToolRegistry, ToolRegistry, ToolSpec},
    };

    #[test]
    fn every_contract_boxes_as_a_trait_object() {
        let providers: Vec<Box<dyn ModelProvider>> = vec![
            Box::new(EchoProvider::new()),
            Box::new(ScriptedProvider::new("scripted")),
        ];
        assert_eq!(
            providers.iter().map(|p| p.id()).collect::<Vec<_>>(),
            vec!["echo", "scripted"]
        );

        let sink: Box<dyn EventSink> = Box::new(CollectingSink::new());
        drop(sink);

        let registry: Box<dyn ToolRegistry> =
            Box::new(StaticToolRegistry::new().with_spec(ToolSpec::new("read_file")));
        assert_eq!(registry.len(), 1);

        let policy: Box<dyn ExecutionPolicy> = Box::new(DenyAllPolicy);
        assert_eq!(policy.name(), "deny-all");

        let audit: Box<dyn AuditSink> = Box::new(MemoryAuditSink::new());
        drop(audit);
    }
}
