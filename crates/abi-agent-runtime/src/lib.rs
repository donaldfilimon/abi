//! Provider-neutral agent runtime **contracts** for ABI.
//!
//! This crate defines the interfaces an agent host needs — a model provider, a
//! typed event stream, tool description, authorization, audit, metering, and
//! cooperative cancellation — plus two deterministic providers so downstream
//! code can be tested without a network.
//!
//! # What this crate deliberately does not own
//!
//! It is a contracts crate, and the negative space is the point:
//!
//! - **No model.** No weights, no tokenizer, no prompt templates, no model
//!   catalogue or default model. [`ModelRequest::model`] is an opaque string.
//! - **No tool behavior.** [`ToolSpec`] describes tools and [`ToolRegistry`]
//!   catalogues them; nothing here invokes one.
//! - **No shell, filesystem, or network.** Not in the providers, not in the
//!   policies, not in the audit sinks. [`MemoryAuditSink`] records in memory;
//!   persistence is the host's job.
//! - **No memory or persistence.** Nothing is stored across runs.
//! - **No pricing.** [`Usage`] carries token counts and elapsed time only.
//!   This crate never converts usage into money, and the token counts it
//!   reports are the provider's claim, not a measurement it can verify.
//! - **No dependencies.** `[dependencies]` is empty — `std` only — and nothing
//!   here depends on Abbey or on any Abbey type. Adding a dependency to this
//!   crate should require a reason that survives review.
//! - **No `unsafe`.** The workspace denies it and this crate has none.
//!
//! # The three enforced guarantees
//!
//! [`RunContext::checkpoint`] is the single cancellation/budget checkpoint;
//! [`RunContext::emit`] calls it before delivery, and bounded blocking adapters
//! may call it directly from a polling loop. Capture stays on the event path:
//!
//! - **Streaming** — output arrives as ordered [`ModelEvent`] values, with tool
//!   use carried as structured [`ToolCall`] items rather than text the caller
//!   must parse. Exactly one [`ModelEvent::Finished`] is delivered, last, even
//!   when the provider fails.
//! - **Cancellation** — a [`CancellationToken`] stops the run at the next
//!   provider checkpoint. It is cooperative and cannot interrupt a provider
//!   step unless that adapter polls. A [`RunBudget`] is enforced at the same
//!   checkpoints.
//! - **Bounded capture** — the run's retained transcript stops at
//!   [`MAX_CAPTURED_BYTES`], truncating on a `UTF-8` character boundary and
//!   disclosing the loss through [`CapturedText::truncated`]. The event itself
//!   still reaches the sink in full; only the retained copy is capped.
//!
//! # Example
//!
//! ```
//! use abi_agent_runtime::{
//!     CancellationToken, CollectingSink, EchoProvider, ModelRequest, RunBudget, run_provider,
//! };
//!
//! let provider = EchoProvider::new();
//! let request = ModelRequest::new("test-model").with_user("hello there");
//! let mut sink = CollectingSink::new();
//! let cancel = CancellationToken::new();
//!
//! let report = run_provider(
//!     &provider,
//!     &request,
//!     &mut sink,
//!     &cancel,
//!     RunBudget::unlimited(),
//! )?;
//!
//! assert_eq!(report.text(), "hello there");
//! assert_eq!(report.usage.output_tokens, 2);
//! assert_eq!(sink.kinds(), vec!["started", "text_delta", "text_delta", "finished"]);
//! # Ok::<(), abi_agent_runtime::RuntimeError>(())
//! ```

mod cancel;
mod capture;
mod error;
mod event;
mod policy;
mod provider;
mod request;
mod run;
mod testing;
mod tool;
mod usage;

pub use cancel::CancellationToken;
pub use capture::{CapturedText, MAX_CAPTURED_BYTES};
pub use error::RuntimeError;
pub use event::{CollectingSink, EventSink, ModelEvent, NullSink, StopReason};
pub use policy::{
    AuditEntry, AuditSink, DenyAllPolicy, EffectScopedPolicy, ExecutionPolicy, MemoryAuditSink,
    NullAuditSink, PolicyDecision, authorize_and_audit,
};
pub use provider::ModelProvider;
pub use request::{Message, ModelRequest, Role};
pub use run::{Flow, RunContext, RunReport, run_provider};
pub use testing::{ECHO_PROVIDER_ID, EchoProvider, ScriptedProvider};
pub use tool::{
    StaticToolRegistry, ToolCall, ToolEffect, ToolRegistry, ToolResult, ToolSpec, ToolStatus,
};
pub use usage::{BudgetLimit, RunBudget, Usage};
