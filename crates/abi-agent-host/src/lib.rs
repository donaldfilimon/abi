//! Bounded orchestration across ABI's provider-neutral runtime contracts.
//!
//! `abi-agent-host` is the execution boundary deliberately absent from
//! `abi-agent-runtime`. A host is assembled from a model provider, a startup-
//! owned tool registry, an execution policy, an audit sink, and an object-safe
//! [`ToolExecutor`]. Model output may select only a registered tool and JSON
//! arguments; it cannot select an executable, environment, or workspace.
//!
//! Every call is resolved and JSON-Schema validated before policy evaluation.
//! Every policy decision is audited before allowed execution. Raw executor
//! output is rejected if it exceeds its ceiling, then represented as a bounded
//! [`ToolResult`](abi_agent_runtime::ToolResult) and fed back to the provider
//! through a correlated tool message. Duplicate call identifiers, provider-
//! fabricated tool results, post-terminal events, and recursive call loops fail
//! closed.
//!
//! Cancellation and all budgets are cooperative boundaries: a provider or
//! executor already inside one synchronous call cannot be pre-empted, but no
//! subsequent event, tool execution, or provider continuation is admitted
//! after the boundary is observed.

mod budget;
mod error;
mod executor;
mod host;
mod report;
mod schema;

pub use budget::{HostBudget, HostBudgetLimit};
pub use error::HostError;
pub use executor::{ToolExecutionContext, ToolExecutionError, ToolExecutor, ToolOutput};
pub use host::AgentHost;
pub use report::{HostRunReport, HostStopReason};
