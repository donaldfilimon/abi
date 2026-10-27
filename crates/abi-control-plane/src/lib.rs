//! Durable tenant authority for public accounts and user-owned node enrollment.

mod authority;
mod metrics;
pub use metrics::OperatorMetrics;
mod enrollment;
mod oidc;
mod private;
mod service;
mod types;
pub use oidc::OidcConfig;
pub use service::{router, router_with_limits, router_with_operator};
mod operator_http;

pub use authority::*;
pub use types::*;

mod relay_crypto;
pub use relay_crypto::{RelayKey, SealedPayload, seal};

mod relay_types;
pub use relay_types::*;

mod relay;
pub use relay::Relay;

mod relay_runtime;
pub use relay_runtime::{InferenceJob, RelayRuntime};

mod relay_http;

mod relay_client;
pub use relay_client::{RelayClient, prepare_job, prepare_tool_job};

mod relay_stream;
pub use relay_stream::StreamFrame;
mod relay_stream_client;

mod delivery_types;
pub use delivery_types::*;

mod relay_recovery;

mod delivery_outbox;
pub use delivery_outbox::{FileRelayOutbox, OutboxEntry, OutboxLimits};
mod delivery_consumer;
pub use delivery_consumer::VerifiedDeliveryStore;
pub use relay_runtime::DurableRelayRuntime;

mod relay_recovery_client;
pub use relay_recovery_client::{RecoverySummary, reconnect_delay};

pub use relay_runtime::{GovernedToolContext, GovernedToolOutcome, ToolJob};
