//! Authenticated, bounded network gateway for the synchronous WDBX v2 facade.
//!
//! The gateway keeps network policy separate from [`abi_wdbx`]: requests are
//! validated and admitted before synchronous store work reaches a bounded
//! `spawn_blocking` executor. Mutation events contain metadata only.

mod auth;
mod config;
mod events;
mod executor;
mod server;
mod service;
mod tls;

#[allow(missing_docs, clippy::all, clippy::pedantic)]
pub mod proto {
    //! Generated gRPC protocol types.
    tonic::include_proto!("abi.wdbx.gateway.v1");
}

pub use auth::BearerToken;
pub use config::{GatewayConfig, GatewayError, Limits, TlsFiles};
pub use events::{EventHub, MutationNotice, validate_websocket_headers, websocket_router};
pub use executor::StoreExecutor;
pub use server::{BoundAddresses, PreparedGateway};
pub use service::GatewayService;
pub use tls::{grpc_tls_config, websocket_tls_config};
