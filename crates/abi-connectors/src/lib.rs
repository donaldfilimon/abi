//! External-service connectors for the ABI framework.
//!
//! The Rust successor to `src/connectors/`. Covers `OpenAI`, Anthropic, Grok,
//! Discord and Twilio.
//!
//! ## The local/live split
//!
//! Every connector has two transports, and the distinction is a safety property
//! rather than a convenience:
//!
//! - **Local** ([`TransportMode::Local`], the default) synthesizes a deterministic
//!   response with no network call. This is what makes the connector tests
//!   hermetic and what `abi complete` uses without credentials.
//! - **Live** ([`TransportMode::Live`]) issues a real request.
//!
//! A *local* call on a live-configured client returns
//! [`ConnectorError::LiveTransportUnavailable`] rather than synthesizing. That
//! asymmetry is deliberate and carried over from Zig: a synthesized response
//! returned in place of a real one is indistinguishable downstream, so it would
//! look like the provider answered.
//!
//! ## Security invariants
//!
//! - A live `base_url` must be HTTPS, or loopback. Enforced by
//!   [`url::require_https_base_url`], which checks the loopback prefix ends at a
//!   real host boundary — `http://127.0.0.1.evil.com` must not pass.
//!   [`url::join_url`] re-checks, so a URL that would leak a key cannot be built.
//! - The live transport does not follow redirects. A redirect from an HTTPS base
//!   URL to an `http://` location would send the API key in cleartext.
//!
//! ## Improvement over the Zig structure
//!
//! The Zig helpers called `std.http.Client` directly, so nothing about the live
//! path was testable without a network. [`transport::Transport`] is a trait, and
//! [`transport::RecordingTransport`] asserts on the exact URL, headers and body a
//! call *would* send — see the provider tests, which pin Anthropic's
//! `x-api-key` + `anthropic-version` pair and Twilio's `%2B` phone-number
//! encoding.
//!
//! Also gone: `connectors/json.zig` carried a duplicate of the foundation JSON
//! escaper, with a comment explaining that the connectors compiled as an isolated
//! module root and could not import it. Cargo has no such constraint, so
//! [`payload`] uses [`abi_foundation::json`] and the "keep these two tables
//! synchronized" hazard is gone.

pub mod connector;
pub mod payload;
pub mod providers;
pub mod sse;
pub mod transport;
pub mod url;

pub use connector::{ConnectorConfig, ConnectorError, Response, Result, TransportMode};
pub use providers::{Client, DiscordClient, Provider, TwilioClient};
pub use sse::{StreamChunk, collect_stream, parse_stream};
pub use transport::{
    DefaultTransport, Header, Method, RecordingTransport, Request, Transport, UnavailableTransport,
};

#[cfg(feature = "live")]
pub use transport::HttpTransport;
