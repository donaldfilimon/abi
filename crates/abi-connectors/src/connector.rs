//! Shared connector types: errors, transport mode, configuration, responses.
//!
//! Ported from `src/connectors/connector.zig`.

use std::fmt;

/// Why a connector call failed.
///
/// Variant names are preserved from the Zig error set because they surface in
/// `connector_test` MCP output and in `abi auth status`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectorError {
    /// The response could not be read.
    ReadFailed,
    /// The connection could not be established or was lost.
    ConnectionFailed,
    /// A live `base_url` is not HTTPS (cleartext or an unknown scheme).
    ///
    /// Fatal on purpose: sending an API key over cleartext HTTP would leak it.
    InsecureBaseUrl,
    /// Credentials were absent or rejected (HTTP 401/403).
    AuthenticationError,
    /// The provider applied rate limiting (HTTP 429).
    RateLimited,
    /// The response was not the expected shape.
    InvalidResponse,
    /// The request exceeded its deadline (HTTP 408/504).
    Timeout,
    /// A live call was attempted but the live transport is not available.
    LiveTransportUnavailable,
    /// On-device Apple `FoundationModels` is not reachable.
    ///
    /// Either this build lacks the feature, the host is not macOS, or the model
    /// runtime is unavailable (Apple Intelligence off, model not downloaded, or
    /// the OS is too old).
    FmUnavailable,
    /// The `FoundationModels` session was reachable but failed while responding.
    FmSessionFailed,
}

impl fmt::Display for ConnectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::ReadFailed => "failed to read response",
            Self::ConnectionFailed => "connection failed",
            Self::InsecureBaseUrl => {
                "live base_url must use https:// (or be loopback) so the API key is not sent in cleartext"
            }
            Self::AuthenticationError => "authentication failed",
            Self::RateLimited => "rate limited by the provider",
            Self::InvalidResponse => "invalid response",
            Self::Timeout => "request timed out",
            Self::LiveTransportUnavailable => "live transport unavailable",
            Self::FmUnavailable => "on-device FoundationModels is unavailable",
            Self::FmSessionFailed => "the FoundationModels session failed",
        };
        f.write_str(text)
    }
}

impl std::error::Error for ConnectorError {}

/// Shorthand for a connector result.
pub type Result<T> = std::result::Result<T, ConnectorError>;

/// Which transport a connector call uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransportMode {
    /// Synthesize a deterministic response without any network call.
    ///
    /// The default, and what makes the connector tests hermetic.
    #[default]
    Local,
    /// Perform a real HTTP request against the provider.
    Live,
}

impl TransportMode {
    /// The lowercase name reported in CLI and MCP output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Live => "live",
        }
    }
}

impl fmt::Display for TransportMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Per-connector configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectorConfig {
    /// Provider credential.
    pub api_key: String,
    /// Provider base URL, without a trailing path.
    pub base_url: String,
    /// Request deadline in milliseconds.
    pub timeout_ms: u32,
    /// Which transport to use.
    pub transport: TransportMode,
}

impl Default for ConnectorConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            base_url: String::new(),
            timeout_ms: 30_000,
            transport: TransportMode::Local,
        }
    }
}

impl ConnectorConfig {
    /// A config with a key and base URL, on the local transport.
    #[must_use]
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            ..Self::default()
        }
    }

    /// Switch to the live transport.
    #[must_use]
    pub fn live(mut self) -> Self {
        self.transport = TransportMode::Live;
        self
    }

    /// Set the request deadline.
    #[must_use]
    pub fn with_timeout_ms(mut self, timeout_ms: u32) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Reject a config that cannot produce a request.
    ///
    /// Error mapping is preserved from Zig: a missing key is an
    /// [`ConnectorError::AuthenticationError`], a missing URL is a
    /// [`ConnectorError::ConnectionFailed`], and a zero timeout is a
    /// [`ConnectorError::Timeout`].
    pub fn validate(&self) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(ConnectorError::AuthenticationError);
        }
        if self.base_url.is_empty() {
            return Err(ConnectorError::ConnectionFailed);
        }
        if self.timeout_ms == 0 {
            return Err(ConnectorError::Timeout);
        }
        Ok(())
    }

    /// The request deadline as a [`std::time::Duration`].
    #[must_use]
    pub const fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.timeout_ms as u64)
    }
}

/// A connector response.
///
/// The Zig struct carried an `owned` flag and a manual `deinit`, because the body
/// might be borrowed or heap-allocated. Rust ownership makes both unnecessary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// HTTP status, or 200 for a synthesized local response.
    pub status: u16,
    /// Response body.
    pub body: String,
}

impl Response {
    /// A `200 OK` response with `body`.
    #[must_use]
    pub fn ok(body: impl Into<String>) -> Self {
        Self {
            status: 200,
            body: body.into(),
        }
    }

    /// Whether the status is 2xx.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// Parse the body as JSON.
    pub fn json(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.body).map_err(|_| ConnectorError::InvalidResponse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_defaults_to_local() {
        // The default must be the hermetic transport: a config that accidentally
        // reaches the network is worse than one that does not.
        assert_eq!(TransportMode::default(), TransportMode::Local);
        assert_eq!(ConnectorConfig::default().transport, TransportMode::Local);
    }

    #[test]
    fn transport_names_are_stable() {
        assert_eq!(TransportMode::Local.name(), "local");
        assert_eq!(TransportMode::Live.name(), "live");
        assert_eq!(TransportMode::Live.to_string(), "live");
    }

    #[test]
    fn default_timeout_is_thirty_seconds() {
        assert_eq!(ConnectorConfig::default().timeout_ms, 30_000);
        assert_eq!(
            ConnectorConfig::default().timeout(),
            std::time::Duration::from_secs(30)
        );
    }

    #[test]
    fn validate_maps_each_missing_field_to_its_zig_error() {
        let base = ConnectorConfig::new("key", "https://api.example.com");
        assert!(base.validate().is_ok());

        let mut no_key = base.clone();
        no_key.api_key.clear();
        assert_eq!(
            no_key.validate().unwrap_err(),
            ConnectorError::AuthenticationError
        );

        let mut no_url = base.clone();
        no_url.base_url.clear();
        assert_eq!(
            no_url.validate().unwrap_err(),
            ConnectorError::ConnectionFailed
        );

        let no_timeout = base.with_timeout_ms(0);
        assert_eq!(no_timeout.validate().unwrap_err(), ConnectorError::Timeout);
    }

    #[test]
    fn builders_compose() {
        let config = ConnectorConfig::new("k", "https://x")
            .live()
            .with_timeout_ms(5_000);
        assert_eq!(config.transport, TransportMode::Live);
        assert_eq!(config.timeout_ms, 5_000);
    }

    #[test]
    fn response_reports_success_by_status_class() {
        assert!(Response::ok("{}").is_success());
        assert!(
            Response {
                status: 299,
                body: String::new()
            }
            .is_success()
        );
        assert!(
            !Response {
                status: 300,
                body: String::new()
            }
            .is_success()
        );
        assert!(
            !Response {
                status: 500,
                body: String::new()
            }
            .is_success()
        );
    }

    #[test]
    fn response_json_parses_or_reports_invalid() {
        assert_eq!(Response::ok(r#"{"a":1}"#).json().unwrap()["a"], 1);
        assert_eq!(
            Response::ok("not json").json().unwrap_err(),
            ConnectorError::InvalidResponse
        );
    }

    #[test]
    fn insecure_base_url_explains_the_risk() {
        // This message reaches the operator; it should say why, not just what.
        let text = ConnectorError::InsecureBaseUrl.to_string();
        assert!(text.contains("https://"));
        assert!(text.contains("cleartext"));
    }
}
