//! Framework configuration.
//!
//! Ported from `src/core/config.zig`. Every default is reproduced exactly — the
//! Zig tests asserted several of them, and `abi backends` prints some, so a
//! changed default is an observable behaviour change.
//!
//! The Zig `Config` carried an `allocator` field and a `deinit` that did nothing;
//! both are dropped. `Default` replaces `init`.

use abi_foundation::logger::Level;
use serde::{Deserialize, Serialize};

/// Resource and protocol ceilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct LimitConfig {
    /// Maximum concurrent streams.
    pub max_streams: u32,
    /// Memory ceiling in mebibytes.
    pub max_memory_mb: u32,
    /// CPU share ceiling, as a percentage.
    pub max_cpu_percent: u8,
    /// Maximum tasks in flight.
    pub max_concurrent_tasks: u32,
    /// Largest accepted vector dimensionality.
    pub max_vector_dim: u32,
    /// Largest accepted block size in kibibytes.
    pub max_block_size_kb: u32,
    /// Request deadline in milliseconds.
    pub request_timeout_ms: u32,
    /// Ring capacity of the in-memory log buffer.
    pub max_log_entries: usize,
}

impl Default for LimitConfig {
    fn default() -> Self {
        Self {
            max_streams: 10,
            max_memory_mb: 1024,
            max_cpu_percent: 50,
            max_concurrent_tasks: 100,
            max_vector_dim: 4096,
            max_block_size_kb: 64,
            request_timeout_ms: 30_000,
            max_log_entries: 10_000,
        }
    }
}

/// AI pipeline defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AiConfig {
    /// Persona profile used when none is requested. One of `abi`, `abbey`, `aviva`.
    pub default_profile: String,
    /// Generation token ceiling.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Nucleus-sampling cutoff.
    pub top_p: f32,
    /// Whether the constitution audit runs.
    pub enable_constitution: bool,
    /// Retry attempts on a transient provider failure.
    pub max_retries: u8,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            default_profile: "abi".to_string(),
            max_tokens: 4096,
            temperature: 0.7,
            top_p: 0.9,
            enable_constitution: true,
            max_retries: 3,
        }
    }
}

/// Listener defaults for the MCP HTTP transport and the WDBX REST surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Bind address.
    ///
    /// Defaults to loopback. Changing this to `0.0.0.0` exposes surfaces whose
    /// only authentication is a bearer token, so it is a deliberate act.
    pub host: String,
    /// Bind port.
    pub port: u16,
    /// Maximum simultaneous connections.
    pub max_connections: u32,
    /// Whether CORS headers are emitted.
    pub enable_cors: bool,
    /// Whether TLS is terminated in-process.
    pub tls_enabled: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            enable_cors: true,
            tls_enabled: false,
        }
    }
}

impl ServerConfig {
    /// Whether `host` is a loopback address.
    ///
    /// Not in the Zig original. The listeners each open-coded a loopback check,
    /// and having one place to ask makes the "is this exposed?" question
    /// answerable in `abi backends` output.
    #[must_use]
    pub fn is_loopback(&self) -> bool {
        self.host == "localhost"
            || self
                .host
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    }
}

/// Top-level framework configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Resource ceilings.
    pub limits: LimitConfig,
    /// AI pipeline defaults.
    pub ai: AiConfig,
    /// Listener defaults.
    pub server: ServerConfig,
    /// Minimum log level.
    pub log_level: LogLevel,
    /// Framework version string.
    pub version: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            limits: LimitConfig::default(),
            ai: AiConfig::default(),
            server: ServerConfig::default(),
            log_level: LogLevel::Info,
            version: "0.1.0-dev".to_string(),
        }
    }
}

impl Config {
    /// Parse configuration from JSON, filling in defaults for absent fields.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Minimum severity to log.
///
/// A serializable mirror of [`abi_foundation::logger::Level`]. The foundation
/// crate does not depend on serde for its logger, so this is the boundary type;
/// [`LogLevel::to_level`] converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    /// Verbose developer detail.
    Debug,
    /// Normal operational milestones.
    #[default]
    Info,
    /// Unexpected but non-fatal.
    Warn,
    /// Failure.
    ///
    /// Serializes as `err`, matching the Zig enum's field name — a config file
    /// written for the Zig build must still parse.
    #[serde(rename = "err", alias = "error")]
    Err,
}

impl LogLevel {
    /// Convert to the logger's level type.
    #[must_use]
    pub const fn to_level(self) -> Level {
        match self {
            Self::Debug => Level::Debug,
            Self::Info => Level::Info,
            Self::Warn => Level::Warn,
            Self::Err => Level::Error,
        }
    }
}

impl From<LogLevel> for Level {
    fn from(value: LogLevel) -> Self {
        value.to_level()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_defaults_match_the_zig_values() {
        let limits = LimitConfig::default();
        assert_eq!(limits.max_streams, 10);
        assert_eq!(limits.max_memory_mb, 1024);
        assert_eq!(limits.max_cpu_percent, 50);
        assert_eq!(limits.max_concurrent_tasks, 100);
        assert_eq!(limits.max_vector_dim, 4096);
        assert_eq!(limits.max_block_size_kb, 64);
        assert_eq!(limits.request_timeout_ms, 30_000);
        assert_eq!(limits.max_log_entries, 10_000);
    }

    #[test]
    fn ai_defaults_match_the_zig_values() {
        let ai = AiConfig::default();
        assert_eq!(ai.default_profile, "abi");
        assert_eq!(ai.max_tokens, 4096);
        assert!((ai.temperature - 0.7).abs() < f32::EPSILON);
        assert!((ai.top_p - 0.9).abs() < f32::EPSILON);
        assert!(ai.enable_constitution);
        assert_eq!(ai.max_retries, 3);
    }

    #[test]
    fn server_defaults_match_the_zig_values() {
        let server = ServerConfig::default();
        assert_eq!(server.host, "127.0.0.1");
        assert_eq!(server.port, 8080);
        assert_eq!(server.max_connections, 100);
        assert!(server.enable_cors);
        assert!(!server.tls_enabled);
    }

    #[test]
    fn the_default_bind_address_is_loopback() {
        // A non-loopback default would expose the MCP and WDBX surfaces, whose
        // only auth is a bearer token.
        assert!(ServerConfig::default().is_loopback());
    }

    #[test]
    fn loopback_detection_covers_the_forms_that_matter() {
        let loopback = |host: &str| {
            ServerConfig {
                host: host.to_string(),
                ..ServerConfig::default()
            }
            .is_loopback()
        };

        assert!(loopback("127.0.0.1"));
        assert!(loopback("127.0.0.53"));
        assert!(loopback("::1"));
        assert!(loopback("localhost"));

        assert!(!loopback("0.0.0.0"));
        assert!(!loopback("192.168.1.10"));
        assert!(!loopback("::"));
        assert!(!loopback("example.com"));
    }

    #[test]
    fn config_defaults_to_info_and_the_dev_version() {
        let config = Config::default();
        assert_eq!(config.log_level, LogLevel::Info);
        assert_eq!(config.version, "0.1.0-dev");
    }

    #[test]
    fn log_level_maps_onto_the_logger_level() {
        assert_eq!(LogLevel::Debug.to_level(), Level::Debug);
        assert_eq!(LogLevel::Info.to_level(), Level::Info);
        assert_eq!(LogLevel::Warn.to_level(), Level::Warn);
        assert_eq!(LogLevel::Err.to_level(), Level::Error);
    }

    #[test]
    fn log_level_serializes_as_err_like_the_zig_field_name() {
        // A config written for the Zig build must still parse.
        assert_eq!(serde_json::to_string(&LogLevel::Err).unwrap(), "\"err\"");
        assert_eq!(
            serde_json::from_str::<LogLevel>("\"err\"").unwrap(),
            LogLevel::Err
        );
        // The friendlier spelling is accepted too.
        assert_eq!(
            serde_json::from_str::<LogLevel>("\"error\"").unwrap(),
            LogLevel::Err
        );
    }

    #[test]
    fn partial_json_fills_in_defaults() {
        let config = Config::from_json(r#"{"server":{"port":9999}}"#).expect("parse");
        assert_eq!(config.server.port, 9999);
        // Absent fields keep their defaults rather than zeroing.
        assert_eq!(config.server.host, "127.0.0.1");
        assert_eq!(config.limits.max_memory_mb, 1024);
        assert_eq!(config.ai.default_profile, "abi");
    }

    #[test]
    fn empty_json_object_is_the_default_config() {
        assert_eq!(Config::from_json("{}").expect("parse"), Config::default());
    }

    #[test]
    fn config_roundtrips_through_json() {
        let mut config = Config::default();
        config.server.port = 1234;
        config.log_level = LogLevel::Warn;
        config.ai.default_profile = "abbey".to_string();

        let text = config.to_json().expect("serialize");
        assert_eq!(Config::from_json(&text).expect("parse"), config);
    }

    #[test]
    fn malformed_json_is_an_error() {
        assert!(Config::from_json("{not json").is_err());
        assert!(Config::from_json(r#"{"log_level":"verbose"}"#).is_err());
    }
}
