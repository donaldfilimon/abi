//! Gateway configuration and pre-bind validation.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};

/// Resource limits enforced before store dispatch.
#[derive(Debug, Clone)]
pub struct Limits {
    /// Maximum decoded transport message bytes.
    pub message_bytes: usize,
    /// Maximum vectors or KV entries in one mutation request.
    pub batch_items: usize,
    /// Maximum key or node identity bytes.
    pub key_bytes: usize,
    /// Maximum KV value bytes.
    pub value_bytes: usize,
    /// Maximum requests admitted per one-second window.
    pub requests_per_second: u32,
    /// Maximum simultaneously running blocking store jobs.
    pub blocking_jobs: usize,
    /// Maximum concurrent mutation streams.
    pub concurrent_streams: usize,
    /// Maximum process-local gateway membership entries.
    pub membership_entries: usize,
    /// Mutation event queue capacity per gateway process.
    pub event_queue: usize,
    /// Stream idle timeout in seconds.
    pub idle_seconds: u64,
    /// Maximum search result count.
    pub search_limit: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            message_bytes: 1024 * 1024,
            batch_items: 128,
            key_bytes: 1024,
            value_bytes: 64 * 1024,
            requests_per_second: 100,
            blocking_jobs: 8,
            concurrent_streams: 32,
            membership_entries: 1_024,
            event_queue: 64,
            idle_seconds: 30,
            search_limit: 1_000,
        }
    }
}

impl Limits {
    pub(crate) fn validate(&self) -> Result<(), GatewayError> {
        let nonzero = [
            self.message_bytes,
            self.batch_items,
            self.key_bytes,
            self.value_bytes,
            self.blocking_jobs,
            self.concurrent_streams,
            self.membership_entries,
            self.event_queue,
            self.search_limit,
        ];
        if nonzero.contains(&0) || self.requests_per_second == 0 || self.idle_seconds == 0 {
            return Err(GatewayError::Configuration(
                "gateway limits must all be nonzero".into(),
            ));
        }
        Ok(())
    }
}

/// TLS certificate, private key, and optional client trust root.
#[derive(Debug, Clone, Default)]
pub struct TlsFiles {
    /// PEM certificate chain.
    pub certificate: Option<PathBuf>,
    /// Owner-protected PEM private key.
    pub private_key: Option<PathBuf>,
    /// Optional PEM client CA. Its presence enables mandatory mTLS.
    pub client_ca: Option<PathBuf>,
}

impl TlsFiles {
    /// Whether both server TLS files are configured.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.certificate.is_some() && self.private_key.is_some()
    }

    pub(crate) fn validate_shape(&self) -> Result<(), GatewayError> {
        match (&self.certificate, &self.private_key) {
            (Some(_), Some(_)) | (None, None) => {}
            _ => {
                return Err(GatewayError::Configuration(
                    "TLS certificate and private key must be configured together".into(),
                ));
            }
        }
        if self.client_ca.is_some() && !self.is_complete() {
            return Err(GatewayError::Configuration(
                "client CA requires a complete server TLS configuration".into(),
            ));
        }
        Ok(())
    }
}

/// Complete server configuration. Defaults bind both transports to loopback.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// gRPC listener address.
    pub grpc_addr: SocketAddr,
    /// WebSocket event listener address.
    pub events_addr: SocketAddr,
    /// WDBX base path (the same path shape accepted by `StorePaths`).
    pub store_path: PathBuf,
    /// Required bearer-token file.
    pub token_file: PathBuf,
    /// TLS and optional mTLS material.
    pub tls: TlsFiles,
    /// Exact accepted browser origins for WebSocket upgrades.
    pub allowed_origins: Vec<String>,
    /// Resource limits.
    pub limits: Limits,
}

impl GatewayConfig {
    /// Construct a loopback configuration around explicit store and token files.
    #[must_use]
    pub fn loopback(store_path: impl Into<PathBuf>, token_file: impl Into<PathBuf>) -> Self {
        Self {
            grpc_addr: SocketAddr::from(([127, 0, 0, 1], 50051)),
            events_addr: SocketAddr::from(([127, 0, 0, 1], 50052)),
            store_path: store_path.into(),
            token_file: token_file.into(),
            tls: TlsFiles::default(),
            allowed_origins: vec!["http://127.0.0.1".into(), "http://localhost".into()],
            limits: Limits::default(),
        }
    }

    /// Validate every security prerequisite before either socket is bound.
    pub fn validate_pre_bind(&self) -> Result<(), GatewayError> {
        self.limits.validate()?;
        self.tls.validate_shape()?;
        if self.grpc_addr == self.events_addr && self.grpc_addr.port() != 0 {
            return Err(GatewayError::Configuration(
                "gRPC and WebSocket listeners must use distinct addresses".into(),
            ));
        }
        let unique_origins = self
            .allowed_origins
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        if self.allowed_origins.is_empty()
            || unique_origins.len() != self.allowed_origins.len()
            || self.allowed_origins.iter().any(|origin| {
                origin.trim().is_empty() || origin == "*" || origin.eq_ignore_ascii_case("null")
            })
        {
            return Err(GatewayError::Configuration(
                "WebSocket origins must be a unique nonempty exact allowlist without null or wildcard"
                    .into(),
            ));
        }
        validate_regular_file(&self.token_file, true, "bearer token")?;
        if let Some(path) = &self.tls.certificate {
            validate_regular_file(path, false, "TLS certificate")?;
        }
        if let Some(path) = &self.tls.private_key {
            validate_regular_file(path, true, "TLS private key")?;
        }
        if let Some(path) = &self.tls.client_ca {
            validate_regular_file(path, false, "TLS client CA")?;
        }
        if (!is_loopback(self.grpc_addr.ip()) || !is_loopback(self.events_addr.ip()))
            && !self.tls.is_complete()
        {
            return Err(GatewayError::Configuration(
                "non-loopback listeners require TLS and an owner-protected bearer token".into(),
            ));
        }
        Ok(())
    }
}

fn is_loopback(ip: IpAddr) -> bool {
    ip.is_loopback()
}

pub(crate) fn validate_regular_file(
    path: &Path,
    private: bool,
    label: &str,
) -> Result<(), GatewayError> {
    open_validated_file(path, private, label).map(|_| ())
}

pub(crate) fn open_validated_file(
    path: &Path,
    private: bool,
    label: &str,
) -> Result<std::fs::File, GatewayError> {
    let file = open_no_follow(path).map_err(|error| GatewayError::File {
        label: label.into(),
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    let metadata = file.metadata().map_err(|error| GatewayError::File {
        label: label.into(),
        path: path.to_path_buf(),
        message: error.to_string(),
    })?;
    if !metadata.is_file() {
        return Err(GatewayError::File {
            label: label.into(),
            path: path.to_path_buf(),
            message: "must be a regular file and not a symlink".into(),
        });
    }
    if private {
        validate_private_metadata(path, &metadata, label)?;
    }
    Ok(file)
}

#[cfg(unix)]
fn open_no_follow(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt as _;

    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

#[cfg(not(unix))]
fn open_no_follow(path: &Path) -> std::io::Result<std::fs::File> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(std::io::Error::other("symlink rejected"));
    }
    std::fs::File::open(path)
}

#[cfg(unix)]
fn validate_private_metadata(
    path: &Path,
    metadata: &std::fs::Metadata,
    label: &str,
) -> Result<(), GatewayError> {
    use std::os::unix::fs::MetadataExt as _;

    if metadata.uid() != nix::unistd::Uid::effective().as_raw() || metadata.mode() & 0o077 != 0 {
        return Err(GatewayError::File {
            label: label.into(),
            path: path.to_path_buf(),
            message: "must be owned by the current user with no group/other permissions".into(),
        });
    }
    Ok(())
}

#[cfg(not(unix))]
fn validate_private_metadata(
    path: &Path,
    _metadata: &std::fs::Metadata,
    label: &str,
) -> Result<(), GatewayError> {
    Err(GatewayError::File {
        label: label.into(),
        path: path.to_path_buf(),
        message: "private-file ownership checks are not implemented on this host".into(),
    })
}

/// Gateway startup or runtime failure with secret-safe messages.
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    /// Invalid or incomplete configuration.
    #[error("invalid gateway configuration: {0}")]
    Configuration(String),
    /// File validation or loading failed.
    #[error("invalid {label} file {path}: {message}")]
    File {
        /// Redacted role of the file.
        label: String,
        /// File path (never file contents).
        path: PathBuf,
        /// Non-secret failure detail.
        message: String,
    },
    /// Store initialization failed.
    #[error("WDBX gateway store failed: {0}")]
    Store(String),
    /// A transport failed.
    #[error("WDBX gateway transport failed: {0}")]
    Transport(String),
}
