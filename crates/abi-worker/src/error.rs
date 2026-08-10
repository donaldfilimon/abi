//! Secret-safe worker validation and state-machine failures.

/// Worker admission, authentication, quota, or state failure.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum WorkerError {
    /// A field is malformed, unbounded, or internally inconsistent.
    #[error("invalid worker request: {0}")]
    Invalid(String),
    /// Signature, signer, audience, or namespace authentication failed.
    #[error("worker authentication failed: {0}")]
    Authentication(String),
    /// Requested capability is absent or incompatible.
    #[error("worker capability refused: {0}")]
    Capability(String),
    /// A deadline or lease is no longer valid.
    #[error("worker deadline refused: {0}")]
    Deadline(String),
    /// A finite worker quota would be exceeded.
    #[error("worker quota refused: {0}")]
    Quota(String),
    /// Replay or idempotency state conflicts with accepted history.
    #[error("worker replay refused: {0}")]
    Replay(String),
    /// Lease, cancellation, or completion state is invalid.
    #[error("worker state refused: {0}")]
    State(String),
    /// Independent control storage failed.
    #[error("worker control store failed: {0}")]
    ControlStore(String),
    /// Existing gateway TLS/mTLS validation failed.
    #[error("worker transport security failed: {0}")]
    Transport(String),
}
