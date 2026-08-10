//! Mandatory-mTLS worker transport configuration.

use crate::{WorkerError, WorkerLimits};
use abi_wdbx_gateway::{Limits, TlsFiles, grpc_tls_config};

/// Worker transport security validated by the existing gateway TLS loader.
#[derive(Clone, Debug)]
pub struct WorkerTransportSecurity {
    /// Owner-protected server identity and mandatory client trust root.
    pub tls: TlsFiles,
    /// Gateway transport bounds reused for any future worker listener.
    pub transport_limits: Limits,
}

impl WorkerTransportSecurity {
    /// Validate complete TLS identity, mandatory client CA, and finite bounds.
    ///
    /// The returned tonic configuration enforces client certificates. This
    /// crate deliberately does not bind a socket or invent a parallel parser.
    pub fn validate(&self) -> Result<tonic::transport::ServerTlsConfig, WorkerError> {
        if self.tls.client_ca.is_none() {
            return Err(WorkerError::Transport(
                "worker transport requires a client CA for mTLS".into(),
            ));
        }
        validate_gateway_limits(&self.transport_limits)?;
        grpc_tls_config(&self.tls)
            .map_err(|error| WorkerError::Transport(error.to_string()))?
            .ok_or_else(|| WorkerError::Transport("worker transport requires TLS".into()))
    }

    /// Derive worker admission bounds from the same gateway ceilings.
    pub fn worker_limits(&self) -> Result<WorkerLimits, WorkerError> {
        validate_gateway_limits(&self.transport_limits)?;
        let limits = WorkerLimits {
            envelope_bytes: self.transport_limits.message_bytes,
            input_bytes: self.transport_limits.value_bytes,
            result_bytes: self.transport_limits.message_bytes,
            stream_chunks: u32::try_from(self.transport_limits.batch_items).unwrap_or(u32::MAX),
            active_jobs: self.transport_limits.blocking_jobs,
            active_jobs_per_principal: self.transport_limits.blocking_jobs,
            control_events: self.transport_limits.membership_entries,
            lease_ms: i64::try_from(self.transport_limits.idle_seconds)
                .unwrap_or(i64::MAX)
                .saturating_mul(1_000),
            clock_skew_ms: 30_000,
        };
        limits.validate()?;
        Ok(limits)
    }
}

fn validate_gateway_limits(limits: &Limits) -> Result<(), WorkerError> {
    if limits.message_bytes == 0
        || limits.batch_items == 0
        || limits.value_bytes == 0
        || limits.blocking_jobs == 0
        || limits.membership_entries == 0
        || limits.idle_seconds == 0
    {
        return Err(WorkerError::Transport(
            "reused gateway transport limits must be nonzero".into(),
        ));
    }
    Ok(())
}
