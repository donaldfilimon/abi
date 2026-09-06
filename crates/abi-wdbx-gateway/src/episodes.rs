//! Canonical episode gate: WDBX v3 `ProposeWrite` and `Verify` over gRPC.
//!
//! The write vocabulary stays owned by [`abi_wdbx::v3::episode`]; requests carry
//! its JSON encoding and are decoded with unknown fields rejected. Rejections
//! surface as gRPC statuses whose message is the store's stable reason label,
//! so the reason is recorded by every caller that logs the status.

use crate::config::open_validated_file;
use crate::executor::GatewayState;
use crate::proto::{
    EpisodeReceipt as ReceiptMessage, ProposeEpisodeWriteRequest, ProposeEpisodeWriteResponse,
    VerifyEpisodeRequest, VerifyEpisodeResponse,
};
use crate::{GatewayError, StoreExecutor};
use abi_wdbx::v3::episode::{EpisodeReceipt, EpisodeStoreError, EpisodeWrite, StorePolicy};
use std::path::Path;
use tonic::Status;

/// Maximum bytes accepted for one guild reference.
const MAX_GUILD_REF_BYTES: usize = 128;
const DECISION_APPENDED: &str = "appended";
const DECISION_PREVIEW: &str = "preview";
const UNCONFIGURED: &str = "episode policy is not configured";

/// Outcome of one blocking episode job, kept separate from gateway failures so
/// the gRPC status code survives the executor boundary.
enum EpisodeOutcome<T> {
    Ok(T),
    Store(EpisodeStoreError),
    Unconfigured,
}

/// Load and validate the JSON `StorePolicy` that enables the episode RPCs.
pub(crate) fn load_policy(path: &Path) -> Result<StorePolicy, GatewayError> {
    let file = open_validated_file(path, false, "episode policy")?;
    serde_json::from_reader(std::io::BufReader::new(file)).map_err(|error| GatewayError::File {
        label: "episode policy".into(),
        path: path.to_path_buf(),
        message: format!(
            "invalid JSON store policy at line {}, column {}",
            error.line(),
            error.column()
        ),
    })
}

/// Propose one canonical episode append, or preview its commitment.
pub(crate) async fn propose(
    executor: &StoreExecutor,
    maximum_write_bytes: usize,
    request: ProposeEpisodeWriteRequest,
) -> Result<ProposeEpisodeWriteResponse, Status> {
    let write = decode_write(&request.episode_write_json, maximum_write_bytes)?;
    let preview_only = request.preview_only;
    let outcome = executor
        .run_gateway(move |state| Ok(propose_blocking(state, &write, preview_only)))
        .await
        .map_err(Status::from)?;
    let (episode_digest, receipt) = finish(outcome)?;
    Ok(ProposeEpisodeWriteResponse {
        decision: if receipt.is_some() {
            DECISION_APPENDED
        } else {
            DECISION_PREVIEW
        }
        .into(),
        episode_digest: episode_digest.to_vec(),
        receipt: receipt.map(receipt_message),
    })
}

/// Look one commitment up in the verified ledger of its guild.
pub(crate) async fn verify(
    executor: &StoreExecutor,
    request: VerifyEpisodeRequest,
) -> Result<VerifyEpisodeResponse, Status> {
    let digest: [u8; 32] = request
        .episode_digest
        .as_slice()
        .try_into()
        .map_err(|_| Status::invalid_argument("episode digest must be 32 bytes"))?;
    let guild_ref = request.guild_ref;
    if guild_ref.is_empty() || guild_ref.len() > MAX_GUILD_REF_BYTES {
        return Err(Status::invalid_argument(
            "guild reference is outside limits",
        ));
    }
    let outcome = executor
        .run_gateway(move |state| Ok(verify_blocking(state, &guild_ref, digest)))
        .await
        .map_err(Status::from)?;
    let receipt = finish(outcome)?;
    Ok(VerifyEpisodeResponse {
        found: receipt.is_some(),
        receipt: receipt.map(receipt_message),
    })
}

/// Lowercase hex of a commitment, for metadata-only mutation notices.
pub(crate) fn digest_hex(digest: &[u8]) -> String {
    use std::fmt::Write as _;

    digest
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

fn decode_write(bytes: &[u8], maximum_bytes: usize) -> Result<EpisodeWrite, Status> {
    if bytes.is_empty() || bytes.len() > maximum_bytes {
        return Err(Status::invalid_argument("episode write is outside limits"));
    }
    serde_json::from_slice::<EpisodeWrite>(bytes)
        .map_err(|_| Status::invalid_argument("episode write JSON is invalid"))
}

fn propose_blocking(
    state: &mut GatewayState,
    write: &EpisodeWrite,
    preview_only: bool,
) -> EpisodeOutcome<([u8; 32], Option<EpisodeReceipt>)> {
    let Some(store) = state.episodes.as_mut() else {
        return EpisodeOutcome::Unconfigured;
    };
    if preview_only {
        return store
            .preview_commitment(write)
            .map_or_else(EpisodeOutcome::Store, |digest| {
                EpisodeOutcome::Ok((digest, None))
            });
    }
    store
        .propose_write(write)
        .map_or_else(EpisodeOutcome::Store, |receipt| {
            EpisodeOutcome::Ok((receipt.episode_digest, Some(receipt)))
        })
}

fn verify_blocking(
    state: &mut GatewayState,
    guild_ref: &str,
    digest: [u8; 32],
) -> EpisodeOutcome<Option<EpisodeReceipt>> {
    let Some(store) = state.episodes.as_ref() else {
        return EpisodeOutcome::Unconfigured;
    };
    store
        .find_receipt(guild_ref, &digest)
        .map_or_else(EpisodeOutcome::Store, EpisodeOutcome::Ok)
}

fn finish<T>(outcome: EpisodeOutcome<T>) -> Result<T, Status> {
    match outcome {
        EpisodeOutcome::Ok(value) => Ok(value),
        EpisodeOutcome::Store(error) => Err(episode_status(&error)),
        EpisodeOutcome::Unconfigured => Err(Status::failed_precondition(UNCONFIGURED)),
    }
}

/// Map a store rejection to a gRPC code; the message is the stable reason label.
fn episode_status(error: &EpisodeStoreError) -> Status {
    let reason = error.to_string();
    match error {
        EpisodeStoreError::InvalidInput | EpisodeStoreError::Canonical(_) => {
            Status::invalid_argument(reason)
        }
        EpisodeStoreError::Replay => Status::already_exists(reason),
        EpisodeStoreError::WriterBusy => Status::unavailable(reason),
        EpisodeStoreError::Io | EpisodeStoreError::Corrupt => Status::internal(reason),
        EpisodeStoreError::StaleBinding
        | EpisodeStoreError::LearningDisabled
        | EpisodeStoreError::Quiet
        | EpisodeStoreError::InvalidTransition
        | EpisodeStoreError::CommitmentMismatch
        | EpisodeStoreError::TokenBudget
        | EpisodeStoreError::StorageBudget => Status::failed_precondition(reason),
    }
}

fn receipt_message(receipt: EpisodeReceipt) -> ReceiptMessage {
    ReceiptMessage {
        sequence: receipt.sequence,
        request_id: receipt.request_id,
        operation_id: receipt.operation_id,
        guild_ref: receipt.guild_ref,
        episode_digest: receipt.episode_digest.to_vec(),
        previous_digest: receipt
            .previous_digest
            .map(|digest| digest.to_vec())
            .unwrap_or_default(),
        event_kind: receipt.event_kind,
        policy_version: receipt.policy_version,
        evidence_level: receipt.evidence_level.label().to_owned(),
        terminal_status: receipt
            .terminal_status
            .map(|status| status.label().to_owned())
            .unwrap_or_default(),
        redacted: receipt.redacted,
    }
}
