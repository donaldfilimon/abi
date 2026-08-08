//! Authenticated, bounded exact-transaction RPC payloads.

use super::{MAX_CLUSTER_DATA_FRAME_SIZE, RpcError, dial};
use crate::{CommittedTransaction, ConflictSet, V2Mutation, Version, VersionedStore};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::net::TcpStream;
use uuid::Uuid;

/// One authenticated child-node data-plane request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum ClusterDataRequest {
    /// Commit one key version and return its exact sealed transaction.
    CommitKv {
        /// Stable shard-routing bytes.
        shard_key: Vec<u8>,
        /// Logical key.
        key: String,
        /// Stored value.
        value: String,
    },
    /// Import one exact authenticated committed transaction.
    ImportCommitted {
        /// Stable shard-routing bytes.
        shard_key: Vec<u8>,
        /// Exact sealed transaction object.
        transaction: CommittedTransaction,
    },
    /// Export one locally published exact transaction.
    ExportTransaction {
        /// Original writer identity.
        writer_id: Uuid,
        /// Contiguous writer-local sequence.
        sequence: u64,
    },
    /// Read every current causal version of one key.
    ReadKv {
        /// Logical key.
        key: String,
    },
    /// Export the exact transactions assigned to one shard.
    ShardTransactions {
        /// Stable shard-routing bytes.
        shard_key: Vec<u8>,
    },
    /// List this node's assigned shard keys in stable order.
    ShardKeys,
}

/// Serializable conflict-preserving key state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterKvState {
    /// Deterministic presentation preference only.
    pub preferred: Version<String>,
    /// Every unresolved concurrent current version.
    pub conflicts: Vec<Version<String>>,
}

impl From<ConflictSet<String>> for ClusterKvState {
    fn from(value: ConflictSet<String>) -> Self {
        Self {
            preferred: value.preferred,
            conflicts: value.conflicts,
        }
    }
}

impl From<ClusterKvState> for ConflictSet<String> {
    fn from(value: ClusterKvState) -> Self {
        Self {
            preferred: value.preferred,
            conflicts: value.conflicts,
        }
    }
}

/// One bounded child-node data-plane response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "result", rename_all = "snake_case")]
pub enum ClusterDataResponse {
    /// A commit or export returned one exact object.
    Transaction {
        /// Exact sealed transaction object.
        transaction: CommittedTransaction,
    },
    /// An exact import authenticated and completed.
    Imported {
        /// Stable transaction identity.
        transaction_id: Uuid,
    },
    /// Current conflict-preserving key state.
    Kv {
        /// `None` when the key is absent.
        current: Option<ClusterKvState>,
    },
    /// Stable exact shard inventory.
    Transactions {
        /// Transactions ordered by writer then sequence.
        transactions: Vec<CommittedTransaction>,
    },
    /// Stable shard-key inventory.
    ShardKeys {
        /// Assigned shard keys.
        shard_keys: Vec<Vec<u8>>,
    },
    /// Authentication succeeded but the requested store operation failed.
    Error {
        /// Redacted, operator-readable failure.
        message: String,
    },
}

/// Scratch-store state owned by one child RPC server.
#[derive(Debug)]
pub(super) struct ClusterDataPlane {
    store: VersionedStore,
    shards: BTreeMap<Vec<u8>, BTreeSet<(Uuid, u64)>>,
}

impl ClusterDataPlane {
    pub(super) fn new(store: VersionedStore) -> Self {
        Self {
            store,
            shards: BTreeMap::new(),
        }
    }

    pub(super) fn handle(&mut self, body: &str) -> ClusterDataResponse {
        let Ok(request) = serde_json::from_str::<ClusterDataRequest>(body) else {
            return error("malformed data-plane request");
        };
        match request {
            ClusterDataRequest::CommitKv {
                shard_key,
                key,
                value,
            } => {
                if shard_key.is_empty() {
                    return error("shard key must not be empty");
                }
                match self
                    .store
                    .commit_export(vec![V2Mutation::PutKv { key, value }])
                {
                    Ok(transaction) => {
                        self.record(&shard_key, &transaction);
                        ClusterDataResponse::Transaction { transaction }
                    }
                    Err(failure) => error(failure.to_string()),
                }
            }
            ClusterDataRequest::ImportCommitted {
                shard_key,
                transaction,
            } => {
                if shard_key.is_empty() {
                    return error("shard key must not be empty");
                }
                match self.store.import_committed(&transaction) {
                    Ok(transaction_id) => {
                        self.record(&shard_key, &transaction);
                        ClusterDataResponse::Imported { transaction_id }
                    }
                    Err(failure) => error(failure.to_string()),
                }
            }
            ClusterDataRequest::ExportTransaction {
                writer_id,
                sequence,
            } => match self.store.export_transaction(writer_id, sequence) {
                Ok(transaction) => ClusterDataResponse::Transaction { transaction },
                Err(failure) => error(failure.to_string()),
            },
            ClusterDataRequest::ReadKv { key } => ClusterDataResponse::Kv {
                current: self.store.get_conflicts(&key).map(ClusterKvState::from),
            },
            ClusterDataRequest::ShardTransactions { shard_key } => {
                let identities = self.shards.get(&shard_key).cloned().unwrap_or_default();
                let mut transactions = Vec::with_capacity(identities.len());
                for (writer_id, sequence) in identities {
                    match self.store.export_transaction(writer_id, sequence) {
                        Ok(transaction) => transactions.push(transaction),
                        Err(failure) => return error(failure.to_string()),
                    }
                }
                ClusterDataResponse::Transactions { transactions }
            }
            ClusterDataRequest::ShardKeys => ClusterDataResponse::ShardKeys {
                shard_keys: self.shards.keys().cloned().collect(),
            },
        }
    }

    fn record(&mut self, shard_key: &[u8], transaction: &CommittedTransaction) {
        self.shards
            .entry(shard_key.to_vec())
            .or_default()
            .insert((transaction.writer_id(), transaction.sequence()));
    }
}

fn error(message: impl Into<String>) -> ClusterDataResponse {
    ClusterDataResponse::Error {
        message: message.into(),
    }
}

pub(super) fn encode_response(response: &ClusterDataResponse) -> String {
    let body = serde_json::to_string(response).expect("cluster data response serializes");
    let framed = format!("DATA {body}\n");
    if framed.len() < MAX_CLUSTER_DATA_FRAME_SIZE {
        return framed;
    }
    let bounded = error("data-plane response exceeds frame limit");
    format!(
        "DATA {}\n",
        serde_json::to_string(&bounded).expect("bounded error serializes")
    )
}

/// Send one authenticated, bounded child-node data-plane request.
pub fn dial_data(
    host: &str,
    port: u16,
    token: &str,
    request: &ClusterDataRequest,
) -> Result<Option<TcpStream>, RpcError> {
    if token.is_empty() || token.bytes().any(|byte| byte.is_ascii_whitespace()) {
        return Err(RpcError::InvalidToken);
    }
    let body = serde_json::to_string(request).map_err(|_| RpcError::MalformedRequest)?;
    let message = format!("AUTH {token} DATA {body}\n");
    if message.len() >= MAX_CLUSTER_DATA_FRAME_SIZE {
        return Err(RpcError::LineTooLong);
    }
    dial(host, port, &message)
}

/// Read one bounded child-node data-plane response.
pub fn read_data_reply(mut stream: TcpStream) -> Result<ClusterDataResponse, RpcError> {
    let mut buffer = vec![0_u8; MAX_CLUSTER_DATA_FRAME_SIZE];
    let line = crate::net_line::read_line(&mut stream, &mut buffer)?;
    let line = std::str::from_utf8(line).map_err(|_| RpcError::MalformedResponse)?;
    let body = line
        .strip_prefix("DATA ")
        .ok_or(RpcError::MalformedResponse)?;
    serde_json::from_str(body).map_err(|_| RpcError::MalformedResponse)
}
