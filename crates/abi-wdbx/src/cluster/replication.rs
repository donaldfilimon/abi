//! Deterministic exact-transaction replication and vector fanout.

use super::{NodeDescriptor, rendezvous_replicas};
use crate::v2::{CommittedTransaction, ConflictSet, RecordId};
use std::collections::BTreeMap;
use uuid::Uuid;

/// Transport-level failure with no borrowed or implementation-specific state.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TransportError {
    message: String,
}

impl TransportError {
    /// Construct a transport error suitable for crossing an object-safe boundary.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// One stable vector-search result returned by a replica.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplicaSearchHit {
    /// Stable WDBX record identity.
    pub id: RecordId,
    /// Cosine similarity reported by the replica.
    pub score: f32,
}

/// Object-safe data-plane boundary used by cluster orchestration.
///
/// Implementations must import [`CommittedTransaction`] with the v2 exact-byte
/// import API. Creating an independent mutation on the destination is invalid.
pub trait ReplicaTransport {
    /// Import an exact authenticated transaction into `node_id`.
    fn import_committed(
        &mut self,
        node_id: Uuid,
        shard_key: &[u8],
        transaction: &CommittedTransaction,
    ) -> Result<(), TransportError>;

    /// Read every current causal key version visible at `node_id`.
    fn read_kv(
        &self,
        node_id: Uuid,
        key: &str,
    ) -> Result<Option<ConflictSet<String>>, TransportError>;

    /// Export the exact committed objects assigned to one shard.
    ///
    /// The result must include each represented writer's contiguous prefix so
    /// a fresh replica can import later objects without synthesizing history.
    fn shard_transactions(
        &self,
        node_id: Uuid,
        shard_key: &[u8],
    ) -> Result<Vec<CommittedTransaction>, TransportError>;

    /// List shard keys stored by one node in stable byte order.
    fn shard_keys(&self, node_id: Uuid) -> Result<Vec<Vec<u8>>, TransportError>;

    /// Search one replica's current vector view.
    fn search(
        &self,
        node_id: Uuid,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<ReplicaSearchHit>, TransportError>;
}

/// Cluster data-plane validation or quorum failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClusterDataError {
    /// Rendezvous placement produced no eligible replicas.
    #[error("no active replicas selected")]
    NoReplicas,
    /// Fewer than a majority of the selected replica set responded successfully.
    #[error("replica quorum unreachable: required {required}, received {received}")]
    QuorumUnreachable {
        /// Majority required for this selected set.
        required: usize,
        /// Successful responses or imports.
        received: usize,
    },
    /// Replica data was malformed or contradicted an exact object identity.
    #[error("invalid replica data: {0}")]
    InvalidReplicaData(String),
    /// A rebalance plan or resume token failed hash/identity verification.
    #[error("invalid rebalance plan: {0}")]
    InvalidRebalancePlan(String),
}

/// Evidence from one deterministic selected-set replication attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationReceipt {
    /// Rendezvous-selected replicas in deterministic priority order.
    pub selected: Vec<Uuid>,
    /// Nodes that authenticated/imported the exact transaction.
    pub acknowledged: Vec<Uuid>,
    /// Majority required across `selected`.
    pub quorum: usize,
    /// SHA-256 claimed and verified by v2 import for the exact envelope.
    pub transaction_sha256: String,
}

/// Replicate one already committed exact transaction to a rendezvous-selected set.
pub fn replicate_committed(
    transport: &mut dyn ReplicaTransport,
    shard_key: &[u8],
    nodes: &[NodeDescriptor],
    replication_factor: usize,
    transaction: &CommittedTransaction,
) -> Result<ReplicationReceipt, ClusterDataError> {
    let selected = rendezvous_replicas(shard_key, nodes, replication_factor);
    let quorum = selected_quorum(&selected)?;
    let mut acknowledged = Vec::new();
    for node_id in &selected {
        if transport
            .import_committed(*node_id, shard_key, transaction)
            .is_ok()
        {
            acknowledged.push(*node_id);
        }
    }
    if acknowledged.len() < quorum {
        return Err(ClusterDataError::QuorumUnreachable {
            required: quorum,
            received: acknowledged.len(),
        });
    }
    Ok(ReplicationReceipt {
        selected,
        acknowledged,
        quorum,
        transaction_sha256: transaction.sha256().to_owned(),
    })
}

/// Fan out vector search to the selected set and merge by score then `RecordId`.
pub fn search_fanout(
    transport: &dyn ReplicaTransport,
    shard_key: &[u8],
    nodes: &[NodeDescriptor],
    replication_factor: usize,
    query: &[f32],
    limit: usize,
) -> Result<Vec<ReplicaSearchHit>, ClusterDataError> {
    let selected = rendezvous_replicas(shard_key, nodes, replication_factor);
    let quorum = selected_quorum(&selected)?;
    let mut responses = 0_usize;
    let mut merged = BTreeMap::<RecordId, f32>::new();
    for node_id in selected {
        let Ok(hits) = transport.search(node_id, query, limit) else {
            continue;
        };
        responses += 1;
        for hit in hits {
            if !hit.score.is_finite() {
                return Err(ClusterDataError::InvalidReplicaData(format!(
                    "node {node_id} returned a non-finite vector score"
                )));
            }
            merged
                .entry(hit.id)
                .and_modify(|score| *score = score.max(hit.score))
                .or_insert(hit.score);
        }
    }
    if responses < quorum {
        return Err(ClusterDataError::QuorumUnreachable {
            required: quorum,
            received: responses,
        });
    }
    let mut hits = merged
        .into_iter()
        .map(|(id, score)| ReplicaSearchHit { id, score })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    hits.truncate(limit);
    Ok(hits)
}

pub(super) fn selected_quorum(selected: &[Uuid]) -> Result<usize, ClusterDataError> {
    if selected.is_empty() {
        return Err(ClusterDataError::NoReplicas);
    }
    Ok(selected.len() / 2 + 1)
}

pub(super) fn validate_transaction_inventory(
    transactions: impl IntoIterator<Item = CommittedTransaction>,
) -> Result<BTreeMap<(Uuid, u64), CommittedTransaction>, ClusterDataError> {
    let mut exact = BTreeMap::new();
    for transaction in transactions {
        let key = (transaction.writer_id(), transaction.sequence());
        if let Some(existing) = exact.get(&key)
            && existing != &transaction
        {
            return Err(ClusterDataError::InvalidReplicaData(format!(
                "writer {} sequence {} has contradictory exact bytes",
                key.0, key.1
            )));
        }
        exact.insert(key, transaction);
    }
    let mut expected = BTreeMap::<Uuid, u64>::new();
    for &(writer_id, sequence) in exact.keys() {
        let next = expected.entry(writer_id).or_insert(1);
        if sequence != *next {
            return Err(ClusterDataError::InvalidReplicaData(format!(
                "writer {writer_id} inventory is not contiguous at sequence {sequence}"
            )));
        }
        *next = next.saturating_add(1);
    }
    Ok(exact)
}
