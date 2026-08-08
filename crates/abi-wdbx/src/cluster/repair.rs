//! Causal read fanout and exact-transaction read repair.

use super::replication::{selected_quorum, validate_transaction_inventory};
use super::{ClusterDataError, NodeDescriptor, ReplicaTransport, rendezvous_replicas};
use crate::v2::{CommittedTransaction, Version};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use uuid::Uuid;

/// Exact committed objects missing from one replica.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairAction {
    /// Replica that should import the transactions.
    pub node_id: Uuid,
    /// Stable shard key used to route the exact import.
    pub shard_key: Vec<u8>,
    /// Exact source envelopes, ordered by writer then contiguous sequence.
    pub transactions: Vec<CommittedTransaction>,
}

/// Deterministic read-repair plan containing only exact committed envelopes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairPlan {
    /// Per-replica exact imports in stable node order.
    pub actions: Vec<RepairAction>,
}

impl RepairPlan {
    /// Apply all actions. Exact v2 imports make completed work safe to retry.
    pub fn apply(&self, transport: &mut dyn ReplicaTransport) -> Result<(), TransportApplyError> {
        for action in &self.actions {
            for transaction in &action.transactions {
                transport
                    .import_committed(action.node_id, &action.shard_key, transaction)
                    .map_err(|source| TransportApplyError {
                        node_id: action.node_id,
                        object_id: transaction.object_id().to_owned(),
                        source,
                    })?;
            }
        }
        Ok(())
    }
}

/// Exact repair application failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("repair import of {object_id} into {node_id} failed: {source}")]
pub struct TransportApplyError {
    /// Destination replica.
    pub node_id: Uuid,
    /// Authenticated transaction object identity.
    pub object_id: String,
    /// Transport-provided failure.
    #[source]
    pub source: super::TransportError,
}

/// Quorum read result preserving every maximal concurrent version.
#[derive(Debug, Clone, PartialEq)]
pub struct KvReadResult {
    /// Global maximal versions, sorted by stable version identity.
    pub versions: Vec<Version<String>>,
    /// Replicas that returned a readable response.
    pub responding_nodes: Vec<Uuid>,
    /// Exact objects required to converge every responding replica.
    pub repair_plan: RepairPlan,
}

/// Read a rendezvous-selected quorum, retain conflicts, and build exact repair.
pub fn read_kv_fanout(
    transport: &dyn ReplicaTransport,
    shard_key: &[u8],
    key: &str,
    nodes: &[NodeDescriptor],
    replication_factor: usize,
) -> Result<KvReadResult, ClusterDataError> {
    let selected = rendezvous_replicas(shard_key, nodes, replication_factor);
    let quorum = selected_quorum(&selected)?;
    let mut states = BTreeMap::new();
    let mut inventories = BTreeMap::new();
    for node_id in selected {
        let Ok(current) = transport.read_kv(node_id, key) else {
            continue;
        };
        let Ok(transactions) = transport.shard_transactions(node_id, shard_key) else {
            continue;
        };
        states.insert(node_id, current);
        inventories.insert(node_id, transactions);
    }
    if states.len() < quorum {
        return Err(ClusterDataError::QuorumUnreachable {
            required: quorum,
            received: states.len(),
        });
    }

    let versions = merge_current_versions(states.values().flatten())?;
    let repair_plan = plan_repairs(shard_key, &inventories, &versions)?;
    Ok(KvReadResult {
        versions,
        responding_nodes: states.into_keys().collect(),
        repair_plan,
    })
}

fn merge_current_versions<'a>(
    sets: impl Iterator<Item = &'a crate::v2::ConflictSet<String>>,
) -> Result<Vec<Version<String>>, ClusterDataError> {
    let mut by_id = BTreeMap::<Uuid, Version<String>>::new();
    for set in sets {
        for version in std::iter::once(&set.preferred).chain(&set.conflicts) {
            if let Some(existing) = by_id.get(&version.version_id)
                && existing != version
            {
                return Err(ClusterDataError::InvalidReplicaData(format!(
                    "version {} has contradictory contents",
                    version.version_id
                )));
            }
            by_id.insert(version.version_id, version.clone());
        }
    }
    let candidates = by_id.into_values().collect::<Vec<_>>();
    Ok(candidates
        .iter()
        .filter(|candidate| {
            !candidates.iter().any(|other| {
                other.version_id != candidate.version_id && dominates(other, candidate)
            })
        })
        .cloned()
        .collect())
}

fn dominates<T, U>(left: &Version<T>, right: &Version<U>) -> bool {
    left.writer_id == right.writer_id && left.sequence > right.sequence
        || left
            .observed_heads
            .get(&right.writer_id)
            .is_some_and(|sequence| *sequence >= right.sequence)
}

fn plan_repairs(
    shard_key: &[u8],
    inventories: &BTreeMap<Uuid, Vec<CommittedTransaction>>,
    versions: &[Version<String>],
) -> Result<RepairPlan, ClusterDataError> {
    type TransactionKey = (Uuid, u64);
    let mut present = BTreeMap::<Uuid, BTreeSet<TransactionKey>>::new();
    let mut all_transactions = Vec::new();
    for (node_id, transactions) in inventories {
        let node_present = present.entry(*node_id).or_default();
        for transaction in transactions {
            let key = (transaction.writer_id(), transaction.sequence());
            all_transactions.push(transaction.clone());
            node_present.insert(key);
        }
    }
    let complete = validate_transaction_inventory(all_transactions)?;
    for version in versions {
        if !complete.contains_key(&(version.writer_id, version.sequence)) {
            return Err(ClusterDataError::InvalidReplicaData(format!(
                "current version {} has no exact committed transaction",
                version.version_id
            )));
        }
    }
    let mut actions = Vec::new();
    for (node_id, node_present) in present {
        let transactions = complete
            .iter()
            .filter(|(key, _)| !node_present.contains(key))
            .map(|(_, transaction)| transaction.clone())
            .collect::<Vec<_>>();
        if !transactions.is_empty() {
            actions.push(RepairAction {
                node_id,
                shard_key: shard_key.to_vec(),
                transactions,
            });
        }
    }
    Ok(RepairPlan { actions })
}
