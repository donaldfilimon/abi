//! Hash-verified, resumable draining-node rebalance plans.

use super::replication::validate_transaction_inventory;
use super::{ClusterDataError, NodeDescriptor, NodeState, ReplicaTransport, rendezvous_replicas};
use crate::v2::CommittedTransaction;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use uuid::Uuid;

const REBALANCE_FORMAT_VERSION: u16 = 1;

/// One deterministic destination import for a draining shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebalanceStep {
    /// Stable shard-key bytes used by rendezvous placement.
    pub shard_key: Vec<u8>,
    /// Active destination replica.
    pub target_node: Uuid,
    /// Exact source objects ordered by writer then sequence.
    pub transactions: Vec<CommittedTransaction>,
}

/// Immutable plan whose digest covers every destination and transaction byte.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebalancePlan {
    format_version: u16,
    /// Node being drained and excluded from new placement.
    pub draining_node: Uuid,
    /// Stable ordered import steps.
    pub steps: Vec<RebalanceStep>,
    digest: String,
}

impl RebalancePlan {
    /// Lowercase SHA-256 of the canonical plan payload.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.digest
    }

    /// Verify the plan's format, ordering, and full canonical payload hash.
    pub fn verify(&self) -> Result<(), ClusterDataError> {
        if self.format_version != REBALANCE_FORMAT_VERSION {
            return Err(ClusterDataError::InvalidRebalancePlan(
                "unsupported format version".into(),
            ));
        }
        if self
            .steps
            .windows(2)
            .any(|pair| step_key(&pair[0]) >= step_key(&pair[1]))
        {
            return Err(ClusterDataError::InvalidRebalancePlan(
                "steps are not canonically ordered and distinct".into(),
            ));
        }
        for step in &self.steps {
            if step.target_node == self.draining_node {
                return Err(ClusterDataError::InvalidRebalancePlan(
                    "a step targets the draining node".into(),
                ));
            }
            validate_transaction_inventory(step.transactions.iter().cloned())?;
        }
        let expected = plan_digest(self.draining_node, &self.steps)?;
        if expected != self.digest {
            return Err(ClusterDataError::InvalidRebalancePlan(
                "payload digest mismatch".into(),
            ));
        }
        Ok(())
    }
}

/// Resume token bound to one exact rebalance plan.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RebalanceProgress {
    /// Digest of the only plan this token may resume.
    pub plan_digest: String,
    /// Fully imported step indexes.
    pub completed_steps: BTreeSet<usize>,
}

impl RebalanceProgress {
    /// Whether every plan step has completed.
    #[must_use]
    pub fn is_complete(&self, plan: &RebalancePlan) -> bool {
        self.plan_digest == plan.digest
            && self.completed_steps.iter().copied().eq(0..plan.steps.len())
    }
}

/// Inventory a draining member and produce a canonical exact-byte move plan.
pub fn build_rebalance_plan(
    transport: &dyn ReplicaTransport,
    draining_node: Uuid,
    nodes: &[NodeDescriptor],
    replication_factor: usize,
) -> Result<RebalancePlan, ClusterDataError> {
    if nodes
        .iter()
        .find(|node| node.id == draining_node)
        .is_none_or(|node| node.state != NodeState::Draining)
    {
        return Err(ClusterDataError::InvalidRebalancePlan(
            "source node is not a draining member".into(),
        ));
    }
    let mut shard_keys = transport
        .shard_keys(draining_node)
        .map_err(|error| ClusterDataError::InvalidReplicaData(error.to_string()))?;
    shard_keys.sort();
    shard_keys.dedup();
    let mut steps = Vec::new();
    for shard_key in shard_keys {
        let targets = rendezvous_replicas(&shard_key, nodes, replication_factor);
        if targets.is_empty() {
            return Err(ClusterDataError::NoReplicas);
        }
        let mut transactions = transport
            .shard_transactions(draining_node, &shard_key)
            .map_err(|error| ClusterDataError::InvalidReplicaData(error.to_string()))?;
        transactions.sort_by_key(|transaction| (transaction.writer_id(), transaction.sequence()));
        validate_transaction_inventory(transactions.iter().cloned())?;
        for target_node in targets {
            steps.push(RebalanceStep {
                shard_key: shard_key.clone(),
                target_node,
                transactions: transactions.clone(),
            });
        }
    }
    steps.sort_by(|left, right| step_key(left).cmp(&step_key(right)));
    let digest = plan_digest(draining_node, &steps)?;
    Ok(RebalancePlan {
        format_version: REBALANCE_FORMAT_VERSION,
        draining_node,
        steps,
        digest,
    })
}

/// Execute up to `step_budget` unfinished steps and return durable resume state.
pub fn resume_rebalance(
    transport: &mut dyn ReplicaTransport,
    plan: &RebalancePlan,
    mut progress: RebalanceProgress,
    step_budget: usize,
) -> Result<RebalanceProgress, ClusterDataError> {
    plan.verify()?;
    if progress.plan_digest.is_empty() {
        progress.plan_digest.clone_from(&plan.digest);
    }
    if progress.plan_digest != plan.digest {
        return Err(ClusterDataError::InvalidRebalancePlan(
            "resume token belongs to another plan".into(),
        ));
    }
    if progress
        .completed_steps
        .iter()
        .any(|index| *index >= plan.steps.len())
    {
        return Err(ClusterDataError::InvalidRebalancePlan(
            "resume token contains an out-of-range step".into(),
        ));
    }

    let mut completed_now = 0_usize;
    for (index, step) in plan.steps.iter().enumerate() {
        if completed_now >= step_budget || progress.completed_steps.contains(&index) {
            continue;
        }
        for transaction in &step.transactions {
            transport
                .import_committed(step.target_node, &step.shard_key, transaction)
                .map_err(|error| ClusterDataError::InvalidReplicaData(error.to_string()))?;
        }
        progress.completed_steps.insert(index);
        completed_now += 1;
    }
    Ok(progress)
}

fn step_key(step: &RebalanceStep) -> (&[u8], Uuid) {
    (&step.shard_key, step.target_node)
}

fn plan_digest(draining_node: Uuid, steps: &[RebalanceStep]) -> Result<String, ClusterDataError> {
    let payload = serde_json::to_vec(&(REBALANCE_FORMAT_VERSION, draining_node, steps))
        .map_err(|error| ClusterDataError::InvalidRebalancePlan(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}
