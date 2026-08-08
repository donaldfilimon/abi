//! Deterministic in-process Raft-style consensus core.
//!
//! This ports the reference-scoped state machine from `cluster.zig`: caller
//! driven elections, configured-size majority quorum, replication, simulated
//! failure and failover. Signed membership records and rendezvous shard
//! placement are additive coordination primitives; they are not yet integrated
//! into durable consensus, data movement, or production multi-host operation.

mod membership;
mod placement;
mod rebalance;
mod repair;
mod replication;

#[cfg(test)]
mod tests;

pub use membership::{
    MembershipChange, MembershipError, MembershipLedger, MembershipRecord, NodeDescriptor,
    NodeState, SignedMembershipRecord, sign_membership_record,
};
pub use placement::{DEFAULT_REPLICATION_FACTOR, rendezvous_replicas};
pub use rebalance::{
    RebalancePlan, RebalanceProgress, RebalanceStep, build_rebalance_plan, resume_rebalance,
};
pub use repair::{KvReadResult, RepairAction, RepairPlan, TransportApplyError, read_kv_fanout};
pub use replication::{
    ClusterDataError, ReplicaSearchHit, ReplicaTransport, ReplicationReceipt, TransportError,
    replicate_committed, search_fanout,
};

/// A node's current consensus role.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Role {
    /// Passive replica.
    #[default]
    Follower,
    /// Election participant soliciting votes.
    Candidate,
    /// Current elected coordinator.
    Leader,
}

/// One replicated log entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogEntry {
    /// Election term in which this entry was appended.
    pub term: u64,
    /// Opaque command bytes.
    pub data: Vec<u8>,
}

/// `RequestVote` result used by the TCP layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoteReply {
    /// Whether the peer granted its vote.
    pub granted: bool,
    /// Peer's current term.
    pub term: u64,
}

/// `AppendEntries` result used by the TCP layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppendReply {
    /// Whether the peer accepted the entry.
    pub ack: bool,
    /// Peer's current term.
    pub term: u64,
}

/// One configured cluster node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Stable numeric node id.
    pub id: u32,
    /// Highest observed election term.
    pub term: u64,
    /// Candidate voted for in the current term.
    pub voted_for: Option<u32>,
    /// Current role.
    pub role: Role,
    /// Whether the deterministic simulation considers this node reachable.
    pub alive: bool,
    /// Replicated entries.
    pub log: Vec<LogEntry>,
    /// One-past-last committed entry.
    pub commit_index: usize,
}

impl Node {
    /// Construct an alive follower.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self {
            id,
            term: 0,
            voted_for: None,
            role: Role::Follower,
            alive: true,
            log: Vec::new(),
            commit_index: 0,
        }
    }
}

/// Consensus operation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterError {
    /// No alive leader exists.
    NoLeader,
    /// A zero-node cluster is invalid.
    NotEnoughNodes,
    /// The requested node does not exist or is down.
    NodeNotFound,
    /// Replication could not reach the configured majority.
    QuorumUnreachable,
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NoLeader => "no leader",
            Self::NotEnoughNodes => "not enough nodes",
            Self::NodeNotFound => "node not found",
            Self::QuorumUnreachable => "quorum unreachable",
        })
    }
}

impl std::error::Error for ClusterError {}

/// Deterministic, single-process cluster simulation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cluster {
    nodes: Vec<Node>,
}

impl Cluster {
    /// Construct nodes numbered `0..node_count`.
    pub fn new(node_count: usize) -> Result<Self, ClusterError> {
        if node_count == 0 {
            return Err(ClusterError::NotEnoughNodes);
        }
        let nodes = (0..node_count)
            .map(|id| Node::new(u32::try_from(id).unwrap_or(u32::MAX)))
            .collect();
        Ok(Self { nodes })
    }

    /// Configured nodes.
    #[must_use]
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Majority of configured membership.
    #[must_use]
    pub const fn quorum(&self) -> usize {
        self.nodes.len() / 2 + 1
    }

    /// Number of currently reachable nodes.
    #[must_use]
    pub fn alive_count(&self) -> usize {
        self.nodes.iter().filter(|node| node.alive).count()
    }

    /// Current alive leader.
    #[must_use]
    pub fn leader(&self) -> Option<&Node> {
        self.nodes
            .iter()
            .find(|node| node.alive && node.role == Role::Leader)
    }

    /// Run one deterministic election.
    pub fn start_election(&mut self, candidate_id: u32) -> Result<bool, ClusterError> {
        let candidate_index = self
            .nodes
            .iter()
            .position(|node| node.id == candidate_id && node.alive)
            .ok_or(ClusterError::NodeNotFound)?;
        let new_term = self
            .nodes
            .iter()
            .map(|node| node.term)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        {
            let candidate = &mut self.nodes[candidate_index];
            candidate.term = new_term;
            candidate.role = Role::Candidate;
            candidate.voted_for = Some(candidate_id);
        }
        let mut votes = 1;
        for (index, peer) in self.nodes.iter_mut().enumerate() {
            if index != candidate_index && peer.alive && apply_vote(peer, new_term, candidate_id) {
                votes += 1;
            }
        }
        if votes >= self.quorum() {
            self.nodes[candidate_index].role = Role::Leader;
            for (index, peer) in self.nodes.iter_mut().enumerate() {
                if index != candidate_index && peer.alive {
                    peer.role = Role::Follower;
                }
            }
            Ok(true)
        } else {
            self.nodes[candidate_index].role = Role::Candidate;
            Ok(false)
        }
    }

    /// Append on the leader and every alive follower, committing at quorum.
    pub fn replicate(&mut self, data: &[u8]) -> Result<usize, ClusterError> {
        let leader_index = self
            .nodes
            .iter()
            .position(|node| node.alive && node.role == Role::Leader)
            .ok_or(ClusterError::NoLeader)?;
        let term = self.nodes[leader_index].term;
        self.nodes[leader_index].log.push(LogEntry {
            term,
            data: data.to_vec(),
        });
        let mut acknowledgements = 1;
        for (index, peer) in self.nodes.iter_mut().enumerate() {
            if index == leader_index || !peer.alive {
                continue;
            }
            peer.log.push(LogEntry {
                term,
                data: data.to_vec(),
            });
            acknowledgements += 1;
        }

        if acknowledgements < self.quorum() {
            return Err(ClusterError::QuorumUnreachable);
        }
        let committed = self.nodes[leader_index].log.len();
        for peer in self.nodes.iter_mut().filter(|node| node.alive) {
            if peer.log.len() >= committed {
                peer.commit_index = committed;
            }
        }
        Ok(acknowledgements)
    }

    /// Simulate a crash.
    pub fn fail_node(&mut self, id: u32) -> Result<(), ClusterError> {
        let node = self.node_mut(id)?;
        node.alive = false;
        node.role = Role::Follower;
        Ok(())
    }

    /// Make a simulated node reachable again.
    pub fn revive_node(&mut self, id: u32) -> Result<(), ClusterError> {
        self.node_mut(id)?.alive = true;
        Ok(())
    }

    /// Human-readable CLI status contract.
    #[must_use]
    pub fn status_line(&self) -> String {
        let leader = self.leader();
        format!(
            "nodes={} alive={} quorum={} leader={} term={} commit_index={}",
            self.nodes.len(),
            self.alive_count(),
            self.quorum(),
            leader.map_or_else(|| "none".to_string(), |node| node.id.to_string()),
            leader.map_or(0, |node| node.term),
            leader.map_or(0, |node| node.commit_index),
        )
    }

    fn node_mut(&mut self, id: u32) -> Result<&mut Node, ClusterError> {
        self.nodes
            .iter_mut()
            .find(|node| node.id == id)
            .ok_or(ClusterError::NodeNotFound)
    }
}

/// Apply one `RequestVote` under the oracle's term/vote rules.
pub fn apply_vote(node: &mut Node, term: u64, candidate: u32) -> bool {
    if term < node.term {
        return false;
    }
    if term > node.term {
        node.term = term;
        node.voted_for = None;
        node.role = Role::Follower;
    }
    if node.voted_for.is_none() || node.voted_for == Some(candidate) {
        node.voted_for = Some(candidate);
        node.role = Role::Follower;
        true
    } else {
        false
    }
}

/// Apply one `AppendEntries` frame.
pub fn apply_append(node: &mut Node, term: u64, data: &[u8]) -> bool {
    if term < node.term {
        return false;
    }
    node.term = term;
    node.role = Role::Follower;
    node.log.push(LogEntry {
        term,
        data: data.to_vec(),
    });
    true
}
