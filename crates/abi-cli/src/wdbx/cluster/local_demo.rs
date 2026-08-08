//! Authenticated local multi-process cluster proof.

use std::collections::BTreeMap;
use std::io::{BufRead as _, BufReader};
use std::process::{Child, ChildStderr, Command, Stdio};

use abi_wdbx::cluster::{NodeDescriptor, rendezvous_replicas};
use abi_wdbx::{
    ClusterDataRequest, ClusterDataResponse, CommittedTransaction, ConflictSet, ReplicaSearchHit,
    ReplicaTransport, TransportError, dial_data, dial_shutdown, dial_vote, read_data_reply,
    read_kv_fanout, read_shutdown_reply, read_vote_reply, replicate_committed,
};
use uuid::Uuid;

use crate::app::Outcome;

const MIN_NODES: usize = 3;
const MAX_NODES: usize = 9;

#[derive(Debug)]
struct ElectionProof {
    leader: u32,
    term: u64,
    votes: usize,
    quorum: usize,
}

#[derive(Debug)]
struct ReplicatedWriteProof {
    acknowledgements: usize,
    quorum: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProofState {
    Verified,
    Unverified,
}

impl ProofState {
    const fn as_bool(self) -> bool {
        matches!(self, Self::Verified)
    }
}

#[derive(Debug)]
struct LocalProof {
    proof: &'static str,
    storage_proof_scope: &'static str,
    nodes: usize,
    election: ElectionProof,
    replicated_write: ReplicatedWriteProof,
    shard_placement: ProofState,
    failover: ElectionProof,
    conflicts: ProofState,
    read_repair: ProofState,
    child_teardown: ProofState,
}

#[derive(Debug)]
struct NodeProcess {
    id: u32,
    port: u16,
    child: Child,
    _stderr: ChildStderr,
}

struct ChildTransport {
    ports: BTreeMap<Uuid, u16>,
    token: String,
}

impl ChildTransport {
    fn new(nodes: &[NodeProcess], descriptors: &[NodeDescriptor], token: &str) -> Self {
        Self {
            ports: descriptors
                .iter()
                .zip(nodes)
                .map(|(descriptor, node)| (descriptor.id, node.port))
                .collect(),
            token: token.to_owned(),
        }
    }

    fn exchange(
        &self,
        node_id: Uuid,
        request: &ClusterDataRequest,
    ) -> Result<ClusterDataResponse, TransportError> {
        let port = self
            .ports
            .get(&node_id)
            .copied()
            .ok_or_else(|| TransportError::new("unknown replica"))?;
        let stream = dial_data("127.0.0.1", port, &self.token, request)
            .map_err(|error| TransportError::new(error.to_string()))?
            .ok_or_else(|| TransportError::new("replica is unreachable"))?;
        let response =
            read_data_reply(stream).map_err(|error| TransportError::new(error.to_string()))?;
        match response {
            ClusterDataResponse::Error { message } => Err(TransportError::new(message)),
            response => Ok(response),
        }
    }

    fn commit_kv(
        &self,
        node_id: Uuid,
        shard_key: &[u8],
        key: &str,
        value: &str,
    ) -> Result<CommittedTransaction, TransportError> {
        match self.exchange(
            node_id,
            &ClusterDataRequest::CommitKv {
                shard_key: shard_key.to_vec(),
                key: key.to_owned(),
                value: value.to_owned(),
            },
        )? {
            ClusterDataResponse::Transaction { transaction } => Ok(transaction),
            _ => Err(TransportError::new("unexpected commit response")),
        }
    }

    fn export_transaction(
        &self,
        node_id: Uuid,
        transaction: &CommittedTransaction,
    ) -> Result<CommittedTransaction, TransportError> {
        match self.exchange(
            node_id,
            &ClusterDataRequest::ExportTransaction {
                writer_id: transaction.writer_id(),
                sequence: transaction.sequence(),
            },
        )? {
            ClusterDataResponse::Transaction { transaction } => Ok(transaction),
            _ => Err(TransportError::new("unexpected export response")),
        }
    }
}

impl ReplicaTransport for ChildTransport {
    fn import_committed(
        &mut self,
        node_id: Uuid,
        shard_key: &[u8],
        transaction: &CommittedTransaction,
    ) -> Result<(), TransportError> {
        match self.exchange(
            node_id,
            &ClusterDataRequest::ImportCommitted {
                shard_key: shard_key.to_vec(),
                transaction: transaction.clone(),
            },
        )? {
            ClusterDataResponse::Imported { transaction_id }
                if transaction_id == transaction.transaction_id() =>
            {
                Ok(())
            }
            _ => Err(TransportError::new("unexpected import response")),
        }
    }

    fn read_kv(
        &self,
        node_id: Uuid,
        key: &str,
    ) -> Result<Option<ConflictSet<String>>, TransportError> {
        match self.exchange(
            node_id,
            &ClusterDataRequest::ReadKv {
                key: key.to_owned(),
            },
        )? {
            ClusterDataResponse::Kv { current } => Ok(current.map(ConflictSet::from)),
            _ => Err(TransportError::new("unexpected read response")),
        }
    }

    fn shard_transactions(
        &self,
        node_id: Uuid,
        shard_key: &[u8],
    ) -> Result<Vec<CommittedTransaction>, TransportError> {
        match self.exchange(
            node_id,
            &ClusterDataRequest::ShardTransactions {
                shard_key: shard_key.to_vec(),
            },
        )? {
            ClusterDataResponse::Transactions { transactions } => Ok(transactions),
            _ => Err(TransportError::new("unexpected shard response")),
        }
    }

    fn shard_keys(&self, node_id: Uuid) -> Result<Vec<Vec<u8>>, TransportError> {
        match self.exchange(node_id, &ClusterDataRequest::ShardKeys)? {
            ClusterDataResponse::ShardKeys { shard_keys } => Ok(shard_keys),
            _ => Err(TransportError::new("unexpected shard-key response")),
        }
    }

    fn search(
        &self,
        _node_id: Uuid,
        _query: &[f32],
        _limit: usize,
    ) -> Result<Vec<ReplicaSearchHit>, TransportError> {
        Err(TransportError::new(
            "vector search is outside the local data-plane proof",
        ))
    }
}

impl NodeProcess {
    fn spawn(id: u32, node_count: usize, token: &str) -> Result<Self, String> {
        let executable = std::env::current_exe()
            .map_err(|error| format!("cannot resolve current ABI binary: {error}"))?;
        let peers = (0..node_count)
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let mut child = Command::new(executable)
            .args([
                "wdbx",
                "cluster",
                "serve",
                "0",
                &id.to_string(),
                "127.0.0.1",
            ])
            .env("ABI_WDBX_CLUSTER_TOKEN", token)
            .env("ABI_WDBX_CLUSTER_PEERS", peers)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("cannot spawn cluster node {id}: {error}"))?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| format!("cluster node {id} has no readiness stream"))?;
        let mut readiness = String::new();
        BufReader::new(&mut stderr)
            .read_line(&mut readiness)
            .map_err(|error| format!("cannot read cluster node {id} readiness: {error}"))?;
        let marker = "wdbx cluster RPC serving on 127.0.0.1:";
        let port = readiness
            .strip_prefix(marker)
            .and_then(|rest| rest.split_once(' '))
            .and_then(|(port, _)| port.parse::<u16>().ok())
            .filter(|port| *port != 0)
            .ok_or_else(|| format!("cluster node {id} emitted invalid readiness"))?;
        Ok(Self {
            id,
            port,
            child,
            _stderr: stderr,
        })
    }

    fn shutdown(&mut self, token: &str) -> Result<(), String> {
        let stream = dial_shutdown("127.0.0.1", self.port, token)
            .map_err(|error| format!("node {} shutdown request failed: {error}", self.id))?
            .ok_or_else(|| format!("node {} became unreachable", self.id))?;
        if !read_shutdown_reply(stream)
            .map_err(|error| format!("node {} shutdown reply failed: {error}", self.id))?
        {
            return Err(format!("node {} rejected shutdown", self.id));
        }
        let status = self
            .child
            .wait()
            .map_err(|error| format!("node {} wait failed: {error}", self.id))?;
        if !status.success() {
            return Err(format!("node {} exited unsuccessfully", self.id));
        }
        Ok(())
    }
}

impl Drop for NodeProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

pub(super) fn run(node_count: usize, json: bool) -> Outcome {
    if !(MIN_NODES..=MAX_NODES).contains(&node_count) {
        return Outcome::stderr("cluster local-demo supports 3..=9 nodes\n".into(), 2);
    }
    match prove(node_count) {
        Ok(proof) => render(&proof, json),
        Err(detail) => Outcome::stderr(format!("cluster local-demo failed: {detail}\n"), 1),
    }
}

fn prove(node_count: usize) -> Result<LocalProof, String> {
    let token = Uuid::new_v4().simple().to_string();
    let mut nodes = (0..node_count)
        .map(|id| NodeProcess::spawn(u32::try_from(id).unwrap_or(u32::MAX), node_count, &token))
        .collect::<Result<Vec<_>, _>>()?;
    let quorum = node_count / 2 + 1;
    let election = elect(&nodes, 0, 1, &token, quorum)?;
    let descriptors = descriptors(node_count)?;
    let mut transport = ChildTransport::new(&nodes, &descriptors, &token);
    let data_proof = prove_exact_data_plane(
        node_count,
        &descriptors,
        &mut transport,
        &mut nodes,
        &token,
        quorum,
    )?;
    for node in &mut nodes[1..] {
        node.shutdown(&token)?;
    }

    Ok(LocalProof {
        proof: "authenticated_local_multi_process",
        storage_proof_scope: "isolated_child_process_exact_transaction_replicas",
        nodes: node_count,
        election,
        replicated_write: ReplicatedWriteProof {
            acknowledgements: data_proof.acknowledgements,
            quorum: data_proof.quorum,
        },
        shard_placement: data_proof.shard_placement,
        failover: data_proof.failover,
        conflicts: data_proof.conflicts,
        read_repair: data_proof.read_repair,
        child_teardown: if nodes
            .iter_mut()
            .all(|node| node.child.try_wait().ok().flatten().is_some())
        {
            ProofState::Verified
        } else {
            ProofState::Unverified
        },
    })
}

struct ExactDataProof {
    acknowledgements: usize,
    quorum: usize,
    shard_placement: ProofState,
    failover: ElectionProof,
    conflicts: ProofState,
    read_repair: ProofState,
}

fn descriptors(node_count: usize) -> Result<Vec<NodeDescriptor>, String> {
    (1..=node_count)
        .map(|value| {
            NodeDescriptor::active(
                Uuid::from_u128(u128::try_from(value).expect("bounded node id")),
                "127.0.0.1:0",
            )
            .map_err(|error| error.to_string())
        })
        .collect()
}

fn prove_exact_data_plane(
    node_count: usize,
    descriptors: &[NodeDescriptor],
    transport: &mut ChildTransport,
    nodes: &mut [NodeProcess],
    token: &str,
    quorum: usize,
) -> Result<ExactDataProof, String> {
    let shard_key = b"kv:local-proof-conflict";
    let first_transaction = transport
        .commit_kv(
            descriptors[0].id,
            shard_key,
            "local-proof-conflict",
            "first",
        )
        .map_err(|error| error.to_string())?;
    let exported = transport
        .export_transaction(descriptors[0].id, &first_transaction)
        .map_err(|error| error.to_string())?;
    if exported != first_transaction || exported.encoded() != first_transaction.encoded() {
        return Err("leader did not export the exact committed envelope".into());
    }

    // The successor commits without having observed the leader's transaction.
    // Importing the leader object afterwards must therefore retain a conflict.
    let second_transaction = transport
        .commit_kv(
            descriptors[1].id,
            shard_key,
            "local-proof-conflict",
            "second",
        )
        .map_err(|error| error.to_string())?;
    let first_receipt = replicate_committed(
        transport,
        shard_key,
        descriptors,
        node_count,
        &first_transaction,
    )
    .map_err(|error| error.to_string())?;
    let placement = rendezvous_replicas(shard_key, descriptors, node_count);
    if first_receipt.selected != placement
        || placement.len() != node_count
        || first_receipt.acknowledged.len() != node_count
    {
        return Err("rendezvous placement did not import the exact object on every child".into());
    }
    for descriptor in descriptors {
        let exact = transport
            .shard_transactions(descriptor.id, shard_key)
            .map_err(|error| error.to_string())?;
        if !exact
            .iter()
            .any(|transaction| transaction == &first_transaction)
        {
            return Err("child shard export omitted the replicated exact envelope".into());
        }
    }

    nodes[0].shutdown(token)?;
    let failover = elect(&nodes[1..], 1, 2, token, quorum)?;
    let before = read_kv_fanout(
        transport,
        shard_key,
        "local-proof-conflict",
        descriptors,
        node_count,
    )
    .map_err(|error| error.to_string())?;
    if before.versions.len() != 2
        || before.repair_plan.actions.len() != node_count.saturating_sub(2)
        || !before
            .repair_plan
            .actions
            .iter()
            .all(|action| action.transactions == [second_transaction.clone()])
    {
        return Err("post-failover conflicts or exact stale-node repair were not surfaced".into());
    }
    before
        .repair_plan
        .apply(transport)
        .map_err(|error| error.to_string())?;
    let after = read_kv_fanout(
        transport,
        shard_key,
        "local-proof-conflict",
        descriptors,
        node_count,
    )
    .map_err(|error| error.to_string())?;
    if after.versions.len() != 2 || !after.repair_plan.actions.is_empty() {
        return Err("exact read repair did not converge every surviving child".into());
    }
    Ok(ExactDataProof {
        acknowledgements: first_receipt.acknowledged.len(),
        quorum: first_receipt.quorum,
        shard_placement: ProofState::Verified,
        failover,
        conflicts: ProofState::Verified,
        read_repair: ProofState::Verified,
    })
}

fn elect(
    nodes: &[NodeProcess],
    candidate: u32,
    term: u64,
    token: &str,
    quorum: usize,
) -> Result<ElectionProof, String> {
    let mut votes = 0;
    for node in nodes {
        let stream = dial_vote("127.0.0.1", node.port, term, candidate, Some(token))
            .map_err(|error| format!("vote dial failed: {error}"))?
            .ok_or_else(|| format!("node {} unreachable during election", node.id))?;
        votes += usize::from(
            read_vote_reply(stream)
                .map_err(|error| format!("vote reply failed: {error}"))?
                .granted,
        );
    }
    if votes < quorum {
        return Err(format!(
            "election reached {votes} votes; quorum is {quorum}"
        ));
    }
    Ok(ElectionProof {
        leader: candidate,
        term,
        votes,
        quorum,
    })
}

fn render(proof: &LocalProof, json: bool) -> Outcome {
    if json {
        let mut stdout = serde_json::to_string(&serde_json::json!({
            "proof": proof.proof,
            "storage_proof_scope": proof.storage_proof_scope,
            "nodes": proof.nodes,
            "election": {
                "leader": proof.election.leader,
                "term": proof.election.term,
                "votes": proof.election.votes,
                "quorum": proof.election.quorum,
            },
            "replicated_write": {
                "acknowledgements": proof.replicated_write.acknowledgements,
                "quorum": proof.replicated_write.quorum,
            },
            "shard_placement_verified": proof.shard_placement.as_bool(),
            "failover": {
                "leader": proof.failover.leader,
                "term": proof.failover.term,
                "votes": proof.failover.votes,
                "quorum": proof.failover.quorum,
            },
            "conflicts_observed": proof.conflicts.as_bool(),
            "read_repair_completed": proof.read_repair.as_bool(),
            "children_reaped": proof.child_teardown.as_bool(),
        }))
        .expect("proof serializes");
        stdout.push('\n');
        return Outcome {
            stdout,
            stderr: String::new(),
            exit_code: 0,
        };
    }
    Outcome::stderr(
        format!(
            "authenticated local multi-process proof: nodes={} election_term={} write_acks={} failover_term={} read_repair={} children_reaped={}\nproduction multi-host deployment remains unverified\n",
            proof.nodes,
            proof.election.term,
            proof.replicated_write.acknowledgements,
            proof.failover.term,
            proof.read_repair.as_bool(),
            proof.child_teardown.as_bool(),
        ),
        0,
    )
}
