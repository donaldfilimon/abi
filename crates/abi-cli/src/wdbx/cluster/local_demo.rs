//! Authenticated local multi-process cluster proof.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead as _, BufReader};
use std::path::PathBuf;
use std::process::{Child, ChildStderr, Command, Stdio};

use abi_wdbx::cluster::{NodeDescriptor, rendezvous_replicas};
use abi_wdbx::{
    CommittedTransaction, ConflictSet, ReplicaSearchHit, ReplicaTransport, TransportError,
    V2Mutation, V2Store, dial_append, dial_shutdown, dial_vote, read_append_reply, read_kv_fanout,
    read_shutdown_reply, read_vote_reply, replicate_committed,
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

struct DemoTransport {
    stores: BTreeMap<Uuid, V2Store>,
    roots: Vec<PathBuf>,
    inventory: BTreeMap<(Uuid, Vec<u8>), Vec<CommittedTransaction>>,
    failed: BTreeSet<Uuid>,
}

impl DemoTransport {
    fn new(nodes: &[NodeDescriptor]) -> Result<Self, String> {
        let mut stores = BTreeMap::new();
        let mut roots = Vec::new();
        for node in nodes {
            let root = abi_foundation::temp_path::temp_file_path(
                &format!("wdbx-local-proof-node-{}", node.id),
                "store",
            );
            let store = V2Store::open(&root)
                .map_err(|error| format!("cannot open scratch replica {}: {error}", node.id))?;
            stores.insert(node.id, store);
            roots.push(root);
        }
        Ok(Self {
            stores,
            roots,
            inventory: BTreeMap::new(),
            failed: BTreeSet::new(),
        })
    }
}

impl Drop for DemoTransport {
    fn drop(&mut self) {
        self.stores.clear();
        for root in &self.roots {
            let _ = std::fs::remove_dir_all(root);
        }
    }
}

impl ReplicaTransport for DemoTransport {
    fn import_committed(
        &mut self,
        node_id: Uuid,
        shard_key: &[u8],
        transaction: &CommittedTransaction,
    ) -> Result<(), TransportError> {
        if self.failed.contains(&node_id) {
            return Err(TransportError::new("injected replica failure"));
        }
        self.stores
            .get_mut(&node_id)
            .ok_or_else(|| TransportError::new("unknown replica"))?
            .import_committed(transaction)
            .map_err(|error| TransportError::new(error.to_string()))?;
        let inventory = self
            .inventory
            .entry((node_id, shard_key.to_vec()))
            .or_default();
        if !inventory.contains(transaction) {
            inventory.push(transaction.clone());
            inventory.sort_by_key(|item| (item.writer_id(), item.sequence()));
        }
        Ok(())
    }

    fn read_kv(
        &self,
        node_id: Uuid,
        key: &str,
    ) -> Result<Option<ConflictSet<String>>, TransportError> {
        self.stores
            .get(&node_id)
            .map(|store| store.snapshot().get(key))
            .ok_or_else(|| TransportError::new("unknown replica"))
    }

    fn shard_transactions(
        &self,
        node_id: Uuid,
        shard_key: &[u8],
    ) -> Result<Vec<CommittedTransaction>, TransportError> {
        Ok(self
            .inventory
            .get(&(node_id, shard_key.to_vec()))
            .cloned()
            .unwrap_or_default())
    }

    fn shard_keys(&self, node_id: Uuid) -> Result<Vec<Vec<u8>>, TransportError> {
        Ok(self
            .inventory
            .keys()
            .filter(|(candidate, _)| *candidate == node_id)
            .map(|(_, key)| key.clone())
            .collect())
    }

    fn search(
        &self,
        _node_id: Uuid,
        _query: &[f32],
        _limit: usize,
    ) -> Result<Vec<ReplicaSearchHit>, TransportError> {
        Ok(Vec::new())
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
    let acknowledgements = append(&nodes, 0, 1, "set local-proof=true", &token)?;
    if acknowledgements < quorum {
        return Err("replicated write did not reach quorum".into());
    }

    nodes[0].shutdown(&token)?;
    let failover = elect(&nodes[1..], 1, 2, &token, quorum)?;
    let repair_acks = append(&nodes[1..], 1, 2, "repair local-proof=true", &token)?;
    if repair_acks < quorum {
        return Err("read repair did not reach surviving quorum".into());
    }
    for node in &mut nodes[1..] {
        node.shutdown(&token)?;
    }
    let (shard_placement, conflicts, read_repair) = prove_exact_data_plane(node_count)?;

    Ok(LocalProof {
        proof: "authenticated_local_multi_process",
        storage_proof_scope: "isolated_in_process_exact_transaction_replicas",
        nodes: node_count,
        election,
        replicated_write: ReplicatedWriteProof {
            acknowledgements,
            quorum,
        },
        shard_placement,
        failover,
        conflicts,
        read_repair,
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

fn prove_exact_data_plane(
    node_count: usize,
) -> Result<(ProofState, ProofState, ProofState), String> {
    let descriptors = (1..=node_count)
        .map(|value| {
            NodeDescriptor::active(
                Uuid::from_u128(u128::try_from(value).expect("bounded node id")),
                "127.0.0.1:0",
            )
            .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let shard_key = b"kv:local-proof-conflict";
    let first_root =
        abi_foundation::temp_path::temp_file_path("wdbx-local-proof-source-first", "store");
    let second_root =
        abi_foundation::temp_path::temp_file_path("wdbx-local-proof-source-second", "store");
    let mut first = V2Store::open(&first_root).map_err(|error| error.to_string())?;
    let mut second = V2Store::open(&second_root).map_err(|error| error.to_string())?;
    let first_transaction = first
        .commit_export(vec![V2Mutation::PutKv {
            key: "local-proof-conflict".into(),
            value: "first".into(),
        }])
        .map_err(|error| error.to_string())?;
    let second_transaction = second
        .commit_export(vec![V2Mutation::PutKv {
            key: "local-proof-conflict".into(),
            value: "second".into(),
        }])
        .map_err(|error| error.to_string())?;
    drop(first);
    drop(second);
    let _ = std::fs::remove_dir_all(&first_root);
    let _ = std::fs::remove_dir_all(&second_root);
    let mut transport = DemoTransport::new(&descriptors)?;
    let first_receipt = replicate_committed(
        &mut transport,
        shard_key,
        &descriptors,
        3,
        &first_transaction,
    )
    .map_err(|error| error.to_string())?;
    let placement = rendezvous_replicas(shard_key, &descriptors, 3);
    if first_receipt.selected != placement || placement.len() != 3 {
        return Err("rendezvous placement proof did not select three replicas".into());
    }
    let lagging = *placement
        .last()
        .ok_or_else(|| "placement proof selected no replicas".to_string())?;
    transport.failed.insert(lagging);
    let second_receipt = replicate_committed(
        &mut transport,
        shard_key,
        &descriptors,
        3,
        &second_transaction,
    )
    .map_err(|error| error.to_string())?;
    transport.failed.clear();
    if second_receipt.selected != placement || second_receipt.acknowledged.len() != 2 {
        return Err("injected lagging replica did not preserve selected quorum".into());
    }

    let before = read_kv_fanout(
        &transport,
        shard_key,
        "local-proof-conflict",
        &descriptors,
        3,
    )
    .map_err(|error| error.to_string())?;
    if before.versions.len() != 2 || before.repair_plan.actions.len() != 1 {
        return Err("concurrent versions or exact repair plan were not surfaced".into());
    }
    before
        .repair_plan
        .apply(&mut transport)
        .map_err(|error| error.to_string())?;
    let after = read_kv_fanout(
        &transport,
        shard_key,
        "local-proof-conflict",
        &descriptors,
        3,
    )
    .map_err(|error| error.to_string())?;
    if after.versions.len() != 2 || !after.repair_plan.actions.is_empty() {
        return Err("exact read repair did not converge every selected replica".into());
    }
    Ok((
        ProofState::Verified,
        ProofState::Verified,
        ProofState::Verified,
    ))
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

fn append(
    nodes: &[NodeProcess],
    leader: u32,
    term: u64,
    data: &str,
    token: &str,
) -> Result<usize, String> {
    let mut acknowledgements = 0;
    for node in nodes {
        let stream = dial_append(
            "127.0.0.1",
            node.port,
            term,
            Some(leader),
            data,
            Some(token),
        )
        .map_err(|error| format!("append dial failed: {error}"))?
        .ok_or_else(|| format!("node {} unreachable during replication", node.id))?;
        acknowledgements += usize::from(
            read_append_reply(stream)
                .map_err(|error| format!("append reply failed: {error}"))?
                .ack,
        );
    }
    Ok(acknowledgements)
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
