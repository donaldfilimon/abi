use super::*;
use crate::v2::{CommittedTransaction, RecordId, V2Mutation, V2Store};
use abi_foundation::temp_path::temp_file_path;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use uuid::Uuid;

struct MockReplica {
    store: V2Store,
    shards: BTreeMap<Vec<u8>, Vec<CommittedTransaction>>,
    search_hits: Vec<ReplicaSearchHit>,
}

struct MockTransport {
    replicas: BTreeMap<Uuid, MockReplica>,
    failed: BTreeSet<Uuid>,
    roots: Vec<PathBuf>,
    imported: Vec<(Uuid, Vec<u8>, CommittedTransaction)>,
}

impl MockTransport {
    fn new(node_ids: impl IntoIterator<Item = Uuid>) -> Self {
        let mut replicas = BTreeMap::new();
        let mut roots = Vec::new();
        for node_id in node_ids {
            let root = temp_file_path("wdbx-cluster-mock", "store");
            let store = V2Store::open(&root).expect("mock store");
            roots.push(root);
            replicas.insert(
                node_id,
                MockReplica {
                    store,
                    shards: BTreeMap::new(),
                    search_hits: Vec::new(),
                },
            );
        }
        Self {
            replicas,
            failed: BTreeSet::new(),
            roots,
            imported: Vec::new(),
        }
    }

    fn seed(&mut self, node_id: Uuid, shard_key: &[u8], transaction: &CommittedTransaction) {
        self.import_committed(node_id, shard_key, transaction)
            .expect("seed replica");
        self.imported.clear();
    }
}

impl Drop for MockTransport {
    fn drop(&mut self) {
        self.replicas.clear();
        for root in &self.roots {
            let _ = std::fs::remove_dir_all(root);
        }
    }
}

impl ReplicaTransport for MockTransport {
    fn import_committed(
        &mut self,
        node_id: Uuid,
        shard_key: &[u8],
        transaction: &CommittedTransaction,
    ) -> Result<(), TransportError> {
        if self.failed.contains(&node_id) {
            return Err(TransportError::new("injected failure"));
        }
        let replica = self
            .replicas
            .get_mut(&node_id)
            .ok_or_else(|| TransportError::new("unknown node"))?;
        replica
            .store
            .import_committed(transaction)
            .map_err(|error| TransportError::new(error.to_string()))?;
        let transactions = replica.shards.entry(shard_key.to_vec()).or_default();
        if !transactions.iter().any(|existing| {
            existing.writer_id() == transaction.writer_id()
                && existing.sequence() == transaction.sequence()
        }) {
            transactions.push(transaction.clone());
            transactions.sort_by_key(|item| (item.writer_id(), item.sequence()));
        }
        self.imported
            .push((node_id, shard_key.to_vec(), transaction.clone()));
        Ok(())
    }

    fn read_kv(
        &self,
        node_id: Uuid,
        key: &str,
    ) -> Result<Option<crate::v2::ConflictSet<String>>, TransportError> {
        if self.failed.contains(&node_id) {
            return Err(TransportError::new("injected failure"));
        }
        let replica = self
            .replicas
            .get(&node_id)
            .ok_or_else(|| TransportError::new("unknown node"))?;
        Ok(replica.store.snapshot().get(key))
    }

    fn shard_transactions(
        &self,
        node_id: Uuid,
        shard_key: &[u8],
    ) -> Result<Vec<CommittedTransaction>, TransportError> {
        if self.failed.contains(&node_id) {
            return Err(TransportError::new("injected failure"));
        }
        let replica = self
            .replicas
            .get(&node_id)
            .ok_or_else(|| TransportError::new("unknown node"))?;
        Ok(replica.shards.get(shard_key).cloned().unwrap_or_default())
    }

    fn shard_keys(&self, node_id: Uuid) -> Result<Vec<Vec<u8>>, TransportError> {
        let replica = self
            .replicas
            .get(&node_id)
            .ok_or_else(|| TransportError::new("unknown node"))?;
        Ok(replica.shards.keys().cloned().collect())
    }

    fn search(
        &self,
        node_id: Uuid,
        _query: &[f32],
        _limit: usize,
    ) -> Result<Vec<ReplicaSearchHit>, TransportError> {
        if self.failed.contains(&node_id) {
            return Err(TransportError::new("injected failure"));
        }
        self.replicas
            .get(&node_id)
            .map(|replica| replica.search_hits.clone())
            .ok_or_else(|| TransportError::new("unknown node"))
    }
}

struct TransactionFixture {
    transaction: CommittedTransaction,
    store: Option<V2Store>,
    root: PathBuf,
}

impl TransactionFixture {
    fn kv(key: &str, value: &str) -> Self {
        let root = temp_file_path("wdbx-cluster-origin", "store");
        let mut store = V2Store::open(&root).expect("origin store");
        let transaction = store
            .commit_export(vec![V2Mutation::PutKv {
                key: key.to_owned(),
                value: value.to_owned(),
            }])
            .expect("export transaction");
        Self {
            transaction,
            store: Some(store),
            root,
        }
    }
}

impl Drop for TransactionFixture {
    fn drop(&mut self) {
        self.store.take();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn descriptors(count: usize) -> Vec<NodeDescriptor> {
    (1..=count)
        .map(|value| {
            NodeDescriptor::active(
                Uuid::from_u128(u128::try_from(value).expect("small fixture id")),
                format!("127.0.0.1:{}", 9000 + value),
            )
            .expect("descriptor")
        })
        .collect()
}

#[test]
fn single_leader_is_elected_with_a_configured_quorum() {
    let mut cluster = Cluster::new(3).expect("cluster");
    assert!(cluster.start_election(0).expect("election"));
    let leader = cluster.leader().expect("leader");
    assert_eq!(leader.id, 0);
    assert_eq!(leader.term, 1);
    assert_eq!(
        cluster
            .nodes()
            .iter()
            .filter(|node| node.role == Role::Leader)
            .count(),
        1
    );
}

#[test]
fn replication_reaches_every_alive_replica_and_commits() {
    let mut cluster = Cluster::new(3).expect("cluster");
    cluster.start_election(0).expect("election");
    assert_eq!(cluster.replicate(b"set x=1").expect("replicate"), 3);
    for node in cluster.nodes() {
        assert_eq!(node.log.len(), 1);
        assert_eq!(node.commit_index, 1);
        assert_eq!(node.log[0].data, b"set x=1");
    }
}

#[test]
fn leader_failover_uses_a_higher_term_and_surviving_majority() {
    let mut cluster = Cluster::new(3).expect("cluster");
    cluster.start_election(0).expect("election");
    cluster.replicate(b"a").expect("replicate");
    cluster.fail_node(0).expect("fail");
    assert!(cluster.leader().is_none());
    assert!(cluster.start_election(1).expect("re-election"));
    assert_eq!(cluster.leader().expect("leader").term, 2);
    assert_eq!(cluster.replicate(b"b").expect("replicate"), 2);
}

#[test]
fn configured_quorum_is_not_redefined_after_failures() {
    let mut cluster = Cluster::new(3).expect("cluster");
    cluster.start_election(0).expect("election");
    cluster.fail_node(0).expect("fail");
    cluster.fail_node(1).expect("fail");
    assert!(!cluster.start_election(2).expect("failed election"));
    assert_eq!(cluster.replicate(b"c"), Err(ClusterError::NoLeader));
}

#[test]
fn primitives_match_the_cluster_driver() {
    let mut driven = Cluster::new(3).expect("cluster");
    driven.start_election(0).expect("election");
    driven.replicate(b"hello").expect("replicate");

    let mut direct = Cluster::new(3).expect("cluster");
    direct.nodes[0].term = 1;
    direct.nodes[0].role = Role::Candidate;
    direct.nodes[0].voted_for = Some(0);
    assert!(apply_vote(&mut direct.nodes[1], 1, 0));
    assert!(apply_vote(&mut direct.nodes[2], 1, 0));
    direct.nodes[0].role = Role::Leader;
    direct.nodes[0].log.push(LogEntry {
        term: 1,
        data: b"hello".to_vec(),
    });
    assert!(apply_append(&mut direct.nodes[1], 1, b"hello"));
    assert!(apply_append(&mut direct.nodes[2], 1, b"hello"));
    for node in &mut direct.nodes {
        node.commit_index = 1;
    }

    for (left, right) in driven.nodes().iter().zip(direct.nodes()) {
        assert_eq!(left.term, right.term);
        assert_eq!(left.voted_for, right.voted_for);
        assert_eq!(left.role, right.role);
        assert_eq!(left.log, right.log);
        assert_eq!(left.commit_index, right.commit_index);
    }
}

#[test]
fn status_line_matches_the_cli_shape() {
    let mut cluster = Cluster::new(1).expect("cluster");
    assert_eq!(
        cluster.status_line(),
        "nodes=1 alive=1 quorum=1 leader=none term=0 commit_index=0"
    );
    assert!(cluster.start_election(0).expect("election"));
    assert_eq!(
        cluster.status_line(),
        "nodes=1 alive=1 quorum=1 leader=0 term=1 commit_index=0"
    );
}

#[test]
fn selected_replication_requires_quorum_and_preserves_exact_identity() {
    let nodes = descriptors(3);
    let mut transport = MockTransport::new(nodes.iter().map(|node| node.id));
    let fixture = TransactionFixture::kv("key", "value");
    transport.failed.insert(nodes[0].id);

    let receipt = replicate_committed(&mut transport, b"shard-a", &nodes, 3, &fixture.transaction)
        .expect("two acknowledgements form quorum");
    assert_eq!(receipt.quorum, 2);
    assert_eq!(receipt.acknowledged.len(), 2);
    assert_eq!(receipt.transaction_sha256, fixture.transaction.sha256());
    for (_, shard, imported) in &transport.imported {
        assert_eq!(shard, b"shard-a");
        assert_eq!(imported, &fixture.transaction);
        assert_eq!(imported.encoded(), fixture.transaction.encoded());
    }

    transport.failed.insert(nodes[1].id);
    assert!(matches!(
        replicate_committed(&mut transport, b"shard-a", &nodes, 3, &fixture.transaction,),
        Err(ClusterDataError::QuorumUnreachable {
            required: 2,
            received: 1
        })
    ));
}

#[test]
fn read_fanout_retains_conflicts_and_repairs_with_exact_objects() {
    let nodes = descriptors(2);
    let mut transport = MockTransport::new(nodes.iter().map(|node| node.id));
    let left = TransactionFixture::kv("shared", "left");
    let right = TransactionFixture::kv("shared", "right");
    transport.seed(nodes[0].id, b"shared", &left.transaction);
    transport.seed(nodes[1].id, b"shared", &right.transaction);

    let read =
        read_kv_fanout(&transport, b"shared", "shared", &nodes, 2).expect("quorum conflict read");
    assert_eq!(read.versions.len(), 2);
    assert_eq!(
        read.versions
            .iter()
            .map(|version| version.value.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["left", "right"])
    );
    assert_eq!(read.repair_plan.actions.len(), 2);
    assert!(read.repair_plan.actions.iter().all(|action| {
        action.transactions.len() == 1
            && (action.transactions[0] == left.transaction
                || action.transactions[0] == right.transaction)
    }));

    read.repair_plan.apply(&mut transport).expect("repair");
    for node in &nodes {
        let conflicts = transport
            .read_kv(node.id, "shared")
            .expect("read")
            .expect("value");
        assert_eq!(conflicts.conflicts.len(), 1);
    }
    for (_, _, imported) in &transport.imported {
        assert!(imported == &left.transaction || imported == &right.transaction);
    }
}

#[test]
fn read_repair_refuses_a_value_without_its_exact_transaction() {
    let nodes = descriptors(1);
    let mut transport = MockTransport::new(nodes.iter().map(|node| node.id));
    let fixture = TransactionFixture::kv("orphan", "value");
    transport.seed(nodes[0].id, b"orphan", &fixture.transaction);
    transport
        .replicas
        .get_mut(&nodes[0].id)
        .expect("replica")
        .shards
        .clear();

    assert!(matches!(
        read_kv_fanout(&transport, b"orphan", "orphan", &nodes, 1),
        Err(ClusterDataError::InvalidReplicaData(message))
            if message.contains("no exact committed transaction")
    ));
}

#[test]
fn vector_fanout_is_deduplicated_and_stably_sorted() {
    let nodes = descriptors(3);
    let mut transport = MockTransport::new(nodes.iter().map(|node| node.id));
    let low = RecordId::Legacy(1);
    let tie_first = RecordId::Legacy(2);
    let tie_second = RecordId::Legacy(3);
    transport
        .replicas
        .get_mut(&nodes[0].id)
        .unwrap()
        .search_hits = vec![
        ReplicaSearchHit {
            id: low,
            score: 0.5,
        },
        ReplicaSearchHit {
            id: tie_second,
            score: 0.9,
        },
    ];
    transport
        .replicas
        .get_mut(&nodes[1].id)
        .unwrap()
        .search_hits = vec![
        ReplicaSearchHit {
            id: tie_first,
            score: 0.9,
        },
        ReplicaSearchHit {
            id: low,
            score: 0.4,
        },
    ];
    transport
        .replicas
        .get_mut(&nodes[2].id)
        .unwrap()
        .search_hits = vec![ReplicaSearchHit {
        id: low,
        score: 0.5,
    }];

    let hits =
        search_fanout(&transport, b"vectors", &nodes, 3, &[1.0, 0.0], 10).expect("search quorum");
    assert_eq!(
        hits.iter().map(|hit| hit.id).collect::<Vec<_>>(),
        vec![tie_first, tie_second, low]
    );
    assert_eq!(hits[2].score, 0.5);
}

#[test]
fn rebalance_plan_is_hash_verified_and_resumes_without_repeating_steps() {
    let mut nodes = descriptors(3);
    let draining = nodes[0].id;
    nodes[0].state = NodeState::Draining;
    let mut transport = MockTransport::new(nodes.iter().map(|node| node.id));
    let first = TransactionFixture::kv("first", "one");
    let second = TransactionFixture::kv("second", "two");
    transport.seed(draining, b"a", &first.transaction);
    transport.seed(draining, b"b", &second.transaction);

    let plan = build_rebalance_plan(&transport, draining, &nodes, 2).expect("plan");
    plan.verify().expect("hash");
    assert_eq!(plan.steps.len(), 4);
    assert!(plan.steps.iter().all(|step| step.target_node != draining));

    let progress = resume_rebalance(&mut transport, &plan, RebalanceProgress::default(), 1)
        .expect("first slice");
    assert_eq!(progress.completed_steps.len(), 1);
    let imports_after_first = transport.imported.len();
    let progress =
        resume_rebalance(&mut transport, &plan, progress, usize::MAX).expect("resume remainder");
    assert!(progress.is_complete(&plan));
    assert_eq!(transport.imported.len(), imports_after_first + 3);

    let mut tampered = serde_json::to_value(&plan).expect("serialize plan");
    tampered["steps"][0]["target_node"] = serde_json::json!(Uuid::new_v4());
    let tampered: RebalancePlan = serde_json::from_value(tampered).expect("decode plan");
    assert!(matches!(
        tampered.verify(),
        Err(ClusterDataError::InvalidRebalancePlan(_))
    ));
}
