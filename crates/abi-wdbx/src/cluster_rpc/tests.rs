use super::*;
use std::thread;

fn data_request(port: u16, token: &str, request: &ClusterDataRequest) -> ClusterDataResponse {
    let stream = dial_data("127.0.0.1", port, token, request)
        .expect("dial")
        .expect("reachable");
    read_data_reply(stream).expect("data reply")
}

#[test]
fn policy_validation_and_reload_are_explicit() {
    assert!(matches!(
        ClusterPolicy::from_values(Some("bad token"), None),
        Err(RpcError::InvalidToken)
    ));
    assert!(matches!(
        ClusterPolicy::from_values(None, Some("1,,2")),
        Err(RpcError::InvalidPeers)
    ));
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0, 1,2")).expect("policy");
    assert!(policy.allows_peer(2));
    assert!(!policy.allows_peer(99));
    assert!(policy.auth.matches(Some("secret")));
    assert!(!policy.auth.matches(Some("wrong")));
}

#[test]
fn loopback_vote_and_append_mutate_the_node() {
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0,1")).expect("policy");
    let mut server = ClusterRpcServer::bind("127.0.0.1", 0, Node::new(1), policy).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        server.serve_one().expect("vote");
        server.serve_one().expect("append");
        server
    });

    let vote = dial_vote("127.0.0.1", port, 1, 0, Some("secret"))
        .expect("dial")
        .expect("reachable");
    assert!(read_vote_reply(vote).expect("reply").granted);
    let append = dial_append(
        "127.0.0.1",
        port,
        2,
        Some(0),
        "set secure=true",
        Some("secret"),
    )
    .expect("dial")
    .expect("reachable");
    assert!(read_append_reply(append).expect("reply").ack);

    let server = handle.join().expect("server");
    assert_eq!(server.node().term, 2);
    assert_eq!(server.node().voted_for, Some(0));
    assert_eq!(server.node().log[0].data, b"set secure=true");
}

#[test]
fn auth_and_allowlist_rejections_do_not_mutate_state() {
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0,1")).expect("policy");
    let mut node = Node::new(1);
    node.term = 5;
    let mut server = ClusterRpcServer::bind("127.0.0.1", 0, node, policy).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        server.serve_one().expect("wrong token");
        server.serve_one().expect("bad peer");
        server
    });

    let wrong = dial_vote("127.0.0.1", port, 6, 0, Some("wrong"))
        .expect("dial")
        .expect("reachable");
    assert!(!read_vote_reply(wrong).expect("reply").granted);
    let bad_peer = dial_append("127.0.0.1", port, 7, Some(99), "forged", Some("secret"))
        .expect("dial")
        .expect("reachable");
    assert!(!read_append_reply(bad_peer).expect("reply").ack);

    let server = handle.join().expect("server");
    assert_eq!(server.node().term, 5);
    assert_eq!(server.node().voted_for, None);
    assert_eq!(server.node().log.len(), 0);
}

#[test]
fn missing_auth_and_missing_authenticated_leader_are_rejected() {
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0,1")).expect("policy");
    let mut node = Node::new(1);
    node.term = 9;
    let mut server = ClusterRpcServer::bind("127.0.0.1", 0, node, policy).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        server.serve_one().expect("missing token");
        server.serve_one().expect("missing leader");
        server
    });

    let missing = dial_vote("127.0.0.1", port, 10, 0, None)
        .expect("dial")
        .expect("reachable");
    assert!(!read_vote_reply(missing).expect("reply").granted);
    let no_leader = dial(
        "127.0.0.1",
        port,
        "AUTH secret APPEND 10 set missing-leader=true\n",
    )
    .expect("dial")
    .expect("reachable");
    assert!(!read_append_reply(no_leader).expect("reply").ack);

    let server = handle.join().expect("server");
    assert_eq!(server.node().term, 9);
    assert_eq!(server.node().log.len(), 0);
}

#[test]
fn peer_reload_admits_a_previously_rejected_candidate() {
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0,1")).expect("policy");
    let mut server = ClusterRpcServer::bind("127.0.0.1", 0, Node::new(1), policy).expect("bind");
    let port = server.local_port().expect("port");

    let rejected = dial_vote("127.0.0.1", port, 1, 99, Some("secret"))
        .expect("dial")
        .expect("reachable");
    server.serve_one().expect("reject");
    assert!(!read_vote_reply(rejected).expect("reply").granted);
    assert_eq!(server.node().term, 0);

    server.set_peers(Some(BTreeSet::from([0, 1, 99])));
    let admitted = dial_vote("127.0.0.1", port, 2, 99, Some("secret"))
        .expect("dial")
        .expect("reachable");
    server.serve_one().expect("admit");
    assert!(read_vote_reply(admitted).expect("reply").granted);
    assert_eq!(server.node().term, 2);
    assert_eq!(server.node().voted_for, Some(99));
}

#[test]
fn legacy_unauthenticated_append_accepts_an_empty_payload() {
    let mut server = ClusterRpcServer::bind("127.0.0.1", 0, Node::new(1), ClusterPolicy::default())
        .expect("bind");
    let port = server.local_port().expect("port");
    let append = dial_append("127.0.0.1", port, 1, None, "", None)
        .expect("dial")
        .expect("reachable");
    server.serve_one().expect("append");
    assert!(read_append_reply(append).expect("reply").ack);
    assert_eq!(server.node().log[0].data.len(), 0);
}

#[test]
fn non_loopback_bind_fails_closed_before_opening_a_socket() {
    assert!(matches!(
        ClusterRpcServer::bind("0.0.0.0", 0, Node::new(1), ClusterPolicy::default()),
        Err(RpcError::NonLoopbackRequiresToken)
    ));
    let policy = ClusterPolicy::from_values(Some("secret"), None).expect("policy");
    let server =
        ClusterRpcServer::bind("0.0.0.0", 0, Node::new(1), policy).expect("authenticated bind");
    assert!(server.local_port().expect("port") > 0);
}

#[test]
fn authenticated_multi_node_round_reaches_vote_and_append_quorum() {
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0,1,2")).expect("policy");
    let mut first =
        ClusterRpcServer::bind("127.0.0.1", 0, Node::new(1), policy.clone()).expect("bind");
    let mut second = ClusterRpcServer::bind("127.0.0.1", 0, Node::new(2), policy).expect("bind");
    let ports = [
        first.local_port().expect("port"),
        second.local_port().expect("port"),
    ];
    let handle = thread::spawn(move || {
        first.serve_one().expect("vote");
        second.serve_one().expect("vote");
        first.serve_one().expect("append");
        second.serve_one().expect("append");
        [first, second]
    });

    let votes = ports.map(|port| {
        dial_vote("127.0.0.1", port, 1, 0, Some("secret"))
            .expect("dial")
            .expect("reachable")
    });
    let mut vote_count = 1;
    for stream in votes {
        vote_count += usize::from(read_vote_reply(stream).expect("reply").granted);
    }
    assert_eq!(vote_count, 3);

    let appends = ports.map(|port| {
        dial_append(
            "127.0.0.1",
            port,
            2,
            Some(0),
            "set loop=true",
            Some("secret"),
        )
        .expect("dial")
        .expect("reachable")
    });
    let mut append_count = 1;
    for stream in appends {
        append_count += usize::from(read_append_reply(stream).expect("reply").ack);
    }
    assert_eq!(append_count, 3);

    let servers = handle.join().expect("servers");
    assert!(
        servers
            .iter()
            .all(|server| server.node().log[0].data == b"set loop=true")
    );
}

#[test]
fn authenticated_data_plane_preserves_exact_objects_and_conflict_state() {
    let root = abi_foundation::temp_path::temp_file_path("cluster-rpc-data", "store");
    let store = crate::VersionedStore::open(crate::StorePaths::new(&root)).expect("store");
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0,1")).expect("policy");
    let mut server = ClusterRpcServer::bind_with_store("127.0.0.1", 0, Node::new(1), policy, store)
        .expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        for _ in 0..7 {
            server.serve_one().expect("data request");
        }
        server
    });

    let mut denied = dial_data(
        "127.0.0.1",
        port,
        "wrong",
        &ClusterDataRequest::ReadKv { key: "k".into() },
    )
    .expect("dial")
    .expect("reachable");
    assert_eq!(read_reply(&mut denied).unwrap().0, "DENIED");

    let shard_key = b"kv:k".to_vec();
    let committed = match data_request(
        port,
        "secret",
        &ClusterDataRequest::CommitKv {
            shard_key: shard_key.clone(),
            key: "k".into(),
            value: "v".into(),
        },
    ) {
        ClusterDataResponse::Transaction { transaction } => transaction,
        response => panic!("unexpected commit response: {response:?}"),
    };
    let exported = match data_request(
        port,
        "secret",
        &ClusterDataRequest::ExportTransaction {
            writer_id: committed.writer_id(),
            sequence: committed.sequence(),
        },
    ) {
        ClusterDataResponse::Transaction { transaction } => transaction,
        response => panic!("unexpected export response: {response:?}"),
    };
    assert_eq!(exported, committed);
    assert_eq!(exported.encoded(), committed.encoded());

    assert!(matches!(
        data_request(
            port,
            "secret",
            &ClusterDataRequest::ImportCommitted {
                shard_key: shard_key.clone(),
                transaction: committed.clone(),
            }
        ),
        ClusterDataResponse::Imported { transaction_id }
            if transaction_id == committed.transaction_id()
    ));
    assert!(matches!(
        data_request(
            port,
            "secret",
            &ClusterDataRequest::ReadKv { key: "k".into() }
        ),
        ClusterDataResponse::Kv { current: Some(current) }
            if current.preferred.value == "v" && current.conflicts.is_empty()
    ));
    assert!(matches!(
        data_request(
            port,
            "secret",
            &ClusterDataRequest::ShardTransactions {
                shard_key: shard_key.clone()
            }
        ),
        ClusterDataResponse::Transactions { transactions }
            if transactions == [committed]
    ));
    assert!(matches!(
        data_request(port, "secret", &ClusterDataRequest::ShardKeys),
        ClusterDataResponse::ShardKeys { shard_keys } if shard_keys == [shard_key]
    ));

    drop(handle.join().expect("server"));
    std::fs::remove_dir_all(root).expect("cleanup");
}

#[test]
fn data_plane_rejects_an_oversized_request_before_connecting() {
    let request = ClusterDataRequest::CommitKv {
        shard_key: vec![1],
        key: "k".into(),
        value: "x".repeat(MAX_CLUSTER_DATA_FRAME_SIZE),
    };
    assert!(matches!(
        dial_data("127.0.0.1", 1, "secret", &request),
        Err(RpcError::LineTooLong)
    ));
}

#[test]
fn graceful_shutdown_requires_authentication_and_stops_the_server() {
    let policy = ClusterPolicy::from_values(Some("secret"), Some("0,1")).expect("policy");
    let mut server = ClusterRpcServer::bind("127.0.0.1", 0, Node::new(1), policy).expect("bind");
    let port = server.local_port().expect("port");
    let handle = thread::spawn(move || {
        server.serve_one().expect("wrong shutdown token");
        assert!(!server.shutdown_requested());
        server.serve_one().expect("authenticated shutdown");
        server
    });

    let denied = dial_shutdown("127.0.0.1", port, "wrong")
        .expect("dial")
        .expect("reachable");
    assert!(!read_shutdown_reply(denied).expect("denied reply"));
    let accepted = dial_shutdown("127.0.0.1", port, "secret")
        .expect("dial")
        .expect("reachable");
    assert!(read_shutdown_reply(accepted).expect("accepted reply"));
    assert!(handle.join().expect("server").shutdown_requested());
}
