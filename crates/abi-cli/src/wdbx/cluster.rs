//! WDBX `cluster` subcommand: single-node status, in-process consensus demo, and authenticated RPC serving.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use std::sync::atomic::{AtomicBool, Ordering};

mod local_demo;

use crate::app::Outcome;
use abi_wdbx::{ClusterPolicy, ClusterRpcServer, Node, StorePaths, VersionedStore};
use std::fmt::Write;

pub(crate) const CLUSTER_HELP: &str = "usage: abi wdbx cluster <status|demo|local-demo|serve> ...\n\nRun single-node status, in-process consensus demo, authenticated local multi-process proof, or cluster RPC serving.\n";

fn cluster_status() -> Outcome {
    let mut cluster = match abi_wdbx::Cluster::new(1) {
        Ok(cluster) => cluster,
        Err(detail) => return super::error("cluster status failed", detail),
    };
    if let Err(detail) = cluster.start_election(0) {
        return super::error("cluster election failed", detail);
    }
    Outcome::stderr(
        format!(
            "cluster: {}\n(single-node default; in-process multi-node consensus is available — run `abi wdbx cluster demo`)\nnorth-star status: single-node/in-process (Phase 1 landed); multi-host production cluster Proposed (Phase 2) (docs/spec/wdbx-north-star.mdx §2/§3.5)\n",
            cluster.status_line()
        ),
        0,
    )
}

fn cluster_demo(node_count: usize) -> Outcome {
    let mut cluster = match abi_wdbx::Cluster::new(node_count) {
        Ok(cluster) => cluster,
        Err(detail) => return super::error("cluster demo failed", detail),
    };
    let elected = match cluster.start_election(0) {
        Ok(elected) => elected,
        Err(detail) => return super::error("cluster demo failed", detail),
    };
    let mut report = format!(
        "election(node 0): leader_elected={elected}\n  status: {}\n",
        cluster.status_line()
    );
    let acknowledgements = cluster.replicate(b"set k=v").unwrap_or(0);
    writeln!(
        report,
        "replicate(\"set k=v\"): acks={acknowledgements} quorum={}",
        cluster.quorum()
    )
    .expect("writing to a String cannot fail");

    let Some(old_leader) = cluster.leader().map(|leader| leader.id) else {
        return super::error("cluster demo failed", "no elected leader");
    };
    if let Err(detail) = cluster.fail_node(old_leader) {
        return super::error("cluster demo failed", detail);
    }
    writeln!(report, "failover: downed leader node {old_leader}")
        .expect("writing to a String cannot fail");
    let next = u32::from(old_leader == 0);
    let reelected = cluster.start_election(next).unwrap_or(false);
    writeln!(
        report,
        "re-election(node {next}): leader_elected={reelected}\n  status: {}",
        cluster.status_line()
    )
    .expect("writing to a String cannot fail");
    report.push_str(
        "(in-process Raft-style consensus; networked RPC serving is available via `cluster serve`)\n",
    );
    report.push_str(
        "north-star status: in-process (Phase 1 landed); multi-host production cluster Proposed (Phase 2) (docs/spec/wdbx-north-star.mdx §2/§3.5)\n",
    );
    Outcome::stderr(report, 0)
}

pub(crate) fn run_cluster(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "status" => cluster_status(),
        [operation] if operation == "demo" => cluster_demo(3),
        [operation, count] if operation == "demo" => match count.parse::<usize>() {
            Ok(count) if count > 0 => cluster_demo(count),
            _ => super::usage(),
        },
        [operation] if operation == "local-demo" => local_demo::run(3, false),
        [operation, flag] if operation == "local-demo" && flag == "--json" => {
            local_demo::run(3, true)
        }
        [operation, count] if operation == "local-demo" => match count.parse::<usize>() {
            Ok(count @ 3..=9) => local_demo::run(count, false),
            _ => super::usage(),
        },
        [operation, count, flag] if operation == "local-demo" && flag == "--json" => {
            match count.parse::<usize>() {
                Ok(count @ 3..=9) => local_demo::run(count, true),
                _ => super::usage(),
            }
        }
        [operation, port] if operation == "serve" => cluster_serve(port, "0", "127.0.0.1"),
        [operation, port, node] if operation == "serve" => cluster_serve(port, node, "127.0.0.1"),
        [operation, port, node, host] if operation == "serve" => cluster_serve(port, node, host),
        _ => super::usage(),
    }
}

fn cluster_serve(port_raw: &str, node_raw: &str, host: &str) -> Outcome {
    let Ok(port) = port_raw.parse::<u16>() else {
        return super::usage();
    };
    let Ok(node_id) = node_raw.parse::<u32>() else {
        return super::usage();
    };
    let policy = match ClusterPolicy::from_env() {
        Ok(policy) => policy,
        Err(detail) => return super::error("cluster serve failed", detail),
    };
    let store_root = abi_foundation::temp_path::temp_file_path(
        &format!("abi-wdbx-cluster-node-{node_id}"),
        "store",
    );
    let store = match VersionedStore::open(StorePaths::new(&store_root)) {
        Ok(store) => store,
        Err(detail) => return super::error("cluster serve failed", detail),
    };
    let mut server =
        match ClusterRpcServer::bind_with_store(host, port, Node::new(node_id), policy, store) {
            Ok(server) => server,
            Err(detail) => {
                let _ = std::fs::remove_dir_all(&store_root);
                return super::error("cluster serve failed", detail);
            }
        };
    let bound = match server.local_port() {
        Ok(p) => p,
        Err(detail) => return super::error("cluster serve failed", detail),
    };
    let auth = if std::env::var("ABI_WDBX_CLUSTER_TOKEN")
        .ok()
        .as_ref()
        .is_some_and(|v| !v.is_empty())
    {
        "token=set"
    } else {
        "token=none (loopback only without token)"
    };
    eprintln!(
        "wdbx cluster RPC serving on {host}:{bound} node={node_id} ({auth}); non-loopback requires ABI_WDBX_CLUSTER_TOKEN; front multi-host with TLS/mTLS proxy — not production sharding"
    );

    // Stop on Ctrl-C so operators and smoke scripts can tear down cleanly.
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let stop_flag = std::sync::Arc::clone(&stop);
    let _ = ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::SeqCst);
    });

    while !stop.load(Ordering::SeqCst) && !server.shutdown_requested() {
        // Short accept timeout so Ctrl-C is observed without waiting forever.
        if let Err(err) = server.serve_one() {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            // Transient accept/read errors are logged; the listener stays up.
            eprintln!("cluster RPC serve error: {err}");
        }
    }
    drop(server);
    let _ = std::fs::remove_dir_all(store_root);
    Outcome::stderr("wdbx cluster RPC stopped\n".into(), 0)
}
