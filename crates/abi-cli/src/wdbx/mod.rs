//! WDBX CLI adapter.
//!
//! This module translates the path-shaped legacy CLI into the directory/base
//! representation used by `abi-wdbx`. Only attached operations are dispatched;
//! the remaining WDBX runtime surfaces fail explicitly during migration.

mod api;
mod benchmark;
mod block;
mod cluster;
mod compute;
mod db;
mod gpu;
mod query;
mod secure;

use std::path::{Path, PathBuf};

use abi_wdbx::StorePaths;

use crate::app::Outcome;
use crate::usage::is_help_token;

const USAGE: &str = include_str!("../../../../tests/golden/wdbx-stats.txt");

pub(crate) fn paths_from_cli_base(raw: &str) -> Result<StorePaths, String> {
    let path = Path::new(raw);
    let base = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| format!("invalid WDBX base path {raw:?}"))?;
    let dir = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    Ok(StorePaths {
        dir,
        base: base.to_owned(),
    })
}

pub(crate) fn error(context: &str, detail: impl std::fmt::Display) -> Outcome {
    Outcome::stderr(format!("{context}: {detail}\n"), 1)
}

pub(crate) fn usage() -> Outcome {
    Outcome::stderr(USAGE.to_owned(), 2)
}

fn help_for(subcommand: &str) -> Option<&'static str> {
    match subcommand {
        "db" => Some(db::DB_HELP),
        "block" => Some(block::BLOCK_HELP),
        "query" => Some(query::QUERY_HELP),
        "benchmark" => Some(benchmark::BENCHMARK_HELP),
        "cluster" => Some(cluster::CLUSTER_HELP),
        "compute" => Some(compute::COMPUTE_HELP),
        "secure" => Some(secure::SECURE_HELP),
        "gpu" => Some(gpu::GPU_HELP),
        "api" => Some(api::API_HELP),
        _ => None,
    }
}

/// Dispatch arguments following the top-level `wdbx` token.
pub(crate) fn run(args: &[String]) -> Outcome {
    let Some(subcommand) = args.first() else {
        return usage();
    };
    if subcommand == "simulate" {
        return crate::wdbx_simulate::run(&args[1..]);
    }
    if is_help_token(subcommand) {
        return Outcome::stderr(USAGE.to_owned(), 0);
    }
    if args.len() == 2
        && is_help_token(&args[1])
        && let Some(help) = help_for(subcommand)
    {
        return Outcome::stderr(help.to_owned(), 0);
    }
    match subcommand.as_str() {
        "db" => db::run_db(&args[1..]),
        "block" => block::run_block(&args[1..]),
        "query" => query::query(&args[1..]),
        "benchmark" => benchmark::run_benchmark(&args[1..]),
        "cluster" => cluster::run_cluster(&args[1..]),
        "compute" => compute::run_compute(&args[1..]),
        "secure" => secure::run_secure(&args[1..]),
        "gpu" => gpu::run_gpu(&args[1..]),
        "api" => api::run_api(&args[1..]),
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wdbx::db::DB_HELP;
    use crate::wdbx::query::backend_label;
    use abi_ai::text_embedding;
    use abi_wdbx::{
        ClusterPolicy, ClusterRpcServer, DurableStore, Node, RestConfig, RestServer, Snapshot,
        StorePaths,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        dir: PathBuf,
        base: String,
    }

    impl Fixture {
        fn new() -> Self {
            let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "abi_cli_wdbx_{}_{}",
                std::process::id(),
                sequence
            ));
            std::fs::create_dir_all(&dir).expect("fixture directory");
            Self {
                dir,
                base: "cli.jsonl".to_owned(),
            }
        }

        fn raw_path(&self) -> String {
            self.dir.join(&self.base).to_string_lossy().into_owned()
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn nested_help_and_usage_are_explicit() {
        assert_eq!(
            run(&strings(&["--help"])),
            Outcome::stderr(USAGE.to_owned(), 0)
        );
        assert_eq!(
            run(&strings(&["db", "--help"])),
            Outcome::stderr(DB_HELP.to_owned(), 0)
        );
        assert_eq!(run(&[]).exit_code, 2);
        assert_eq!(run(&strings(&["unknown"])).exit_code, 2);
    }

    #[test]
    fn keygen_creates_owner_only_files_without_printing_or_overwriting_keys() {
        let fixture = Fixture::new();
        let key_dir = fixture.dir.join("keys");
        let key_dir_string = key_dir.to_string_lossy().into_owned();
        let generated = run(&strings(&["db", "keygen", &key_dir_string]));
        assert_eq!(generated.exit_code, 0, "{}", generated.stderr);
        assert!(generated.stderr.contains("key bytes not displayed"));
        for name in ["encryption.key", "signing.key", "verify.key"] {
            let path = key_dir.join(name);
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(bytes.len(), 32);
            assert!(!generated.stderr.contains(&format!("{bytes:?}")));
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                assert_eq!(
                    std::fs::metadata(path).unwrap().permissions().mode() & 0o077,
                    0
                );
            }
        }
        let repeated = run(&strings(&["db", "keygen", &key_dir_string]));
        assert_eq!(repeated.exit_code, 1);
        assert!(repeated.stderr.contains("refusing to overwrite"));
    }

    #[test]
    fn init_insert_get_and_verify_round_trip() {
        let fixture = Fixture::new();
        let raw_path = fixture.raw_path();

        let initialized = run(&["db".to_owned(), "init".to_owned(), raw_path.clone()]);
        assert_eq!(initialized.exit_code, 0, "{}", initialized.stderr);

        let inserted = run(&[
            "block".to_owned(),
            "insert".to_owned(),
            raw_path.clone(),
            "abbey".to_owned(),
            r#"{"turn":1}"#.to_owned(),
        ]);
        assert_eq!(inserted.exit_code, 0, "{}", inserted.stderr);
        assert!(inserted.stderr.contains("profile=abbey blocks=1 hash="));

        let fetched = run(&["block".to_owned(), "get".to_owned(), raw_path.clone()]);
        assert_eq!(fetched.exit_code, 0, "{}", fetched.stderr);
        assert!(fetched.stderr.contains("metadata={\"turn\":1}"));

        let verified = run(&["db".to_owned(), "verify".to_owned(), raw_path]);
        assert_eq!(verified.exit_code, 0, "{}", verified.stderr);
        assert!(verified.stderr.contains("kv=0 vectors=0 blocks=1"));
        assert!(verified.stderr.contains("merged_chain_valid=true"));
    }

    #[test]
    fn verify_reads_a_legacy_single_file_checkpoint() {
        let fixture = Fixture::new();
        let raw_path = fixture.raw_path();
        let mut snapshot = Snapshot::new();
        snapshot
            .kv
            .insert("legacy".to_owned(), "visible".to_owned());
        snapshot.recount();
        abi_wdbx::persistence::write_snapshot(&raw_path, &snapshot).expect("legacy checkpoint");

        let verified = run(&["db".to_owned(), "verify".to_owned(), raw_path]);
        assert_eq!(verified.exit_code, 0, "{}", verified.stderr);
        assert!(verified.stderr.contains("source=snapshot epoch=0 kv=1"));
    }

    #[test]
    fn query_stats_and_empty_results_are_fully_attached() {
        let fixture = Fixture::new();
        let raw_path = fixture.raw_path();
        assert_eq!(
            run(&["db".to_owned(), "init".to_owned(), raw_path.clone()]).exit_code,
            0
        );

        let stats = run(&["query".to_owned(), raw_path.clone()]);
        assert_eq!(stats.exit_code, 0);
        assert_eq!(
            stats.stderr,
            format!(
                "{{\"kv_entries\":0,\"vectors\":0,\"blocks\":0,\"spatial_records\":0,\"temporal_nodes\":0,\"temporal_edges\":0,\"vector_dimensions\":null,\"next_vector_id\":1,\"backend\":\"{}\",\"mode\":\"cpu_fallback\"}}\n",
                backend_label()
            )
        );

        let empty = run(&[
            "query".to_owned(),
            raw_path.clone(),
            "memory".to_owned(),
            "--json".to_owned(),
            "--limit".to_owned(),
            "3".to_owned(),
        ]);
        assert_eq!(empty.exit_code, 0);
        let path_json = serde_json::to_string(&raw_path).expect("path serializes");
        assert_eq!(
            empty.stderr,
            format!(
                "{{\"path\":{path_json},\"query\":\"memory\",\"persona\":\"all\",\"ranking\":\"hybrid\",\"limit\":3,\"vectors\":0,\"results\":[]}}\n"
            )
        );
    }

    #[test]
    fn cluster_status_and_demo_are_deterministic() {
        let status = run(&strings(&["cluster", "status"]));
        assert_eq!(status.exit_code, 0);
        assert!(
            status
                .stderr
                .starts_with("cluster: nodes=1 alive=1 quorum=1 leader=0 term=1 commit_index=0\n")
        );

        let demo = run(&strings(&["cluster", "demo", "3"]));
        assert_eq!(demo.exit_code, 0);
        assert!(
            demo.stderr
                .contains("replicate(\"set k=v\"): acks=3 quorum=2")
        );
        assert!(demo.stderr.contains(
            "re-election(node 1): leader_elected=true\n  status: nodes=3 alive=2 quorum=2 leader=1 term=2 commit_index=1"
        ));
    }

    #[test]
    fn compute_report_discloses_cpu_fallback() {
        let outcome = run(&strings(&["compute", "info"]));
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stderr.contains("native dispatch not linked"));
        assert!(outcome.stderr.contains("native=false"));
        assert!(outcome.stderr.contains("request npu-ane -> effective=cpu-"));
        assert!(outcome.stderr.contains("not production TPU"));
    }

    #[test]
    fn secure_demo_composes_every_reference_primitive() {
        let outcome = run(&strings(&["secure", "demo"]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        for marker in [
            "compression:",
            "entropy Huffman:",
            "entropy rANS0:",
            "entropy rANS1:",
            "neural_compress:",
            "additive HE:",
            "homomorphic eval:",
            "not security-audited",
            "production FHE/SOTA codecs remain Proposed",
        ] {
            assert!(outcome.stderr.contains(marker), "missing {marker}");
        }
    }

    #[test]
    fn text_query_ranks_inserted_vectors() {
        let fixture = Fixture::new();
        let raw_path = fixture.raw_path();
        assert_eq!(
            run(&["db".to_owned(), "init".to_owned(), raw_path.clone()]).exit_code,
            0
        );
        let paths = paths_from_cli_base(&raw_path).expect("paths");
        let mut store = DurableStore::open(paths).expect("open");
        let a = text_embedding("hello memory abbey");
        let b = text_embedding("completely unrelated tokens xyz");
        let id_a = store.put_vector(&a).expect("put a");
        let id_b = store.put_vector(&b).expect("put b");
        store
            .put(&format!("wdbx:profile:{id_a}"), "abbey")
            .expect("profile a");
        store
            .put(&format!("wdbx:profile:{id_b}"), "aviva")
            .expect("profile b");
        store.add_temporal_node(id_a, 1_000).expect("temporal a");
        store.add_temporal_node(id_b, 1_000).expect("temporal b");
        drop(store);

        let text = run(&[
            "query".to_owned(),
            raw_path.clone(),
            "hello memory".to_owned(),
            "--limit".to_owned(),
            "2".to_owned(),
        ]);
        assert_eq!(text.exit_code, 0, "{}", text.stderr);
        assert!(text.stderr.contains("hybrid ranking"), "{}", text.stderr);
        assert!(text.stderr.contains("id="), "{}", text.stderr);

        let json = run(&[
            "query".to_owned(),
            raw_path.clone(),
            "hello memory".to_owned(),
            "--json".to_owned(),
            "--limit".to_owned(),
            "2".to_owned(),
        ]);
        assert_eq!(json.exit_code, 0, "{}", json.stderr);
        assert!(
            json.stderr.contains("\"ranking\":\"hybrid\""),
            "{}",
            json.stderr
        );
        assert!(json.stderr.contains("\"results\":["), "{}", json.stderr);
        let v: serde_json::Value =
            serde_json::from_str(json.stderr.trim()).expect("json query parses");
        assert_eq!(v["ranking"], "hybrid");
        assert_ne!(v["results"].as_array().expect("results").len(), 0);

        let scoped = run(&[
            "query".to_owned(),
            raw_path,
            "hello".to_owned(),
            "abbey".to_owned(),
            "--json".to_owned(),
        ]);
        assert_eq!(scoped.exit_code, 0, "{}", scoped.stderr);
        assert!(scoped.stderr.contains("\"persona\":\"abbey\""));
    }

    #[test]
    fn gpu_info_is_claim_honest() {
        let outcome = run(&strings(&["gpu", "info"]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stderr.contains("accelerated="));
        if abi_gpu::metal_kernels::kernels_active() {
            assert!(outcome.stderr.contains("native_linked=true"));
            assert!(outcome.stderr.contains("accelerated=true"));
        } else {
            assert!(outcome.stderr.contains("native_linked=false"));
        }
    }

    #[test]
    fn rest_server_health_is_reachable_on_loopback() {
        use std::io::{Read, Write};

        let fixture = Fixture::new();
        let paths = StorePaths {
            dir: fixture.dir.clone(),
            base: fixture.base.clone(),
        };
        abi_wdbx::segments::reset(&paths).ok();
        let store = DurableStore::open(paths).expect("open");
        let config = RestConfig {
            bearer_token: None,
            rate_limiter: abi_wdbx::RateLimiter::from_env(),
        };
        let mut server = RestServer::bind(0, store, config).expect("bind");
        let port = server.local_port().expect("port");
        let handle = std::thread::spawn(move || {
            server.serve_one().expect("serve");
        });
        // Give the acceptor a moment to listen.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut stream =
            std::net::TcpStream::connect(format!("127.0.0.1:{port}")).expect("connect");
        stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            .expect("write");
        let mut buf = vec![0_u8; 4096];
        let n = stream.read(&mut buf).expect("read");
        let body = String::from_utf8_lossy(&buf[..n]);
        assert!(
            body.contains("200") || body.contains("ok") || body.contains("health"),
            "unexpected health response: {body}"
        );
        handle.join().expect("server thread");
    }

    #[test]
    fn cluster_rpc_server_accepts_loopback_vote() {
        let policy = ClusterPolicy::from_values(None, None).expect("policy");
        let mut server =
            ClusterRpcServer::bind("127.0.0.1", 0, Node::new(0), policy).expect("bind");
        let port = server.local_port().expect("port");
        let handle = std::thread::spawn(move || {
            server.serve_one().expect("serve");
        });
        std::thread::sleep(std::time::Duration::from_millis(20));
        let stream = abi_wdbx::dial_vote("127.0.0.1", port, 1, 0, None).expect("dial");
        assert!(stream.is_some(), "vote should get a reply stream");
        handle.join().expect("server thread");
    }

    #[test]
    fn compact_rejects_zero_at_the_grammar_boundary() {
        let outcome = run(&strings(&["db", "compact", "unused", "0"]));
        assert_eq!(outcome.exit_code, 2);
        assert_eq!(outcome.stderr, USAGE);
    }

    #[test]
    fn benchmark_preserves_the_frozen_workload_and_report_shape() {
        let report = run(&strings(&["benchmark", "4"]));
        assert_eq!(report.exit_code, 0, "{}", report.stderr);
        assert!(
            report
                .stderr
                .starts_with("benchmark (local, in-memory; not a published throughput claim):\n")
        );
        assert!(report.stderr.contains("  inserts: 4 in "));
        assert!(report.stderr.contains("  searches: 4 in "));
        assert!(report.stderr.contains("k=10 over 4 vectors"));
        assert!(report.stderr.contains("p50="));
        assert!(report.stderr.contains("p95="));
        assert!(report.stderr.contains("p99="));
        assert_eq!(run(&strings(&["benchmark", "not-a-number"])).exit_code, 2);
        assert_eq!(run(&strings(&["benchmark", "1", "extra"])).exit_code, 2);
    }
}
