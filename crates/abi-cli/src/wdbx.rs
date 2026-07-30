//! WDBX CLI adapter.
//!
//! This module translates the path-shaped legacy CLI into the directory/base
//! representation used by `abi-wdbx`. Only attached operations are dispatched;
//! the remaining WDBX runtime surfaces fail explicitly during migration.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use abi_ai::text_embedding;
use abi_wdbx::{
    ClusterPolicy, ClusterRpcServer, DurableStore, HnswIndex, HybridScorer, Node, RestConfig,
    RestServer, Snapshot, StorePaths, TemporalCausalGraph, Wal, hybrid_search_scoped,
    hybrid_search_with_persona,
};

use crate::app::Outcome;
use crate::usage::is_help_token;

const USAGE: &str = include_str!("../../../tests/golden/wdbx-stats.txt");

const DB_HELP: &str = "usage: abi wdbx db <init|verify|compact> <path> [keep]\n\nManage segment checkpoints, WAL recovery, and snapshot integrity.\n";
const BLOCK_HELP: &str = "usage: abi wdbx block <insert|get> <path> ...\n\nAppend or inspect SHA-linked conversation blocks in a WDBX checkpoint.\n";
const QUERY_HELP: &str = "usage: abi wdbx query <path> [text] [persona] [--limit N] [--json] [--text T] [--persona P]\n\nPrint store stats (no text) or run hybrid semantic retrieval (semantic × temporal × causal × persona).\nPersona isolates results to that persona's memories. --limit defaults to 10. --json emits a machine-\nreadable result list (ranking=hybrid) with borrowed vector dims (zero-copy getVector view).\n";
const BENCHMARK_HELP: &str = "usage: abi wdbx benchmark [count]\n\nMeasure local insert/search timing for the in-process vector store.\n";
const CLUSTER_HELP: &str = "usage: abi wdbx cluster <status|demo|serve> ...\n\nRun single-node status, in-process consensus demo, or authenticated cluster RPC serving.\n";
const COMPUTE_HELP: &str = "usage: abi wdbx compute info\n\nReport CPU/GPU/NPU/TPU backend selection and fallback state.\n";
const SECURE_HELP: &str = "usage: abi wdbx secure demo\n\nDemonstrate local compression plus reference homomorphic aggregation; not security-audited FHE.\n";
const GPU_HELP: &str =
    "usage: abi wdbx gpu info\n\nReport GPU backend capability and native-kernel status.\n";
const API_HELP: &str = "usage: abi wdbx api serve [port]\n\nServe the loopback WDBX REST API.\n\nEnv:\n  ABI_WDBX_REST_TOKEN     Optional bearer token for request auth.\n  ABI_WDBX_TLS_CERT       Path to PEM certificate (TLS config / proxy deployment).\n  ABI_WDBX_TLS_KEY        Path to PEM private key (TLS config / proxy deployment).\n\nTLS: native termination is not linked; deploy behind nginx/Caddy/haproxy.\n";

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

fn error(context: &str, detail: impl std::fmt::Display) -> Outcome {
    Outcome::stderr(format!("{context}: {detail}\n"), 1)
}

fn usage() -> Outcome {
    Outcome::stderr(USAGE.to_owned(), 2)
}

fn help_for(subcommand: &str) -> Option<&'static str> {
    match subcommand {
        "db" => Some(DB_HELP),
        "block" => Some(BLOCK_HELP),
        "query" => Some(QUERY_HELP),
        "benchmark" => Some(BENCHMARK_HELP),
        "cluster" => Some(CLUSTER_HELP),
        "compute" => Some(COMPUTE_HELP),
        "secure" => Some(SECURE_HELP),
        "gpu" => Some(GPU_HELP),
        "api" => Some(API_HELP),
        _ => None,
    }
}

fn init_db(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return error("db init failed", detail),
    };
    if let Err(detail) = std::fs::create_dir_all(&paths.dir) {
        return error("db init failed", detail);
    }
    if let Err(detail) = abi_wdbx::segments::reset(&paths) {
        return error("db init failed", detail);
    }
    if let Err(detail) = std::fs::remove_file(paths.index())
        && detail.kind() != std::io::ErrorKind::NotFound
    {
        return error("db init failed", detail);
    }
    if let Err(detail) = std::fs::remove_file(paths.mirror_epoch())
        && detail.kind() != std::io::ErrorKind::NotFound
    {
        return error("db init failed", detail);
    }
    let wal = abi_wdbx::wal::wal_path(&paths);
    if let Err(detail) = std::fs::remove_file(&wal)
        && detail.kind() != std::io::ErrorKind::NotFound
    {
        return error("db init failed", detail);
    }
    if let Err(detail) = abi_wdbx::persistence::flush(&paths, &Snapshot::new()) {
        return error("db init failed", detail);
    }
    Outcome::stderr(
        format!("initialized empty WDBX segment checkpoint at {raw_path}\n"),
        0,
    )
}

fn verify_db(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return error("verify FAILED", detail),
    };
    let (snapshot, checkpoint_source) = match abi_wdbx::store::load_checkpoint_with_source(&paths) {
        Ok(loaded) => loaded,
        Err(detail) => return error(&format!("verify FAILED: checkpoint {raw_path}"), detail),
    };
    let chain_valid = snapshot.verify_chain_strict().is_ok();
    let stats = snapshot.stats;
    let source = match checkpoint_source {
        abi_wdbx::store::CheckpointSource::Empty => "empty",
        abi_wdbx::store::CheckpointSource::Legacy { .. } => "snapshot",
        abi_wdbx::store::CheckpointSource::Segment { .. } => "segment",
    };
    let checkpoint_epoch = checkpoint_source.epoch();
    let mut report = format!(
        "checkpoint OK: source={source} epoch={checkpoint_epoch} kv={} vectors={} blocks={} spatial={} temporal_nodes={} temporal_edges={} chain_valid={chain_valid}\n",
        stats.kv_entries,
        stats.vectors,
        stats.blocks,
        stats.spatial_records,
        stats.temporal_nodes,
        stats.temporal_edges
    );

    let wal_path = abi_wdbx::wal::wal_path(&paths);
    if !wal_path.is_file() {
        return Outcome::stderr(report, u8::from(!chain_valid));
    }
    let wal = match Wal::read(&wal_path) {
        Ok(wal) => wal,
        Err(detail) => {
            writeln!(
                report,
                "WAL verify FAILED: {}: {detail}",
                wal_path.display()
            )
            .expect("writing to a String cannot fail");
            return Outcome::stderr(report, 1);
        }
    };
    if wal.base_epoch != checkpoint_epoch {
        writeln!(
            report,
            "WAL note: frames={} base_epoch={} predates checkpoint epoch={checkpoint_epoch}; discarded on recovery",
            wal.len(),
            wal.base_epoch
        )
        .expect("writing to a String cannot fail");
        return Outcome::stderr(report, u8::from(!chain_valid));
    }

    let mut merged = snapshot;
    if let Err(detail) = wal.replay_onto(&mut merged) {
        writeln!(
            report,
            "WAL replay FAILED: {}: {detail}",
            wal_path.display()
        )
        .expect("writing to a String cannot fail");
        return Outcome::stderr(report, 1);
    }
    let merged_valid = merged.verify_chain_strict().is_ok();
    writeln!(
        report,
        "WAL OK: frames={} merged_blocks={} merged_chain_valid={merged_valid}",
        wal.len(),
        merged.blocks.len()
    )
    .expect("writing to a String cannot fail");
    Outcome::stderr(report, u8::from(!(chain_valid && merged_valid)))
}

fn compact_db(raw_path: &str, keep_latest: usize) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return error("compact FAILED", detail),
    };
    let result = match abi_wdbx::segments::compact_retain_latest(&paths, keep_latest) {
        Ok(result) => result,
        Err(detail) => return error(&format!("compact FAILED: {raw_path}"), detail),
    };
    let latest = result
        .latest_epoch
        .map_or_else(|| "none".to_owned(), |epoch| epoch.to_string());
    let mut report = format!(
        "compacted WDBX segments: path={raw_path} keep_latest={} before={} after={} deleted={} latest_epoch={latest}",
        result.keep_latest, result.before, result.after, result.deleted
    );
    if let Some(watermark) = result.watermark_epoch {
        write!(report, " watermark_epoch={watermark}").expect("writing to a String cannot fail");
    }
    report.push('\n');
    Outcome::stderr(report, 0)
}

fn unix_ms() -> Result<i64, String> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|detail| detail.to_string())?;
    i64::try_from(duration.as_millis()).map_err(|_| "Unix timestamp does not fit i64".to_owned())
}

fn insert_block(raw_path: &str, profile: &str, metadata: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return error("block insert failed", detail),
    };
    let mut store = match DurableStore::open(paths) {
        Ok(store) => store,
        Err(detail) => return error(&format!("error: {raw_path}"), detail),
    };
    let timestamp_ms = match unix_ms() {
        Ok(timestamp_ms) => timestamp_ms,
        Err(detail) => return error("block insert failed", detail),
    };
    let block = match store.add_block(profile, 0, 0, metadata, timestamp_ms) {
        Ok(block) => block,
        Err(detail) => return error("block insert failed", detail),
    };
    if let Err(detail) = store.checkpoint() {
        return error("block insert failed", detail);
    }
    Outcome::stderr(
        format!(
            "appended block: profile={profile} blocks={} hash={}\n",
            store.stats().blocks,
            block.hash.to_hex()
        ),
        0,
    )
}

fn get_block(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return error("block get failed", detail),
    };
    let store = match DurableStore::open(paths) {
        Ok(store) => store,
        Err(detail) => return error(&format!("error: {raw_path}"), detail),
    };
    let Some(block) = store.snapshot().blocks.last() else {
        return Outcome::stderr(format!("no blocks in {raw_path}\n"), 0);
    };
    Outcome::stderr(
        format!(
            "block: profile={} query_id={} response_id={} timestamp_ms={}\n  hash={}\n  metadata={}\n",
            block.profile,
            block.query_id,
            block.response_id,
            block.timestamp_ms,
            block.hash.to_hex(),
            block.metadata
        ),
        0,
    )
}

#[derive(Debug, PartialEq, Eq)]
struct QueryOptions<'a> {
    path: &'a str,
    text: Option<&'a str>,
    persona: Option<&'a str>,
    limit: usize,
    json: bool,
}

fn parse_query(args: &[String]) -> Result<QueryOptions<'_>, ()> {
    let path = args.first().ok_or(())?.as_str();
    let mut text = None;
    let mut persona = None;
    let mut positionals = Vec::with_capacity(2);
    let mut limit = 10;
    let mut json = false;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--limit" => {
                index += 1;
                limit = args.get(index).ok_or(())?.parse().map_err(|_| ())?;
                if limit == 0 {
                    return Err(());
                }
            }
            "--text" => {
                index += 1;
                text = Some(args.get(index).ok_or(())?.as_str());
            }
            "--persona" => {
                index += 1;
                persona = Some(args.get(index).ok_or(())?.as_str());
            }
            token if token.starts_with("--") => return Err(()),
            token => {
                if positionals.len() == 2 {
                    return Err(());
                }
                positionals.push(token);
            }
        }
        index += 1;
    }
    if text.is_none() {
        text = positionals.first().copied();
    }
    if persona.is_none() {
        persona = positionals.get(1).copied();
    }
    Ok(QueryOptions {
        path,
        text,
        persona,
        limit,
        json,
    })
}

const fn backend_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cpu"
    }
}

fn store_manifest(store: &DurableStore) -> String {
    let stats = store.stats();
    let dimensions = store
        .snapshot()
        .vector_dimensions()
        .map_or_else(|| "null".to_owned(), |value| value.to_string());
    format!(
        "{{\"kv_entries\":{},\"vectors\":{},\"blocks\":{},\"spatial_records\":{},\"temporal_nodes\":{},\"temporal_edges\":{},\"vector_dimensions\":{dimensions},\"next_vector_id\":{},\"backend\":\"{}\",\"mode\":\"cpu_fallback\"}}",
        stats.kv_entries,
        stats.vectors,
        stats.blocks,
        stats.spatial_records,
        stats.temporal_nodes,
        stats.temporal_edges,
        store.next_vector_id(),
        backend_label()
    )
}

fn query(args: &[String]) -> Outcome {
    let Ok(options) = parse_query(args) else {
        return usage();
    };
    let paths = match paths_from_cli_base(options.path) {
        Ok(paths) => paths,
        Err(detail) => return error("query failed", detail),
    };
    let store = match DurableStore::open(paths) {
        Ok(store) => store,
        Err(detail) => return error(&format!("error: {}", options.path), detail),
    };
    let manifest = store_manifest(&store);
    let Some(text) = options.text else {
        if options.json {
            let path = serde_json::to_string(options.path).expect("a string always serializes");
            return Outcome::stderr(
                format!(
                    "{{\"path\":{path},\"mode\":\"stats\",\"ranking\":null,\"manifest\":{manifest}}}\n"
                ),
                0,
            );
        }
        return Outcome::stderr(format!("{manifest}\n"), 0);
    };

    if store.stats().vectors == 0 {
        if options.json {
            let path = serde_json::to_string(options.path).expect("a string always serializes");
            let query = serde_json::to_string(text).expect("a string always serializes");
            let persona = options.persona.unwrap_or("all");
            return Outcome::stderr(
                format!(
                    "{{\"path\":{path},\"query\":{query},\"persona\":{},\"ranking\":\"hybrid\",\"limit\":{},\"vectors\":0,\"results\":[]}}\n",
                    serde_json::to_string(persona).expect("string"),
                    options.limit
                ),
                0,
            );
        }
        return Outcome::stderr(
            format!(
                "no vectors in {}; nothing to rank (populate with `abi complete`)\n",
                options.path
            ),
            0,
        );
    }

    query_with_text(
        &store,
        options.path,
        text,
        options.persona,
        options.limit,
        options.json,
    )
}

/// Hybrid semantic×temporal×causal×persona ranking over a recovered store.
fn query_with_text(
    store: &DurableStore,
    path: &str,
    text: &str,
    persona: Option<&str>,
    limit: usize,
    json: bool,
) -> Outcome {
    let query_vec = text_embedding(text);
    let graph = TemporalCausalGraph::from_records(&store.snapshot().temporal);
    let now_ms = abi_foundation::time::unix_ms();
    let scorer = HybridScorer {
        now_ms,
        ..HybridScorer::new(now_ms)
    };
    let focus_id = store.snapshot().vectors.keys().next().copied().unwrap_or(1);

    let ranked = if let Some(filter) = persona {
        let filter_owned = filter.to_ascii_lowercase();
        hybrid_search_scoped(store, &query_vec, limit, &graph, scorer, focus_id, |id| {
            vector_persona_matches(store, id, &filter_owned)
        })
    } else {
        hybrid_search_with_persona(store, &query_vec, limit, &graph, scorer, focus_id, |_id| {
            0.5
        })
    };

    let ranked = match ranked {
        Ok(rows) => rows,
        Err(detail) => return error("query failed", detail),
    };

    let persona_label = persona.unwrap_or("all");
    if json {
        let path_json = serde_json::to_string(path).expect("path serializes");
        let query_json = serde_json::to_string(text).expect("query serializes");
        let persona_json = serde_json::to_string(persona_label).expect("persona serializes");
        let mut results = String::from("[");
        for (i, row) in ranked.iter().enumerate() {
            if i > 0 {
                results.push(',');
            }
            let dims = row.vector.len();
            let _ = write!(
                results,
                "{{\"id\":{},\"score\":{:.6},\"semantic\":{:.6},\"temporal\":{:.6},\"causal\":{:.6},\"persona\":{:.6},\"dims\":{dims}}}",
                row.id,
                row.score,
                row.components.semantic,
                row.components.temporal,
                row.components.causal,
                row.components.persona,
            );
        }
        results.push(']');
        return Outcome::stderr(
            format!(
                "{{\"path\":{path_json},\"query\":{query_json},\"persona\":{persona_json},\"ranking\":\"hybrid\",\"limit\":{limit},\"vectors\":{},\"results\":{results}}}\n",
                store.stats().vectors
            ),
            0,
        );
    }

    if ranked.is_empty() {
        return Outcome::stderr(
            format!("no hybrid matches in {path} for query (persona={persona_label})\n"),
            0,
        );
    }

    let mut out =
        format!("query results (hybrid ranking, limit={limit}, persona={persona_label}):\n");
    for (i, row) in ranked.iter().enumerate() {
        let _ = writeln!(
            out,
            "  #{} id={} score={:.4} semantic={:.4} temporal={:.4} causal={:.4} persona={:.4} dims={}",
            i + 1,
            row.id,
            row.score,
            row.components.semantic,
            row.components.temporal,
            row.components.causal,
            row.components.persona,
            row.vector.len(),
        );
    }
    Outcome::stderr(out, 0)
}

fn vector_persona_matches(store: &DurableStore, id: u64, filter_lower: &str) -> bool {
    if let Some(label) = store.get(&format!("wdbx:profile:{id}")) {
        return label.eq_ignore_ascii_case(filter_lower);
    }
    // Fall back to block profile tags that reference this vector as query or response.
    for block in &store.snapshot().blocks {
        if (block.query_id == id || block.response_id == id)
            && block.profile.eq_ignore_ascii_case(filter_lower)
        {
            return true;
        }
    }
    false
}

fn cluster_status() -> Outcome {
    let mut cluster = match abi_wdbx::Cluster::new(1) {
        Ok(cluster) => cluster,
        Err(detail) => return error("cluster status failed", detail),
    };
    if let Err(detail) = cluster.start_election(0) {
        return error("cluster election failed", detail);
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
        Err(detail) => return error("cluster demo failed", detail),
    };
    let elected = match cluster.start_election(0) {
        Ok(elected) => elected,
        Err(detail) => return error("cluster demo failed", detail),
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
        return error("cluster demo failed", "no elected leader");
    };
    if let Err(detail) = cluster.fail_node(old_leader) {
        return error("cluster demo failed", detail);
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

fn run_cluster(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "status" => cluster_status(),
        [operation] if operation == "demo" => cluster_demo(3),
        [operation, count] if operation == "demo" => match count.parse::<usize>() {
            Ok(count) if count > 0 => cluster_demo(count),
            _ => usage(),
        },
        [operation, port] if operation == "serve" => cluster_serve(port, "0", "127.0.0.1"),
        [operation, port, node] if operation == "serve" => cluster_serve(port, node, "127.0.0.1"),
        [operation, port, node, host] if operation == "serve" => cluster_serve(port, node, host),
        _ => usage(),
    }
}

fn cluster_serve(port_raw: &str, node_raw: &str, host: &str) -> Outcome {
    let Ok(port) = port_raw.parse::<u16>() else {
        return usage();
    };
    let Ok(node_id) = node_raw.parse::<u32>() else {
        return usage();
    };
    let policy = match ClusterPolicy::from_env() {
        Ok(policy) => policy,
        Err(detail) => return error("cluster serve failed", detail),
    };
    let mut server = match ClusterRpcServer::bind(host, port, Node::new(node_id), policy) {
        Ok(server) => server,
        Err(detail) => return error("cluster serve failed", detail),
    };
    let bound = match server.local_port() {
        Ok(p) => p,
        Err(detail) => return error("cluster serve failed", detail),
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

    while !stop.load(Ordering::SeqCst) {
        // Short accept timeout so Ctrl-C is observed without waiting forever.
        if let Err(err) = server.serve_one() {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            // Transient accept/read errors are logged; the listener stays up.
            eprintln!("cluster RPC serve error: {err}");
        }
    }
    Outcome::stderr("wdbx cluster RPC stopped\n".into(), 0)
}

fn open_default_store() -> Result<DurableStore, String> {
    if let Ok(path) = std::env::var("ABI_WDBX_PATH") {
        if path == ":memory:" {
            return Err(
                "ABI_WDBX_PATH=:memory: is not valid for durable REST/cluster serving".into(),
            );
        }
        return DurableStore::open(StorePaths::new(path)).map_err(|e| e.to_string());
    }
    if matches!(
        std::env::var("ABI_WDBX_PERSIST").as_deref(),
        Ok("0" | "false" | "no" | "off")
    ) {
        return Err("WDBX persistence disabled (ABI_WDBX_PERSIST=0)".into());
    }
    let home = std::env::var("HOME").map_err(|_| "HOME is unset".to_string())?;
    DurableStore::open(StorePaths::new(format!("{home}/.abi/wdbx"))).map_err(|e| e.to_string())
}

fn api_serve(port_raw: Option<&str>) -> Outcome {
    let port: u16 = match port_raw {
        None => 8081,
        Some(raw) => match raw.parse() {
            Ok(p) => p,
            Err(_) => return usage(),
        },
    };
    let store = match open_default_store() {
        Ok(store) => store,
        Err(detail) => return error("api serve failed", detail),
    };
    // Disclose TLS env presence without claiming native TLS termination.
    if let Some(tls) = abi_wdbx::TlsConfig::from_env() {
        let _ = tls;
        eprintln!(
            "note: ABI_WDBX_TLS_CERT/KEY present — native TLS termination is not linked; deploy behind nginx/Caddy/haproxy"
        );
    }
    let config = RestConfig::from_env();
    let auth = if config.bearer_token.is_some() {
        "auth=bearer"
    } else {
        "auth=off"
    };
    let mut server = match RestServer::bind(port, store, config) {
        Ok(server) => server,
        Err(detail) => return error("api serve failed", detail),
    };
    let bound = match server.local_port() {
        Ok(p) => p,
        Err(detail) => return error("api serve failed", detail),
    };
    eprintln!(
        "wdbx REST serving on 127.0.0.1:{bound} ({auth}); routes: POST /insert /query /verify, GET /health /stats; loopback only"
    );

    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let stop_flag = std::sync::Arc::clone(&stop);
    let _ = ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::SeqCst);
    });

    while !stop.load(Ordering::SeqCst) {
        if let Err(err) = server.serve_one() {
            if stop.load(Ordering::SeqCst) {
                break;
            }
            eprintln!("wdbx REST serve error: {err}");
        }
    }
    Outcome::stderr("wdbx REST stopped\n".into(), 0)
}

fn run_api(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "serve" => api_serve(None),
        [operation, port] if operation == "serve" => api_serve(Some(port)),
        _ => usage(),
    }
}

fn gpu_info() -> Outcome {
    let gpu = abi_gpu::detect_backend();
    Outcome::stderr(
        format!(
            "gpu backend={} accelerated={} native_linked=false\nmessage={}\nnote: native kernels are not linked in the Rust port; vector ops use deterministic CPU SIMD fallback\n",
            gpu.backend.name(),
            gpu.accelerated,
            gpu.message,
        ),
        0,
    )
}

fn run_gpu(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "info" => gpu_info(),
        _ => usage(),
    }
}

fn compute_info() -> Outcome {
    let mut report = String::from(
        "compute backends (native dispatch not linked in this build; CPU fallback active):\n",
    );
    for capability in abi_wdbx::capabilities() {
        writeln!(
            report,
            "  {:<10} class={:<3} available={} native={}",
            capability.backend.name(),
            capability.backend.class(),
            capability.available,
            capability.native
        )
        .expect("writing to a String cannot fail");
    }
    let best = abi_wdbx::best_cpu_backend();
    let selection = abi_wdbx::select(abi_wdbx::Backend::NpuAne);
    writeln!(
        report,
        "dynamic selection: best_cpu={}; request npu-ane -> effective={} ({})",
        best.name(),
        selection.effective.name(),
        selection.message
    )
    .expect("writing to a String cannot fail");
    writeln!(
        report,
        "apple neural engine: hardware_present={} native_dispatch=false (CoreML/ANE path requires Apple frameworks, not linked; CPU fallback)",
        abi_wdbx::ane_hardware_present()
    )
    .expect("writing to a String cannot fail");
    let endpoint = abi_wdbx::remote_compute_endpoint();
    writeln!(
        report,
        "remote compute: endpoint={} ({}; dotOrLocal attempts dial then CPU fallback — not production TPU)",
        endpoint.as_deref().unwrap_or("none"),
        abi_wdbx::REMOTE_COMPUTE_ENDPOINT_ENV
    )
    .expect("writing to a String cannot fail");
    if endpoint.is_some() {
        match abi_wdbx::dot_or_local(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) {
            Ok(value) => {
                writeln!(
                    report,
                    "remote compute probe: dot([1,2,3],[4,5,6])={value:.4} (local ref=32)"
                )
                .expect("writing to a String cannot fail");
            }
            Err(detail) => {
                writeln!(report, "remote compute probe: error={detail}")
                    .expect("writing to a String cannot fail");
                report
                    .push_str("remote compute probe: dot([1,2,3],[4,5,6])=0.0000 (local ref=32)\n");
            }
        }
    }
    Outcome::stderr(report, 0)
}

fn run_compute(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "info" => compute_info(),
        _ => usage(),
    }
}

const fn entropy_mode_name(mode: abi_wdbx::entropy::EntropyMode) -> &'static str {
    match mode {
        abi_wdbx::entropy::EntropyMode::Stored => "stored",
        abi_wdbx::entropy::EntropyMode::Huffman => "huffman",
    }
}

const fn ans_mode_name(mode: abi_wdbx::AnsMode) -> &'static str {
    match mode {
        abi_wdbx::AnsMode::Stored => "stored",
        abi_wdbx::AnsMode::Rans0 => "rans0",
        abi_wdbx::AnsMode::Rans1 => "rans1",
    }
}

fn append_entropy_demo(report: &mut String) -> Result<(), String> {
    let entropy_source = b"WDBX-entropy-demo-aaaaaaaaaa-bbbbbbbbbb-cccccccccc-HELLO";
    let huffman = abi_wdbx::entropy_encode(entropy_source);
    let huffman_round_trip =
        abi_wdbx::entropy_decode(&huffman).map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "entropy Huffman: mode={} {}B -> serialized {}B ratio={:.2}x roundtrip={}",
        entropy_mode_name(huffman.mode),
        entropy_source.len(),
        huffman.serialized_len(),
        huffman.compression_ratio(),
        huffman_round_trip == entropy_source
    )
    .expect("writing to a String cannot fail");

    let rans0 = abi_wdbx::ans_encode(entropy_source).map_err(|detail| detail.to_string())?;
    let rans0_round_trip = abi_wdbx::ans_decode(&rans0).map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "entropy rANS0: mode={} {}B -> serialized {}B ratio={:.2}x roundtrip={}",
        ans_mode_name(rans0.mode),
        entropy_source.len(),
        rans0.serialized_len(),
        rans0.compression_ratio(),
        rans0_round_trip == entropy_source
    )
    .expect("writing to a String cannot fail");

    let order_one_source = b"the the the cat sat on the mat the the cat sat";
    let rans1 =
        abi_wdbx::ans_encode_order1(order_one_source).map_err(|detail| detail.to_string())?;
    let rans1_round_trip = abi_wdbx::ans_decode(&rans1).map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "entropy rANS1: mode={} {}B -> serialized {}B ratio={:.2}x roundtrip={} (demo; not SOTA)",
        ans_mode_name(rans1.mode),
        order_one_source.len(),
        rans1.serialized_len(),
        rans1.compression_ratio(),
        rans1_round_trip == order_one_source
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn append_autoencoder_demo(report: &mut String) -> Result<(), String> {
    let mut autoencoder =
        abi_wdbx::Autoencoder::new(8, 4, 0xC0DE_C0DE).map_err(|detail| detail.to_string())?;
    let sample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    autoencoder
        .train_step(&sample, 0.05)
        .map_err(|detail| detail.to_string())?;
    let mut latent = [0.0; 4];
    let mut reconstruction = [0.0; 8];
    autoencoder
        .encode_into(&sample, &mut latent)
        .map_err(|detail| detail.to_string())?;
    autoencoder
        .decode_into(&latent, &mut reconstruction)
        .map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "neural_compress: autoencoder 8->4->8 recon[0]={:.4} (reference demo, not SOTA)",
        reconstruction[0]
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn append_additive_he_demo(report: &mut String) -> Result<(), String> {
    let key = abi_wdbx::HeKey::new(0xAB_CDEF);
    let mut accumulated = key.encrypt(0, 0).map_err(|detail| detail.to_string())?;
    let mut plain_sum = 0;
    for value in 1_u64..=5 {
        let cipher = key
            .encrypt(value * 100, 1000 + value)
            .map_err(|detail| detail.to_string())?;
        accumulated = abi_wdbx::he_add(&accumulated, &cipher);
        plain_sum += value * 100;
    }
    let decrypted = key
        .decrypt(&accumulated)
        .map_err(|detail| detail.to_string())?;
    writeln!(
        report,
        "additive HE: sum of 5 encrypted values decrypts to {decrypted} (expected {plain_sum}, match={})",
        decrypted == plain_sum
    )
    .expect("writing to a String cannot fail");
    Ok(())
}

fn append_dghv_demo(report: &mut String) {
    let mut random = abi_wdbx::DghvRng::new(0x5EED_F00D_C0FF_EE11);
    let keypair = abi_wdbx::dghv_keygen(&mut random);
    let encrypted_one = abi_wdbx::dghv_encrypt(&keypair, &mut random, true);
    let encrypted_one_again = abi_wdbx::dghv_encrypt(&keypair, &mut random, true);
    let encrypted_zero = abi_wdbx::dghv_encrypt(&keypair, &mut random, false);
    let product = abi_wdbx::dghv_mul(&keypair, &encrypted_one, &encrypted_one_again);
    let evaluated = abi_wdbx::dghv_add(&keypair, &product, &encrypted_zero);
    let evaluated_bit = u8::from(abi_wdbx::dghv_decrypt(&keypair, &evaluated));
    writeln!(
        report,
        "homomorphic eval: enc((1 AND 1) XOR 0) decrypts to {evaluated_bit} (expected 1, match={})",
        evaluated_bit == 1
    )
    .expect("writing to a String cannot fail");
    report.push_str(
        "(DGHV somewhat-homomorphic scheme: real encrypted add+multiply on ciphertexts, reference parameters / bounded depth — not security-audited)\n",
    );
}

fn secure_demo_result() -> Result<String, String> {
    let vector: Vec<f32> = (0_u16..128)
        .map(|index| (f32::from(index) * 0.1).sin())
        .collect();
    let quantized = abi_wdbx::quantize(&vector).map_err(|detail| detail.to_string())?;
    let reconstructed = abi_wdbx::dequantize(&quantized);
    let mut report = format!(
        "compression: {} f32 -> int8 codes, ratio={:.2}x, max_error={:.5}\n",
        vector.len(),
        quantized.compression_ratio(),
        abi_wdbx::max_error(&vector, &reconstructed)
    );
    append_entropy_demo(&mut report)?;
    append_autoencoder_demo(&mut report)?;
    append_additive_he_demo(&mut report)?;
    append_dghv_demo(&mut report);
    report.push_str(
        "north-star status: Partial — int8 + Huffman + rANS/order-1 demos + autoencoder + additive HE + reference DGHV SHE (not audited, not SOTA); production FHE/SOTA codecs remain Proposed\n",
    );
    Ok(report)
}

fn run_secure(args: &[String]) -> Outcome {
    match args {
        [operation] if operation == "demo" => match secure_demo_result() {
            Ok(report) => Outcome::stderr(report, 0),
            Err(detail) => error("secure demo failed", detail),
        },
        _ => usage(),
    }
}

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn percentile_sorted(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let rank = percentile.saturating_mul(samples.len()).saturating_add(99) / 100;
    samples[rank.clamp(1, samples.len()) - 1]
}

fn benchmark_result(count: usize) -> Result<String, String> {
    let mut index = HnswIndex::new(4).map_err(|detail| detail.to_string())?;
    let mut insert_samples = Vec::new();
    insert_samples
        .try_reserve_exact(count)
        .map_err(|detail| detail.to_string())?;
    let insert_start = Instant::now();
    for position in 0..count {
        let x = u16::try_from(position % 97).expect("modulo result fits u16");
        let y = u16::try_from(position % 31).expect("modulo result fits u16");
        let vector = [f32::from(x), f32::from(y), 0.0, 0.0];
        let operation_start = Instant::now();
        let id = u64::try_from(position)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| "benchmark vector id overflow".to_owned())?;
        index
            .insert(id, &vector)
            .map_err(|detail| detail.to_string())?;
        index.commit_last_insert(id);
        insert_samples.push(elapsed_ns(operation_start));
    }
    let insert_ns = elapsed_ns(insert_start);

    let query_count = count.min(200);
    let mut search_samples = Vec::new();
    search_samples
        .try_reserve_exact(query_count)
        .map_err(|detail| detail.to_string())?;
    let search_start = Instant::now();
    for _ in 0..query_count {
        let operation_start = Instant::now();
        let results = index
            .search(&[1.0, 0.0, 0.0, 0.0], 10)
            .map_err(|detail| detail.to_string())?;
        search_samples.push(elapsed_ns(operation_start));
        drop(results);
    }
    let search_ns = elapsed_ns(search_start);

    insert_samples.sort_unstable();
    search_samples.sort_unstable();
    let insert_average = if count == 0 {
        0
    } else {
        insert_ns / u64::try_from(count).unwrap_or(u64::MAX)
    };
    let search_average = if query_count == 0 {
        0
    } else {
        search_ns / u64::try_from(query_count).unwrap_or(u64::MAX)
    };
    Ok(format!(
        "benchmark (local, in-memory; not a published throughput claim):\n  inserts: {count} in {insert_ns} ns  (avg {insert_average} ns/op; includes per-op acceleration-kernel dispatch)\n    p50={} ns  p95={} ns  p99={} ns\n  searches: {query_count} in {search_ns} ns (avg {search_average} ns/op, k=10 over {} vectors)\n    p50={} ns  p95={} ns  p99={} ns\n",
        percentile_sorted(&insert_samples, 50),
        percentile_sorted(&insert_samples, 95),
        percentile_sorted(&insert_samples, 99),
        index.len(),
        percentile_sorted(&search_samples, 50),
        percentile_sorted(&search_samples, 95),
        percentile_sorted(&search_samples, 99),
    ))
}

fn run_benchmark(args: &[String]) -> Outcome {
    let count = match args {
        [] => 256,
        [count] => match count.parse::<usize>() {
            Ok(count) => count,
            Err(_) => return usage(),
        },
        _ => return usage(),
    };
    match benchmark_result(count) {
        Ok(report) => Outcome::stderr(report, 0),
        Err(detail) => error("benchmark failed", detail),
    }
}

fn run_db(args: &[String]) -> Outcome {
    if args.len() == 1 && is_help_token(&args[0]) {
        return Outcome::stderr(DB_HELP.to_owned(), 0);
    }
    match args {
        [operation, path] if operation == "init" => init_db(path),
        [operation, path] if operation == "verify" => verify_db(path),
        [operation, path] if operation == "compact" => compact_db(path, 2),
        [operation, path, keep] if operation == "compact" => match keep.parse::<usize>() {
            Ok(keep) if keep > 0 => compact_db(path, keep),
            _ => usage(),
        },
        _ => usage(),
    }
}

fn run_block(args: &[String]) -> Outcome {
    if args.len() == 1 && is_help_token(&args[0]) {
        return Outcome::stderr(BLOCK_HELP.to_owned(), 0);
    }
    match args {
        [operation, path, profile, metadata] if operation == "insert" => {
            insert_block(path, profile, metadata)
        }
        [operation, path] if operation == "get" => get_block(path),
        _ => usage(),
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
        "db" => run_db(&args[1..]),
        "block" => run_block(&args[1..]),
        "query" => query(&args[1..]),
        "benchmark" => run_benchmark(&args[1..]),
        "cluster" => run_cluster(&args[1..]),
        "compute" => run_compute(&args[1..]),
        "secure" => run_secure(&args[1..]),
        "gpu" => run_gpu(&args[1..]),
        "api" => run_api(&args[1..]),
        _ => usage(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(!v["results"].as_array().expect("results").is_empty());

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
        assert!(outcome.stderr.contains("native_linked=false"));
        assert!(outcome.stderr.contains("accelerated="));
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
