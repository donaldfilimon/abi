//! WDBX CLI adapter.
//!
//! This module translates the path-shaped legacy CLI into the directory/base
//! representation used by `abi-wdbx`. Only attached operations are dispatched;
//! the remaining WDBX runtime surfaces fail explicitly during migration.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use abi_wdbx::{DurableStore, Snapshot, StorePaths, Wal};

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

fn paths_from_cli_base(raw: &str) -> Result<StorePaths, String> {
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
    let manifest = match paths.read_manifest() {
        Ok(manifest) => manifest,
        Err(detail) => return error(&format!("verify FAILED: checkpoint {raw_path}"), detail),
    };
    let (snapshot, epoch) = match abi_wdbx::store::load_newest_valid_with_epoch(&paths, &manifest) {
        Ok(loaded) => loaded,
        Err(detail) => return error(&format!("verify FAILED: checkpoint {raw_path}"), detail),
    };
    let chain_valid = snapshot.verify_chain_strict().is_ok();
    let stats = snapshot.stats;
    let source = if epoch.is_some() { "segment" } else { "empty" };
    let checkpoint_epoch = epoch.unwrap_or(0);
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
            return Outcome::stderr(
                format!(
                    "{{\"path\":{path},\"query\":{query},\"persona\":\"all\",\"ranking\":\"hybrid\",\"limit\":{},\"vectors\":0,\"results\":[]}}\n",
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

    Outcome::stderr(
        "error: Rust WDBX text-query embedding handler is not yet ported\n".to_owned(),
        1,
    )
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
        known
            if matches!(
                known,
                "benchmark" | "simulate" | "cluster" | "compute" | "secure" | "gpu" | "api"
            ) =>
        {
            Outcome::stderr(
                format!("error: Rust WDBX handler for `{known}` is not yet ported\n"),
                1,
            )
        }
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
    fn compact_rejects_zero_at_the_grammar_boundary() {
        let outcome = run(&strings(&["db", "compact", "unused", "0"]));
        assert_eq!(outcome.exit_code, 2);
        assert_eq!(outcome.stderr, USAGE);
    }
}
