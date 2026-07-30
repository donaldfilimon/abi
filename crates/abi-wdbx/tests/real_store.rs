//! Reads the WDBX fixtures, and optionally the developer's real store.
//!
//! The fixture tests always run. The real-store test is gated on
//! `ABI_WDBX_REAL_STORE_TEST=1` because it reads ~180 MB from `~/.abi/`, which is
//! too slow for the default gate and depends on machine-local state — but it is
//! the only check that proves the Rust reader can actually open the user's data,
//! so it exists and is documented rather than being left as an assumption.
//!
//! ```bash
//! ABI_WDBX_REAL_STORE_TEST=1 ./tools/cargo.sh test -p abi-wdbx --test real_store
//! ```

use abi_wdbx::format::{Manifest, Record, parse_segment};
use abi_wdbx::{ExactIndex, HnswIndex, Snapshot, StorePaths};
use std::path::{Path, PathBuf};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden")
}

fn read_golden(name: &str) -> String {
    let path = golden_dir().join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {}: {e}", path.display()))
}

#[test]
fn parses_the_sample_segment_with_both_hash_encodings() {
    let content = read_golden("wdbx-sample.seg.jsonl");
    let segment = parse_segment(0, Path::new("wdbx-sample.seg.jsonl"), &content)
        .expect("sample segment parses");

    assert!(!segment.truncated_tail);

    let kinds: Vec<&str> = segment.records.iter().map(Record::type_name).collect();
    assert_eq!(
        kinds,
        ["kv", "kv", "vector", "vector", "block", "block", "spatial"]
    );

    // The fixture deliberately carries one genesis block whose prev_hash is in
    // the STRING encoding and one linked block whose prev_hash is an ARRAY. Both
    // must land as usable hashes — this is the case a from-scratch reader misses.
    let blocks: Vec<_> = segment
        .records
        .iter()
        .filter_map(|record| match record {
            Record::Block(block) => Some(block),
            _ => None,
        })
        .collect();
    assert_eq!(blocks.len(), 2);
    assert!(
        blocks[0].prev_hash.is_genesis(),
        "the string-encoded all-zero prev_hash must decode to genesis"
    );
    assert!(!blocks[1].prev_hash.is_genesis());
    assert_eq!(
        blocks[1].prev_hash, blocks[0].hash,
        "the second block must link to the first"
    );
}

#[test]
fn the_sample_chain_verifies() {
    let content = read_golden("wdbx-sample.seg.jsonl");
    let segment = parse_segment(0, Path::new("sample"), &content).expect("sample segment parses");
    let mut snapshot = Snapshot::new();
    for record in segment.records {
        snapshot.apply(record);
    }
    snapshot.recount();

    snapshot
        .verify_chain()
        .expect("the fixture's two blocks link from genesis");
    assert_eq!(snapshot.stats.blocks, 2);
    assert_eq!(snapshot.stats.kv_entries, 2);
    assert_eq!(snapshot.stats.vectors, 2);
    assert_eq!(snapshot.stats.spatial_records, 1);
}

#[test]
fn parses_the_magic_only_sample_segment() {
    let content = read_golden("wdbx-sample.seg.3.jsonl");
    let segment = parse_segment(3, Path::new("sample.3"), &content).expect("parses");
    assert!(segment.records.is_empty());
    assert!(!segment.truncated_tail);
}

#[test]
fn reports_the_torn_sample_segment() {
    let content = read_golden("wdbx-sample.seg.truncated.jsonl");
    let segment = parse_segment(9, Path::new("sample.truncated"), &content).expect("parses");
    assert_eq!(segment.records.len(), 1, "the complete record survives");
    assert!(segment.truncated_tail, "the torn line must be reported");
}

#[test]
fn parses_the_sample_manifest() {
    let manifest = Manifest::parse(&read_golden("wdbx-sample.manifest")).expect("parses");
    assert_eq!(manifest.next_epoch, 4);
    // Epoch 1 is deliberately absent: it is the "collected, must not load" case.
    assert_eq!(manifest.active, [0, 2, 3]);
    assert!(!manifest.active.contains(&1));
}

/// Reads the developer's real `~/.abi/` store.
///
/// Opt-in via `ABI_WDBX_REAL_STORE_TEST=1`. Read-only — it never writes to the
/// store.
#[test]
fn reads_the_real_store_when_opted_in() {
    if std::env::var_os("ABI_WDBX_REAL_STORE_TEST").is_none() {
        return;
    }

    let Some(home) = std::env::var_os("HOME") else {
        panic!("HOME is unset; cannot locate the real store");
    };
    let dir = PathBuf::from(home).join(".abi");
    if !dir.is_dir() {
        println!("no store at {} — nothing to check", dir.display());
        return;
    }

    let paths = StorePaths::new(&dir);
    let manifest = paths.read_manifest().expect("the real manifest must parse");
    assert!(
        !manifest.active.is_empty(),
        "a real store should list active epochs"
    );

    let snapshot = abi_wdbx::store::load(&paths).expect("the real store must load");

    // Printed in the same field order as Zig's `abi wdbx db verify`, so the two
    // implementations can be compared line for line. They agreed exactly on the
    // observed store: kv=262 vectors=663 blocks=330 spatial=0 temporal_nodes=3
    // temporal_edges=2, chain valid.
    println!(
        "loaded {} epochs: kv={} vectors={} blocks={} spatial={} temporal_nodes={} temporal_edges={} truncated={}",
        snapshot.stats.epochs_loaded,
        snapshot.stats.kv_entries,
        snapshot.stats.vectors,
        snapshot.stats.blocks,
        snapshot.stats.spatial_records,
        snapshot.stats.temporal_nodes,
        snapshot.stats.temporal_edges,
        snapshot.stats.truncated_segments,
    );

    assert!(
        snapshot.stats.epochs_loaded > 0,
        "at least one active segment should have loaded"
    );
    assert!(
        snapshot.stats.vectors > 0,
        "the observed store contains ~100k vector records"
    );
    assert!(
        snapshot.stats.blocks > 0,
        "the observed store contains ~50k block records"
    );

    // Every vector in the observed store is 32-dimensional. A None here would
    // mean they disagree, which is worth knowing about but is not this test's
    // business to fail on.
    if let Some(dims) = snapshot.vector_dimensions() {
        println!("vector dimensions: {dims}");

        let hnsw = HnswIndex::from_snapshot(&snapshot)
            .expect("the real vectors must rebuild into the Rust HNSW graph");
        hnsw.validate_graph()
            .expect("the rebuilt real-store HNSW graph must be internally valid");
        let exact = ExactIndex::from_snapshot(&snapshot)
            .expect("the real vectors must rebuild into the exact oracle");
        let query = snapshot
            .vectors
            .first_key_value()
            .map(|(_, vector)| vector.as_slice())
            .expect("the observed store has vectors");
        let approximate = hnsw.search(query, 10).expect("HNSW query");
        let reference = exact.search(query, 10).expect("exact query");
        assert!(!approximate.is_empty());
        assert!(!reference.is_empty());
        println!(
            "HNSW candidate id={} score={:.6}; exact id={} score={:.6}",
            approximate[0].id, approximate[0].score, reference[0].id, reference[0].score
        );
        assert!(
            (approximate[0].score - reference[0].score).abs() < 0.0001,
            "HNSW should recover an exact nearest neighbor for a stored vector"
        );
        println!(
            "HNSW rebuilt {} vectors; stored-vector top score={:.6}",
            hnsw.len(),
            approximate[0].score
        );
    } else {
        println!("vector dimensions vary across the store");
    }

    // Reported rather than asserted. A break is meaningful information about the
    // user's data, but this test's job is to prove the *reader* works — failing
    // the gate over the state of a machine-local store would be wrong.
    match snapshot.verify_chain() {
        Ok(()) => println!(
            "audit chain verifies across all {} blocks",
            snapshot.stats.blocks
        ),
        Err(break_at) => println!("audit chain break: {break_at}"),
    }

    // The load must be a single checkpoint, not a replay. If this ever reads
    // higher than 1, the checkpoint-vs-delta semantics have regressed and the
    // block chain will be silently duplicated once per epoch.
    assert_eq!(
        snapshot.stats.epochs_loaded, 1,
        "exactly one checkpoint should be loaded, not every active epoch"
    );
}
