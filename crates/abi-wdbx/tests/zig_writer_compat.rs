//! Transitional differential test: Zig must read checkpoints written by Rust.
//!
//! Set `ABI_ZIG_BIN` to the full Zig CLI binary. The test is intentionally
//! opt-in because the final Rust-only cutover removes that binary and this test
//! with it.

use abi_wdbx::wal::{append_kv, create_with_epoch, wal_path};
use abi_wdbx::{
    BlockRecord, Hash, Snapshot, SpatialRecord, StorePaths, TemporalKind, TemporalRecord, flush,
};
use std::collections::BTreeMap;
use std::process::Command;

#[test]
fn zig_verifies_a_rust_written_checkpoint_when_opted_in() {
    let Some(zig_bin) = std::env::var_os("ABI_ZIG_BIN") else {
        return;
    };

    let dir = abi_foundation::temp_path::temp_file_path("abi_wdbx_zig_compat", "store");
    std::fs::create_dir_all(&dir).expect("create fixture directory");
    let _cleanup = Cleanup(dir.clone());
    let paths = StorePaths {
        dir: dir.clone(),
        base: "compat".to_string(),
    };

    let mut temporal_node = serde_json::Map::new();
    temporal_node.insert("id".to_string(), serde_json::json!(3));
    temporal_node.insert("timestamp_ms".to_string(), serde_json::json!(99));
    let mut temporal_edge = serde_json::Map::new();
    temporal_edge.insert("cause".to_string(), serde_json::json!(3));
    temporal_edge.insert("effect".to_string(), serde_json::json!(4));

    let mut snapshot = Snapshot {
        kv: BTreeMap::from([("compat".to_string(), "rust-writer".to_string())]),
        vectors: BTreeMap::from([(1, vec![0.25, 0.5, 0.75])]),
        blocks: vec![BlockRecord {
            hash: Hash([1; 32]),
            prev_hash: Hash::GENESIS,
            timestamp_ms: 1_753_000_000_000,
            sequence: 0,
            profile: "abi".to_string(),
            query_id: 1,
            response_id: 2,
            metadata: "written by Rust".to_string(),
        }],
        spatial: BTreeMap::from([(
            5,
            SpatialRecord {
                id: 5,
                x: 1.0,
                y: -2.0,
                z: 3.5,
                payload: "point".to_string(),
            },
        )]),
        temporal: vec![
            TemporalRecord {
                kind: TemporalKind::Node,
                fields: temporal_node,
            },
            TemporalRecord {
                kind: TemporalKind::Edge,
                fields: temporal_edge,
            },
        ],
        stats: abi_wdbx::SnapshotStats::default(),
    };
    snapshot.recount();
    flush(&paths, &snapshot).expect("Rust flush");
    let wal = wal_path(&paths);
    create_with_epoch(&wal, 0).expect("create Rust WAL");
    append_kv(&wal, "wal-delta", "yes").expect("append Rust WAL frame");

    let base = dir.join("compat");
    let repository_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output = Command::new(zig_bin)
        .current_dir(repository_root)
        .args(["wdbx", "db", "verify"])
        .arg(&base)
        .output()
        .expect("run Zig verifier");
    assert!(
        output.status.success(),
        "Zig rejected Rust checkpoint\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("checkpoint OK: source=segment epoch=0"));
    assert!(stderr.contains("kv=1 vectors=1 blocks=1 spatial=1"));
    assert!(stderr.contains("temporal_nodes=1 temporal_edges=1"));
    assert!(stderr.contains("chain_valid=true"));
    assert!(stderr.contains("WAL OK: frames=1"));
}

struct Cleanup(std::path::PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
