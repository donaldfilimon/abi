//! Opt-in byte compatibility between Rust and the restored Zig multiway oracle.

use std::path::PathBuf;
use std::process::Command;

use abi_wdbx::{
    MultiwayConfig, MultiwayRule, compute_multiway_metrics, export_multiway_json, run_multiway,
};

#[test]
fn rust_canonical_export_matches_zig_when_opted_in() {
    let Some(zig_binary) = std::env::var_os("ABI_ZIG_BIN") else {
        eprintln!("skipping Zig multiway compatibility; set ABI_ZIG_BIN to opt in");
        return;
    };
    let directory = abi_foundation::temp_path::temp_file_path("abi_multiway_zig_compat", "fixture");
    std::fs::create_dir_all(&directory).expect("fixture directory");
    let export_path: PathBuf = directory.join("zig-export.json");
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root")
        .to_owned();

    let status = Command::new(zig_binary)
        .current_dir(workspace_root)
        .args([
            "wdbx",
            "simulate",
            "--initial",
            "A",
            "--rule",
            "A->AB",
            "--depth",
            "3",
            "--max-states",
            "100",
            "--max-events",
            "1000",
            "--max-payload",
            "64",
            "--format",
            "json",
            "--output",
        ])
        .arg(&export_path)
        .arg("--quiet")
        .status()
        .expect("run Zig oracle");
    assert!(status.success(), "Zig oracle failed");

    let mut config = MultiwayConfig::new(vec![b"A".to_vec()], vec![MultiwayRule::new(b"A", b"AB")]);
    config.max_depth = 3;
    config.max_states = 100;
    config.max_events = 1_000;
    config.max_payload = 64;
    let result = run_multiway(&config, None);
    let rust_export = export_multiway_json(&config, &result, &compute_multiway_metrics(&result))
        .expect("Rust export");
    let zig_export = std::fs::read_to_string(&export_path).expect("Zig export");

    assert_eq!(rust_export, zig_export);
    std::fs::remove_dir_all(directory).expect("cleanup");
}
