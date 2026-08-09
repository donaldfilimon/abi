//! Persist executed `abi agent os` commands to the WDBX audit chain.
//!
//! Before this, an executed command left only a single `[os-cmd]` line on
//! stderr — gone the moment the terminal scrolled. The audit chain is the
//! store's append-only, hash-linked record, which is what "auditable command
//! execution" has to mean if it is to mean anything.
//!
//! **Only `execute` is recorded.** `dry-run` spawns nothing, so writing a
//! block for it would put commands in the chain that never ran — the opposite
//! of an audit trail.
//!
//! The store is injected rather than resolved here so tests use a scratch
//! `VersionedStore` and never open the user's real `~/.abi/`.
//!
//! The three WAL writes are ordered but not transactional. A failure after the
//! vector or metadata write is disclosed to the caller and may leave that
//! earlier record without a block; this module does not claim atomic append.

use abi_wdbx::{RecordId, VersionedStore};

/// Profile label for os-control blocks, distinguishing them from the persona
/// blocks (`Abbey` / `Aviva` / `ABI`) that share the chain.
const AUDIT_PROFILE: &str = "os-control";

/// What a successful audit write produced.
pub(crate) struct Recorded {
    pub(crate) vector_id: RecordId,
    pub(crate) block_hash: String,
}

/// Canonical JSON for one executed command.
///
/// `serde_json` does the escaping, so a command argument containing quotes,
/// backslashes, newlines, or non-ASCII text cannot corrupt the record or
/// forge extra fields.
pub(crate) fn metadata_json(
    argv: &[&str],
    cwd: &str,
    exit_code: u8,
    elapsed_ms: u128,
    timed_out: bool,
) -> String {
    let value = serde_json::json!({
        "kind": "os-cmd",
        "argv": argv,
        "cwd": cwd,
        "exit_code": exit_code,
        "elapsed_ms": u64::try_from(elapsed_ms).unwrap_or(u64::MAX),
        "timed_out": timed_out,
        "env_filtered": true,
    });
    value.to_string()
}

/// Append one os-control block to the audit chain.
///
/// The command line is embedded so the record is searchable alongside the
/// persona vectors already in the store, and the same id is used for both
/// endpoints of the block: an executed command is a single event, not a
/// query/response pair.
///
/// # Errors
///
/// Any store write failure, returned as a string for direct disclosure.
pub(crate) fn record(
    store: &mut VersionedStore,
    argv: &[&str],
    cwd: &str,
    exit_code: u8,
    elapsed_ms: u128,
    timed_out: bool,
    now_ms: i64,
) -> Result<Recorded, String> {
    let metadata = metadata_json(argv, cwd, exit_code, elapsed_ms, timed_out);
    let embedding = abi_ai::text_embedding(&argv.join(" "));

    let vector_id = store
        .put_vector(&embedding)
        .map_err(|e| format!("put_vector failed: {e}"))?;
    store
        .put(&format!("os-cmd:{vector_id}"), &metadata)
        .map_err(|e| format!("metadata write failed: {e}"))?;
    let block = store
        .add_block(AUDIT_PROFILE, vector_id, vector_id, &metadata, now_ms)
        .map_err(|e| format!("add_block failed: {e}"))?;

    Ok(Recorded {
        vector_id,
        block_hash: block.hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_store(tag: &str) -> (VersionedStore, std::path::PathBuf) {
        // Never `~/.abi/`: a scratch path under the temp dir, unique per test.
        // PID+ThreadId alone can collide with a leftover dir from a *panicked*
        // prior run (the libtest thread pool reuses ThreadIds, and a panic
        // skips this test's own cleanup) — the shared counter-based helper
        // guarantees no collision with anything this process has made before.
        let dir =
            abi_foundation::temp_path::temp_file_path(&format!("abi-os-audit-{tag}"), "store");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let store = VersionedStore::open(abi_wdbx::StorePaths::new(dir.display().to_string()))
            .expect("scratch store opens");
        (store, dir)
    }

    #[test]
    fn metadata_is_canonical_json_with_the_executed_fields() {
        let json = metadata_json(&["ls", "-la"], "/tmp", 0, 12, false);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["kind"], "os-cmd");
        assert_eq!(parsed["argv"][0], "ls");
        assert_eq!(parsed["argv"][1], "-la");
        assert_eq!(parsed["cwd"], "/tmp");
        assert_eq!(parsed["exit_code"], 0);
        assert_eq!(parsed["elapsed_ms"], 12);
        assert_eq!(parsed["timed_out"], false);
        assert_eq!(parsed["env_filtered"], true);
    }

    #[test]
    fn metadata_escapes_hostile_arguments_instead_of_breaking_the_record() {
        // A quote-and-newline argument must stay one JSON string value rather
        // than injecting structure into the audit record.
        let json = metadata_json(&["ls", "a\" , \"kind\": \"forged\n"], "/tmp", 0, 1, false);
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("still valid JSON");
        assert_eq!(parsed["kind"], "os-cmd", "kind was not overwritten");
        assert_eq!(parsed["argv"].as_array().expect("array").len(), 2);
    }

    #[test]
    fn recording_appends_a_block_and_a_vector_to_a_scratch_store() {
        let (mut store, dir) = scratch_store("append");
        let before = store.stats();

        let rec = record(&mut store, &["ls", "-la"], "/tmp", 0, 7, false, 1_000)
            .expect("audit write succeeds");

        let after = store.stats();
        assert_eq!(after.blocks, before.blocks + 1, "one audit block appended");
        assert_eq!(after.vectors, before.vectors + 1, "one vector appended");
        assert_eq!(
            after.kv_entries,
            before.kv_entries + 1,
            "one metadata entry"
        );
        assert_ne!(rec.block_hash, "");

        let stored = store
            .get(&format!("os-cmd:{}", rec.vector_id))
            .expect("metadata is retrievable by key");
        let parsed: serde_json::Value = serde_json::from_str(&stored).expect("valid JSON");
        assert_eq!(parsed["argv"][1], "-la");

        drop(store);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_recorded_block_is_reachable_after_reopening_the_store() {
        // Durability is the point: a record that lives only in memory is the
        // stderr line this replaced.
        let (mut store, dir) = scratch_store("reopen");
        let rec = record(&mut store, &["whoami"], "/tmp", 0, 3, false, 2_000)
            .expect("audit write succeeds");
        drop(store);

        let reopened = VersionedStore::open(abi_wdbx::StorePaths::new(dir.display().to_string()))
            .expect("store reopens");
        let stored = reopened
            .get(&format!("os-cmd:{}", rec.vector_id))
            .expect("metadata survived the reopen");
        assert!(stored.contains("whoami"));
        assert_eq!(reopened.stats().blocks, 1);

        drop(reopened);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
