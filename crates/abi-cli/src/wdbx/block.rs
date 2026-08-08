//! WDBX `block` subcommand: SHA-linked conversation block insert/get.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use crate::app::Outcome;
use crate::usage::is_help_token;
use crate::wdbx::paths_from_cli_base;
use abi_foundation::time::unix_ms;
use abi_wdbx::{RecordId, VersionedSnapshot, VersionedStore, open_versioned_read_only};

pub(crate) const BLOCK_HELP: &str = "usage: abi wdbx block <insert|get> <path> ...\n\nAppend or inspect SHA-linked conversation blocks in a WDBX checkpoint.\n";

fn insert_block(raw_path: &str, profile: &str, metadata: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("block insert failed", detail),
    };
    let mut store = match VersionedStore::open(paths) {
        Ok(store) => store,
        Err(detail) => return super::error(&format!("error: {raw_path}"), detail),
    };
    let timestamp_ms = unix_ms();
    let block = match store.add_block(
        profile,
        RecordId::new_v2(),
        RecordId::new_v2(),
        metadata,
        timestamp_ms,
    ) {
        Ok(block) => block,
        Err(detail) => return super::error("block insert failed", detail),
    };
    if let Err(detail) = store.compact() {
        return super::error("block insert failed", detail);
    }
    Outcome::stderr(
        format!(
            "appended block: profile={profile} blocks={} hash={}\n",
            store.stats().blocks,
            block.hash
        ),
        0,
    )
}

fn get_block(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("block get failed", detail),
    };
    let snapshot = match open_versioned_read_only(&paths) {
        Ok(snapshot) => snapshot,
        Err(detail) => return super::error(&format!("error: {raw_path}"), detail),
    };
    match snapshot {
        VersionedSnapshot::Empty => Outcome::stderr(format!("no blocks in {raw_path}\n"), 0),
        VersionedSnapshot::V1(snapshot) => {
            let Some(block) = snapshot.blocks.last() else {
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
        VersionedSnapshot::V2(snapshot) => {
            let Some(block) = snapshot.audit_blocks().last() else {
                return Outcome::stderr(format!("no blocks in {raw_path}\n"), 0);
            };
            Outcome::stderr(
                format!(
                    "block: profile={} query_id={} response_id={} timestamp_ms={}\n  hash={}\n  metadata={}\n",
                    block.profile,
                    block.query_id,
                    block.response_id,
                    block.timestamp_ms,
                    block.hash,
                    block.metadata
                ),
                0,
            )
        }
    }
}

pub(crate) fn run_block(args: &[String]) -> Outcome {
    if args.len() == 1 && is_help_token(&args[0]) {
        return Outcome::stderr(BLOCK_HELP.to_owned(), 0);
    }
    match args {
        [operation, path, profile, metadata] if operation == "insert" => {
            insert_block(path, profile, metadata)
        }
        [operation, path] if operation == "get" => get_block(path),
        _ => super::usage(),
    }
}
