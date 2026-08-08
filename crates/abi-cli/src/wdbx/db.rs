//! WDBX `db` subcommand: segment checkpoints, WAL recovery, and snapshot integrity.
//!
//! Split from the flat `wdbx` CLI module; dispatch lives in `super::run`.

use crate::app::Outcome;
use crate::usage::is_help_token;
use crate::wdbx::paths_from_cli_base;
use abi_wdbx::v2::{
    ABI_WDBX_ENCRYPTION_KEY_FILE, ABI_WDBX_SIGNING_KEY_FILE, ABI_WDBX_VERIFY_KEY_FILE,
    MigrationStatus, ObjectSecurity, VersionedSnapshot, gc_versioned, generate_key_material,
    migration_status, open_versioned_read_only, rekey_versioned, verify_versioned,
    write_key_material,
};
use abi_wdbx::{
    AutoencoderConfigV1, CompactionReport, PqConfigV1, SegmentCodecKind, SegmentCodecPolicy,
    VersionedStore, Wal,
};
use std::fmt::Write;
use std::path::Path;

pub(crate) const DB_HELP: &str = "usage: abi wdbx db <init|verify> <path>\n       abi wdbx db compact <path> [--codec none|pq|autoencoder]\n       abi wdbx db verify <path> --require-signature\n       abi wdbx db migration-status <path>\n       abi wdbx db migration-dry-run <path>\n       abi wdbx db keygen <directory>\n       abi wdbx db rekey <path> <new-key-directory> --confirm\n       abi wdbx db gc <path> --confirm\n\nManage segment checkpoints, WAL recovery, snapshot integrity, and v2 object keys.\nCodec none preserves vector bits exactly. PQ and autoencoder are explicit lossy vector representations and report persisted quality evidence.\n";

fn keygen(directory: &str) -> Outcome {
    let directory = Path::new(directory);
    if let Err(detail) = std::fs::create_dir_all(directory) {
        return super::error("keygen failed", detail);
    }
    let targets = [
        directory.join("encryption.key"),
        directory.join("signing.key"),
        directory.join("verify.key"),
    ];
    if let Some(existing) = targets.iter().find(|path| path.exists()) {
        return super::error(
            "keygen failed",
            format!("refusing to overwrite {}", existing.display()),
        );
    }
    let (encryption, signing, verifying) = generate_key_material();
    for (path, material) in targets.iter().zip([&encryption, &signing, &verifying]) {
        if let Err(detail) = write_key_material(path, material) {
            for created in targets.iter().take_while(|created| *created != path) {
                std::fs::remove_file(created).ok();
            }
            return super::error("keygen failed", detail);
        }
    }
    Outcome::stderr(
        format!(
            "generated WDBX v2 keys (key bytes not displayed)\n{ABI_WDBX_ENCRYPTION_KEY_FILE}={}\n{ABI_WDBX_SIGNING_KEY_FILE}={}\n{ABI_WDBX_VERIFY_KEY_FILE}={}\n",
            targets[0].display(),
            targets[1].display(),
            targets[2].display()
        ),
        0,
    )
}

fn rekey(raw_path: &str, key_directory: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("rekey failed", detail),
    };
    let key_directory = Path::new(key_directory);
    let replacement = match ObjectSecurity::from_key_files(
        Some(&key_directory.join("encryption.key")),
        Some(&key_directory.join("signing.key")),
        Some(&key_directory.join("verify.key")),
    ) {
        Ok(security) => security,
        Err(detail) => return super::error("rekey failed", detail),
    };
    let (store, report) = match rekey_versioned(&paths, replacement) {
        Ok(result) => result,
        Err(detail) => return super::error("rekey failed", detail),
    };
    drop(store);
    Outcome::stderr(
        format!(
            "rekeyed WDBX v2 generation: previous={} retained=true active={} digest={}\n",
            report.previous_generation.display(),
            report.generation.display(),
            report.digest
        ),
        0,
    )
}

fn init_db(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("db init failed", detail),
    };
    match migration_status(&paths) {
        Ok(MigrationStatus::Empty) => {}
        Ok(MigrationStatus::V1 { .. } | MigrationStatus::V2 { .. }) => {
            return super::error(
                "db init failed",
                "refusing to replace an initialized store; choose a new path",
            );
        }
        Err(detail) => return super::error("db init failed", detail),
    }
    let store = match abi_wdbx::VersionedStore::open(paths) {
        Ok(store) => store,
        Err(detail) => return super::error("db init failed", detail),
    };
    drop(store);
    Outcome::stderr(
        format!("initialized empty WDBX v2 store at {raw_path}\n"),
        0,
    )
}

fn verify_db(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("verify FAILED", detail),
    };
    match migration_status(&paths) {
        Ok(MigrationStatus::V2 { .. }) => return verify_v2(raw_path, &paths, false),
        Ok(MigrationStatus::Empty | MigrationStatus::V1 { .. }) => {}
        Err(detail) => return super::error("verify FAILED", detail),
    }
    let (snapshot, checkpoint_source) = match abi_wdbx::store::load_checkpoint_with_source(&paths) {
        Ok(loaded) => loaded,
        Err(detail) => {
            return super::error(&format!("verify FAILED: checkpoint {raw_path}"), detail);
        }
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

fn verify_v2(raw_path: &str, paths: &abi_wdbx::StorePaths, require_signature: bool) -> Outcome {
    let report = match verify_versioned(paths, require_signature) {
        Ok(report) => report,
        Err(detail) => return super::error(&format!("verify FAILED: {raw_path}"), detail),
    };
    Outcome::stderr(
        format!(
            "v2 verify OK: path={raw_path} require_signature={} writer_heads={} transactions={} kv={} vectors={} blocks={} spatial={} temporal={} merged_chain_valid=true recovered_digest={}\n",
            report.required_signature,
            report.writer_heads,
            report.committed_transactions,
            report.kv_keys,
            report.vectors,
            report.audit_blocks,
            report.spatial,
            report.temporal,
            report.recovered_digest
        ),
        0,
    )
}

fn verify_required_signature(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("verify FAILED", detail),
    };
    verify_v2(raw_path, &paths, true)
}

fn migration_status_report(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("migration status failed", detail),
    };
    let report = match migration_status(&paths) {
        Ok(MigrationStatus::Empty) => {
            format!("migration status: path={raw_path} format=empty writable_open=new-v2\n")
        }
        Ok(MigrationStatus::V1 { objects }) => format!(
            "migration status: path={raw_path} format=v1 objects={objects} writable_open=migrate-with-backup\n"
        ),
        Ok(MigrationStatus::V2 {
            generation,
            backup,
            digest,
        }) => format!(
            "migration status: path={raw_path} format=v2 generation={} backup={} digest={digest}\n",
            generation.display(),
            backup
                .as_ref()
                .map_or_else(|| "none".to_owned(), |path| path.display().to_string())
        ),
        Err(detail) => return super::error("migration status failed", detail),
    };
    Outcome::stderr(report, 0)
}

fn migration_dry_run(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("migration dry-run failed", detail),
    };
    let snapshot = match open_versioned_read_only(&paths) {
        Ok(snapshot) => snapshot,
        Err(detail) => return super::error("migration dry-run failed", detail),
    };
    let report = match snapshot {
        VersionedSnapshot::Empty => format!(
            "migration dry-run: path={raw_path} format=empty would_create_v2=true mutated=false\n"
        ),
        VersionedSnapshot::V1(snapshot) => format!(
            "migration dry-run: path={raw_path} format=v1 would_migrate=true backup=retained kv={} vectors={} blocks={} spatial={} temporal={} mutated=false\n",
            snapshot.kv.len(),
            snapshot.vectors.len(),
            snapshot.blocks.len(),
            snapshot.spatial.len(),
            snapshot.temporal.len()
        ),
        VersionedSnapshot::V2(snapshot) => format!(
            "migration dry-run: path={raw_path} format=v2 would_migrate=false writer_heads={} transactions={} vectors={} mutated=false\n",
            snapshot.heads().len(),
            snapshot.committed_transactions(),
            snapshot.vector_ids().count()
        ),
    };
    Outcome::stderr(report, 0)
}

fn gc(raw_path: &str) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("gc failed", detail),
    };
    let report = match gc_versioned(&paths, true) {
        Ok(report) => report,
        Err(detail) => return super::error("gc failed", detail),
    };
    Outcome::stderr(
        format!(
            "garbage-collected WDBX v2 generation: path={} retained_segment={} removed_objects={} removed_bytes={} prior_generations_retained=true\n",
            report.generation.display(),
            report.retained_segment.display(),
            report.removed_objects,
            report.removed_bytes
        ),
        0,
    )
}

fn codec_label(codec: SegmentCodecKind) -> &'static str {
    match codec {
        SegmentCodecKind::None => "none",
        SegmentCodecKind::Pq => "pq",
        SegmentCodecKind::Autoencoder => "autoencoder",
    }
}

fn learned_codec_policy(
    store: &VersionedStore,
    codec: SegmentCodecKind,
) -> Result<SegmentCodecPolicy, String> {
    if codec == SegmentCodecKind::None {
        return Ok(SegmentCodecPolicy::None);
    }
    let snapshot = store.snapshot();
    let dimensions = snapshot.vector_dimensions().ok_or_else(|| {
        "lossy codec compaction requires a non-empty, single-dimension vector space".to_owned()
    })?;
    let records = snapshot.vector_count();
    if records < 2 {
        return Err("lossy codec compaction requires at least two vector records".to_owned());
    }
    let recall_k = records.saturating_sub(1).min(3);
    match codec {
        SegmentCodecKind::None => Ok(SegmentCodecPolicy::None),
        SegmentCodecKind::Pq => Ok(SegmentCodecPolicy::Pq(PqConfigV1 {
            dimensions,
            subspaces: dimensions.min(3),
            centroids: records.min(8),
            iterations: 20,
            seed: 0xD15C_A11E,
            recall_k,
        })),
        SegmentCodecKind::Autoencoder if dimensions < 2 => {
            Err("autoencoder compaction requires vectors with at least two dimensions".to_owned())
        }
        SegmentCodecKind::Autoencoder => Ok(SegmentCodecPolicy::Autoencoder(AutoencoderConfigV1 {
            input_dimensions: dimensions,
            latent_dimensions: (dimensions / 2).max(1),
            seed: 0xA070_EC0D,
            epochs: 200,
            learning_rate: 0.08,
            recall_k,
        })),
    }
}

fn compaction_report(raw_path: &str, report: &CompactionReport) -> String {
    let codec = codec_label(report.codec);
    let lossy = report.codec != SegmentCodecKind::None;
    let artifact = report.artifact_id.as_deref().unwrap_or("none");
    let mut output = format!(
        "compacted WDBX v2: path={raw_path} segment={} object_id={} codec={codec} lossy={lossy} artifact={artifact} encrypted={} signed={} covered_writer_heads={}",
        report.path.display(),
        report.object_id,
        report.encrypted,
        report.signed,
        report.covered_heads.len()
    );
    if let Some(metrics) = &report.codec_metrics {
        write!(
            output,
            " warning=lossy-vector-reconstruction reconstruction_mse={} compression_ratio={} recall_at_k={} recall_k={}",
            metrics.reconstruction_mse,
            metrics.compression_ratio,
            metrics.recall_at_k,
            metrics.recall_k
        )
        .expect("writing to a String cannot fail");
    }
    output.push('\n');
    output
}

fn compact_db(raw_path: &str, codec: SegmentCodecKind) -> Outcome {
    let paths = match paths_from_cli_base(raw_path) {
        Ok(paths) => paths,
        Err(detail) => return super::error("compact FAILED", detail),
    };
    let mut store = match VersionedStore::open(paths) {
        Ok(store) => store,
        Err(detail) => return super::error(&format!("compact FAILED: {raw_path}"), detail),
    };
    let policy = match learned_codec_policy(&store, codec) {
        Ok(policy) => policy,
        Err(detail) => return super::error(&format!("compact FAILED: {raw_path}"), detail),
    };
    let report = match store.compact_with_codec(policy) {
        Ok(report) => report,
        Err(detail) => return super::error(&format!("compact FAILED: {raw_path}"), detail),
    };
    Outcome::stderr(compaction_report(raw_path, &report), 0)
}

pub(crate) fn run_db(args: &[String]) -> Outcome {
    if args.len() == 1 && is_help_token(&args[0]) {
        return Outcome::stderr(DB_HELP.to_owned(), 0);
    }
    match args {
        [operation, path] if operation == "init" => init_db(path),
        [operation, path] if operation == "verify" => verify_db(path),
        [operation, path, require] if operation == "verify" && require == "--require-signature" => {
            verify_required_signature(path)
        }
        [operation, path] if operation == "migration-status" => migration_status_report(path),
        [operation, path] if operation == "migration-dry-run" => migration_dry_run(path),
        [operation, directory] if operation == "keygen" => keygen(directory),
        [operation, path, key_directory, confirm]
            if operation == "rekey" && confirm == "--confirm" =>
        {
            rekey(path, key_directory)
        }
        [operation, path, confirm] if operation == "gc" && confirm == "--confirm" => gc(path),
        [operation, path] if operation == "compact" => compact_db(path, SegmentCodecKind::None),
        [operation, path, flag, codec] if operation == "compact" && flag == "--codec" => {
            match codec.as_str() {
                "none" => compact_db(path, SegmentCodecKind::None),
                "pq" => compact_db(path, SegmentCodecKind::Pq),
                "autoencoder" => compact_db(path, SegmentCodecKind::Autoencoder),
                _ => super::usage(),
            }
        }
        _ => super::usage(),
    }
}
