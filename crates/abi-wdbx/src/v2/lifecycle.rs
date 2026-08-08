//! Version detection and crash-safe v1-to-v2 activation.
//!
//! The only activation point is one atomically replaced pointer file. Migration
//! builds and verifies a sibling v2 generation and a byte-exact v1 backup before
//! publishing that pointer. Until publication, every legacy object remains the
//! authoritative store.

use super::{
    ObjectSecurity, RecordId, V2AuditBlock, V2Error, V2Mutation, V2Snapshot, V2SpatialRecord,
    V2Store, V2TemporalKind, V2TemporalRecord,
};
use crate::durable::acquire_writer_lock;
use crate::store::{CheckpointSource, Snapshot, load_checkpoint_with_source};
use crate::wal::{Recovered, RecoverySource, Wal, wal_path};
use crate::{StorePaths, TemporalKind};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

const ACTIVE_HEADER: &str = "ABI-WDBX-ACTIVE 1\n";

/// Version detection or migration failure.
#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    /// Legacy recovery or writer locking failed.
    #[error("WDBX v1 migration source failed: {0}")]
    V1(String),
    /// V2 recovery or persistence failed.
    #[error(transparent)]
    V2(#[from] V2Error),
    /// Activation metadata is malformed or unsafe.
    #[error("invalid WDBX activation metadata: {0}")]
    InvalidActivation(String),
    /// Source and replacement generations are not equivalent.
    #[error("WDBX migration verification failed: {0}")]
    Verification(String),
    /// A lifecycle filesystem operation failed.
    #[error("WDBX migration I/O failed for {path}: {message}")]
    Io {
        /// Object being accessed.
        path: PathBuf,
        /// Underlying error.
        message: String,
    },
}

/// Non-mutating version probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MigrationStatus {
    /// No v1 objects and no active v2 pointer exist.
    Empty,
    /// Legacy objects exist and remain authoritative.
    V1 {
        /// Number of source objects that would be backed up.
        objects: usize,
    },
    /// An atomically published v2 generation is authoritative.
    V2 {
        /// Active generation directory.
        generation: PathBuf,
        /// Permanent byte-exact legacy backup, when this was a migration.
        backup: Option<PathBuf>,
        /// Canonical migrated-state digest, or the empty-store digest.
        digest: String,
    },
}

/// Read-only versioned result. Opening v1 through this path never edits its WAL.
#[derive(Debug, Clone)]
pub enum VersionedSnapshot {
    /// No store exists.
    Empty,
    /// Read-only legacy snapshot, including a matching WAL delta.
    V1(Arc<Snapshot>),
    /// Immutable v2 causal snapshot.
    V2(Arc<V2Snapshot>),
}

/// Result of opening a writable v2 generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    /// Whether this call migrated authoritative v1 state.
    pub migrated: bool,
    /// Active generation directory.
    pub generation: PathBuf,
    /// Permanent byte-exact legacy backup.
    pub backup: Option<PathBuf>,
    /// Canonical state digest verified before activation.
    pub digest: String,
}

/// Verified generation replacement produced by an explicit rekey operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RekeyReport {
    /// Previously active generation, retained without modification.
    pub previous_generation: PathBuf,
    /// Newly activated generation.
    pub generation: PathBuf,
    /// Full causal snapshot digest compared before activation.
    pub digest: String,
}

/// Read-only verification summary for an active v2 generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationReport {
    /// Verified active generation.
    pub generation: PathBuf,
    /// Whether every immutable object was required to carry a signature.
    pub required_signature: bool,
    /// Activation digest recorded when the generation was published.
    pub activation_digest: String,
    /// Digest of the fully recovered causal snapshot.
    pub recovered_digest: String,
    /// Number of writer heads in the recovered frontier.
    pub writer_heads: usize,
    /// Number of committed transactions represented by the snapshot.
    pub committed_transactions: usize,
    /// Logical key count, including conflict sets as one key each.
    pub kv_keys: usize,
    /// Stable vector identity count.
    pub vectors: usize,
    /// Spatial identity count.
    pub spatial: usize,
    /// Temporal logical-key count.
    pub temporal: usize,
    /// Immutable audit block count.
    pub audit_blocks: usize,
}

/// Objects reclaimed by an explicitly confirmed, coverage-verified v2 GC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcReport {
    /// Active generation cleaned in place.
    pub generation: PathBuf,
    /// Newly published segment that covers the recovered frontier.
    pub retained_segment: PathBuf,
    /// Dominated journal, head, and older-segment objects removed.
    pub removed_objects: usize,
    /// Sum of removed regular-file sizes.
    pub removed_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Activation {
    version: u8,
    generation: String,
    backup: Option<String>,
    digest: String,
}

/// Detect the active store format without creating, removing, or rewriting files.
pub fn migration_status(paths: &StorePaths) -> Result<MigrationStatus, MigrationError> {
    if paths_active(paths).is_file() {
        let activation = read_activation(paths)?;
        let generation = resolve_child(paths, &activation.generation)?;
        verify_v2_root(&generation)?;
        let backup = activation
            .backup
            .as_deref()
            .map(|name| resolve_child(paths, name))
            .transpose()?;
        return Ok(MigrationStatus::V2 {
            generation,
            backup,
            digest: activation.digest,
        });
    }
    let objects = v1_objects(paths)?;
    if objects.is_empty() {
        Ok(MigrationStatus::Empty)
    } else {
        Ok(MigrationStatus::V1 {
            objects: objects.len(),
        })
    }
}

/// Open the authoritative store without mutating v1 or creating an empty v2 store.
pub fn open_versioned_read_only(paths: &StorePaths) -> Result<VersionedSnapshot, MigrationError> {
    match migration_status(paths)? {
        MigrationStatus::Empty => Ok(VersionedSnapshot::Empty),
        MigrationStatus::V1 { .. } => Ok(VersionedSnapshot::V1(Arc::new(
            recover_v1_read_only(paths)?.snapshot,
        ))),
        MigrationStatus::V2 { generation, .. } => {
            let security = ObjectSecurity::from_env().map_err(|error| V2Error::Security {
                path: generation.clone(),
                reason: error.to_string(),
            })?;
            Ok(VersionedSnapshot::V2(Arc::new(super::recover(
                &generation,
                &security,
            )?)))
        }
    }
}

/// Open a writable v2 generation, migrating authoritative v1 state when needed.
///
/// New stores use v2. Existing v1 stores are locked only for the migration
/// window, copied to a permanent backup, verified record-for-record, and then
/// activated through a single atomic pointer write.
pub fn open_versioned_writable(
    paths: &StorePaths,
) -> Result<(V2Store, MigrationReport), MigrationError> {
    match migration_status(paths)? {
        MigrationStatus::V2 {
            generation,
            backup,
            digest,
        } => {
            let store = V2Store::open(&generation)?;
            Ok((
                store,
                MigrationReport {
                    migrated: false,
                    generation,
                    backup,
                    digest,
                },
            ))
        }
        MigrationStatus::Empty => create_empty_v2(paths),
        MigrationStatus::V1 { .. } => migrate_v1(paths),
    }
}

/// Re-encrypt/re-sign an active v2 store through verified generation replacement.
///
/// The caller supplies replacement keys explicitly. Current-generation keys are
/// loaded from the documented environment variables. No source or replacement
/// generation is deleted by this operation.
pub fn rekey_versioned(
    paths: &StorePaths,
    replacement_security: ObjectSecurity,
) -> Result<(V2Store, RekeyReport), MigrationError> {
    ensure_parent(paths)?;
    let MigrationStatus::V2 {
        generation: previous_generation,
        backup,
        ..
    } = migration_status(paths)?
    else {
        return Err(MigrationError::InvalidActivation(
            "rekey requires an active WDBX v2 generation".into(),
        ));
    };
    discard_rekey_staging(paths)?;
    let maintenance = super::lease::begin_maintenance(&previous_generation)?;
    let current_security = ObjectSecurity::from_env().map_err(|error| V2Error::Security {
        path: previous_generation.clone(),
        reason: error.to_string(),
    })?;
    let source = super::recover(&previous_generation, &current_security)?;
    let digest = snapshot_digest(&source)?;

    let nonce = Uuid::new_v4();
    let staging_name = format!(".{}.rekey-{nonce}", paths.base);
    let staging = paths.dir.join(&staging_name);
    initialize_v2_root(&staging)?;
    super::segment::write_segment(&staging, &source, &replacement_security)?;
    let replacement = super::recover(&staging, &replacement_security)?;
    if source != replacement || digest != snapshot_digest(&replacement)? {
        return Err(MigrationError::Verification(
            "rekey replacement changed causal heads, versions, records, or digest".into(),
        ));
    }

    let generation_name = format!("{}.v2-{nonce}", paths.base);
    let generation = paths.dir.join(&generation_name);
    std::fs::rename(&staging, &generation).map_err(|error| io_error(&generation, &error))?;
    let backup_name = backup.as_deref().map(file_name_string).transpose()?;
    publish_activation(
        paths,
        &Activation {
            version: 2,
            generation: generation_name,
            backup: backup_name,
            digest: digest.clone(),
        },
    )?;
    drop(maintenance);
    let store = V2Store::open_with_security(&generation, replacement_security)?;
    Ok((
        store,
        RekeyReport {
            previous_generation,
            generation,
            digest,
        },
    ))
}

/// Authenticate and causally replay the active v2 generation without mutation.
pub fn verify_versioned(
    paths: &StorePaths,
    require_signature: bool,
) -> Result<VerificationReport, MigrationError> {
    let MigrationStatus::V2 {
        generation,
        digest: activation_digest,
        ..
    } = migration_status(paths)?
    else {
        return Err(MigrationError::InvalidActivation(
            "signature-aware verification requires an active WDBX v2 generation".into(),
        ));
    };
    let security = ObjectSecurity::from_env().map_err(|error| V2Error::Security {
        path: generation.clone(),
        reason: error.to_string(),
    })?;
    let snapshot = super::recover_with_policy(&generation, &security, require_signature)?;
    Ok(VerificationReport {
        generation,
        required_signature: require_signature,
        activation_digest,
        recovered_digest: snapshot_digest(&snapshot)?,
        writer_heads: snapshot.heads.len(),
        committed_transactions: snapshot.committed_transactions,
        kv_keys: snapshot.kv.len(),
        vectors: snapshot.vectors.len(),
        spatial: snapshot.spatial.len(),
        temporal: snapshot.temporal.len(),
        audit_blocks: snapshot.audit.len(),
    })
}

/// Reclaim objects dominated by one newly verified covering segment.
///
/// This never removes v1 backups, prior rekey generations, unknown files, or
/// anything outside the active generation.
pub fn gc_versioned(paths: &StorePaths, confirm: bool) -> Result<GcReport, MigrationError> {
    if !confirm {
        return Err(MigrationError::InvalidActivation(
            "v2 garbage collection requires explicit confirmation".into(),
        ));
    }
    let MigrationStatus::V2 { generation, .. } = migration_status(paths)? else {
        return Err(MigrationError::InvalidActivation(
            "garbage collection requires an active WDBX v2 generation".into(),
        ));
    };
    let maintenance = super::lease::begin_maintenance(&generation)?;
    let security = ObjectSecurity::from_env().map_err(|error| V2Error::Security {
        path: generation.clone(),
        reason: error.to_string(),
    })?;
    let source = super::recover(&generation, &security)?;
    let covering = super::segment::write_segment(&generation, &source, &security)?;
    if super::recover(&generation, &security)? != source {
        return Err(MigrationError::Verification(
            "covering segment changed recovered state before garbage collection".into(),
        ));
    }

    let mut removed_objects = 0_usize;
    let mut removed_bytes = 0_u64;
    for (directory, extensions) in [
        (generation.join("journals"), &["objects", "jsonl"][..]),
        (generation.join("heads"), &["head"][..]),
        (generation.join("segments"), &["object"][..]),
    ] {
        if !directory.is_dir() {
            continue;
        }
        let mut paths: Vec<_> = std::fs::read_dir(&directory)
            .map_err(|error| io_error(&directory, &error))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extensions.contains(&extension))
            })
            .collect();
        paths.sort();
        for path in paths {
            if path == covering.path {
                continue;
            }
            removed_bytes = removed_bytes.saturating_add(
                std::fs::metadata(&path)
                    .map_err(|error| io_error(&path, &error))?
                    .len(),
            );
            std::fs::remove_file(&path).map_err(|error| io_error(&path, &error))?;
            removed_objects = removed_objects.saturating_add(1);
        }
    }
    if super::recover(&generation, &security)? != source {
        return Err(MigrationError::Verification(
            "retained covering segment did not survive post-GC recovery".into(),
        ));
    }
    drop(maintenance);
    Ok(GcReport {
        generation,
        retained_segment: covering.path,
        removed_objects,
        removed_bytes,
    })
}

fn create_empty_v2(paths: &StorePaths) -> Result<(V2Store, MigrationReport), MigrationError> {
    ensure_parent(paths)?;
    discard_unpublished_generations(paths)?;
    let generation_name = format!("{}.v2-{}", paths.base, Uuid::new_v4());
    let generation = paths.dir.join(&generation_name);
    let store = V2Store::open(&generation)?;
    let digest = MigrationImage::default().digest()?;
    publish_activation(
        paths,
        &Activation {
            version: 2,
            generation: generation_name,
            backup: None,
            digest: digest.clone(),
        },
    )?;
    Ok((
        store,
        MigrationReport {
            migrated: false,
            generation,
            backup: None,
            digest,
        },
    ))
}

fn migrate_v1(paths: &StorePaths) -> Result<(V2Store, MigrationReport), MigrationError> {
    ensure_parent(paths)?;
    let legacy_lock =
        acquire_writer_lock(paths).map_err(|error| MigrationError::V1(error.to_string()))?;
    // Re-check after acquiring the v1 lock so two migrators cannot both publish.
    if let MigrationStatus::V2 { .. } = migration_status(paths)? {
        drop(legacy_lock);
        return open_versioned_writable(paths);
    }

    discard_unpublished_generations(paths)?;
    let recovered = recover_v1_read_only(paths)?;
    recovered
        .snapshot
        .verify_chain_strict()
        .map_err(|error| MigrationError::Verification(error.to_string()))?;
    let source = MigrationImage::from_v1(&recovered.snapshot);
    let digest = source.digest()?;

    let nonce = Uuid::new_v4();
    let staging_name = format!(".{}.migration-{nonce}", paths.base);
    let staging = paths.dir.join(&staging_name);
    let mut replacement = V2Store::open(&staging)?;
    let mutations = source.mutations();
    if !mutations.is_empty() {
        replacement.commit(mutations)?;
    }
    let rebuilt = MigrationImage::from_v2(&replacement.snapshot())?;
    if source != rebuilt || digest != rebuilt.digest()? {
        return Err(MigrationError::Verification(
            "record families, identities, vector contents, audit chain, or canonical digest differ"
                .into(),
        ));
    }
    drop(replacement);

    let backup = create_backup(paths)?;
    let generation_name = format!("{}.v2-{nonce}", paths.base);
    let generation = paths.dir.join(&generation_name);
    std::fs::rename(&staging, &generation).map_err(|error| io_error(&generation, &error))?;
    let backup_name = file_name_string(&backup)?;
    publish_activation(
        paths,
        &Activation {
            version: 2,
            generation: generation_name,
            backup: Some(backup_name),
            digest: digest.clone(),
        },
    )?;
    let store = V2Store::open(&generation)?;
    Ok((
        store,
        MigrationReport {
            migrated: true,
            generation,
            backup: Some(backup),
            digest,
        },
    ))
}

fn recover_v1_read_only(paths: &StorePaths) -> Result<Recovered, MigrationError> {
    let (mut snapshot, checkpoint_source) = load_checkpoint_with_source(paths)
        .map_err(|error| MigrationError::V1(error.to_string()))?;
    let checkpoint_epoch = checkpoint_source.epoch();
    let source = match checkpoint_source {
        CheckpointSource::Empty => RecoverySource::Empty,
        CheckpointSource::Legacy { .. } => RecoverySource::Snapshot,
        CheckpointSource::Segment { .. } => RecoverySource::Segment,
    };
    let path = wal_path(paths);
    if !path.is_file() {
        return Ok(Recovered {
            snapshot,
            source,
            checkpoint_epoch,
            frames_applied: 0,
            torn_wal_tail: false,
        });
    }
    let wal = Wal::read(&path).map_err(|error| MigrationError::V1(error.to_string()))?;
    if wal.base_epoch != checkpoint_epoch {
        return Ok(Recovered {
            snapshot,
            source,
            checkpoint_epoch,
            frames_applied: 0,
            torn_wal_tail: false,
        });
    }
    let frames_applied = wal
        .replay_onto(&mut snapshot)
        .map_err(|error| MigrationError::V1(error.to_string()))?;
    Ok(Recovered {
        snapshot,
        source: RecoverySource::Merged,
        checkpoint_epoch,
        frames_applied,
        torn_wal_tail: wal.torn_tail,
    })
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
struct MigrationImage {
    kv: BTreeMap<String, String>,
    vectors: BTreeMap<RecordId, Vec<f32>>,
    audit: Vec<V2AuditBlock>,
    spatial: BTreeMap<RecordId, V2SpatialRecord>,
    temporal: Vec<V2TemporalRecord>,
}

impl MigrationImage {
    fn from_v1(snapshot: &Snapshot) -> Self {
        let vectors = snapshot
            .vectors
            .iter()
            .map(|(id, values)| (RecordId::Legacy(*id), values.clone()))
            .collect();
        let audit = snapshot
            .blocks
            .iter()
            .map(|block| V2AuditBlock {
                hash: block.hash.to_hex(),
                parents: if block.prev_hash.is_genesis() {
                    Vec::new()
                } else {
                    vec![block.prev_hash.to_hex()]
                },
                timestamp_ms: block.timestamp_ms,
                sequence: block.sequence,
                profile: block.profile.clone(),
                query_id: RecordId::Legacy(block.query_id),
                response_id: RecordId::Legacy(block.response_id),
                metadata: block.metadata.clone(),
            })
            .collect();
        let spatial = snapshot
            .spatial
            .iter()
            .map(|(id, record)| {
                let id = RecordId::Legacy(*id);
                (
                    id,
                    V2SpatialRecord {
                        id,
                        x: record.x,
                        y: record.y,
                        z: record.z,
                        payload: record.payload.clone(),
                    },
                )
            })
            .collect();
        let temporal = snapshot
            .temporal
            .iter()
            .map(|record| V2TemporalRecord {
                kind: match record.kind {
                    TemporalKind::Node => V2TemporalKind::Node,
                    TemporalKind::Edge => V2TemporalKind::Edge,
                },
                fields: record.fields.clone(),
            })
            .collect();
        Self {
            kv: snapshot.kv.clone(),
            vectors,
            audit,
            spatial,
            temporal,
        }
    }

    fn from_v2(snapshot: &V2Snapshot) -> Result<Self, MigrationError> {
        let mut image = Self::default();
        for (key, versions) in &snapshot.kv {
            image.kv.insert(key.clone(), only_value(versions, key)?);
        }
        for (id, versions) in &snapshot.vectors {
            image
                .vectors
                .insert(*id, only_value(versions, &id.to_string())?);
        }
        for (id, versions) in &snapshot.spatial {
            image
                .spatial
                .insert(*id, only_value(versions, &id.to_string())?);
        }
        for (key, versions) in &snapshot.temporal {
            image.temporal.push(only_value(versions, key)?);
        }
        image.audit.extend(snapshot.audit_blocks().cloned());
        Ok(image)
    }

    fn mutations(&self) -> Vec<V2Mutation> {
        let mut mutations = Vec::with_capacity(
            self.kv.len()
                + self.vectors.len()
                + self.audit.len()
                + self.spatial.len()
                + self.temporal.len(),
        );
        mutations.extend(self.kv.iter().map(|(key, value)| V2Mutation::PutKv {
            key: key.clone(),
            value: value.clone(),
        }));
        mutations.extend(
            self.vectors
                .iter()
                .map(|(id, values)| V2Mutation::PutVector {
                    id: *id,
                    values: values.clone(),
                }),
        );
        mutations.extend(
            self.audit
                .iter()
                .cloned()
                .map(|block| V2Mutation::PutAudit { block }),
        );
        mutations.extend(
            self.spatial
                .values()
                .cloned()
                .map(|record| V2Mutation::PutSpatial { record }),
        );
        mutations.extend(
            self.temporal
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, record)| V2Mutation::PutTemporal {
                    key: format!("legacy:{index:020}"),
                    record,
                }),
        );
        mutations
    }

    fn digest(&self) -> Result<String, MigrationError> {
        let bytes = serde_json::to_vec(self)
            .map_err(|error| MigrationError::Verification(error.to_string()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

fn only_value<T: Clone>(
    versions: &[super::Version<T>],
    identity: &str,
) -> Result<T, MigrationError> {
    let current = super::types::current_versions(versions).ok_or_else(|| {
        MigrationError::Verification(format!("{identity} has no current v2 value"))
    })?;
    if !current.conflicts.is_empty() {
        return Err(MigrationError::Verification(format!(
            "{identity} unexpectedly has concurrent versions immediately after migration"
        )));
    }
    Ok(current.preferred.value)
}

fn create_backup(paths: &StorePaths) -> Result<PathBuf, MigrationError> {
    let objects = v1_objects(paths)?;
    if objects.is_empty() {
        return Err(MigrationError::Verification(
            "legacy objects disappeared while the migration lock was held".into(),
        ));
    }
    let nonce = Uuid::new_v4();
    let staging = paths.dir.join(format!(".{}.backup-{nonce}", paths.base));
    std::fs::create_dir(&staging).map_err(|error| io_error(&staging, &error))?;
    for source in objects {
        let name = source.file_name().ok_or_else(|| {
            MigrationError::Verification(format!("{} has no file name", source.display()))
        })?;
        copy_file_verified(&source, &staging.join(name))?;
    }
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| MigrationError::Verification(error.to_string()))?
        .as_nanos();
    let backup = paths
        .dir
        .join(format!("{}.v1-backup-{timestamp}Z", paths.base));
    std::fs::rename(&staging, &backup).map_err(|error| io_error(&backup, &error))?;
    Ok(backup)
}

fn copy_file_verified(source: &Path, destination: &Path) -> Result<(), MigrationError> {
    let source_digest = file_digest(source)?;
    std::fs::copy(source, destination).map_err(|error| io_error(destination, &error))?;
    let destination_file = std::fs::OpenOptions::new()
        .read(true)
        .open(destination)
        .map_err(|error| io_error(destination, &error))?;
    destination_file
        .sync_all()
        .map_err(|error| io_error(destination, &error))?;
    if source_digest != file_digest(destination)? {
        return Err(MigrationError::Verification(format!(
            "backup copy of {} differs byte-for-byte",
            source.display()
        )));
    }
    Ok(())
}

fn file_digest(path: &Path) -> Result<[u8; 32], MigrationError> {
    let mut file = std::fs::File::open(path).map_err(|error| io_error(path, &error))?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| io_error(path, &error))?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(hash.finalize().into())
}

fn v1_objects(paths: &StorePaths) -> Result<Vec<PathBuf>, MigrationError> {
    if !paths.dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut objects = Vec::new();
    let exact = paths.base.as_str();
    let prefix = format!("{}.", paths.base);
    for entry in std::fs::read_dir(&paths.dir).map_err(|error| io_error(&paths.dir, &error))? {
        let entry = entry.map_err(|error| io_error(&paths.dir, &error))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let is_store_object = name == exact
            || name.starts_with(&prefix)
                && !name.ends_with(".writer.lock")
                && name != format!("{}.active", paths.base);
        if is_store_object {
            objects.push(path);
        }
    }
    objects.sort();
    Ok(objects)
}

fn discard_unpublished_generations(paths: &StorePaths) -> Result<(), MigrationError> {
    if !paths.dir.is_dir() {
        return Ok(());
    }
    let migration_prefix = format!(".{}.migration-", paths.base);
    let generation_prefix = format!("{}.v2-", paths.base);
    for entry in std::fs::read_dir(&paths.dir).map_err(|error| io_error(&paths.dir, &error))? {
        let entry = entry.map_err(|error| io_error(&paths.dir, &error))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if path.is_dir()
            && (name.starts_with(&migration_prefix) || name.starts_with(&generation_prefix))
        {
            std::fs::remove_dir_all(&path).map_err(|error| io_error(&path, &error))?;
        }
    }
    Ok(())
}

fn discard_rekey_staging(paths: &StorePaths) -> Result<(), MigrationError> {
    if !paths.dir.is_dir() {
        return Ok(());
    }
    let prefix = format!(".{}.rekey-", paths.base);
    for entry in std::fs::read_dir(&paths.dir).map_err(|error| io_error(&paths.dir, &error))? {
        let entry = entry.map_err(|error| io_error(&paths.dir, &error))?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if path.is_dir() && name.starts_with(&prefix) {
            std::fs::remove_dir_all(&path).map_err(|error| io_error(&path, &error))?;
        }
    }
    Ok(())
}

fn initialize_v2_root(root: &Path) -> Result<(), MigrationError> {
    std::fs::create_dir_all(root).map_err(|error| io_error(root, &error))?;
    for child in ["journals", "heads", "segments", "leases"] {
        let directory = root.join(child);
        std::fs::create_dir_all(&directory).map_err(|error| io_error(&directory, &error))?;
    }
    super::atomic_write(&root.join("VERSION"), b"ABI-WDBX 2\n")?;
    Ok(())
}

fn snapshot_digest(snapshot: &V2Snapshot) -> Result<String, MigrationError> {
    let bytes = serde_json::to_vec(snapshot)
        .map_err(|error| MigrationError::Verification(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn publish_activation(paths: &StorePaths, activation: &Activation) -> Result<(), MigrationError> {
    let mut bytes = ACTIVE_HEADER.as_bytes().to_vec();
    serde_json::to_writer(&mut bytes, activation)
        .map_err(|error| MigrationError::InvalidActivation(error.to_string()))?;
    bytes.push(b'\n');
    super::atomic_write(&paths_active(paths), &bytes)?;
    Ok(())
}

fn read_activation(paths: &StorePaths) -> Result<Activation, MigrationError> {
    let path = paths_active(paths);
    let content = std::fs::read_to_string(&path).map_err(|error| io_error(&path, &error))?;
    let body = content.strip_prefix(ACTIVE_HEADER).ok_or_else(|| {
        MigrationError::InvalidActivation(format!("{} has an invalid header", path.display()))
    })?;
    let activation: Activation = serde_json::from_str(body)
        .map_err(|error| MigrationError::InvalidActivation(error.to_string()))?;
    if activation.version != 2 {
        return Err(MigrationError::InvalidActivation(format!(
            "unsupported version {}",
            activation.version
        )));
    }
    Ok(activation)
}

fn resolve_child(paths: &StorePaths, name: &str) -> Result<PathBuf, MigrationError> {
    let path = Path::new(name);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(MigrationError::InvalidActivation(format!(
            "unsafe child path {name:?}"
        )));
    }
    Ok(paths.dir.join(path))
}

fn verify_v2_root(root: &Path) -> Result<(), MigrationError> {
    let version = root.join("VERSION");
    let content = std::fs::read(&version).map_err(|error| io_error(&version, &error))?;
    if content != b"ABI-WDBX 2\n" {
        return Err(MigrationError::InvalidActivation(format!(
            "{} is not a WDBX v2 generation",
            root.display()
        )));
    }
    Ok(())
}

fn paths_active(paths: &StorePaths) -> PathBuf {
    paths.dir.join(format!("{}.active", paths.base))
}

fn ensure_parent(paths: &StorePaths) -> Result<(), MigrationError> {
    std::fs::create_dir_all(&paths.dir).map_err(|error| io_error(&paths.dir, &error))
}

fn file_name_string(path: &Path) -> Result<String, MigrationError> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            MigrationError::InvalidActivation(format!("{} has no UTF-8 file name", path.display()))
        })
}

fn io_error(path: &Path, error: &std::io::Error) -> MigrationError {
    MigrationError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod tests;
