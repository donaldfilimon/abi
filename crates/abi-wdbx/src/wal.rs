//! CRC32-framed WDBX write-ahead log and epoch-gated recovery.
//!
//! A WAL contains one header followed by append-only frames:
//!
//! ```text
//! # ABI-WDBX-WAL v1 base_epoch=7
//! 89abcdef {"type":"kv","key":"k","value":"v"}
//! ```
//!
//! The checksum covers only the minified JSON bytes. A malformed incomplete
//! final frame is treated as a crash-torn tail; a bad CRC is always corruption,
//! including on the final frame.

use crate::format::{
    BlockRecord, FormatError, Hash, Record, SpatialRecord, StorePaths, TemporalKind, VectorRecord,
};
use crate::store::{CheckpointSource, Snapshot, load_checkpoint_with_source};
use serde::Serialize;
use sha2::{Digest as _, Sha256};
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Header prefix shared by legacy and epoch-tagged WALs.
pub const WAL_HEADER_PREFIX: &str = "# ABI-WDBX-WAL v1";

/// Maximum WAL size accepted by the in-memory verifier/replayer.
pub const MAX_WAL_BYTES: u64 = 256 * 1024 * 1024;

/// Maximum vector width accepted by the Zig WDBX store.
pub const MAX_VECTOR_DIMENSIONS: usize = 128;

/// A WAL or recovery failure.
#[derive(Debug)]
pub enum WalError {
    /// The first line was not the WDBX WAL header.
    InvalidHeader,
    /// A frame, epoch token, or mutation was corrupt.
    Corruption {
        /// One-based line number when known.
        line: Option<usize>,
        /// Why the data was rejected.
        reason: String,
    },
    /// A WAL exceeded [`MAX_WAL_BYTES`].
    TooLarge {
        /// Actual byte length.
        bytes: u64,
    },
    /// An absolute vector id did not continue the checkpoint sequence.
    CorruptVectorId {
        /// Next id required by the recovered checkpoint.
        expected: u64,
        /// Id carried by the WAL.
        found: u64,
    },
    /// Filesystem operation failure.
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// OS error text.
        message: String,
    },
    /// Checkpoint format failure.
    Format(FormatError),
}

impl std::fmt::Display for WalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeader => formatter.write_str("invalid WDBX WAL header"),
            Self::Corruption { line, reason } => match line {
                Some(line) => write!(formatter, "WAL corruption at line {line}: {reason}"),
                None => write!(formatter, "WAL corruption: {reason}"),
            },
            Self::TooLarge { bytes } => {
                write!(formatter, "WAL is {bytes} bytes; limit is {MAX_WAL_BYTES}")
            }
            Self::CorruptVectorId { expected, found } => {
                write!(
                    formatter,
                    "WAL vector id {found} does not continue at {expected}"
                )
            }
            Self::Io { path, message } => {
                write!(
                    formatter,
                    "WAL I/O failed for {}: {message}",
                    path.display()
                )
            }
            Self::Format(error) => write!(formatter, "checkpoint failure: {error}"),
        }
    }
}

impl std::error::Error for WalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Format(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FormatError> for WalError {
    fn from(error: FormatError) -> Self {
        Self::Format(error)
    }
}

/// WAL result alias.
pub type Result<T> = std::result::Result<T, WalError>;

/// A verified WAL held in memory for deterministic replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wal {
    /// Checkpoint epoch this log is a delta from.
    pub base_epoch: u64,
    /// Verified JSON frames in append order.
    pub frames: Vec<String>,
    /// Whether an incomplete final frame was ignored.
    pub torn_tail: bool,
}

impl Wal {
    /// Parse and verify a WAL body.
    pub fn parse(content: &str) -> Result<Self> {
        let raw_lines: Vec<&str> = content.split('\n').collect();
        let header = raw_lines.first().copied().ok_or(WalError::InvalidHeader)?;
        let header = header.trim_matches([' ', '\t', '\r']);
        if !header.starts_with(WAL_HEADER_PREFIX) {
            return Err(WalError::InvalidHeader);
        }

        let rest = header[WAL_HEADER_PREFIX.len()..].trim_matches([' ', '\t', '\r']);
        let base_epoch = if let Some(value) = rest.strip_prefix("base_epoch=") {
            value.parse::<u64>().map_err(|_| WalError::Corruption {
                line: Some(1),
                reason: format!("invalid base_epoch {value:?}"),
            })?
        } else {
            0
        };

        let mut frames = Vec::new();
        let mut torn_tail = false;
        for (index, raw) in raw_lines.iter().enumerate().skip(1) {
            let line_number = index + 1;
            let line = raw.trim_matches([' ', '\t', '\r']);
            if line.is_empty() {
                continue;
            }
            let later_non_empty = raw_lines[index + 1..]
                .iter()
                .any(|later| !later.trim_matches([' ', '\t', '\r']).is_empty());
            let Some((crc, json)) = line.split_once(' ') else {
                if later_non_empty {
                    return Err(WalError::Corruption {
                        line: Some(line_number),
                        reason: "frame is missing the CRC/JSON separator".to_string(),
                    });
                }
                torn_tail = true;
                break;
            };
            if json.is_empty() {
                if later_non_empty {
                    return Err(WalError::Corruption {
                        line: Some(line_number),
                        reason: "frame has no JSON payload".to_string(),
                    });
                }
                torn_tail = true;
                break;
            }
            let expected = crc32_hex(json.as_bytes());
            if crc != expected {
                return Err(WalError::Corruption {
                    line: Some(line_number),
                    reason: format!("CRC mismatch: expected {expected}, found {crc}"),
                });
            }
            frames.push(json.to_string());
        }

        Ok(Self {
            base_epoch,
            frames,
            torn_tail,
        })
    }

    /// Read and verify a WAL from disk.
    pub fn read(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let metadata = std::fs::metadata(path).map_err(|error| io_error(path, &error))?;
        if metadata.len() > MAX_WAL_BYTES {
            return Err(WalError::TooLarge {
                bytes: metadata.len(),
            });
        }
        let content = std::fs::read_to_string(path).map_err(|error| io_error(path, &error))?;
        Self::parse(&content)
    }

    /// Number of verified frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether no verified frames exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Replay every verified frame onto `snapshot`.
    pub fn replay_onto(&self, snapshot: &mut Snapshot) -> Result<usize> {
        for (index, json) in self.frames.iter().enumerate() {
            apply_frame(snapshot, json, index + 2)?;
        }
        snapshot.recount();
        Ok(self.frames.len())
    }
}

/// Create an epoch-tagged WAL if absent, without truncating an existing delta.
pub fn create_with_epoch(path: impl AsRef<Path>, base_epoch: u64) -> Result<()> {
    let path = path.as_ref();
    if path.exists() {
        return Ok(());
    }
    ensure_parent(path)?;
    let header = format!("{WAL_HEADER_PREFIX} base_epoch={base_epoch}\n");
    match std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
    {
        Ok(mut file) => {
            file.write_all(header.as_bytes())
                .map_err(|error| io_error(path, &error))?;
            file.sync_all().map_err(|error| io_error(path, &error))?;
            sync_parent_best_effort(path);
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(io_error(path, &error)),
    }
}

/// Append one minified JSON record, creating a legacy epoch-0 WAL when absent.
pub fn append_record(path: impl AsRef<Path>, json: &str) -> Result<()> {
    let path = path.as_ref();
    if json.contains(['\n', '\r']) {
        return Err(WalError::Corruption {
            line: None,
            reason: "a WAL JSON frame cannot contain a newline".to_string(),
        });
    }
    ensure_parent(path)?;

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .read(true)
        .open(path)
        .map_err(|error| io_error(path, &error))?;
    if file
        .metadata()
        .map_err(|error| io_error(path, &error))?
        .len()
        == 0
    {
        writeln!(file, "{WAL_HEADER_PREFIX}").map_err(|error| io_error(path, &error))?;
    }
    writeln!(file, "{} {json}", crc32_hex(json.as_bytes()))
        .map_err(|error| io_error(path, &error))?;
    file.sync_all().map_err(|error| io_error(path, &error))?;
    sync_parent_best_effort(path);
    Ok(())
}

/// Shared JSON shape for every appendable WAL mutation.
///
/// Serde's externally-tagged `type` field matches the Zig/Rust WAL frame payload
/// (`{"type":"kv",...}`, `{"type":"vector",...}`, …).
#[derive(Serialize)]
#[serde(tag = "type")]
enum Mutation<'a> {
    #[serde(rename = "kv")]
    Kv { key: &'a str, value: &'a str },
    #[serde(rename = "vector")]
    Vector { id: u64, values: &'a [f32] },
    #[serde(rename = "block")]
    Block {
        profile: &'a str,
        query_id: u64,
        response_id: u64,
        metadata: &'a str,
        timestamp_ms: i64,
    },
    #[serde(rename = "spatial")]
    Spatial {
        id: u64,
        x: f32,
        y: f32,
        z: f32,
        payload: &'a str,
    },
    #[serde(rename = "temporal_node")]
    TemporalNode { id: u64, timestamp_ms: i64 },
    #[serde(rename = "temporal_edge")]
    TemporalEdge { cause: u64, effect: u64 },
}

/// Append a key/value mutation.
pub fn append_kv(path: impl AsRef<Path>, key: &str, value: &str) -> Result<()> {
    append_json(path, &Mutation::Kv { key, value })
}

/// Append a vector mutation with its absolute id.
pub fn append_vector(path: impl AsRef<Path>, id: u64, values: &[f32]) -> Result<()> {
    if values.is_empty()
        || values.len() > MAX_VECTOR_DIMENSIONS
        || values.iter().any(|value| !value.is_finite())
    {
        return Err(WalError::Corruption {
            line: None,
            reason: "vector dimensions/components are invalid".to_string(),
        });
    }
    append_json(path, &Mutation::Vector { id, values })
}

/// Append an audit-block mutation with an explicit timestamp.
pub fn append_block(
    path: impl AsRef<Path>,
    profile: &str,
    query_id: u64,
    response_id: u64,
    metadata: &str,
    timestamp_ms: i64,
) -> Result<()> {
    if profile.is_empty() || query_id > u64::from(u32::MAX) || response_id > u64::from(u32::MAX) {
        return Err(WalError::Corruption {
            line: None,
            reason: "block profile or vector ids are invalid".to_string(),
        });
    }
    append_json(
        path,
        &Mutation::Block {
            profile,
            query_id,
            response_id,
            metadata,
            timestamp_ms,
        },
    )
}

/// Build the next deterministic audit block without mutating `snapshot`.
pub fn build_block(
    snapshot: &Snapshot,
    profile: &str,
    query_id: u64,
    response_id: u64,
    metadata: &str,
    timestamp_ms: i64,
) -> Result<BlockRecord> {
    if profile.is_empty() || query_id > u64::from(u32::MAX) || response_id > u64::from(u32::MAX) {
        return Err(WalError::Corruption {
            line: None,
            reason: "block profile or vector ids are invalid".to_string(),
        });
    }
    let sequence = u64::try_from(snapshot.blocks.len()).map_err(|_| WalError::Corruption {
        line: None,
        reason: "block sequence overflow".to_string(),
    })?;
    let prev_hash = snapshot
        .blocks
        .last()
        .map_or(Hash::GENESIS, |block| block.hash);
    let hash = compute_block_hash(prev_hash, timestamp_ms, sequence, profile, metadata);
    Ok(BlockRecord {
        hash,
        prev_hash,
        timestamp_ms,
        sequence,
        profile: profile.to_string(),
        query_id,
        response_id,
        metadata: metadata.to_string(),
    })
}

/// Append a spatial mutation.
pub fn append_spatial(path: impl AsRef<Path>, record: &SpatialRecord) -> Result<()> {
    if !record.x.is_finite() || !record.y.is_finite() || !record.z.is_finite() {
        return Err(WalError::Corruption {
            line: None,
            reason: "spatial coordinates must be finite".to_string(),
        });
    }
    append_json(
        path,
        &Mutation::Spatial {
            id: record.id,
            x: record.x,
            y: record.y,
            z: record.z,
            payload: &record.payload,
        },
    )
}

/// Append a temporal-node mutation.
pub fn append_temporal_node(path: impl AsRef<Path>, id: u64, timestamp_ms: i64) -> Result<()> {
    append_json(path, &Mutation::TemporalNode { id, timestamp_ms })
}

/// Append a temporal-edge mutation.
pub fn append_temporal_edge(path: impl AsRef<Path>, cause: u64, effect: u64) -> Result<()> {
    append_json(path, &Mutation::TemporalEdge { cause, effect })
}

/// Read only the WAL base epoch. Missing files are legacy epoch 0.
pub fn read_base_epoch(path: impl AsRef<Path>) -> Result<u64> {
    let path = path.as_ref();
    match Wal::read(path) {
        Ok(wal) => Ok(wal.base_epoch),
        Err(WalError::Io { .. }) if !path.exists() => Ok(0),
        Err(error) => Err(error),
    }
}

/// Verify every frame and return the frame count.
pub fn verify(path: impl AsRef<Path>) -> Result<usize> {
    Wal::read(path).map(|wal| wal.len())
}

/// Location of the sidecar WAL for a store.
#[must_use]
pub fn wal_path(paths: &StorePaths) -> PathBuf {
    paths.dir.join(format!("{}.wal", paths.base))
}

/// Source selected by durable recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoverySource {
    /// Neither checkpoint nor WAL held mutations.
    Empty,
    /// A pre-segment single-file snapshot checkpoint.
    Snapshot,
    /// A manifest-listed segment checkpoint.
    Segment,
    /// A checkpoint plus a matching WAL delta.
    Merged,
}

/// Result of opening durable WDBX state.
#[derive(Debug, Clone, PartialEq)]
pub struct Recovered {
    /// Reconstructed state.
    pub snapshot: Snapshot,
    /// Durable source selected.
    pub source: RecoverySource,
    /// Epoch of the loaded checkpoint, or 0 for an empty baseline.
    pub checkpoint_epoch: u64,
    /// Verified WAL frames applied.
    pub frames_applied: usize,
    /// Whether an incomplete WAL tail was ignored.
    pub torn_wal_tail: bool,
}

/// Recover the newest valid checkpoint and fold a matching WAL delta onto it.
///
/// A WAL from an older or otherwise different epoch is superseded by the
/// manifest-committed checkpoint and is removed without replay.
pub fn recover(paths: &StorePaths) -> Result<Recovered> {
    let (mut snapshot, checkpoint_source) = load_checkpoint_with_source(paths)?;
    let checkpoint_epoch = checkpoint_source.epoch();
    let mut source = match checkpoint_source {
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

    let wal = Wal::read(&path)?;
    if wal.base_epoch != checkpoint_epoch {
        std::fs::remove_file(&path).map_err(|error| io_error(&path, &error))?;
        return Ok(Recovered {
            snapshot,
            source,
            checkpoint_epoch,
            frames_applied: 0,
            torn_wal_tail: false,
        });
    }

    let frames_applied = wal.replay_onto(&mut snapshot)?;
    source = RecoverySource::Merged;
    Ok(Recovered {
        snapshot,
        source,
        checkpoint_epoch,
        frames_applied,
        torn_wal_tail: wal.torn_tail,
    })
}

/// Fold `snapshot` into a new checkpoint and reset the WAL to that epoch.
pub fn checkpoint(paths: &StorePaths, snapshot: &Snapshot) -> Result<u64> {
    let epoch = crate::persistence::flush(paths, snapshot)?;
    crate::persistence::write_snapshot(paths.index(), snapshot)?;
    let mirror_epoch = paths.mirror_epoch();
    let provenance = format!("{}\nepoch={epoch}\n", crate::format::MIRROR_EPOCH_HEADER);
    abi_foundation::io::write_file_atomic(&mirror_epoch, provenance)
        .map_err(|error| io_error(&mirror_epoch, &error))?;
    let path = wal_path(paths);
    match std::fs::remove_file(&path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error(&path, &error)),
    }
    create_with_epoch(path, epoch)?;
    Ok(epoch)
}

fn append_json<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<()> {
    let json = serde_json::to_string(value).map_err(|error| WalError::Corruption {
        line: None,
        reason: error.to_string(),
    })?;
    append_record(path, &json)
}

fn apply_frame(snapshot: &mut Snapshot, json: &str, line: usize) -> Result<()> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|error| WalError::Corruption {
            line: Some(line),
            reason: error.to_string(),
        })?;
    let object = value.as_object().ok_or_else(|| WalError::Corruption {
        line: Some(line),
        reason: "mutation is not a JSON object".to_string(),
    })?;
    let kind = object
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| WalError::Corruption {
            line: Some(line),
            reason: "mutation has no string type".to_string(),
        })?;

    match kind {
        "block" => apply_block(snapshot, object, line),
        "vector" => {
            let record = Record::parse(json).map_err(WalError::Format)?;
            let Record::Vector(vector) = record else {
                unreachable!("type dispatch already selected vector");
            };
            apply_vector(snapshot, vector)
        }
        "kv" | "spatial" | "temporal_node" | "temporal_edge" => {
            let record = Record::parse(json).map_err(WalError::Format)?;
            apply_regular_record(snapshot, record);
            Ok(())
        }
        other => Err(WalError::Corruption {
            line: Some(line),
            reason: format!("unknown mutation type {other:?}"),
        }),
    }
}

fn apply_vector(snapshot: &mut Snapshot, vector: VectorRecord) -> Result<()> {
    if vector.values.is_empty()
        || vector.values.len() > MAX_VECTOR_DIMENSIONS
        || vector.values.iter().any(|value| !value.is_finite())
    {
        return Err(WalError::Corruption {
            line: None,
            reason: "vector dimensions/components are invalid".to_string(),
        });
    }
    if let Some(dimensions) = snapshot.vector_dimensions()
        && dimensions != vector.values.len()
    {
        return Err(WalError::Corruption {
            line: None,
            reason: format!(
                "vector dimension mismatch: expected {dimensions}, found {}",
                vector.values.len()
            ),
        });
    }
    let expected = snapshot
        .max_vector_id()
        .map_or(1, |last| last.saturating_add(1));
    if vector.id != expected {
        return Err(WalError::CorruptVectorId {
            expected,
            found: vector.id,
        });
    }
    snapshot.vectors.insert(vector.id, vector.values);
    Ok(())
}

fn apply_block(
    snapshot: &mut Snapshot,
    object: &serde_json::Map<String, serde_json::Value>,
    line: usize,
) -> Result<()> {
    let profile = json_string(object, "profile", line)?;
    if profile.is_empty() {
        return Err(corruption(line, "block profile is empty"));
    }
    let query_id = json_u32(object, "query_id", line)?;
    let response_id = json_u32(object, "response_id", line)?;
    let metadata = json_string(object, "metadata", line)?;
    let timestamp_ms = object
        .get("timestamp_ms")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| corruption(line, "block timestamp_ms is not an i64"))?;
    let block = build_block(
        snapshot,
        profile,
        u64::from(query_id),
        u64::from(response_id),
        metadata,
        timestamp_ms,
    )?;
    snapshot.blocks.push(block);
    Ok(())
}

fn apply_regular_record(snapshot: &mut Snapshot, record: Record) {
    match record {
        Record::Temporal(temporal) if temporal.kind == TemporalKind::Node => {
            let id = temporal
                .fields
                .get("id")
                .and_then(serde_json::Value::as_u64);
            if let Some(id) = id
                && let Some(existing) = snapshot.temporal.iter_mut().find(|entry| {
                    entry.kind == TemporalKind::Node
                        && entry.fields.get("id").and_then(serde_json::Value::as_u64) == Some(id)
                })
            {
                *existing = temporal;
            } else {
                snapshot.temporal.push(temporal);
            }
        }
        other => snapshot.apply(other),
    }
}

pub(crate) fn compute_block_hash(
    prev_hash: Hash,
    timestamp_ms: i64,
    sequence: u64,
    profile: &str,
    metadata: &str,
) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(prev_hash.bytes());
    hasher.update(timestamp_ms.to_le_bytes());
    hasher.update(sequence.to_le_bytes());
    hasher.update(profile.as_bytes());
    hasher.update(metadata.as_bytes());
    Hash(hasher.finalize().into())
}

fn json_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
    line: usize,
) -> Result<&'a str> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| corruption(line, &format!("field {field:?} is not a string")))
}

fn json_u32(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    line: usize,
) -> Result<u32> {
    let value = object
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| corruption(line, &format!("field {field:?} is not an integer")))?;
    u32::try_from(value).map_err(|_| corruption(line, &format!("field {field:?} is out of range")))
}

fn corruption(line: usize, reason: &str) -> WalError {
    WalError::Corruption {
        line: Some(line),
        reason: reason.to_string(),
    }
}

fn crc32_hex(bytes: &[u8]) -> String {
    format!("{:08x}", crc32fast::hash(bytes))
}

fn ensure_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|error| io_error(parent, &error))?;
    }
    Ok(())
}

#[cfg(unix)]
fn sync_parent_best_effort(path: &Path) {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        && let Ok(directory) = std::fs::File::open(parent)
    {
        let _ = directory.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_parent_best_effort(_path: &Path) {}

fn io_error(path: &Path, error: &std::io::Error) -> WalError {
    WalError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    struct Fixture {
        dir: PathBuf,
    }

    impl Fixture {
        fn new(name: &str) -> Self {
            let dir = abi_foundation::temp_path::temp_file_path(name, "wal");
            std::fs::create_dir_all(&dir).expect("create fixture directory");
            Self { dir }
        }

        fn paths(&self) -> StorePaths {
            StorePaths::new(&self.dir)
        }

        fn wal(&self) -> PathBuf {
            wal_path(&self.paths())
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }

    #[test]
    fn append_verify_and_replay_all_mutation_families() {
        let fixture = Fixture::new("abi_wal_replay");
        create_with_epoch(fixture.wal(), 0).expect("create");
        append_kv(fixture.wal(), "agent:abbey", "trained").expect("kv");
        append_vector(fixture.wal(), 1, &[1.0, 0.0, 0.0, 0.0]).expect("vector");
        append_block(fixture.wal(), "abbey", 1, 2, "turn", 1000).expect("block");
        append_spatial(
            fixture.wal(),
            &SpatialRecord {
                id: 7,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                payload: "point".to_string(),
            },
        )
        .expect("spatial");
        append_temporal_node(fixture.wal(), 1, 1000).expect("node");
        append_temporal_edge(fixture.wal(), 1, 2).expect("edge");

        let wal = Wal::read(fixture.wal()).expect("read");
        assert_eq!(wal.base_epoch, 0);
        assert_eq!(wal.len(), 6);
        let mut snapshot = Snapshot::new();
        assert_eq!(wal.replay_onto(&mut snapshot).expect("replay"), 6);
        assert_eq!(snapshot.kv["agent:abbey"], "trained");
        assert_eq!(snapshot.vectors.len(), 1);
        assert_eq!(snapshot.blocks.len(), 1);
        assert_eq!(snapshot.spatial.len(), 1);
        assert_eq!(snapshot.stats.temporal_nodes, 1);
        assert_eq!(snapshot.stats.temporal_edges, 1);
        snapshot.verify_chain().expect("linked chain");
    }

    #[test]
    fn crc_matches_the_frozen_zig_algorithm() {
        assert_eq!(
            crc32_hex(br#"{"type":"kv","key":"k","value":"v"}"#),
            "be3abb8d"
        );
    }

    #[test]
    fn flipped_payload_byte_is_corruption_even_on_the_last_frame() {
        let fixture = Fixture::new("abi_wal_crc");
        append_kv(fixture.wal(), "k1", "v1").expect("append");
        let mut content = std::fs::read_to_string(fixture.wal()).expect("read");
        content = content.replacen("k1", "X1", 1);
        std::fs::write(fixture.wal(), content).expect("tamper");
        assert!(matches!(
            Wal::read(fixture.wal()),
            Err(WalError::Corruption { .. })
        ));
    }

    #[test]
    fn incomplete_last_frame_is_a_soft_torn_tail() {
        let fixture = Fixture::new("abi_wal_torn");
        append_kv(fixture.wal(), "keep", "yes").expect("append");
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(fixture.wal())
            .expect("open");
        write!(file, "deadbeef-incomplete").expect("tear");

        let wal = Wal::read(fixture.wal()).expect("verified prefix");
        assert_eq!(wal.len(), 1);
        assert!(wal.torn_tail);
    }

    #[test]
    fn incomplete_mid_log_frame_is_corruption() {
        let content = format!(
            "{WAL_HEADER_PREFIX}\nno-separator\n{} {}\n",
            crc32_hex(br#"{"type":"kv","key":"k","value":"v"}"#),
            r#"{"type":"kv","key":"k","value":"v"}"#
        );
        assert!(matches!(
            Wal::parse(&content),
            Err(WalError::Corruption { line: Some(2), .. })
        ));
    }

    #[test]
    fn create_with_epoch_never_truncates_an_existing_delta() {
        let fixture = Fixture::new("abi_wal_epoch");
        create_with_epoch(fixture.wal(), 42).expect("create");
        append_kv(fixture.wal(), "k", "v").expect("append");
        create_with_epoch(fixture.wal(), 99).expect("must preserve");
        let wal = Wal::read(fixture.wal()).expect("read");
        assert_eq!(wal.base_epoch, 42);
        assert_eq!(wal.len(), 1);
    }

    #[test]
    fn vector_ids_must_continue_the_checkpoint() {
        let fixture = Fixture::new("abi_wal_vector_id");
        create_with_epoch(fixture.wal(), 0).expect("create");
        append_vector(fixture.wal(), 3, &[0.0, 0.0]).expect("append");
        let wal = Wal::read(fixture.wal()).expect("read");
        let mut snapshot = Snapshot {
            vectors: BTreeMap::from([(1, vec![1.0, 0.0])]),
            ..Snapshot::new()
        };
        assert!(matches!(
            wal.replay_onto(&mut snapshot),
            Err(WalError::CorruptVectorId {
                expected: 2,
                found: 3
            })
        ));
    }

    #[test]
    fn legacy_snapshot_without_wal_reports_snapshot_source() {
        let fixture = Fixture::new("abi_wal_legacy_snapshot_source");
        let mut legacy = Snapshot::new();
        legacy
            .kv
            .insert("legacy".to_string(), "readable".to_string());
        crate::persistence::write_snapshot(fixture.paths().index(), &legacy)
            .expect("legacy checkpoint");

        let recovered = recover(&fixture.paths()).expect("recover legacy checkpoint");
        assert_eq!(recovered.source, RecoverySource::Snapshot);
        assert_eq!(recovered.checkpoint_epoch, 0);
        assert_eq!(recovered.frames_applied, 0);
        assert_eq!(
            recovered.snapshot.kv.get("legacy").map(String::as_str),
            Some("readable")
        );
    }

    #[test]
    fn legacy_checkpoint_is_recovered_then_refreshed_on_segment_cutover() {
        let fixture = Fixture::new("abi_wal_legacy_checkpoint");
        let mut legacy = Snapshot::new();
        legacy.kv.insert(
            "stale:multiway:experiment".to_string(),
            "old-export".to_string(),
        );
        crate::persistence::write_snapshot(fixture.paths().index(), &legacy)
            .expect("legacy checkpoint");
        create_with_epoch(fixture.wal(), 0).expect("legacy wal");

        let recovered = recover(&fixture.paths()).expect("recover legacy checkpoint");
        assert_eq!(recovered.source, RecoverySource::Merged);
        assert_eq!(recovered.checkpoint_epoch, 0);
        assert_eq!(recovered.frames_applied, 0);
        assert_eq!(
            recovered
                .snapshot
                .kv
                .get("stale:multiway:experiment")
                .map(String::as_str),
            Some("old-export")
        );

        let mut current = recovered.snapshot;
        current.kv.remove("stale:multiway:experiment");
        current.kv.insert(
            "multiway:experiment:latest".to_string(),
            "current-export".to_string(),
        );
        checkpoint(&fixture.paths(), &current).expect("cut over to segments");
        assert!(
            fixture.paths().index().exists(),
            "the compatibility mirror must be refreshed after publication"
        );
        std::fs::remove_file(fixture.paths().manifest()).expect("simulate manifest loss");
        let after_manifest_loss = recover(&fixture.paths()).expect("recover without manifest");
        assert_eq!(after_manifest_loss.source, RecoverySource::Merged);
        assert_eq!(after_manifest_loss.frames_applied, 0);
        assert_eq!(
            after_manifest_loss
                .snapshot
                .kv
                .get("multiway:experiment:latest")
                .map(String::as_str),
            Some("current-export"),
            "manifest loss must preserve the newly published checkpoint"
        );
        assert!(
            !after_manifest_loss
                .snapshot
                .kv
                .contains_key("stale:multiway:experiment"),
            "manifest loss must not resurrect stale mirror data"
        );
    }

    #[test]
    fn mirror_epoch_preserves_a_newer_wal_delta_without_a_manifest() {
        let fixture = Fixture::new("abi_wal_mirror_epoch");
        let mut first = Snapshot::new();
        first.kv.insert("base".to_string(), "epoch-0".to_string());
        assert_eq!(
            checkpoint(&fixture.paths(), &first).expect("epoch 0 checkpoint"),
            0
        );

        let mut second = Snapshot::new();
        second.kv.insert("base".to_string(), "epoch-1".to_string());
        assert_eq!(
            checkpoint(&fixture.paths(), &second).expect("epoch 1 checkpoint"),
            1
        );
        append_kv(fixture.wal(), "delta", "after-epoch-1").expect("WAL delta");
        std::fs::remove_file(fixture.paths().manifest()).expect("simulate manifest loss");

        let recovered = recover(&fixture.paths()).expect("recover mirror plus matching WAL");
        assert_eq!(recovered.source, RecoverySource::Merged);
        assert_eq!(recovered.checkpoint_epoch, 1);
        assert_eq!(recovered.frames_applied, 1);
        assert_eq!(recovered.snapshot.kv["base"], "epoch-1");
        assert_eq!(recovered.snapshot.kv["delta"], "after-epoch-1");
    }

    #[test]
    fn matching_epoch_wal_merges_onto_checkpoint() {
        let fixture = Fixture::new("abi_wal_recover");
        let mut checkpoint_snapshot = Snapshot::new();
        checkpoint_snapshot
            .kv
            .insert("checkpoint".to_string(), "yes".to_string());
        crate::persistence::flush(&fixture.paths(), &checkpoint_snapshot).expect("checkpoint");
        create_with_epoch(fixture.wal(), 0).expect("wal");
        append_kv(fixture.wal(), "delta", "yes").expect("delta");

        let recovered = recover(&fixture.paths()).expect("recover");
        assert_eq!(recovered.source, RecoverySource::Merged);
        assert_eq!(recovered.checkpoint_epoch, 0);
        assert_eq!(recovered.frames_applied, 1);
        assert_eq!(recovered.snapshot.kv["checkpoint"], "yes");
        assert_eq!(recovered.snapshot.kv["delta"], "yes");
    }

    #[test]
    fn stale_wal_is_removed_without_double_applying() {
        let fixture = Fixture::new("abi_wal_stale");
        let mut first = Snapshot::new();
        first.kv.insert("old".to_string(), "1".to_string());
        crate::persistence::flush(&fixture.paths(), &first).expect("epoch 0");
        create_with_epoch(fixture.wal(), 0).expect("old wal");
        append_kv(fixture.wal(), "must-not-return", "stale").expect("old delta");

        let mut second = Snapshot::new();
        second.kv.insert("new".to_string(), "2".to_string());
        crate::persistence::flush(&fixture.paths(), &second).expect("epoch 1");

        let recovered = recover(&fixture.paths()).expect("recover");
        assert_eq!(recovered.source, RecoverySource::Segment);
        assert_eq!(recovered.checkpoint_epoch, 1);
        assert_eq!(recovered.snapshot.kv, second.kv);
        assert!(!fixture.wal().exists());
    }

    #[test]
    fn checkpoint_resets_wal_to_the_new_epoch() {
        let fixture = Fixture::new("abi_wal_checkpoint");
        create_with_epoch(fixture.wal(), 0).expect("wal");
        append_kv(fixture.wal(), "delta", "value").expect("append");
        let mut snapshot = Snapshot::new();
        snapshot.kv.insert("folded".to_string(), "yes".to_string());

        let epoch = checkpoint(&fixture.paths(), &snapshot).expect("checkpoint");
        assert_eq!(epoch, 0);
        let wal = Wal::read(fixture.wal()).expect("fresh wal");
        assert_eq!(wal.base_epoch, 0);
        assert_eq!(wal.len(), 0);
    }
}
