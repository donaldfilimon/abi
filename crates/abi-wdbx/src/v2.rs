//! WDBX v2 causal, multi-writer journal primitives.
//!
//! V2 deliberately lives beside the legacy checkpoint/WAL reader. Each writer
//! owns one append-only journal and publishes only its own head file, so no
//! shared manifest or process-wide writer lock is required. A transaction is
//! visible only after a hash-covered commit frame has been durably appended.

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// Historical numeric identities and v2 UUID identities share one public type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RecordId {
    /// Identity read from a v1 store or accepted at a compatibility boundary.
    Legacy(u64),
    /// Stable v2 identity, serialized as a UUID string.
    V2(Uuid),
}

impl RecordId {
    /// Allocate a new v2 identity.
    #[must_use]
    pub fn new_v2() -> Self {
        Self::V2(Uuid::new_v4())
    }
}

impl std::fmt::Display for RecordId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Legacy(id) => write!(formatter, "{id}"),
            Self::V2(id) => write!(formatter, "{id}"),
        }
    }
}

/// Highest committed sequence observed for each writer.
pub type CausalHeads = BTreeMap<Uuid, u64>;

/// Maximum vector width accepted by v2 stores.
pub const MAX_V2_VECTOR_DIMENSIONS: usize = 4096;

/// A mutation carried by one causal transaction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum V2Mutation {
    /// Write one key/value version.
    PutKv {
        /// Logical key.
        key: String,
        /// Versioned value.
        value: String,
    },
    /// Write one vector under a stable public identity.
    PutVector {
        /// Stable public vector identity.
        id: RecordId,
        /// Finite vector components.
        values: Vec<f32>,
    },
}

/// One committed multi-version value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Version<T> {
    /// Stable version identity used by explicit conflict resolution.
    pub version_id: Uuid,
    /// Writer that created this value.
    pub writer_id: Uuid,
    /// Writer-local committed sequence.
    pub sequence: u64,
    /// Causal frontier observed before the transaction was written.
    pub observed_heads: CausalHeads,
    /// Stored value.
    pub value: T,
}

impl<T> Version<T> {
    fn dominates<U>(&self, other: &Version<U>) -> bool {
        self.writer_id == other.writer_id && self.sequence >= other.sequence
            || self
                .observed_heads
                .get(&other.writer_id)
                .is_some_and(|sequence| *sequence >= other.sequence)
    }
}

/// Preferred current version plus every unresolved concurrent version.
#[derive(Debug, Clone, PartialEq)]
pub struct ConflictSet<T> {
    /// Deterministic preferred version. Preference is presentation only; it
    /// does not resolve or discard the other versions.
    pub preferred: Version<T>,
    /// Concurrent current versions in deterministic identity order.
    pub conflicts: Vec<Version<T>>,
}

/// Immutable recovered v2 state. Views retain an `Arc` to this snapshot.
#[derive(Debug, Clone, Default)]
pub struct V2Snapshot {
    heads: CausalHeads,
    kv: BTreeMap<String, Vec<Version<String>>>,
    vectors: BTreeMap<RecordId, Vec<Version<Vec<f32>>>>,
    committed_transactions: usize,
}

impl V2Snapshot {
    /// Causal frontier represented by this snapshot.
    #[must_use]
    pub fn heads(&self) -> &CausalHeads {
        &self.heads
    }

    /// Number of verified committed transactions replayed.
    #[must_use]
    pub fn committed_transactions(&self) -> usize {
        self.committed_transactions
    }

    /// Return the maximal causal versions for a key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<ConflictSet<String>> {
        current_versions(self.kv.get(key)?)
    }

    /// Return the maximal causal versions for one vector identity.
    #[must_use]
    pub fn get_vector(&self, id: RecordId) -> Option<ConflictSet<Vec<f32>>> {
        current_versions(self.vectors.get(&id)?)
    }

    /// All stable vector identities in this immutable view.
    pub fn vector_ids(&self) -> impl Iterator<Item = RecordId> + '_ {
        self.vectors.keys().copied()
    }
}

fn current_versions<T: Clone>(versions: &[Version<T>]) -> Option<ConflictSet<T>> {
    let mut current: Vec<_> = versions
        .iter()
        .filter(|candidate| {
            !versions
                .iter()
                .any(|other| other.version_id != candidate.version_id && other.dominates(candidate))
        })
        .cloned()
        .collect();
    current.sort_by_key(|version| version.version_id);
    let preferred = current.pop()?;
    Some(ConflictSet {
        preferred,
        conflicts: current,
    })
}

/// V2 persistence or validation failure.
#[derive(Debug, thiserror::Error)]
pub enum V2Error {
    /// Filesystem operation failed.
    #[error("WDBX v2 I/O failed for {path}: {message}")]
    Io {
        /// Object being accessed.
        path: PathBuf,
        /// Underlying I/O detail.
        message: String,
    },
    /// A committed journal transaction was malformed or failed verification.
    #[error("WDBX v2 journal {path} is corrupt at line {line}: {reason}")]
    CorruptJournal {
        /// Journal being replayed.
        path: PathBuf,
        /// One-based frame line.
        line: usize,
        /// Verification failure.
        reason: String,
    },
    /// A mutation was rejected before any bytes were appended.
    #[error("invalid WDBX v2 mutation: {0}")]
    InvalidMutation(String),
    /// Explicit resolution did not name exactly the current conflicting set.
    #[error("conflict resolution set does not match the current versions")]
    StaleResolution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BeginFrame {
    writer_id: Uuid,
    transaction_id: Uuid,
    sequence: u64,
    observed_heads: CausalHeads,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "frame", rename_all = "snake_case")]
enum JournalFrame {
    Begin(BeginFrame),
    Mutation {
        transaction_id: Uuid,
        mutation: V2Mutation,
    },
    Commit {
        transaction_id: Uuid,
        transaction_hash: String,
    },
}

/// One process-local v2 writer. Different instances never share a writer ID.
pub struct V2Store {
    root: PathBuf,
    writer_id: Uuid,
    next_sequence: u64,
    snapshot: Arc<V2Snapshot>,
}

impl V2Store {
    /// Open or create a v2 directory and recover every writer journal.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, V2Error> {
        let root = root.into();
        ensure_dir(&root)?;
        ensure_dir(&root.join("journals"))?;
        ensure_dir(&root.join("heads"))?;
        let version_path = root.join("VERSION");
        if !version_path.exists() {
            match OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&version_path)
            {
                Ok(mut file) => {
                    file.write_all(b"ABI-WDBX 2\n")
                        .map_err(|error| io_error(&version_path, &error))?;
                    file.sync_all()
                        .map_err(|error| io_error(&version_path, &error))?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(io_error(&version_path, &error)),
            }
        }
        let writer_id = Uuid::new_v4();
        let snapshot = Arc::new(recover(&root)?);
        Ok(Self {
            root,
            writer_id,
            next_sequence: 1,
            snapshot,
        })
    }

    /// Unique identity for this writer session.
    #[must_use]
    pub fn writer_id(&self) -> Uuid {
        self.writer_id
    }

    /// Retain an immutable, mutation-safe view.
    #[must_use]
    pub fn snapshot(&self) -> Arc<V2Snapshot> {
        Arc::clone(&self.snapshot)
    }

    /// Rescan all journals and atomically replace the in-process view.
    pub fn refresh(&mut self) -> Result<(), V2Error> {
        self.snapshot = Arc::new(recover(&self.root)?);
        self.next_sequence = self
            .snapshot
            .heads
            .get(&self.writer_id)
            .copied()
            .unwrap_or(0)
            .saturating_add(1);
        Ok(())
    }

    /// Commit one or more mutations under the currently observed frontier.
    pub fn commit(&mut self, mutations: Vec<V2Mutation>) -> Result<Uuid, V2Error> {
        validate_mutations(&mutations)?;
        if mutations.is_empty() {
            return Err(V2Error::InvalidMutation(
                "transactions must contain at least one mutation".into(),
            ));
        }
        self.refresh()?;
        let transaction_id = Uuid::new_v4();
        let begin = BeginFrame {
            writer_id: self.writer_id,
            transaction_id,
            sequence: self.next_sequence,
            observed_heads: self.snapshot.heads.clone(),
        };
        let hash = transaction_hash(&begin, &mutations)?;
        let mut frames = Vec::with_capacity(mutations.len() + 2);
        frames.push(JournalFrame::Begin(begin));
        frames.extend(
            mutations
                .into_iter()
                .map(|mutation| JournalFrame::Mutation {
                    transaction_id,
                    mutation,
                }),
        );
        frames.push(JournalFrame::Commit {
            transaction_id,
            transaction_hash: hash,
        });
        append_frames(&self.journal_path(), &frames)?;
        atomic_write(&self.head_path(self.next_sequence), b"committed\n")?;
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.refresh()?;
        Ok(transaction_id)
    }

    /// Resolve the explicitly named current key versions with a dominating write.
    pub fn resolve(
        &mut self,
        key: &str,
        version_ids: &[Uuid],
        value: String,
    ) -> Result<Uuid, V2Error> {
        self.refresh()?;
        let Some(current) = self.snapshot.get(key) else {
            return Err(V2Error::StaleResolution);
        };
        let expected: BTreeSet<_> = std::iter::once(current.preferred.version_id)
            .chain(current.conflicts.iter().map(|version| version.version_id))
            .collect();
        let provided: BTreeSet<_> = version_ids.iter().copied().collect();
        if expected != provided || expected.len() < 2 {
            return Err(V2Error::StaleResolution);
        }
        self.commit(vec![V2Mutation::PutKv {
            key: key.to_string(),
            value,
        }])
    }

    fn journal_path(&self) -> PathBuf {
        self.root
            .join("journals")
            .join(format!("{}.jsonl", self.writer_id))
    }

    fn head_path(&self, sequence: u64) -> PathBuf {
        self.root
            .join("heads")
            .join(format!("{}-{sequence:020}.head", self.writer_id))
    }
}

fn validate_mutations(mutations: &[V2Mutation]) -> Result<(), V2Error> {
    for mutation in mutations {
        match mutation {
            V2Mutation::PutKv { key, .. } if key.is_empty() => {
                return Err(V2Error::InvalidMutation("key must not be empty".into()));
            }
            V2Mutation::PutVector { values, .. }
                if values.is_empty() || values.len() > MAX_V2_VECTOR_DIMENSIONS =>
            {
                return Err(V2Error::InvalidMutation(format!(
                    "vector dimensions must be 1..={MAX_V2_VECTOR_DIMENSIONS}"
                )));
            }
            V2Mutation::PutVector { values, .. }
                if values.iter().any(|value| !value.is_finite()) =>
            {
                return Err(V2Error::InvalidMutation(
                    "vectors must contain only finite values".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn transaction_hash(begin: &BeginFrame, mutations: &[V2Mutation]) -> Result<String, V2Error> {
    let bytes = serde_json::to_vec(&(begin, mutations))
        .map_err(|error| V2Error::InvalidMutation(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn append_frames(path: &Path, frames: &[JournalFrame]) -> Result<(), V2Error> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| io_error(path, &error))?;
    for frame in frames {
        serde_json::to_writer(&mut file, frame).map_err(|error| V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: 0,
            reason: error.to_string(),
        })?;
        file.write_all(b"\n")
            .map_err(|error| io_error(path, &error))?;
    }
    file.sync_all().map_err(|error| io_error(path, &error))
}

fn recover(root: &Path) -> Result<V2Snapshot, V2Error> {
    let mut snapshot = V2Snapshot::default();
    let journal_dir = root.join("journals");
    let mut journals: Vec<_> = std::fs::read_dir(&journal_dir)
        .map_err(|error| io_error(&journal_dir, &error))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .collect();
    journals.sort();
    for journal in journals {
        replay_journal(&journal, &mut snapshot)?;
    }
    Ok(snapshot)
}

fn replay_journal(path: &Path, snapshot: &mut V2Snapshot) -> Result<(), V2Error> {
    let content = std::fs::read_to_string(path).map_err(|error| io_error(path, &error))?;
    let mut pending: Option<(BeginFrame, Vec<V2Mutation>)> = None;
    // A concurrent writer or crashed process may leave bytes after the final
    // newline. Those bytes are not a frame yet and therefore cannot be
    // corruption or visible state. Every newline-terminated frame still fails
    // closed below if its JSON, ordering, sequence, or hash is invalid.
    let complete_len = content.rfind('\n').map_or(0, |index| index + 1);
    for (index, line) in content[..complete_len].lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() {
            continue;
        }
        let frame: JournalFrame =
            serde_json::from_str(line).map_err(|error| V2Error::CorruptJournal {
                path: path.to_path_buf(),
                line: line_number,
                reason: error.to_string(),
            })?;
        match frame {
            JournalFrame::Begin(begin) => {
                if pending.is_some() {
                    return corrupt(path, line_number, "nested transaction begin");
                }
                pending = Some((begin, Vec::new()));
            }
            JournalFrame::Mutation {
                transaction_id,
                mutation,
            } => {
                let Some((begin, mutations)) = pending.as_mut() else {
                    return corrupt(path, line_number, "mutation without begin");
                };
                if begin.transaction_id != transaction_id {
                    return corrupt(path, line_number, "transaction id changed mid-frame");
                }
                validate_mutations(std::slice::from_ref(&mutation))?;
                mutations.push(mutation);
            }
            JournalFrame::Commit {
                transaction_id,
                transaction_hash: found,
            } => {
                let Some((begin, mutations)) = pending.take() else {
                    return corrupt(path, line_number, "commit without begin");
                };
                if begin.transaction_id != transaction_id {
                    return corrupt(path, line_number, "commit transaction id mismatch");
                }
                if begin.sequence != snapshot.heads.get(&begin.writer_id).copied().unwrap_or(0) + 1
                {
                    return corrupt(path, line_number, "writer sequence is not contiguous");
                }
                let expected = transaction_hash(&begin, &mutations)?;
                if expected != found {
                    return corrupt(path, line_number, "transaction hash mismatch");
                }
                apply_transaction(snapshot, &begin, mutations);
            }
        }
    }
    // A crash may leave a begin/mutation suffix. It is intentionally invisible.
    Ok(())
}

fn apply_transaction(snapshot: &mut V2Snapshot, begin: &BeginFrame, mutations: Vec<V2Mutation>) {
    for (mutation_index, mutation) in mutations.into_iter().enumerate() {
        let version_id = Uuid::new_v5(&begin.transaction_id, mutation_index.to_string().as_bytes());
        match mutation {
            V2Mutation::PutKv { key, value } => snapshot.kv.entry(key).or_default().push(Version {
                version_id,
                writer_id: begin.writer_id,
                sequence: begin.sequence,
                observed_heads: begin.observed_heads.clone(),
                value,
            }),
            V2Mutation::PutVector { id, values } => {
                snapshot.vectors.entry(id).or_default().push(Version {
                    version_id,
                    writer_id: begin.writer_id,
                    sequence: begin.sequence,
                    observed_heads: begin.observed_heads.clone(),
                    value: values,
                });
            }
        }
    }
    snapshot.heads.insert(begin.writer_id, begin.sequence);
    snapshot.committed_transactions += 1;
}

fn corrupt<T>(path: &Path, line: usize, reason: &str) -> Result<T, V2Error> {
    Err(V2Error::CorruptJournal {
        path: path.to_path_buf(),
        line,
        reason: reason.to_string(),
    })
}

fn ensure_dir(path: &Path) -> Result<(), V2Error> {
    std::fs::create_dir_all(path).map_err(|error| io_error(path, &error))
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), V2Error> {
    let temp = path.with_extension(format!("tmp-{}", Uuid::new_v4()));
    {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(|error| io_error(&temp, &error))?;
        file.write_all(bytes)
            .map_err(|error| io_error(&temp, &error))?;
        file.sync_all().map_err(|error| io_error(&temp, &error))?;
    }
    std::fs::rename(&temp, path).map_err(|error| io_error(path, &error))
}

fn io_error(path: &Path, error: &std::io::Error) -> V2Error {
    V2Error::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use abi_foundation::temp_path::temp_file_path;
    use std::sync::{Arc, Barrier};

    fn scratch() -> PathBuf {
        temp_file_path("wdbx-v2", "store")
    }

    #[test]
    fn record_ids_keep_legacy_numbers_and_v2_strings() {
        assert_eq!(serde_json::to_string(&RecordId::Legacy(7)).unwrap(), "7");
        let id = RecordId::new_v2();
        let json = serde_json::to_string(&id).unwrap();
        assert!(json.starts_with('"'));
        assert_eq!(serde_json::from_str::<RecordId>(&json).unwrap(), id);
    }

    #[test]
    fn concurrent_writers_surface_conflicts_and_resolution_dominates_them() {
        let root = scratch();
        let mut left = V2Store::open(&root).unwrap();
        let mut right = V2Store::open(&root).unwrap();
        left.commit(vec![V2Mutation::PutKv {
            key: "k".into(),
            value: "left".into(),
        }])
        .unwrap();
        // Force right's observation back to its open-time empty frontier to model
        // a genuinely concurrent transaction written without refreshing peers.
        let right_begin = BeginFrame {
            writer_id: right.writer_id,
            transaction_id: Uuid::new_v4(),
            sequence: 1,
            observed_heads: CausalHeads::new(),
        };
        let mutations = vec![V2Mutation::PutKv {
            key: "k".into(),
            value: "right".into(),
        }];
        let hash = transaction_hash(&right_begin, &mutations).unwrap();
        append_frames(
            &right.journal_path(),
            &[
                JournalFrame::Begin(right_begin.clone()),
                JournalFrame::Mutation {
                    transaction_id: right_begin.transaction_id,
                    mutation: mutations[0].clone(),
                },
                JournalFrame::Commit {
                    transaction_id: right_begin.transaction_id,
                    transaction_hash: hash,
                },
            ],
        )
        .unwrap();
        right.refresh().unwrap();
        let set = right.snapshot().get("k").unwrap();
        assert_eq!(set.conflicts.len(), 1);
        let ids = [set.preferred.version_id, set.conflicts[0].version_id];
        right.resolve("k", &ids, "resolved".into()).unwrap();
        let resolved = right.snapshot().get("k").unwrap();
        assert_eq!(resolved.preferred.value, "resolved");
        assert_eq!(resolved.conflicts.len(), 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incomplete_transaction_is_ignored_and_old_views_survive_mutation() {
        let root = scratch();
        let mut store = V2Store::open(&root).unwrap();
        store
            .commit(vec![V2Mutation::PutKv {
                key: "k".into(),
                value: "old".into(),
            }])
            .unwrap();
        let old = store.snapshot();
        let begin = JournalFrame::Begin(BeginFrame {
            writer_id: store.writer_id,
            transaction_id: Uuid::new_v4(),
            sequence: 2,
            observed_heads: old.heads.clone(),
        });
        append_frames(&store.journal_path(), &[begin]).unwrap();
        store.refresh().unwrap();
        assert_eq!(store.snapshot().get("k").unwrap().preferred.value, "old");
        store
            .commit(vec![V2Mutation::PutKv {
                key: "k".into(),
                value: "new".into(),
            }])
            .unwrap_err();
        assert_eq!(old.get("k").unwrap().preferred.value, "old");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn v2_accepts_4096_dimensions_but_rejects_bad_vectors() {
        let root = scratch();
        let mut store = V2Store::open(&root).unwrap();
        let id = RecordId::new_v2();
        store
            .commit(vec![V2Mutation::PutVector {
                id,
                values: vec![0.25; 4096],
            }])
            .unwrap();
        assert_eq!(
            store
                .snapshot()
                .get_vector(id)
                .unwrap()
                .preferred
                .value
                .len(),
            4096
        );
        assert!(
            store
                .commit(vec![V2Mutation::PutVector {
                    id: RecordId::new_v2(),
                    values: vec![]
                }])
                .is_err()
        );
        assert!(
            store
                .commit(vec![V2Mutation::PutVector {
                    id: RecordId::new_v2(),
                    values: vec![f32::NAN]
                }])
                .is_err()
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_tampered_committed_transaction_fails_closed() {
        let root = scratch();
        let mut store = V2Store::open(&root).unwrap();
        store
            .commit(vec![V2Mutation::PutKv {
                key: "k".into(),
                value: "value".into(),
            }])
            .unwrap();
        let journal = store.journal_path();
        drop(store);
        let mut content = std::fs::read_to_string(&journal).unwrap();
        let marker = "\"transaction_hash\":\"";
        let hash_start = content.find(marker).unwrap() + marker.len();
        let replacement = if content.as_bytes()[hash_start] == b'0' {
            "1"
        } else {
            "0"
        };
        content.replace_range(hash_start..=hash_start, replacement);
        std::fs::write(&journal, content).unwrap();
        let Err(error) = V2Store::open(&root) else {
            panic!("tampered commit must not open");
        };
        assert!(error.to_string().contains("transaction hash mismatch"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fifty_independent_writers_recover_every_commit_without_a_global_lock() {
        let root = scratch();
        let barrier = Arc::new(Barrier::new(50));
        let mut workers = Vec::new();
        for index in 0_u16..50 {
            let root = root.clone();
            let barrier = Arc::clone(&barrier);
            workers.push(std::thread::spawn(move || {
                let mut store = V2Store::open(root).unwrap();
                barrier.wait();
                store
                    .commit(vec![
                        V2Mutation::PutKv {
                            key: format!("writer-{index}"),
                            value: index.to_string(),
                        },
                        V2Mutation::PutVector {
                            id: RecordId::new_v2(),
                            values: vec![f32::from(index), 1.0],
                        },
                    ])
                    .unwrap();
            }));
        }
        for worker in workers {
            worker.join().unwrap();
        }
        let recovered = V2Store::open(&root).unwrap().snapshot();
        assert_eq!(recovered.committed_transactions(), 50);
        assert_eq!(recovered.vector_ids().count(), 50);
        for index in 0_u16..50 {
            assert_eq!(
                recovered
                    .get(&format!("writer-{index}"))
                    .unwrap()
                    .preferred
                    .value,
                index.to_string()
            );
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
