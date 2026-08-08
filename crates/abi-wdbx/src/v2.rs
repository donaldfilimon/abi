//! WDBX v2 causal, multi-writer journal primitives.
//!
//! V2 deliberately lives beside the legacy checkpoint/WAL reader. Each writer
//! owns one append-only journal and publishes only its own head file, so no
//! shared manifest or process-wide writer lock is required. A transaction is
//! visible only after a hash-covered commit frame has been durably appended.

mod index;
mod lifecycle;
mod types;

pub use index::{V2SearchResult, V2VectorIndex};
pub use lifecycle::{
    MigrationError, MigrationReport, MigrationStatus, VersionedSnapshot, migration_status,
    open_versioned_read_only, open_versioned_writable,
};
pub use types::{
    CausalHeads, ConflictSet, MAX_V2_VECTOR_DIMENSIONS, RecordId, V2AuditBlock, V2Error,
    V2Mutation, V2Snapshot, V2SpatialRecord, V2TemporalKind, V2TemporalRecord, Version,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

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
        if !version_path.exists()
            && let Err(error) = atomic_write(&version_path, b"ABI-WDBX 2\n")
            && !version_path.exists()
        {
            return Err(error);
        }
        let found =
            std::fs::read(&version_path).map_err(|error| io_error(&version_path, &error))?;
        if found != b"ABI-WDBX 2\n" {
            return Err(V2Error::UnsupportedVersion {
                path: version_path,
                found: String::from_utf8_lossy(&found).into_owned(),
            });
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
            V2Mutation::PutSpatial { record }
                if [record.x, record.y, record.z]
                    .iter()
                    .any(|value| !value.is_finite()) =>
            {
                return Err(V2Error::InvalidMutation(
                    "spatial coordinates must contain only finite values".into(),
                ));
            }
            V2Mutation::PutTemporal { key, .. } if key.is_empty() => {
                return Err(V2Error::InvalidMutation(
                    "temporal keys must not be empty".into(),
                ));
            }
            V2Mutation::PutAudit { block }
                if !valid_hash(&block.hash)
                    || block.parents.iter().any(|parent| !valid_hash(parent)) =>
            {
                return Err(V2Error::InvalidMutation(
                    "audit hashes must be 64 lowercase hexadecimal characters".into(),
                ));
            }
            V2Mutation::PutAudit { block }
                if block.parents.iter().any(|parent| parent == &block.hash) =>
            {
                return Err(V2Error::InvalidMutation(
                    "an audit block cannot name itself as a parent".into(),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
    let published = published_heads(root)?;
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
        replay_journal(&journal, &published, &mut snapshot)?;
    }
    Ok(snapshot)
}

fn published_heads(root: &Path) -> Result<CausalHeads, V2Error> {
    let head_dir = root.join("heads");
    let mut heads = CausalHeads::new();
    for entry in std::fs::read_dir(&head_dir).map_err(|error| io_error(&head_dir, &error))? {
        let entry = entry.map_err(|error| io_error(&head_dir, &error))?;
        let path = entry.path();
        if path.extension().is_none_or(|extension| extension != "head") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let Some((writer, sequence)) = stem.rsplit_once('-') else {
            continue;
        };
        let (Ok(writer), Ok(sequence)) = (Uuid::parse_str(writer), sequence.parse::<u64>()) else {
            continue;
        };
        heads
            .entry(writer)
            .and_modify(|current| *current = (*current).max(sequence))
            .or_insert(sequence);
    }
    Ok(heads)
}

fn replay_journal(
    path: &Path,
    published: &CausalHeads,
    snapshot: &mut V2Snapshot,
) -> Result<(), V2Error> {
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
                if published
                    .get(&begin.writer_id)
                    .is_none_or(|sequence| begin.sequence > *sequence)
                {
                    // Durability alone is insufficient: until this writer's
                    // own atomic head object exists, the transaction is not
                    // published and therefore not visible.
                    continue;
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
            V2Mutation::PutSpatial { record } => {
                snapshot
                    .spatial
                    .entry(record.id)
                    .or_default()
                    .push(Version {
                        version_id,
                        writer_id: begin.writer_id,
                        sequence: begin.sequence,
                        observed_heads: begin.observed_heads.clone(),
                        value: record,
                    });
            }
            V2Mutation::PutTemporal { key, record } => {
                snapshot.temporal.entry(key).or_default().push(Version {
                    version_id,
                    writer_id: begin.writer_id,
                    sequence: begin.sequence,
                    observed_heads: begin.observed_heads.clone(),
                    value: record,
                });
            }
            V2Mutation::PutAudit { block } => {
                if !snapshot.audit.contains_key(&block.hash) {
                    snapshot.audit_order.push(block.hash.clone());
                }
                snapshot.audit.insert(block.hash.clone(), block);
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
        atomic_write(&right.head_path(1), b"committed\n").unwrap();
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
    fn a_synced_commit_is_invisible_until_its_writer_head_is_published() {
        let root = scratch();
        let mut store = V2Store::open(&root).unwrap();
        let begin = BeginFrame {
            writer_id: store.writer_id,
            transaction_id: Uuid::new_v4(),
            sequence: 1,
            observed_heads: CausalHeads::new(),
        };
        let mutations = vec![V2Mutation::PutKv {
            key: "unpublished".into(),
            value: "hidden".into(),
        }];
        let hash = transaction_hash(&begin, &mutations).unwrap();
        append_frames(
            &store.journal_path(),
            &[
                JournalFrame::Begin(begin.clone()),
                JournalFrame::Mutation {
                    transaction_id: begin.transaction_id,
                    mutation: mutations[0].clone(),
                },
                JournalFrame::Commit {
                    transaction_id: begin.transaction_id,
                    transaction_hash: hash,
                },
            ],
        )
        .unwrap();
        store.refresh().unwrap();
        assert!(store.snapshot().get("unpublished").is_none());
        atomic_write(&store.head_path(1), b"committed\n").unwrap();
        store.refresh().unwrap();
        assert_eq!(
            store.snapshot().get("unpublished").unwrap().preferred.value,
            "hidden"
        );
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
