//! WDBX v2 causal, multi-writer journal primitives.
//!
//! V2 deliberately lives beside the legacy checkpoint/WAL reader. Each writer
//! owns one append-only journal and publishes only its own head file, so no
//! shared manifest or process-wide writer lock is required. New journals store
//! every committed transaction in an independently authenticated object; the
//! legacy plaintext JSONL v2 journal remains readable. A transaction is visible
//! only after its hash-covered object is durable and its writer head is published.

mod index;
mod lease;
mod lifecycle;
mod security;
mod segment;
mod types;

pub use index::{V2SearchResult, V2VectorIndex};
pub use lifecycle::{
    GcReport, MigrationError, MigrationReport, MigrationStatus, RekeyReport, VerificationReport,
    VersionedSnapshot, gc_versioned, migration_status, open_versioned_read_only,
    open_versioned_writable, rekey_versioned, verify_versioned,
};
pub use security::{
    ABI_WDBX_ENCRYPTION_KEY_FILE, ABI_WDBX_SIGNING_KEY_FILE, ABI_WDBX_VERIFY_KEY_FILE, KeyMaterial,
    ObjectKind, ObjectSecurity, OpenedObject, SecurityError, generate_key_material, open_object,
    seal_object, write_key_material,
};
pub use segment::CompactionReport;
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

const MAX_JOURNAL_OBJECT_BYTES: usize = 128 * 1024 * 1024 + 32 * 1024;

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
    security: ObjectSecurity,
    _lease: lease::WriterLease,
}

impl V2Store {
    /// Open or create a v2 directory and recover every writer journal.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, V2Error> {
        let security = ObjectSecurity::from_env().map_err(|error| V2Error::Security {
            path: PathBuf::from("<configured-key-file>"),
            reason: error.to_string(),
        })?;
        Self::open_with_security(root, security)
    }

    /// Open with explicit object-security policy instead of process environment keys.
    pub fn open_with_security(
        root: impl Into<PathBuf>,
        security: ObjectSecurity,
    ) -> Result<Self, V2Error> {
        let root = root.into();
        ensure_dir(&root)?;
        ensure_dir(&root.join("journals"))?;
        ensure_dir(&root.join("heads"))?;
        ensure_dir(&root.join("segments"))?;
        ensure_dir(&root.join("leases"))?;
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
        let lease = lease::acquire_writer_lease(&root, writer_id)?;
        let snapshot = Arc::new(recover(&root, &security)?);
        Ok(Self {
            root,
            writer_id,
            next_sequence: 1,
            snapshot,
            security,
            _lease: lease,
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
        self.snapshot = Arc::new(recover(&self.root, &self.security)?);
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
        append_journal_object(
            &self.object_journal_path(),
            &begin_object_id(&frames)?,
            &frames,
            &self.security,
        )?;
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

    /// Publish an immutable snapshot covering the currently observed heads.
    ///
    /// Compaction never deletes journals or older segments. Recovery replays
    /// every transaction not covered by the selected causal frontier.
    pub fn compact(&mut self) -> Result<CompactionReport, V2Error> {
        self.refresh()?;
        segment::write_segment(&self.root, &self.snapshot, &self.security)
    }

    #[cfg(test)]
    fn journal_path(&self) -> PathBuf {
        self.root
            .join("journals")
            .join(format!("{}.jsonl", self.writer_id))
    }

    fn object_journal_path(&self) -> PathBuf {
        self.root
            .join("journals")
            .join(format!("{}.objects", self.writer_id))
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

#[cfg(test)]
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

fn begin_object_id(frames: &[JournalFrame]) -> Result<String, V2Error> {
    let Some(JournalFrame::Begin(begin)) = frames.first() else {
        return Err(V2Error::InvalidMutation(
            "journal object must begin with a begin frame".into(),
        ));
    };
    Ok(format!(
        "{}:{:020}:{}",
        begin.writer_id, begin.sequence, begin.transaction_id
    ))
}

fn append_journal_object(
    path: &Path,
    object_id: &str,
    frames: &[JournalFrame],
    security: &ObjectSecurity,
) -> Result<(), V2Error> {
    let plaintext = serde_json::to_vec(frames).map_err(|error| V2Error::CorruptJournal {
        path: path.to_path_buf(),
        line: 0,
        reason: error.to_string(),
    })?;
    let encoded =
        seal_object(ObjectKind::Journal, object_id, &plaintext, security).map_err(|error| {
            V2Error::Security {
                path: path.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
    let length = u64::try_from(encoded.len()).map_err(|_| V2Error::CorruptJournal {
        path: path.to_path_buf(),
        line: 0,
        reason: "journal object length overflow".into(),
    })?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| io_error(path, &error))?;
    file.write_all(&length.to_be_bytes())
        .and_then(|()| file.write_all(&encoded))
        .map_err(|error| io_error(path, &error))?;
    file.sync_all().map_err(|error| io_error(path, &error))
}

fn recover(root: &Path, security: &ObjectSecurity) -> Result<V2Snapshot, V2Error> {
    recover_with_policy(root, security, false)
}

fn recover_with_policy(
    root: &Path,
    security: &ObjectSecurity,
    require_signature: bool,
) -> Result<V2Snapshot, V2Error> {
    let mut snapshot =
        segment::load_segment(root, security, require_signature)?.unwrap_or_default();
    let published = published_heads(root)?;
    let journal_dir = root.join("journals");
    let mut journals: Vec<_> = std::fs::read_dir(&journal_dir)
        .map_err(|error| io_error(&journal_dir, &error))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl" || extension == "objects")
        })
        .collect();
    journals.sort();
    for journal in journals {
        if journal
            .extension()
            .is_some_and(|extension| extension == "objects")
        {
            replay_object_journal(
                &journal,
                &published,
                &mut snapshot,
                security,
                require_signature,
            )?;
        } else {
            if require_signature {
                return Err(V2Error::Security {
                    path: journal,
                    reason: SecurityError::SignatureRequired.to_string(),
                });
            }
            replay_journal(&journal, &published, &mut snapshot)?;
        }
    }
    for (writer, sequence) in &published {
        if snapshot.heads.get(writer).copied().unwrap_or(0) < *sequence {
            return Err(V2Error::CorruptJournal {
                path: root.join("heads"),
                line: 0,
                reason: format!(
                    "published head {writer}:{sequence} has no recoverable transaction"
                ),
            });
        }
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
    // A concurrent writer or crashed process may leave bytes after the final
    // newline. Those bytes are not a frame yet and therefore cannot be
    // corruption or visible state. Every newline-terminated frame still fails
    // closed below if its JSON, ordering, sequence, or hash is invalid.
    let complete_len = content.rfind('\n').map_or(0, |index| index + 1);
    let mut frames = Vec::new();
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
        frames.push((line_number, frame));
    }
    process_frames(path, published, snapshot, frames, true)
}

fn replay_object_journal(
    path: &Path,
    published: &CausalHeads,
    snapshot: &mut V2Snapshot,
    security: &ObjectSecurity,
    require_signature: bool,
) -> Result<(), V2Error> {
    let bytes = std::fs::read(path).map_err(|error| io_error(path, &error))?;
    let mut offset = 0_usize;
    let mut object_number = 0_usize;
    loop {
        let Some(prefix_end) = offset.checked_add(8) else {
            return corrupt(path, object_number + 1, "journal object offset overflow");
        };
        let Some(prefix) = bytes.get(offset..prefix_end) else {
            break;
        };
        let length = usize::try_from(u64::from_be_bytes(
            prefix
                .try_into()
                .expect("journal length prefix is exactly eight bytes"),
        ))
        .map_err(|_| V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: object_number + 1,
            reason: "journal object length overflow".into(),
        })?;
        if length == 0 || length > MAX_JOURNAL_OBJECT_BYTES {
            return corrupt(
                path,
                object_number + 1,
                "journal object exceeds the size limit",
            );
        }
        let Some(object_end) = prefix_end.checked_add(length) else {
            return corrupt(path, object_number + 1, "journal object offset overflow");
        };
        let Some(encoded) = bytes.get(prefix_end..object_end) else {
            // A crash between the length prefix and complete envelope leaves an
            // unpublished tail. The writer head is not created until after sync.
            break;
        };
        object_number += 1;
        let opened = open_object(encoded, security, require_signature).map_err(|error| {
            V2Error::Security {
                path: path.to_path_buf(),
                reason: error.to_string(),
            }
        })?;
        if opened.kind != ObjectKind::Journal {
            return corrupt(path, object_number, "object kind is not journal");
        }
        let frames: Vec<JournalFrame> =
            serde_json::from_slice(&opened.plaintext).map_err(|error| V2Error::CorruptJournal {
                path: path.to_path_buf(),
                line: object_number,
                reason: error.to_string(),
            })?;
        let begin_count = frames
            .iter()
            .filter(|frame| matches!(frame, JournalFrame::Begin(_)))
            .count();
        let commit_count = frames
            .iter()
            .filter(|frame| matches!(frame, JournalFrame::Commit { .. }))
            .count();
        if begin_count != 1
            || commit_count != 1
            || !matches!(frames.first(), Some(JournalFrame::Begin(_)))
            || !matches!(frames.last(), Some(JournalFrame::Commit { .. }))
        {
            return corrupt(
                path,
                object_number,
                "journal object must contain exactly one complete transaction",
            );
        }
        if begin_object_id(&frames)? != opened.object_id {
            return corrupt(path, object_number, "journal object identity mismatch");
        }
        let numbered = frames
            .into_iter()
            .enumerate()
            .map(|(index, frame)| (index + 1, frame))
            .collect();
        process_frames(path, published, snapshot, numbered, false)?;
        offset = object_end;
    }
    Ok(())
}

fn process_frames(
    path: &Path,
    published: &CausalHeads,
    snapshot: &mut V2Snapshot,
    frames: Vec<(usize, JournalFrame)>,
    allow_incomplete_tail: bool,
) -> Result<(), V2Error> {
    let mut pending: Option<(BeginFrame, Vec<V2Mutation>)> = None;
    for (line_number, frame) in frames {
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
                let expected = transaction_hash(&begin, &mutations)?;
                if expected != found {
                    return corrupt(path, line_number, "transaction hash mismatch");
                }
                let recovered_sequence = snapshot.heads.get(&begin.writer_id).copied().unwrap_or(0);
                if begin.sequence <= recovered_sequence {
                    // The selected immutable segment already authenticated and
                    // retained this transaction's state.
                    continue;
                }
                if begin.sequence != recovered_sequence + 1 {
                    return corrupt(path, line_number, "writer sequence is not contiguous");
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
    if pending.is_some() && !allow_incomplete_tail {
        return corrupt(path, 0, "journal object has no commit frame");
    }
    // A legacy JSONL crash may leave a begin/mutation suffix. It is invisible.
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
mod tests;
