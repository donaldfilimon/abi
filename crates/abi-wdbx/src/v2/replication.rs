//! Exact authenticated committed-transaction portability.

use super::{
    BeginFrame, JournalFrame, MAX_JOURNAL_OBJECT_BYTES, ObjectKind, ObjectSecurity, V2Error,
    V2Mutation, V2Store, apply_transaction, atomic_write, begin_object_id, corrupt, io_error,
    open_object, recover, transaction_hash, validate_mutations,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use uuid::Uuid;

/// One exact sealed v2 journal transaction suitable for authenticated transport.
///
/// Fields are intentionally opaque: callers can inspect metadata and serialize
/// the object, but only `V2Store` can create or import it. Import authenticates
/// the exact sealed envelope and cross-checks every metadata field.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommittedTransaction {
    writer_id: Uuid,
    transaction_id: Uuid,
    sequence: u64,
    object_id: String,
    sha256: String,
    encoded: Vec<u8>,
}

impl std::fmt::Debug for CommittedTransaction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CommittedTransaction")
            .field("writer_id", &self.writer_id)
            .field("transaction_id", &self.transaction_id)
            .field("sequence", &self.sequence)
            .field("object_id", &self.object_id)
            .field("sha256", &self.sha256)
            .field(
                "encoded",
                &format_args!("[REDACTED; {} bytes]", self.encoded.len()),
            )
            .finish()
    }
}

impl CommittedTransaction {
    /// Original writer identity.
    #[must_use]
    pub const fn writer_id(&self) -> Uuid {
        self.writer_id
    }

    /// Stable transaction identity used to derive version IDs.
    #[must_use]
    pub const fn transaction_id(&self) -> Uuid {
        self.transaction_id
    }

    /// Contiguous writer-local sequence.
    #[must_use]
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Authenticated object identity.
    #[must_use]
    pub fn object_id(&self) -> &str {
        &self.object_id
    }

    /// Lowercase SHA-256 of the exact encoded envelope.
    #[must_use]
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Exact sealed envelope bytes; imports append these bytes without resealing.
    #[must_use]
    pub fn encoded(&self) -> &[u8] {
        &self.encoded
    }
}

#[derive(Clone)]
struct VerifiedTransaction {
    begin: BeginFrame,
    mutations: Vec<V2Mutation>,
}

impl V2Store {
    /// Commit mutations and return the exact durable sealed transaction object.
    pub fn commit_export(
        &mut self,
        mutations: Vec<V2Mutation>,
    ) -> Result<CommittedTransaction, V2Error> {
        let transaction_id = self.commit(mutations)?;
        let sequence = self
            .snapshot
            .heads
            .get(&self.writer_id)
            .copied()
            .ok_or_else(|| V2Error::CorruptJournal {
                path: self.root.join("heads"),
                line: 0,
                reason: "committed writer head is missing after commit".into(),
            })?;
        let exported = self.export_transaction(self.writer_id, sequence)?;
        if exported.transaction_id != transaction_id {
            return corrupt(
                &self.object_journal_path(),
                0,
                "committed transaction identity changed during export",
            );
        }
        Ok(exported)
    }

    /// Export one published transaction as its exact authenticated envelope.
    pub fn export_transaction(
        &self,
        writer_id: Uuid,
        sequence: u64,
    ) -> Result<CommittedTransaction, V2Error> {
        let recovered = recover(&self.root, &self.security)?;
        if sequence == 0
            || recovered
                .heads
                .get(&writer_id)
                .is_none_or(|published| sequence > *published)
        {
            return corrupt(
                &journal_path(&self.root, writer_id),
                0,
                "transaction is not published in the recovered frontier",
            );
        }
        let path = journal_path(&self.root, writer_id);
        let stream = read_object_stream(&path)?;
        for encoded in stream.objects {
            let verified = verify_encoded(&path, &encoded, &self.security)?;
            if verified.begin.writer_id == writer_id && verified.begin.sequence == sequence {
                return Ok(committed_from_verified(&verified, encoded));
            }
        }
        corrupt(
            &path,
            0,
            "published transaction has no portable sealed journal object",
        )
    }

    /// Import an exact committed object without resealing it.
    ///
    /// Verification and sequence checks complete before any filesystem write.
    /// The envelope is then appended and synced before only that writer's head
    /// is atomically published.
    pub fn import_committed(
        &mut self,
        transaction: &CommittedTransaction,
    ) -> Result<Uuid, V2Error> {
        let path = journal_path(&self.root, transaction.writer_id);
        verify_outer(&path, transaction)?;
        let verified = verify_encoded(&path, &transaction.encoded, &self.security)?;
        cross_check_metadata(&path, transaction, &verified)?;

        // Recover into a local value. A rejected import never replaces the
        // caller-visible Arc snapshot.
        let mut recovered = recover(&self.root, &self.security)?;
        let current = recovered
            .heads
            .get(&transaction.writer_id)
            .copied()
            .unwrap_or(0);
        let stream = read_object_stream(&path)?;
        if stream.incomplete_tail {
            return corrupt(
                &path,
                stream.objects.len() + 1,
                "cannot import after an incomplete journal-object tail",
            );
        }
        let mut existing = None;
        for encoded in &stream.objects {
            let found = verify_encoded(&path, encoded, &self.security)?;
            if found.begin.writer_id != transaction.writer_id {
                return corrupt(&path, 0, "writer journal contains another writer identity");
            }
            if found.begin.sequence == transaction.sequence {
                if existing.is_some() {
                    return corrupt(&path, 0, "writer sequence appears more than once");
                }
                existing = Some(encoded.as_slice());
            }
        }

        if current >= transaction.sequence {
            return if existing == Some(transaction.encoded.as_slice()) {
                Ok(transaction.transaction_id)
            } else {
                corrupt(
                    &path,
                    0,
                    "conflicting repeat of an imported writer sequence",
                )
            };
        }
        if transaction.sequence != current.saturating_add(1) {
            return corrupt(&path, 0, "imported writer sequence is not contiguous");
        }
        if let Some(existing) = existing
            && existing != transaction.encoded
        {
            return corrupt(
                &path,
                0,
                "conflicting repeat of an imported writer sequence",
            );
        }

        let append_state = if existing.is_none() {
            Some(append_exact(&path, &transaction.encoded)?)
        } else {
            None
        };
        let head = head_path(&self.root, transaction.writer_id, transaction.sequence);
        if let Err(error) = atomic_write(&head, b"committed\n") {
            if let Some(append_state) = append_state {
                rollback_append(&path, append_state)?;
            }
            return Err(error);
        }

        apply_transaction(&mut recovered, &verified.begin, verified.mutations);
        self.snapshot = Arc::new(recovered);
        if transaction.writer_id == self.writer_id {
            self.next_sequence = transaction.sequence.saturating_add(1);
        }
        Ok(transaction.transaction_id)
    }
}

fn committed_from_verified(
    verified: &VerifiedTransaction,
    encoded: Vec<u8>,
) -> CommittedTransaction {
    CommittedTransaction {
        writer_id: verified.begin.writer_id,
        transaction_id: verified.begin.transaction_id,
        sequence: verified.begin.sequence,
        object_id: format!(
            "{}:{:020}:{}",
            verified.begin.writer_id, verified.begin.sequence, verified.begin.transaction_id
        ),
        sha256: encoded_sha256(&encoded),
        encoded,
    }
}

fn verify_outer(path: &Path, transaction: &CommittedTransaction) -> Result<(), V2Error> {
    if transaction.encoded.is_empty()
        || transaction.encoded.len() > MAX_JOURNAL_OBJECT_BYTES
        || transaction.sequence == 0
        || transaction.sha256.len() != 64
        || !transaction
            .sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || encoded_sha256(&transaction.encoded) != transaction.sha256
    {
        return corrupt(
            path,
            0,
            "committed transaction outer digest or bounds are invalid",
        );
    }
    Ok(())
}

fn verify_encoded(
    path: &Path,
    encoded: &[u8],
    security: &ObjectSecurity,
) -> Result<VerifiedTransaction, V2Error> {
    if encoded.is_empty() || encoded.len() > MAX_JOURNAL_OBJECT_BYTES {
        return corrupt(path, 0, "journal object exceeds the size limit");
    }
    let opened = open_object(encoded, security, false).map_err(|error| V2Error::Security {
        path: path.to_path_buf(),
        reason: error.to_string(),
    })?;
    if opened.kind != ObjectKind::Journal {
        return corrupt(path, 0, "object kind is not journal");
    }
    let frames: Vec<JournalFrame> =
        serde_json::from_slice(&opened.plaintext).map_err(|error| V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: 0,
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
        || frames.len() < 3
        || !matches!(frames.first(), Some(JournalFrame::Begin(_)))
        || !matches!(frames.last(), Some(JournalFrame::Commit { .. }))
    {
        return corrupt(
            path,
            0,
            "journal object must contain exactly one non-empty transaction",
        );
    }
    if begin_object_id(&frames)? != opened.object_id {
        return corrupt(path, 0, "journal object identity mismatch");
    }
    let JournalFrame::Begin(begin) = &frames[0] else {
        unreachable!("first frame checked above")
    };
    if begin.sequence == 0 {
        return corrupt(path, 0, "writer sequence starts at one");
    }
    let mut mutations = Vec::with_capacity(frames.len() - 2);
    for frame in &frames[1..frames.len() - 1] {
        let JournalFrame::Mutation {
            transaction_id,
            mutation,
        } = frame
        else {
            return corrupt(path, 0, "non-mutation frame inside transaction");
        };
        if *transaction_id != begin.transaction_id {
            return corrupt(path, 0, "transaction id changed mid-frame");
        }
        mutations.push(mutation.clone());
    }
    validate_mutations(&mutations)?;
    let JournalFrame::Commit {
        transaction_id,
        transaction_hash: found_hash,
    } = frames.last().expect("non-empty frames checked above")
    else {
        unreachable!("last frame checked above")
    };
    if *transaction_id != begin.transaction_id {
        return corrupt(path, 0, "commit transaction id mismatch");
    }
    if transaction_hash(begin, &mutations)? != *found_hash {
        return corrupt(path, 0, "transaction hash mismatch");
    }
    Ok(VerifiedTransaction {
        begin: begin.clone(),
        mutations,
    })
}

fn cross_check_metadata(
    path: &Path,
    transaction: &CommittedTransaction,
    verified: &VerifiedTransaction,
) -> Result<(), V2Error> {
    let expected_object_id = format!(
        "{}:{:020}:{}",
        verified.begin.writer_id, verified.begin.sequence, verified.begin.transaction_id
    );
    if transaction.writer_id != verified.begin.writer_id
        || transaction.transaction_id != verified.begin.transaction_id
        || transaction.sequence != verified.begin.sequence
        || transaction.object_id != expected_object_id
    {
        return corrupt(
            path,
            0,
            "committed transaction metadata does not match its envelope",
        );
    }
    Ok(())
}

struct ObjectStream {
    objects: Vec<Vec<u8>>,
    incomplete_tail: bool,
}

fn read_object_stream(path: &Path) -> Result<ObjectStream, V2Error> {
    if !path.exists() {
        return Ok(ObjectStream {
            objects: Vec::new(),
            incomplete_tail: false,
        });
    }
    let bytes = std::fs::read(path).map_err(|error| io_error(path, &error))?;
    let mut objects = Vec::new();
    let mut offset = 0_usize;
    while offset < bytes.len() {
        let Some(prefix_end) = offset.checked_add(8) else {
            return corrupt(path, objects.len() + 1, "journal object offset overflow");
        };
        let Some(prefix) = bytes.get(offset..prefix_end) else {
            return Ok(ObjectStream {
                objects,
                incomplete_tail: true,
            });
        };
        let length = usize::try_from(u64::from_be_bytes(
            prefix
                .try_into()
                .expect("journal length prefix is exactly eight bytes"),
        ))
        .map_err(|_| V2Error::CorruptJournal {
            path: path.to_path_buf(),
            line: objects.len() + 1,
            reason: "journal object length overflow".into(),
        })?;
        if length == 0 || length > MAX_JOURNAL_OBJECT_BYTES {
            return corrupt(
                path,
                objects.len() + 1,
                "journal object exceeds the size limit",
            );
        }
        let Some(end) = prefix_end.checked_add(length) else {
            return corrupt(path, objects.len() + 1, "journal object offset overflow");
        };
        let Some(encoded) = bytes.get(prefix_end..end) else {
            return Ok(ObjectStream {
                objects,
                incomplete_tail: true,
            });
        };
        objects.push(encoded.to_vec());
        offset = end;
    }
    Ok(ObjectStream {
        objects,
        incomplete_tail: false,
    })
}

#[derive(Clone, Copy)]
struct AppendState {
    old_length: u64,
    existed: bool,
}

fn append_exact(path: &Path, encoded: &[u8]) -> Result<AppendState, V2Error> {
    let length = u64::try_from(encoded.len()).map_err(|_| V2Error::CorruptJournal {
        path: path.to_path_buf(),
        line: 0,
        reason: "journal object length overflow".into(),
    })?;
    let state = match std::fs::metadata(path) {
        Ok(metadata) => AppendState {
            old_length: metadata.len(),
            existed: true,
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => AppendState {
            old_length: 0,
            existed: false,
        },
        Err(error) => return Err(io_error(path, &error)),
    };
    let mut frame = Vec::with_capacity(8 + encoded.len());
    frame.extend_from_slice(&length.to_be_bytes());
    frame.extend_from_slice(encoded);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| io_error(path, &error))?;
    if let Err(error) = file.write_all(&frame).and_then(|()| file.sync_all()) {
        drop(file);
        rollback_append(path, state)?;
        return Err(io_error(path, &error));
    }
    Ok(state)
}

fn rollback_append(path: &Path, state: AppendState) -> Result<(), V2Error> {
    if !state.existed {
        return std::fs::remove_file(path).map_err(|error| io_error(path, &error));
    }
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| io_error(path, &error))?;
    file.set_len(state.old_length)
        .map_err(|error| io_error(path, &error))?;
    file.sync_all().map_err(|error| io_error(path, &error))
}

fn journal_path(root: &Path, writer_id: Uuid) -> PathBuf {
    root.join("journals").join(format!("{writer_id}.objects"))
}

fn head_path(root: &Path, writer_id: Uuid, sequence: u64) -> PathBuf {
    root.join("heads")
        .join(format!("{writer_id}-{sequence:020}.head"))
}

fn encoded_sha256(encoded: &[u8]) -> String {
    format!("{:x}", Sha256::digest(encoded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{CausalHeads, KeyMaterial, RecordId, generate_key_material, seal_object};
    use abi_foundation::temp_path::temp_file_path;

    fn scratch(name: &str) -> PathBuf {
        temp_file_path(name, "store")
    }

    fn assert_empty_import_target(root: &Path, store: &V2Store) {
        assert_eq!(store.snapshot().committed_transactions(), 0);
        assert_eq!(store.snapshot().kv_count(), 0);
        assert_eq!(store.snapshot().vector_count(), 0);
        assert_eq!(std::fs::read_dir(root.join("journals")).unwrap().count(), 0);
        assert_eq!(std::fs::read_dir(root.join("heads")).unwrap().count(), 0);
    }

    fn records(key: &str, value: &str, vector_id: RecordId) -> Vec<V2Mutation> {
        vec![
            V2Mutation::PutKv {
                key: key.into(),
                value: value.into(),
            },
            V2Mutation::PutVector {
                id: vector_id,
                values: vec![0.25, -0.5, 1.0],
            },
        ]
    }

    fn conflicting_repeat(
        original: &CommittedTransaction,
        security: &ObjectSecurity,
    ) -> CommittedTransaction {
        let begin = BeginFrame {
            writer_id: original.writer_id,
            transaction_id: Uuid::new_v4(),
            sequence: original.sequence,
            observed_heads: CausalHeads::new(),
        };
        let mutations = vec![V2Mutation::PutKv {
            key: "portable".into(),
            value: "fork".into(),
        }];
        let frames = vec![
            JournalFrame::Begin(begin.clone()),
            JournalFrame::Mutation {
                transaction_id: begin.transaction_id,
                mutation: mutations[0].clone(),
            },
            JournalFrame::Commit {
                transaction_id: begin.transaction_id,
                transaction_hash: transaction_hash(&begin, &mutations).unwrap(),
            },
        ];
        let object_id = begin_object_id(&frames).unwrap();
        let plaintext = serde_json::to_vec(&frames).unwrap();
        let encoded = seal_object(ObjectKind::Journal, &object_id, &plaintext, security).unwrap();
        CommittedTransaction {
            writer_id: begin.writer_id,
            transaction_id: begin.transaction_id,
            sequence: begin.sequence,
            object_id,
            sha256: encoded_sha256(&encoded),
            encoded,
        }
    }

    #[test]
    fn exact_export_import_preserves_versions_values_and_bytes() {
        let source_root = scratch("wdbx-v2-portable-source");
        let target_root = scratch("wdbx-v2-portable-target");
        let mut source = V2Store::open(&source_root).unwrap();
        let vector_id = RecordId::new_v2();
        let portable = source
            .commit_export(records("portable", "source", vector_id))
            .unwrap();
        let wire = serde_json::to_vec(&portable).unwrap();
        let portable: CommittedTransaction = serde_json::from_slice(&wire).unwrap();
        assert!(!format!("{portable:?}").contains("source"));
        assert_eq!(
            source
                .export_transaction(portable.writer_id(), portable.sequence())
                .unwrap(),
            portable
        );
        let source_kv = source.snapshot().get("portable").unwrap().preferred;
        let source_vector = source.snapshot().get_vector(vector_id).unwrap().preferred;

        let mut target = V2Store::open(&target_root).unwrap();
        assert_eq!(
            target.import_committed(&portable).unwrap(),
            portable.transaction_id()
        );
        let target_kv = target.snapshot().get("portable").unwrap().preferred;
        let target_vector = target.snapshot().get_vector(vector_id).unwrap().preferred;
        assert_eq!(target_kv, source_kv);
        assert_eq!(target_vector, source_vector);
        assert_eq!(
            target
                .export_transaction(portable.writer_id(), portable.sequence())
                .unwrap(),
            portable
        );
        let journal = std::fs::read(journal_path(&target_root, portable.writer_id())).unwrap();
        assert_eq!(&journal[8..], portable.encoded());
        std::fs::remove_dir_all(source_root).unwrap();
        std::fs::remove_dir_all(target_root).unwrap();
    }

    #[test]
    fn identical_repeat_is_idempotent_but_conflicting_repeat_fails_closed() {
        let source_root = scratch("wdbx-v2-portable-repeat-source");
        let target_root = scratch("wdbx-v2-portable-repeat-target");
        let mut source = V2Store::open(&source_root).unwrap();
        let portable = source
            .commit_export(vec![V2Mutation::PutKv {
                key: "portable".into(),
                value: "first".into(),
            }])
            .unwrap();
        let mut target = V2Store::open(&target_root).unwrap();
        target.import_committed(&portable).unwrap();
        let before_snapshot = target.snapshot();
        let journal_path = journal_path(&target_root, portable.writer_id());
        let before_journal = std::fs::read(&journal_path).unwrap();
        target.import_committed(&portable).unwrap();
        assert_eq!(target.snapshot().as_ref(), before_snapshot.as_ref());
        assert_eq!(std::fs::read(&journal_path).unwrap(), before_journal);

        let fork = conflicting_repeat(&portable, &ObjectSecurity::plaintext());
        assert!(target.import_committed(&fork).is_err());
        assert_eq!(target.snapshot().as_ref(), before_snapshot.as_ref());
        assert_eq!(std::fs::read(&journal_path).unwrap(), before_journal);
        std::fs::remove_dir_all(source_root).unwrap();
        std::fs::remove_dir_all(target_root).unwrap();
    }

    #[test]
    fn tamper_invalid_mutation_and_gap_leave_target_unchanged() {
        let source_root = scratch("wdbx-v2-portable-reject-source");
        let target_root = scratch("wdbx-v2-portable-reject-target");
        let mut source = V2Store::open(&source_root).unwrap();
        let first = source
            .commit_export(vec![V2Mutation::PutKv {
                key: "first".into(),
                value: "one".into(),
            }])
            .unwrap();
        let second = source
            .commit_export(vec![V2Mutation::PutKv {
                key: "second".into(),
                value: "two".into(),
            }])
            .unwrap();
        let mut target = V2Store::open(&target_root).unwrap();
        assert!(target.import_committed(&second).is_err());
        assert_empty_import_target(&target_root, &target);

        let mut tampered = first.clone();
        let last = tampered.encoded.len() - 1;
        tampered.encoded[last] ^= 1;
        assert!(target.import_committed(&tampered).is_err());
        assert_empty_import_target(&target_root, &target);

        let begin = BeginFrame {
            writer_id: first.writer_id,
            transaction_id: Uuid::new_v4(),
            sequence: 1,
            observed_heads: CausalHeads::new(),
        };
        let mutations = vec![V2Mutation::PutKv {
            key: String::new(),
            value: "invalid".into(),
        }];
        let frames = vec![
            JournalFrame::Begin(begin.clone()),
            JournalFrame::Mutation {
                transaction_id: begin.transaction_id,
                mutation: mutations[0].clone(),
            },
            JournalFrame::Commit {
                transaction_id: begin.transaction_id,
                transaction_hash: transaction_hash(&begin, &mutations).unwrap(),
            },
        ];
        let object_id = begin_object_id(&frames).unwrap();
        let encoded = seal_object(
            ObjectKind::Journal,
            &object_id,
            &serde_json::to_vec(&frames).unwrap(),
            &ObjectSecurity::plaintext(),
        )
        .unwrap();
        let invalid = CommittedTransaction {
            writer_id: begin.writer_id,
            transaction_id: begin.transaction_id,
            sequence: begin.sequence,
            object_id,
            sha256: encoded_sha256(&encoded),
            encoded,
        };
        assert!(target.import_committed(&invalid).is_err());
        assert_empty_import_target(&target_root, &target);
        std::fs::remove_dir_all(source_root).unwrap();
        std::fs::remove_dir_all(target_root).unwrap();
    }

    #[test]
    fn encrypted_signed_import_preserves_envelope_and_wrong_key_fails_closed() {
        let source_root = scratch("wdbx-v2-portable-secure-source");
        let target_root = scratch("wdbx-v2-portable-secure-target");
        let wrong_root = scratch("wdbx-v2-portable-secure-wrong");
        let (encryption, signing, verifying) = generate_key_material();
        let secure =
            ObjectSecurity::from_material(Some(&encryption), Some(&signing), Some(&verifying))
                .unwrap();
        let mut source = V2Store::open_with_security(&source_root, secure.clone()).unwrap();
        let portable = source
            .commit_export(vec![V2Mutation::PutKv {
                key: "classified".into(),
                value: "preserved".into(),
            }])
            .unwrap();

        let (wrong_encryption, _wrong_signing, _wrong_verifying): (
            KeyMaterial,
            KeyMaterial,
            KeyMaterial,
        ) = generate_key_material();
        let wrong_security =
            ObjectSecurity::from_material(Some(&wrong_encryption), None, Some(&verifying)).unwrap();
        let mut wrong = V2Store::open_with_security(&wrong_root, wrong_security).unwrap();
        assert!(wrong.import_committed(&portable).is_err());
        assert_empty_import_target(&wrong_root, &wrong);

        let mut target = V2Store::open_with_security(&target_root, secure).unwrap();
        target.import_committed(&portable).unwrap();
        let journal = std::fs::read(journal_path(&target_root, portable.writer_id())).unwrap();
        assert_eq!(&journal[8..], portable.encoded());
        assert!(
            !journal
                .windows(b"preserved".len())
                .any(|window| window == b"preserved")
        );
        std::fs::remove_dir_all(source_root).unwrap();
        std::fs::remove_dir_all(target_root).unwrap();
        std::fs::remove_dir_all(wrong_root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn failed_head_publication_rolls_back_a_new_journal() {
        use std::os::unix::fs::PermissionsExt as _;

        let source_root = scratch("wdbx-v2-portable-head-failure-source");
        let target_root = scratch("wdbx-v2-portable-head-failure-target");
        let mut source = V2Store::open(&source_root).unwrap();
        let portable = source
            .commit_export(vec![V2Mutation::PutKv {
                key: "rollback".into(),
                value: "required".into(),
            }])
            .unwrap();
        let mut target = V2Store::open(&target_root).unwrap();
        let heads = target_root.join("heads");
        std::fs::set_permissions(&heads, std::fs::Permissions::from_mode(0o500)).unwrap();
        let result = target.import_committed(&portable);
        std::fs::set_permissions(&heads, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(result.is_err());
        assert_empty_import_target(&target_root, &target);
        std::fs::remove_dir_all(source_root).unwrap();
        std::fs::remove_dir_all(target_root).unwrap();
    }
}
