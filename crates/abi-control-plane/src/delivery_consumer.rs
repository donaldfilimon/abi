//! Verified ciphertext journal; rendering can be replayed after a client restart.
use crate::{
    DeliveryAck, DeliveryPage, Error, JobBinding, OutboxPacket, PacketKind, RelayKey, Result,
    TerminalManifest, delivery_outbox::PrivateDb,
};
use rusqlite::{OptionalExtension as _, params};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Journal {
    binding: JobBinding,
    node_key: [u8; 32],
    authority_key: [u8; 32],
    packets: Vec<OutboxPacket>,
    manifest: Option<TerminalManifest>,
    cursor: u32,
    terminal: Option<String>,
    cancelled: bool,
}
/// Owner-private encrypted packets and a transactionally committed verified cursor.
pub struct VerifiedDeliveryStore {
    store: PrivateDb,
    #[cfg(test)]
    import_checkpoint: Option<fn(&str)>,
}
impl VerifiedDeliveryStore {
    /// Exclusively create a private journal bound to one job, node key and pinned authority key.
    pub fn create(
        path: &Path,
        binding: &JobBinding,
        node_key: [u8; 32],
        authority_key: [u8; 32],
    ) -> Result<Self> {
        let store = PrivateDb::open(path, true, 2 * 1024 * 1024)?;
        store.db.execute_batch("CREATE TABLE consumer_version(version INTEGER NOT NULL);INSERT INTO consumer_version VALUES(2);CREATE TABLE journal(id INTEGER PRIMARY KEY CHECK(id=1),body BLOB NOT NULL);")?;
        let mut out = Self {
            store,
            #[cfg(test)]
            import_checkpoint: None,
        };
        out.save(&Journal {
            binding: binding.clone(),
            node_key,
            authority_key,
            packets: vec![],
            manifest: None,
            cursor: 0,
            terminal: None,
            cancelled: false,
        })?;
        Ok(out)
    }
    /// Validate and reopen saved packets without creating missing state.
    pub fn open(path: &Path) -> Result<Self> {
        let store = PrivateDb::open(path, false, 2 * 1024 * 1024)?;
        if store
            .db
            .query_row("SELECT version FROM consumer_version", [], |r| {
                r.get::<_, i64>(0)
            })?
            != 2
            || store
                .db
                .query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0))?
                != "ok"
        {
            return Err(Error::Denied);
        }
        let out = Self {
            store,
            #[cfg(test)]
            import_checkpoint: None,
        };
        let j = out.read()?;
        Self::check_saved(&j)?;
        Ok(out)
    }
    /// Verify signatures, scope, prefix and result before atomically committing packets and cursor.
    pub fn import(&mut self, page: &DeliveryPage, reply_key: &RelayKey) -> Result<DeliveryAck> {
        let mut j = self.read()?;
        Self::check_saved(&j)?;
        Self::validate_page_scope(&j, page, reply_key)?;
        let manifest = page.manifest.as_ref().ok_or(Error::Denied)?;
        manifest.verify(&j.node_key)?;
        j.cancelled |= page.cancelled || manifest.outcome == crate::RecordedOutcome::Cancelled;
        if j.cancelled
            && page
                .packets
                .iter()
                .any(|p| matches!(p, OutboxPacket::Terminal { .. }))
        {
            return Err(Error::Denied);
        }
        if manifest.binding != j.binding
            || manifest.submission_digest != page.receipt.submission_digest
        {
            return Err(Error::Denied);
        }
        if let Some(old) = &j.manifest
            && (manifest.frame_digests.len() < old.frame_digests.len()
                || !manifest.frame_digests.starts_with(&old.frame_digests)
                || (old.terminal_digest.is_some() && old.digest()? != manifest.digest()?))
        {
            return Err(Error::Denied);
        }
        let allowed = manifest.packets();
        for packet in &page.packets {
            let id = packet.id(&j.binding)?;
            if !allowed.contains(&id) {
                return Err(Error::Denied);
            }
            if let Some(old) = j.packets.iter().find(|p| {
                p.id(&j.binding)
                    .is_ok_and(|p| p.kind == id.kind && p.sequence == id.sequence)
            }) {
                if encode(old)? != encode(packet)? {
                    return Err(Error::Denied);
                }
                continue;
            }
            match packet {
                OutboxPacket::Delta { frame } => {
                    if frame.sequence != j.cursor || j.terminal.is_some() {
                        return Err(Error::Denied);
                    }
                    let bytes = frame.open(&j.binding, &j.node_key, reply_key)?;
                    if bytes.is_empty() {
                        return Err(Error::Denied);
                    }
                    j.cursor += 1;
                }
                OutboxPacket::Terminal { packet } => {
                    if j.cursor as usize != manifest.frame_digests.len() {
                        return Err(Error::Denied);
                    }
                    let bytes = packet.open(&j.binding, &j.node_key, reply_key)?;
                    let result: abi_worker::JobResult =
                        serde_json::from_slice(&bytes).map_err(|_| Error::Denied)?;
                    if !result.verify_digest() || result.job_id.to_string() != j.binding.job_id {
                        return Err(Error::Denied);
                    }
                    if manifest.completed_at != Some(result.completed_at_unix_ms / 1000)
                        || result
                            .chunks
                            .iter()
                            .enumerate()
                            .any(|(i, c)| usize::try_from(c.sequence) != Ok(i))
                    {
                        return Err(Error::Denied);
                    }
                    let prefix = Self::open_packets(&j, reply_key)?;
                    let full = result
                        .chunks
                        .iter()
                        .flat_map(|c| c.bytes.iter().copied())
                        .collect::<Vec<_>>();
                    if !full.starts_with(&prefix) || full.len() > j.binding.max_result_bytes {
                        return Err(Error::Denied);
                    }
                    j.terminal = Some(id.digest);
                }
            }
            j.packets.push(packet.clone());
        }
        let bytes = Self::open_packets(&j, reply_key)?;
        if bytes.len() > j.binding.max_result_bytes {
            return Err(Error::Full);
        }
        j.manifest = Some(manifest.clone());
        let ack = DeliveryAck {
            manifest_digest: manifest.digest()?,
            contiguous: j.cursor,
            terminal_digest: j.terminal.clone(),
        };
        self.save_import(&j)?;
        Ok(ack)
    }
    /// Replay the verified saved view; stdout effects are intentionally replayable.
    /// Later cancellation blocks new terminal imports but cannot revoke output already imported.
    pub fn output(&self, reply_key: &RelayKey) -> Result<Vec<u8>> {
        let j = self.read()?;
        Self::check_saved(&j)?;
        Self::open_packets(&j, reply_key)
    }
    /// Return the durable owning-client receipt metadata after import.
    pub fn ack(&self) -> Result<Option<DeliveryAck>> {
        let j = self.read()?;
        j.manifest
            .as_ref()
            .map(|m| {
                Ok(DeliveryAck {
                    manifest_digest: m.digest()?,
                    contiguous: j.cursor,
                    terminal_digest: j.terminal.clone(),
                })
            })
            .transpose()
    }
    fn validate_page_scope(j: &Journal, page: &DeliveryPage, reply_key: &RelayKey) -> Result<()> {
        page.policy.verify(
            &j.authority_key,
            &j.binding,
            &page.receipt.submission_digest,
        )?;
        if page.receipt.binding != j.binding
            || reply_key.public_key() != j.binding.reply_key
            || page.policy.policy.submission_digest != page.receipt.submission_digest
        {
            return Err(Error::Denied);
        }
        if page.policy.policy.binding_digest
            != crate::authority::hash(&crate::enrollment::json(&j.binding)?)
            || page.policy.policy.until_unix_s < j.binding.submitted_at + 60
            || page.policy.policy.until_unix_s > j.binding.submitted_at + 7 * 86400
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
    fn read(&self) -> Result<Journal> {
        self.store.check()?;
        let b: Option<Vec<u8>> = self
            .store
            .db
            .query_row("SELECT body FROM journal WHERE id=1", [], |r| r.get(0))
            .optional()?;
        serde_json::from_slice(&b.ok_or(Error::Denied)?).map_err(|_| Error::Denied)
    }
    fn save_import(&mut self, j: &Journal) -> Result<()> {
        self.save(j)?;
        #[cfg(test)]
        if let Some(checkpoint) = self.import_checkpoint {
            checkpoint("after_commit");
        }
        Ok(())
    }
    fn save(&mut self, j: &Journal) -> Result<()> {
        let body = encode(j)?;
        if body.len() > 1024 * 1024 {
            return Err(Error::Full);
        }
        #[cfg(test)]
        let checkpoint = self.import_checkpoint;
        self.store.write(|tx| {
            tx.execute("INSERT INTO journal(id,body) VALUES(1,?1) ON CONFLICT(id) DO UPDATE SET body=excluded.body", params![body])?;
            #[cfg(test)]
            if let Some(checkpoint) = checkpoint {
                checkpoint("before_commit");
            }
            Ok(())
        })
    }
    fn check_saved(j: &Journal) -> Result<()> {
        if j.packets.len() > 17 || j.cursor > 16 {
            return Err(Error::Denied);
        }
        let mut cursor = 0;
        let mut terminal = None;
        for p in &j.packets {
            p.verify(&j.binding, &j.node_key)?;
            let id = p.id(&j.binding)?;
            if id.kind == PacketKind::Delta {
                if id.sequence != cursor || terminal.is_some() {
                    return Err(Error::Denied);
                }
                cursor += 1;
            } else {
                if terminal.is_some() {
                    return Err(Error::Denied);
                }
                terminal = Some(id.digest);
            }
        }
        if cursor != j.cursor
            || terminal != j.terminal
            || (j.manifest.is_none() && !j.packets.is_empty())
        {
            return Err(Error::Denied);
        }
        if let Some(m) = &j.manifest {
            m.verify(&j.node_key)?;
            if m.binding != j.binding {
                return Err(Error::Denied);
            }
            let allowed = m.packets();
            if j.packets
                .iter()
                .any(|p| p.id(&j.binding).map_or(true, |id| !allowed.contains(&id)))
            {
                return Err(Error::Denied);
            }
        }
        Ok(())
    }
    fn open_packets(j: &Journal, key: &RelayKey) -> Result<Vec<u8>> {
        if key.public_key() != j.binding.reply_key {
            return Err(Error::Denied);
        }
        let mut bytes = vec![];
        for p in &j.packets {
            match p {
                OutboxPacket::Delta { frame } => {
                    bytes.extend(frame.open(&j.binding, &j.node_key, key)?);
                }
                OutboxPacket::Terminal { packet } => {
                    let plain = packet.open(&j.binding, &j.node_key, key)?;
                    let result: abi_worker::JobResult =
                        serde_json::from_slice(&plain).map_err(|_| Error::Denied)?;
                    if !result.verify_digest() {
                        return Err(Error::Denied);
                    }
                    return Ok(result.chunks.into_iter().flat_map(|c| c.bytes).collect());
                }
            }
        }
        Ok(bytes)
    }
}
fn encode<T: Serialize>(v: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(v).map_err(|_| Error::Invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DeliveryPolicy, Dispatch, JobReceipt, RecordedOutcome, SignedDeliveryPolicy, StreamFrame,
        Submission, TerminalPacket, seal,
    };
    use ed25519_dalek::SigningKey;
    pub(super) fn fixture() -> (tempfile::TempDir, DeliveryPage, RelayKey, SigningKey) {
        let dir = tempfile::tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let reply = RelayKey::from_seed(&[8; 32]).unwrap();
        let signing = SigningKey::from_bytes(&[9; 32]);
        let binding = JobBinding {
            protocol: "abi.relay-job.v1".into(),
            audience: "http://localhost".into(),
            job_id: uuid::Uuid::new_v4().to_string(),
            account_id: "account".into(),
            tenant_id: "tenant".into(),
            node_id: "node".into(),
            capability: "model.infer".into(),
            expires_at: 300,
            submitted_at: 1,
            max_result_bytes: 65536,
            reply_key: reply.public_key(),
        };
        let submission = Submission {
            payload: seal(
                &reply.public_key(),
                &binding.aad("request").unwrap(),
                b"prompt-canary",
            )
            .unwrap(),
            binding: binding.clone(),
        };
        let policy = SignedDeliveryPolicy::sign(
            DeliveryPolicy::for_submission(&submission, 86401).unwrap(),
            &signing,
        )
        .unwrap();
        let packet = OutboxPacket::Delta {
            frame: StreamFrame::seal(&binding, 0, b"private-verified-prefix", &signing).unwrap(),
        };
        let manifest = TerminalManifest {
            binding: binding.clone(),
            submission_digest: policy.policy.submission_digest.clone(),
            frame_digests: vec![packet.id(&binding).unwrap().digest],
            terminal_digest: None,
            outcome: RecordedOutcome::Unresolved,
            completed_at: None,
            signature: vec![],
        }
        .sign(&signing)
        .unwrap();
        let _dispatch = Dispatch {
            submission,
            signature: vec![],
        };
        let page = DeliveryPage {
            receipt: JobReceipt {
                binding,
                state: "unresolved".into(),
                submission_digest: policy.policy.submission_digest.clone(),
                result_available: true,
            },
            policy,
            manifest: Some(manifest),
            packets: vec![packet],
            next: 1,
            epoch: "epoch".into(),
            cancelled: false,
        };
        (dir, page, reply, signing)
    }
    #[test]
    fn duplicate_import_reopen_before_render_is_atomic_and_ciphertext_only() {
        let (dir, page, key, signing) = fixture();
        let path = dir.path().join("client.sqlite");
        let mut store = VerifiedDeliveryStore::create(
            &path,
            &page.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        let first = store.import(&page, &key).unwrap();
        let second = store.import(&page, &key).unwrap();
        assert_eq!(encode(&first).unwrap(), encode(&second).unwrap());
        drop(store);
        let store = VerifiedDeliveryStore::open(&path).unwrap();
        assert_eq!(store.ack().unwrap().unwrap().contiguous, 1);
        assert_eq!(store.output(&key).unwrap(), b"private-verified-prefix");
        assert!(
            !std::fs::read(&path)
                .unwrap()
                .windows(b"private-verified-prefix".len())
                .any(|w| w == b"private-verified-prefix")
        );
    }
    #[test]
    fn wrong_key_scope_conflict_and_missing_prefix_preserve_cursor() {
        let (dir, page, key, signing) = fixture();
        let path = dir.path().join("client.sqlite");
        let mut store = VerifiedDeliveryStore::create(
            &path,
            &page.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        assert!(store.import(&page, &RelayKey::generate().unwrap()).is_err());
        let mut wrong = page.clone();
        wrong.receipt.binding.tenant_id = "foreign".into();
        assert!(store.import(&wrong, &key).is_err());
        assert!(store.ack().unwrap().is_none());
        store.import(&page, &key).unwrap();
        let mut conflict = page.clone();
        conflict.packets[0] = OutboxPacket::Delta {
            frame: StreamFrame::seal(&page.receipt.binding, 0, b"changed", &signing).unwrap(),
        };
        assert!(store.import(&conflict, &key).is_err());
        let mut gap = page.clone();
        gap.packets = vec![OutboxPacket::Delta {
            frame: StreamFrame::seal(&page.receipt.binding, 2, b"gap", &signing).unwrap(),
        }];
        assert!(store.import(&gap, &key).is_err());
        assert_eq!(store.output(&key).unwrap(), b"private-verified-prefix");
    }
    #[test]
    fn terminal_with_inconsistent_result_digest_refuses_without_advancing() {
        let (dir, mut page, key, signing) = fixture();
        let path = dir.path().join("client.sqlite");
        let mut store = VerifiedDeliveryStore::create(
            &path,
            &page.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        store.import(&page, &key).unwrap();
        let packet = OutboxPacket::Terminal {
            packet: TerminalPacket::seal(&page.receipt.binding, b"{\"unverified\":true}", &signing)
                .unwrap(),
        };
        let mut manifest = page.manifest.take().unwrap();
        manifest.outcome = RecordedOutcome::Completed;
        manifest.completed_at = Some(100);
        manifest.terminal_digest = Some(packet.id(&page.receipt.binding).unwrap().digest);
        page.manifest = Some(manifest.sign(&signing).unwrap());
        page.packets = vec![packet];
        assert!(store.import(&page, &key).is_err());
        drop(store);
        let store = VerifiedDeliveryStore::open(&path).unwrap();
        assert!(store.ack().unwrap().unwrap().terminal_digest.is_none());
        assert_eq!(store.output(&key).unwrap(), b"private-verified-prefix");
    }

    #[test]
    fn signed_terminal_requires_exact_verified_prefix_and_reopens_complete_view() {
        #[derive(Serialize)]
        struct Canonical<'a> {
            job_id: uuid::Uuid,
            request_digest: &'a str,
            status: abi_worker::JobStatus,
            chunks: &'a [abi_worker::ResultChunk],
            completed_at_unix_ms: i64,
        }
        let (dir, mut page, key, signing) = fixture();
        let path = dir.path().join("client.sqlite");
        let mut store = VerifiedDeliveryStore::create(
            &path,
            &page.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        store.import(&page, &key).unwrap();
        let make = |text: &[u8]| {
            let job_id = uuid::Uuid::parse_str(&page.receipt.binding.job_id).unwrap();
            let request_digest = "a".repeat(64);
            let chunks = vec![abi_worker::ResultChunk {
                sequence: 0,
                bytes: text.to_vec(),
            }];
            let canonical = serde_json::to_string(&Canonical {
                job_id,
                request_digest: &request_digest,
                status: abi_worker::JobStatus::Succeeded,
                chunks: &chunks,
                completed_at_unix_ms: 100_000,
            })
            .unwrap();
            abi_worker::JobResult {
                job_id,
                request_digest: abi_worker::ResultDigest::parse(request_digest).unwrap(),
                status: abi_worker::JobStatus::Succeeded,
                chunks,
                result_digest: abi_worker::ResultDigest::parse({
                    use sha2::Digest as _;
                    use std::fmt::Write as _;
                    let mut hex = String::new();
                    for b in sha2::Sha256::digest(canonical.as_bytes()) {
                        write!(&mut hex, "{b:02x}").unwrap();
                    }
                    hex
                })
                .unwrap(),
                completed_at_unix_ms: 100_000,
            }
        };
        let bad = make(b"different-prefix");
        assert!(bad.verify_digest());
        let good = make(b"private-verified-prefix complete");
        assert!(good.verify_digest());
        let mut old = page.manifest.take().unwrap();
        for (result, accept) in [(bad, false), (good, true)] {
            let packet = OutboxPacket::Terminal {
                packet: TerminalPacket::seal(
                    &page.receipt.binding,
                    &serde_json::to_vec(&result).unwrap(),
                    &signing,
                )
                .unwrap(),
            };
            old.outcome = RecordedOutcome::Completed;
            old.completed_at = Some(100);
            old.terminal_digest = Some(packet.id(&page.receipt.binding).unwrap().digest);
            page.manifest = Some(old.clone().sign(&signing).unwrap());
            page.packets = vec![packet];
            assert_eq!(store.import(&page, &key).is_ok(), accept);
        }
        drop(store);
        let mut store = VerifiedDeliveryStore::open(&path).unwrap();
        assert_eq!(
            store.output(&key).unwrap(),
            b"private-verified-prefix complete"
        );
        assert!(store.ack().unwrap().unwrap().terminal_digest.is_some());
        page.cancelled = true;
        page.packets.clear();
        store.import(&page, &key).unwrap();
        assert_eq!(
            store.output(&key).unwrap(),
            b"private-verified-prefix complete"
        );
    }

    #[test]
    fn failed_cursor_sync_never_exposes_uncommitted_import() {
        let (dir, page, key, signing) = fixture();
        let path = dir.path().join("client.sqlite");
        let guard = crate::delivery_outbox::faults::Guard::new(&path);
        let mut store = VerifiedDeliveryStore::create(
            &path,
            &page.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        guard.arm(crate::delivery_outbox::faults::Fault::Sync);
        assert!(store.import(&page, &key).is_err());
        assert!(store.ack().is_err());
        assert!(store.output(&key).is_err());
        drop(store);
        if let Ok(store) = VerifiedDeliveryStore::open(&path) {
            match store.ack().unwrap() {
                Some(ack) => {
                    assert_eq!(ack.contiguous, 1);
                    assert_eq!(store.output(&key).unwrap(), b"private-verified-prefix");
                }
                None => assert_eq!(store.output(&key).unwrap(), Vec::<u8>::new()),
            }
        }
    }
    #[test]
    fn cancelled_delivery_imports_only_prefix_of_historical_completed_manifest() {
        let (dir, mut page, key, signing) = fixture();
        let terminal = OutboxPacket::Terminal {
            packet: TerminalPacket::seal(
                &page.receipt.binding,
                b"never imported terminal",
                &signing,
            )
            .unwrap(),
        };
        let mut manifest = page.manifest.take().unwrap();
        manifest.outcome = RecordedOutcome::Completed;
        manifest.completed_at = Some(100);
        manifest.terminal_digest = Some(terminal.id(&page.receipt.binding).unwrap().digest);
        page.manifest = Some(manifest.sign(&signing).unwrap());
        page.cancelled = true;
        let mut store = VerifiedDeliveryStore::create(
            &dir.path().join("cancel-prefix.sqlite"),
            &page.receipt.binding,
            signing.verifying_key().to_bytes(),
            signing.verifying_key().to_bytes(),
        )
        .unwrap();
        let ack = store.import(&page, &key).unwrap();
        assert_eq!(ack.contiguous, 1);
        assert!(ack.terminal_digest.is_none());
        page.packets = vec![terminal];
        assert!(store.import(&page, &key).is_err());
    }
    #[test]
    fn authority_policy_signature_is_pinned_and_checked_before_import_after_reopen() {
        let (dir, mut page, key, node) = fixture();
        let authority = SigningKey::from_bytes(&[43; 32]);
        page.policy = SignedDeliveryPolicy::sign(page.policy.policy.clone(), &authority).unwrap();
        let path = dir.path().join("pinned-authority.sqlite");
        let mut store = VerifiedDeliveryStore::create(
            &path,
            &page.receipt.binding,
            node.verifying_key().to_bytes(),
            authority.verifying_key().to_bytes(),
        )
        .unwrap();
        let mut changed = page.clone();
        changed.policy.signature.clear();
        assert!(store.import(&changed, &key).is_err());
        assert!(store.ack().unwrap().is_none());
        changed.policy = SignedDeliveryPolicy::sign(page.policy.policy.clone(), &node).unwrap();
        assert!(store.import(&changed, &key).is_err());
        assert!(store.ack().unwrap().is_none());
        drop(store);
        let mut store = VerifiedDeliveryStore::open(&path).unwrap();
        assert!(store.import(&changed, &key).is_err());
        let ack = store.import(&page, &key).unwrap();
        assert_eq!(ack.contiguous, 1);
        let mut changed = page;
        changed.policy.policy.until_unix_s += 1;
        assert!(store.import(&changed, &key).is_err());
        assert_eq!(store.ack().unwrap().unwrap().contiguous, 1);
    }
}

#[cfg(test)]
mod interruption_tests;
