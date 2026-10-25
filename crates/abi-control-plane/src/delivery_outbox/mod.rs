//! Private ciphertext-only, write-before-send relay journal.
#[cfg(test)]
pub(crate) mod faults;
mod store;
#[cfg(test)]
mod tests;
use crate::{
    Dispatch, Error, OutboxPacket, PacketId, RecoveryReply, Result, SignedDeliveryPolicy,
    TerminalManifest,
};
use rusqlite::{OptionalExtension as _, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
pub(crate) use store::PrivateDb;

/// Fixed reservation and physical storage ceilings.
#[derive(Clone, Copy, Debug)]
pub struct OutboxLimits {
    /// Maximum allocated SQLite database bytes.
    pub database_bytes: usize,
    /// Encoded artifact reservation charged before admission.
    pub job_bytes: usize,
    /// Maximum retained identity tombstones and active jobs.
    pub retained_jobs: usize,
}
impl Default for OutboxLimits {
    fn default() -> Self {
        Self {
            database_bytes: 64 * 1024 * 1024,
            job_bytes: 1024 * 1024,
            retained_jobs: 256,
        }
    }
}
/// Exact private dispatch history and encrypted artifacts.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutboxEntry {
    /// Original authority-signed encrypted request.
    pub dispatch: Dispatch,
    /// Immutable authority-signed delivery retention.
    pub policy: SignedDeliveryPolicy,
    /// Exact independently encrypted signed transport artifacts.
    pub packets: Vec<OutboxPacket>,
    /// Sticky observation of authority cancellation.
    pub cancelled: bool,
    /// Local recovery found no resumable execution.
    pub abandoned: bool,
    /// Node-signed closed outcome and ordered commitments.
    pub manifest: Option<TerminalManifest>,
    /// Artifacts were deleted at their immutable deadline.
    pub expired: bool,
    /// Private local delta hashes for recorded-result prefix reconciliation.
    pub plaintext_digests: Vec<String>,
}
/// Exclusive, synced user-node outbox. A failed commit poisons this handle.
pub struct FileRelayOutbox {
    store: PrivateDb,
    limits: OutboxLimits,
}
impl FileRelayOutbox {
    /// Exclusively create a fresh private journal; existing paths refuse.
    pub fn create(path: &Path, limits: OutboxLimits) -> Result<Self> {
        Self::load(path, limits, true)
    }
    /// Reopen retained history without creating missing state.
    pub fn open(path: &Path, limits: OutboxLimits) -> Result<Self> {
        Self::load(path, limits, false)
    }
    fn load(path: &Path, limits: OutboxLimits, create: bool) -> Result<Self> {
        if limits.database_bytes > 64 * 1024 * 1024
            || limits.job_bytes > 1024 * 1024
            || limits.job_bytes == 0
            || limits.retained_jobs > 256
            || limits.retained_jobs == 0
        {
            return Err(Error::Invalid);
        }
        let store = PrivateDb::open(path, create, limits.database_bytes)?;
        if create {
            store.db.execute_batch("CREATE TABLE outbox_version(version INTEGER NOT NULL);INSERT INTO outbox_version VALUES(1);CREATE TABLE entries(job TEXT PRIMARY KEY, until_s INTEGER NOT NULL, body BLOB NOT NULL, checksum TEXT NOT NULL, reserved INTEGER NOT NULL, ack BLOB,cancel_seen INTEGER NOT NULL DEFAULT 0);")?;
        }
        if store
            .db
            .query_row("SELECT version FROM outbox_version", [], |r| {
                r.get::<_, i64>(0)
            })?
            != 1
        {
            return Err(Error::Denied);
        }
        if store
            .db
            .query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0))?
            != "ok"
        {
            return Err(Error::Denied);
        }
        let out = Self { store, limits };
        for entry in out.all()? {
            out.validate(&entry)?;
        }
        Ok(out)
    }
    /// Read exact retained history, including expired identity tombstones.
    pub fn entry(&self, id: &str) -> Result<Option<OutboxEntry>> {
        self.store.check()?;
        let row: Option<(Vec<u8>, bool, String)> = self
            .store
            .db
            .query_row(
                "SELECT body,cancel_seen,checksum FROM entries WHERE job=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        row.map(|(bytes, cancelled, checksum)| {
            if body_digest(&bytes)? != checksum {
                return Err(Error::Denied);
            }
            let mut entry: OutboxEntry =
                serde_json::from_slice(&bytes).map_err(|_| Error::Denied)?;
            entry.cancelled |= cancelled;
            Ok(entry)
        })
        .transpose()
    }
    fn all(&self) -> Result<Vec<OutboxEntry>> {
        self.store.check()?;
        let mut q = self
            .store
            .db
            .prepare("SELECT job FROM entries ORDER BY job LIMIT 257")?;
        let ids = q
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if ids.len() > self.limits.retained_jobs {
            return Err(Error::Full);
        }
        ids.into_iter()
            .map(|id| self.entry(&id)?.ok_or(Error::Denied))
            .collect()
    }
    /// Obtain an independent monotonic cancellation writer while execution owns the journal.
    pub(crate) fn cancellation_recorder(&self, id: &str) -> Result<CancelRecorder> {
        if self.entry(id)?.is_none() {
            return Err(Error::Denied);
        }
        Ok(CancelRecorder {
            store: self.store.sibling()?,
            job: id.into(),
        })
    }
    /// List unexpired retained encrypted jobs before fresh polling.
    pub fn entries(&self, now: i64) -> Result<Vec<OutboxEntry>> {
        Ok(self
            .all()?
            .into_iter()
            .filter(|e| !e.expired && e.policy.policy.until_unix_s > now)
            .collect())
    }
    /// Sync immutable dispatch and reserve the entire job budget before admission.
    pub fn prepare(&mut self, dispatch: &Dispatch, policy: &SignedDeliveryPolicy) -> Result<()> {
        policy.policy.validate(&dispatch.submission)?;
        let id = &dispatch.submission.binding.job_id;
        if let Some(old) = self.entry(id)? {
            if encode(&old.dispatch)? == encode(dispatch)?
                && encode(&old.policy)? == encode(policy)?
            {
                return Ok(());
            }
            return Err(Error::Denied);
        }
        let entry = OutboxEntry {
            dispatch: dispatch.clone(),
            policy: policy.clone(),
            packets: vec![],
            cancelled: false,
            abandoned: false,
            manifest: None,
            expired: false,
            plaintext_digests: vec![],
        };
        self.validate(&entry)?;
        let body = encode(&entry)?;
        let reservation = i64::try_from(self.limits.job_bytes).map_err(|_| Error::Invalid)?;
        let max = i64::try_from(self.limits.database_bytes.saturating_sub(65536))
            .map_err(|_| Error::Invalid)?;
        let jobs = i64::try_from(self.limits.retained_jobs).map_err(|_| Error::Invalid)?;
        self.store.write(|tx| {
            let (count, reserved): (i64, i64) = tx.query_row(
                "SELECT count(*),coalesce(sum(reserved),0) FROM entries",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?;
            if count >= jobs || reserved.saturating_add(reservation) > max {
                return Err(Error::Full);
            }
            tx.execute(
                "INSERT INTO entries(job,until_s,body,reserved,checksum) VALUES(?1,?2,?3,?4,?5)",
                params![
                    id,
                    policy.policy.until_unix_s,
                    body,
                    reservation,
                    body_digest(&body)?
                ],
            )?;
            Ok(())
        })
    }
    /// Sync one exact packet before transport; conflicting duplicates refuse.
    pub fn append(&mut self, id: &str, packet: &OutboxPacket) -> Result<()> {
        let mut entry = self.entry(id)?.ok_or(Error::Denied)?;
        if entry.expired {
            return Err(Error::Denied);
        }
        let packet_id = packet.id(&entry.dispatch.submission.binding)?;
        for old in &entry.packets {
            let old_id = old.id(&entry.dispatch.submission.binding)?;
            if old_id.kind == packet_id.kind && old_id.sequence == packet_id.sequence {
                return if encode(old)? == encode(packet)? {
                    Ok(())
                } else {
                    Err(Error::Denied)
                };
            }
        }
        if entry.manifest.is_some()
            || entry
                .packets
                .iter()
                .any(|p| matches!(p, OutboxPacket::Terminal { .. }))
        {
            return Err(Error::Denied);
        }
        if let OutboxPacket::Delta { frame } = packet
            && (frame.sequence as usize != entry.packets.len() || frame.sequence >= 16)
        {
            return Err(Error::Denied);
        }
        entry.packets.push(packet.clone());
        self.save(id, &entry)
    }
    /// Atomically retain a sealed delta and its private plaintext commitment.
    pub fn append_frame(
        &mut self,
        id: &str,
        frame: &crate::StreamFrame,
        plaintext_digest: &str,
    ) -> Result<()> {
        if plaintext_digest.len() != 64 || !plaintext_digest.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(Error::Invalid);
        }
        let mut e = self.entry(id)?.ok_or(Error::Denied)?;
        let sequence = frame.sequence as usize;
        if sequence < e.packets.len() {
            return if encode(&e.packets[sequence])?
                == encode(&OutboxPacket::Delta {
                    frame: frame.clone(),
                })?
                && e.plaintext_digests.get(sequence).map(String::as_str) == Some(plaintext_digest)
            {
                Ok(())
            } else {
                Err(Error::Denied)
            };
        }
        if e.expired
            || e.manifest.is_some()
            || sequence != e.packets.len()
            || sequence >= 16
            || e.packets
                .iter()
                .any(|p| matches!(p, OutboxPacket::Terminal { .. }))
        {
            return Err(Error::Denied);
        }
        e.packets.push(OutboxPacket::Delta {
            frame: frame.clone(),
        });
        e.plaintext_digests.push(plaintext_digest.into());
        self.save(id, &e)
    }
    /// Whether uncertain persistence prevents further use.
    pub fn poisoned(&self) -> bool {
        self.store.poisoned
    }
    /// Read the exact saved ciphertext for a committed packet identity.
    #[allow(clippy::needless_pass_by_value)] // Public packet identity is intentionally consumed.
    pub fn packet(&self, id: &str, packet: PacketId) -> Result<Option<OutboxPacket>> {
        let Some(entry) = self.entry(id)? else {
            return Ok(None);
        };
        for p in entry.packets {
            if p.id(&entry.dispatch.submission.binding)? == packet {
                return Ok(Some(p));
            }
        }
        Ok(None)
    }
    /// List signed manifests retained for recovery inspection.
    pub fn pending(&self, now: i64) -> Result<Vec<TerminalManifest>> {
        Ok(self
            .entries(now)?
            .into_iter()
            .filter_map(|e| e.manifest)
            .collect())
    }
    /// Close the saved prefix with immutable signed outcome evidence.
    pub fn set_manifest(&mut self, id: &str, manifest: &TerminalManifest) -> Result<()> {
        let mut e = self.entry(id)?.ok_or(Error::Denied)?;
        if let Some(old) = &e.manifest {
            if encode(old)? == encode(manifest)? {
                return Ok(());
            }
            return Err(Error::Denied);
        }
        if manifest.binding != e.dispatch.submission.binding
            || manifest.submission_digest != e.policy.policy.submission_digest
            || manifest.packets()
                != e.packets
                    .iter()
                    .map(|p| p.id(&manifest.binding))
                    .collect::<Result<Vec<_>>>()?
        {
            return Err(Error::Denied);
        }
        e.manifest = Some(manifest.clone());
        self.save(id, &e)
    }
    /// Record that recovery must never dispatch this prepared identity.
    pub fn mark_unresolved(&mut self, id: &str, _now: i64) -> Result<()> {
        let mut e = self.entry(id)?.ok_or(Error::Denied)?;
        e.abandoned = true;
        self.save(id, &e)
    }
    /// Persist a sticky locally observed cancellation.
    pub fn observe_cancel(&mut self, id: &str) -> Result<()> {
        let mut e = self.entry(id)?.ok_or(Error::Denied)?;
        e.cancelled = true;
        self.save(id, &e)
    }
    /// Retain upload metadata without deleting the durable ciphertext.
    pub fn record_ack(&mut self, ack: &RecoveryReply) -> Result<()> {
        let bytes = encode(ack)?;
        if bytes.len() > 384 * 1024 {
            return Err(Error::Full);
        }
        self.store.write(|tx| {
            for p in &ack.committed {
                tx.execute(
                    "UPDATE entries SET ack=?1 WHERE job=?2",
                    params![bytes, p.job_id],
                )?;
            }
            Ok(())
        })
    }
    /// Purge expired transport artifacts while retaining identity tombstones.
    pub fn expire(&mut self, now: i64) -> Result<()> {
        for mut e in self.all()? {
            if !e.expired && e.policy.policy.until_unix_s <= now {
                e.packets.clear();
                e.plaintext_digests.clear();
                e.dispatch.submission.payload.ciphertext.clear();
                e.dispatch.submission.payload.encapsulated_key.clear();
                e.expired = true;
                let id = e.dispatch.submission.binding.job_id.clone();
                self.save(&id, &e)?;
                self.store.write(|tx| {
                    tx.execute("UPDATE entries SET reserved=0 WHERE job=?1", [&id])?;
                    Ok(())
                })?;
            }
        }
        Ok(())
    }
    fn save(&mut self, id: &str, entry: &OutboxEntry) -> Result<()> {
        self.validate(entry)?;
        let body = encode(entry)?;
        self.store.write(|tx| {
            tx.execute(
                "UPDATE entries SET body=?1,checksum=?3 WHERE job=?2",
                params![body, id, body_digest(&body)?],
            )?;
            Ok(())
        })
    }
    fn validate(&self, e: &OutboxEntry) -> Result<()> {
        if e.expired {
            return if e.packets.is_empty() && e.dispatch.submission.payload.ciphertext.is_empty() {
                Ok(())
            } else {
                Err(Error::Denied)
            };
        }
        e.policy.policy.validate(&e.dispatch.submission)?;
        if encode(e)?.len() > self.limits.job_bytes {
            return Err(Error::Full);
        }
        let mut total = 0;
        let mut terminal = false;
        for (i, p) in e.packets.iter().enumerate() {
            match p {
                OutboxPacket::Delta { frame } => {
                    if terminal || frame.sequence as usize != i || i >= 16 {
                        return Err(Error::Denied);
                    }
                    frame.payload.validate()?;
                    total += frame.payload.ciphertext.len() - 16;
                }
                OutboxPacket::Terminal { packet } => {
                    if terminal {
                        return Err(Error::Denied);
                    }
                    packet.payload.validate()?;
                    if packet.payload.ciphertext.len()
                        > e.dispatch.submission.binding.max_result_bytes + 16
                    {
                        return Err(Error::Full);
                    }
                    terminal = true;
                }
            }
        }
        if total > e.dispatch.submission.binding.max_result_bytes {
            return Err(Error::Full);
        }
        if let Some(manifest) = &e.manifest
            && (manifest.binding != e.dispatch.submission.binding
                || manifest.submission_digest != e.policy.policy.submission_digest
                || manifest.packets()
                    != e.packets
                        .iter()
                        .map(|p| p.id(&manifest.binding))
                        .collect::<Result<Vec<_>>>()?)
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
}
/// A scope-bound local stop marker; it never grants execution or uploads packets.
pub(crate) struct CancelRecorder {
    store: PrivateDb,
    job: String,
}
impl CancelRecorder {
    /// Commit observation before signalling the provider cancellation token.
    pub(crate) fn observe(&mut self) -> Result<()> {
        self.store.write(|tx| {
            if tx.execute("UPDATE entries SET cancel_seen=1 WHERE job=?1", [&self.job])? != 1 {
                return Err(Error::Denied);
            }
            Ok(())
        })
    }
}
fn encode<T: Serialize>(v: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(v).map_err(|_| Error::Invalid)
}

fn body_digest(bytes: &[u8]) -> Result<String> {
    Ok(crate::authority::hash(
        std::str::from_utf8(bytes).map_err(|_| Error::Denied)?,
    ))
}
