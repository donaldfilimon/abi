//! Replay authority persists commitments only; every payload remains transient.
use crate::{
    Account, Authority, DeliveryAck, DeliveryPage, DurableSubmission, Error, JobReceipt,
    NodeCertificate, NodeReply, OutboxPacket, PacketId, PacketKind, RecordedOutcome,
    RecoveryAction, RecoveryReply, RecoveryRequest, RecoveryStatus, Relay, ReplayGrant, Result,
    SignedDeliveryPolicy, TerminalManifest,
    authority::{audit, hash},
    enrollment::{decode, json},
    relay::{authorize_binding, lookup, node_receipt, owned_node},
};
use ed25519_dalek::Signer as _;
use rusqlite::{OptionalExtension as _, TransactionBehavior, params};
impl Relay {
    pub(crate) fn delivery_policy(
        db: &rusqlite::Connection,
        id: &str,
    ) -> Result<Option<SignedDeliveryPolicy>> {
        let encoded: Option<String> = db
            .query_row(
                "SELECT delivery_policy FROM relay_jobs WHERE id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        encoded.map(|s| decode(&s)).transpose()
    }
    /// Submit a durable job with authority-attested immutable retention.
    pub fn submit_durable(
        &mut self,
        a: &mut Authority,
        account: &Account,
        s: DurableSubmission,
        now: i64,
    ) -> Result<JobReceipt> {
        s.delivery.validate(&s.submission)?;
        if now >= s.delivery.until_unix_s {
            return Err(Error::Denied);
        }
        let id = s.submission.binding.job_id.clone();
        if let Some(existing) = Self::delivery_policy(&a.db, &id)?
            && existing.policy != s.delivery
        {
            return Err(Error::Denied);
        }
        let signed = SignedDeliveryPolicy::sign(s.delivery, &a.key)?;
        self.submit_with_policy(a, account, s.submission, Some(&signed), now)
    }
    fn cancelled(db: &rusqlite::Connection, id: &str) -> Result<bool> {
        Ok(
            db.query_row("SELECT cancelled FROM relay_jobs WHERE id=?1", [id], |r| {
                r.get::<_, i64>(0)
            })? != 0,
        )
    }
    fn manifest(db: &rusqlite::Connection, id: &str) -> Result<Option<TerminalManifest>> {
        let encoded: Option<String> =
            db.query_row("SELECT manifest FROM relay_jobs WHERE id=?1", [id], |r| {
                r.get(0)
            })?;
        encoded.map(|s| decode(&s)).transpose()
    }
    fn cached_ids(&self, b: &crate::JobBinding) -> Result<Vec<PacketId>> {
        let mut ids = vec![];
        for f in self.frames.get(&b.job_id).into_iter().flatten() {
            ids.push(OutboxPacket::Delta { frame: f.clone() }.id(b)?);
        }
        if let Some(p) = self.terminals.get(&b.job_id) {
            ids.push(OutboxPacket::Terminal { packet: p.clone() }.id(b)?);
        }
        Ok(ids)
    }
    #[allow(clippy::too_many_lines)] // One immediate transaction validates all replay authority before commit.
    pub(crate) fn recover(
        &mut self,
        a: &mut Authority,
        cert: &NodeCertificate,
        request: &RecoveryRequest,
        now: i64,
    ) -> Result<NodeReply> {
        if request.protocol != "abi.relay-recovery.v1" {
            return Err(Error::Invalid);
        }
        let mut restore = None;
        let mut reply = RecoveryReply {
            status: RecoveryStatus::Refused,
            epoch: self.epoch.clone(),
            committed: vec![],
            cached: vec![],
            grant: None,
            cancelled: false,
        };
        let tx =
            a.db.transaction_with_behavior(TransactionBehavior::Immediate)?;
        owned_node(
            &tx,
            &cert.node_id,
            None,
            &a.key.verifying_key().to_bytes(),
            &a.audience,
            now,
        )?;
        let id = match &request.action {
            RecoveryAction::Inspect { job_id, .. } => job_id,
            RecoveryAction::Publish { grant, .. } => &grant.binding.job_id,
        };
        let mut receipt = node_receipt(&tx, cert, id)?;
        let Some(policy) = Self::delivery_policy(&tx, id)? else {
            reply.status = RecoveryStatus::Unavailable;
            return Ok(NodeReply {
                dispatch: None,
                receipt: Some(receipt),
                permit: None,
                delivery: None,
                recovery: Some(reply),
            });
        };
        reply.cancelled = Self::cancelled(&tx, id)?;
        if now >= policy.policy.until_unix_s {
            reply.status = RecoveryStatus::Expired;
            return Ok(NodeReply {
                dispatch: None,
                receipt: Some(receipt),
                permit: None,
                delivery: Some(policy),
                recovery: Some(reply),
            });
        }
        match &request.action {
            RecoveryAction::Inspect { manifest, .. } => {
                manifest.verify(&cert.claim.public_key)?;
                if manifest.binding != receipt.binding
                    || manifest.submission_digest != receipt.submission_digest
                {
                    return Err(Error::Denied);
                }
                let admitted: Option<i64> = tx.query_row(
                    "SELECT permit_issued FROM relay_jobs WHERE id=?1",
                    [id],
                    |r| r.get(0),
                )?;
                if manifest.outcome == RecordedOutcome::Completed && admitted.is_none() {
                    return Err(Error::Denied);
                }
                if matches!(receipt.state.as_str(), "queued" | "offered") {
                    tx.execute("UPDATE relay_jobs SET state='unresolved' WHERE id=?1", [id])?;
                    receipt.state = "unresolved".into();
                    self.pending.remove(id);
                }
                if manifest.outcome == RecordedOutcome::Cancelled {
                    tx.execute("UPDATE relay_jobs SET cancelled=1 WHERE id=?1", [id])?;
                    reply.cancelled = true;
                }
                let previous = Self::manifest(&tx, id)?;
                if let Some(old) = &previous
                    && old.digest()? != manifest.digest()?
                    && (old.terminal_digest.is_some()
                        || matches!(
                            old.outcome,
                            RecordedOutcome::Cancelled | RecordedOutcome::Abandoned
                        )
                        || !manifest.frame_digests.starts_with(&old.frame_digests))
                {
                    reply.status = RecoveryStatus::Contradictory;
                    return Ok(NodeReply {
                        dispatch: None,
                        receipt: Some(receipt),
                        permit: None,
                        delivery: Some(policy),
                        recovery: Some(reply),
                    });
                }
                let mut stmt = tx.prepare(
                    "SELECT sequence,digest FROM relay_frames WHERE job=?1 ORDER BY sequence",
                )?;
                let existing = stmt
                    .query_map([id], |r| Ok((r.get::<_, u32>(0)?, r.get::<_, String>(1)?)))?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                drop(stmt);
                if existing
                    .iter()
                    .any(|(n, d)| manifest.frame_digests.get(*n as usize) != Some(d))
                {
                    return Err(Error::Denied);
                }
                let old_result: Option<String> = tx.query_row(
                    "SELECT result_digest FROM relay_jobs WHERE id=?1",
                    [id],
                    |r| r.get(0),
                )?;
                if old_result.is_some()
                    && previous.as_ref().and_then(|m| m.terminal_digest.as_ref())
                        != manifest.terminal_digest.as_ref()
                {
                    return Err(Error::Denied);
                }
                tx.execute(
                    "UPDATE relay_jobs SET manifest=?1 WHERE id=?2",
                    params![json(manifest)?, id],
                )?;
                if !reply.cancelled
                    && manifest.outcome == RecordedOutcome::Completed
                    && matches!(
                        receipt.state.as_str(),
                        "dispatched" | "unresolved" | "completed"
                    )
                {
                    tx.execute("UPDATE relay_jobs SET state='completed' WHERE id=?1", [id])?;
                    receipt.state = "completed".into();
                }
                if reply.cancelled {
                    tx.execute("UPDATE relay_jobs SET state='cancelled' WHERE id=?1", [id])?;
                    receipt.state = "cancelled".into();
                }
                reply.committed = manifest.packets();
                reply.cached = self.cached_ids(&receipt.binding)?;
                let mut grant = ReplayGrant {
                    binding: receipt.binding.clone(),
                    policy: policy.policy.clone(),
                    manifest_digest: manifest.digest()?,
                    epoch: self.epoch.clone(),
                    packets: manifest.packets(),
                    cancelled: reply.cancelled,
                    issued_at: now,
                    expires_at: (now + 30).min(policy.policy.until_unix_s),
                    signature: vec![],
                };
                if reply.cancelled {
                    grant.packets.retain(|p| p.kind == PacketKind::Delta);
                }
                grant.signature = a.key.sign(&grant.bytes()?).to_bytes().to_vec();
                reply.grant = Some(grant);
                reply.status = if reply.committed == reply.cached {
                    RecoveryStatus::Cached
                } else {
                    RecoveryStatus::Missing
                };
            }
            RecoveryAction::Publish { grant, packet } => {
                grant.verify(&a.key.verifying_key().to_bytes(), now)?;
                let manifest = Self::manifest(&tx, id)?.ok_or(Error::Denied)?;
                let packet_id = packet.id(&receipt.binding)?;
                if grant.epoch != self.epoch
                    || grant.binding != receipt.binding
                    || grant.policy != policy.policy
                    || grant.manifest_digest != manifest.digest()?
                    || !grant.packets.contains(&packet_id)
                    || (reply.cancelled && packet_id.kind == PacketKind::Terminal)
                {
                    return Err(Error::Denied);
                }
                packet.verify(&receipt.binding, &cert.claim.public_key)?;
                let cached = self.cached_ids(&receipt.binding)?;
                if !cached.contains(&packet_id) {
                    let bytes = packet.encoded_bytes()?;
                    let encoded_memory = self
                        .frames
                        .values()
                        .flatten()
                        .map(|f| {
                            serde_json::to_vec(&OutboxPacket::Delta { frame: f.clone() })
                                .map_or(usize::MAX, |v| v.len())
                        })
                        .try_fold(0usize, usize::checked_add)
                        .ok_or(Error::Full)?
                        + self
                            .terminals
                            .values()
                            .map(|p| {
                                serde_json::to_vec(&OutboxPacket::Terminal { packet: p.clone() })
                                    .map_or(usize::MAX, |v| v.len())
                            })
                            .try_fold(0usize, usize::checked_add)
                            .ok_or(Error::Full)?;
                    let raw_packets = self
                        .frames
                        .values()
                        .flatten()
                        .map(|f| f.payload.bytes())
                        .sum::<usize>()
                        + self
                            .terminals
                            .values()
                            .map(|p| p.payload.bytes())
                            .sum::<usize>();
                    if self.memory_bytes() + encoded_memory.saturating_sub(raw_packets) + bytes
                        > self.limits.memory_bytes
                    {
                        reply.status = RecoveryStatus::CapacityFull;
                        return Ok(NodeReply {
                            dispatch: None,
                            receipt: Some(receipt),
                            permit: None,
                            delivery: Some(policy),
                            recovery: Some(reply),
                        });
                    }
                    match packet {
                        OutboxPacket::Delta { frame } => {
                            let buffered = self.frames.get(id).map_or(0, Vec::len);
                            if frame.sequence as usize != buffered {
                                return Err(Error::Denied);
                            }
                            let size = i64::try_from(frame.payload.ciphertext.len() - 16)
                                .map_err(|_| Error::Invalid)?;
                            let previous: Option<String> = tx
                                .query_row(
                                    "SELECT digest FROM relay_frames WHERE job=?1 AND sequence=?2",
                                    params![id, frame.sequence],
                                    |r| r.get(0),
                                )
                                .optional()?;
                            if previous.as_ref().is_some_and(|d| *d != packet_id.digest) {
                                return Err(Error::Denied);
                            }
                            let total: i64 = tx.query_row(
                                "SELECT coalesce(sum(bytes),0) FROM relay_frames WHERE job=?1",
                                [id],
                                |r| r.get(0),
                            )?;
                            if total + size
                                > i64::try_from(receipt.binding.max_result_bytes)
                                    .map_err(|_| Error::Invalid)?
                                && previous.is_none()
                            {
                                return Err(Error::Invalid);
                            }
                            tx.execute("INSERT OR IGNORE INTO relay_frames(job,sequence,digest,bytes) VALUES(?1,?2,?3,?4)",params![id,frame.sequence,packet_id.digest,size])?;
                            if previous.is_none() {
                                audit(&tx, now, "job.frame", &cert.claim.account_id, id)?;
                            }
                        }
                        OutboxPacket::Terminal { packet } => {
                            if self.frames.get(id).map_or(0, Vec::len)
                                != manifest.frame_digests.len()
                            {
                                return Err(Error::Denied);
                            }
                            let previous: Option<String> = tx.query_row(
                                "SELECT result_digest FROM relay_jobs WHERE id=?1",
                                [id],
                                |r| r.get(0),
                            )?;
                            if previous.is_none() {
                                audit(&tx, now, "job.finished", &cert.claim.account_id, id)?;
                            }
                            tx.execute(
                                "UPDATE relay_jobs SET result_digest=?1 WHERE id=?2",
                                params![hash(&json(&packet.payload)?), id],
                            )?;
                        }
                    }
                    restore = Some(packet.clone());
                }
                reply.status = RecoveryStatus::Recorded;
                reply.committed = manifest.packets();
                reply.cached = self.cached_ids(&receipt.binding)?;
            }
        }
        tx.commit()?;
        if let Some(packet) = restore {
            match packet {
                OutboxPacket::Delta { frame } => {
                    let frames = self.frames.entry(id.clone()).or_default();
                    frames.push(frame);
                    frames.sort_by_key(|f| f.sequence);
                }
                OutboxPacket::Terminal { packet } => {
                    self.results.insert(id.clone(), packet.payload.clone());
                    self.terminals.insert(id.clone(), packet);
                }
            }
            reply.cached = self.cached_ids(&receipt.binding)?;
        }
        Ok(NodeReply {
            dispatch: None,
            receipt: Some(receipt),
            permit: None,
            delivery: Some(policy),
            recovery: Some(reply),
        })
    }
    /// Fetch an encoded-size bounded page after checking current ownership and node authority.
    pub fn delivery(
        &mut self,
        a: &mut Authority,
        account: &Account,
        id: &str,
        after: u32,
        now: i64,
    ) -> Result<DeliveryPage> {
        if after > 16 {
            return Err(Error::Invalid);
        }
        self.cleanup(a, now)?;
        let tx = a.db.transaction()?;
        let receipt = lookup(&tx, id)?.ok_or(Error::Denied)?;
        if receipt.binding.account_id != account.account_id
            || receipt.binding.tenant_id != account.tenant_id
        {
            return Err(Error::Denied);
        }
        let cert = owned_node(
            &tx,
            &receipt.binding.node_id,
            Some(account),
            &a.key.verifying_key().to_bytes(),
            &a.audience,
            now,
        )?;
        authorize_binding(&cert, &receipt.binding)?;
        let policy = Self::delivery_policy(&tx, id)?.ok_or(Error::Denied)?;
        if now >= policy.policy.until_unix_s {
            return Err(Error::Denied);
        }
        let mut page = DeliveryPage {
            receipt,
            policy,
            manifest: Self::manifest(&tx, id)?,
            packets: vec![],
            next: after,
            epoch: self.epoch.clone(),
            cancelled: Self::cancelled(&tx, id)?,
        };
        let mut packets: Vec<_> = self
            .frames
            .get(id)
            .into_iter()
            .flatten()
            .filter(|f| f.sequence >= after)
            .cloned()
            .map(|frame| OutboxPacket::Delta { frame })
            .collect();
        if !page.cancelled
            && let Some(packet) = self.terminals.get(id)
        {
            packets.push(OutboxPacket::Terminal {
                packet: packet.clone(),
            });
        }
        for packet in packets {
            let old = page.next;
            if let OutboxPacket::Delta { frame } = &packet {
                if frame.sequence != page.next {
                    break;
                }
                page.next = frame.sequence + 1;
            }
            page.packets.push(packet);
            if serde_json::to_vec(&page).map_err(|_| Error::Invalid)?.len() > 384 * 1024 {
                page.packets.pop();
                page.next = old;
                break;
            }
        }
        Ok(page)
    }
    /// Record a client's exact imported manifest and contiguous prefix.
    pub fn acknowledge_delivery(
        &mut self,
        a: &mut Authority,
        account: &Account,
        id: &str,
        ack: &DeliveryAck,
        now: i64,
    ) -> Result<()> {
        let page = self.delivery(a, account, id, 0, now)?;
        let manifest = page.manifest.ok_or(Error::Denied)?;
        if ack.manifest_digest != manifest.digest()?
            || ack.contiguous
                > u32::try_from(manifest.frame_digests.len()).map_err(|_| Error::Invalid)?
            || (ack.terminal_digest.is_some()
                && (ack.terminal_digest != manifest.terminal_digest
                    || ack.contiguous
                        != u32::try_from(manifest.frame_digests.len())
                            .map_err(|_| Error::Invalid)?))
        {
            return Err(Error::Denied);
        }
        a.db.execute(
            "UPDATE relay_jobs SET delivery_ack=?1 WHERE id=?2",
            params![json(ack)?, id],
        )?;
        Ok(())
    }
}
