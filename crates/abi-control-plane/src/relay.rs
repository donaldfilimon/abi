//! Durable scheduling metadata and bounded ephemeral ciphertext. Never redispatch.
use crate::{
    Account, Authority, Dispatch, DispatchPermit, Error, JobReceipt, JobView, NodeAction,
    NodeCertificate, NodeMessage, NodeReply, RelayLimits, Result, SealedPayload, Submission,
    authority::{audit, hash},
    enrollment::{decode, json},
};
use ed25519_dalek::Signer as _;
use rusqlite::{OptionalExtension as _, Transaction, TransactionBehavior, params};
use std::{collections::BTreeMap, fs::File, path::PathBuf};
/// One relay epoch, protected against competing processes by a private file lock.
pub struct Relay {
    _lock: File,
    pub(crate) epoch: String,
    pub(crate) terminals: BTreeMap<String, crate::TerminalPacket>,
    pub(crate) limits: RelayLimits,
    pub(crate) pending: BTreeMap<String, Submission>,
    pub(crate) results: BTreeMap<String, SealedPayload>,
    pub(crate) frames: BTreeMap<String, Vec<crate::StreamFrame>>,
}
impl Relay {
    /// Open an explicit relay epoch. Interrupted handoffs are unresolved, never queued again.
    pub fn open(authority: &mut Authority, limits: RelayLimits, now: i64) -> Result<Self> {
        if limits.active_jobs == 0
            || limits.active_jobs > 64
            || limits.queued_jobs == 0
            || limits.queued_jobs > 1024
            || limits.job_records == 0
            || limits.job_records > 100_000
            || !(65_584..=268_435_456).contains(&limits.memory_bytes)
        {
            return Err(Error::Invalid);
        }
        let mut path = PathBuf::from(authority.db.path().ok_or(Error::Invalid)?).into_os_string();
        path.push(".relay.lock");
        let path = PathBuf::from(path);
        let initialize = !path.try_exists()?;
        let (lock, _) = crate::private::prepare(&path, initialize)?;
        lock.try_lock().map_err(|_| Error::Denied)?;
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(include_str!("relay_schema.sql"))?;
        let version: i64 = tx.query_row("SELECT version FROM relay_meta WHERE id=1", [], |r| {
            r.get(0)
        })?;
        if !matches!(version, 1 | 2) {
            return Err(Error::Denied);
        }
        if version == 1 {
            tx.execute_batch(include_str!("relay_recovery/migration.sql"))?;
        }
        let changed=tx.execute("UPDATE relay_jobs SET state=CASE WHEN cancelled=1 AND delivery_policy IS NOT NULL THEN 'cancelled' ELSE 'unresolved' END WHERE state IN ('queued','offered','dispatched','cancel_requested')",[])?;
        if changed > 0 {
            audit(&tx, now, "relay.recovered", "", "")?;
        }
        tx.commit()?;
        Ok(Self {
            _lock: lock,
            epoch: uuid::Uuid::new_v4().to_string(),
            terminals: BTreeMap::new(),
            limits,
            pending: BTreeMap::new(),
            results: BTreeMap::new(),
            frames: BTreeMap::new(),
        })
    }
    /// Admit only ciphertext targeting the verified account's enrolled private node.
    pub fn submit(
        &mut self,
        authority: &mut Authority,
        account: &Account,
        submission: Submission,
        now: i64,
    ) -> Result<JobReceipt> {
        self.submit_with_policy(authority, account, submission, None, now)
    }
    pub(crate) fn submit_with_policy(
        &mut self,
        authority: &mut Authority,
        account: &Account,
        submission: Submission,
        policy: Option<&crate::SignedDeliveryPolicy>,
        now: i64,
    ) -> Result<JobReceipt> {
        self.cleanup(authority, now)?;
        validate(&submission, account, &authority.audience, now)?;
        if submission.binding.capability == "tool.execute" && policy.is_none() {
            return Err(Error::Denied);
        }
        let binding = &submission.binding;
        let digest = hash(&json(&submission)?);
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let cert = owned_node(
            &tx,
            &binding.node_id,
            Some(account),
            &authority.key.verifying_key().to_bytes(),
            &authority.audience,
            now,
        )?;
        authorize_binding(&cert, binding)?;
        if let Some(mut receipt) = lookup(&tx, &binding.job_id)? {
            if receipt.binding != *binding
                || receipt.submission_digest != digest
                || Self::delivery_policy(&tx, &binding.job_id)?
                    .as_ref()
                    .map(|p| &p.policy)
                    != policy.map(|p| &p.policy)
            {
                return Err(Error::Denied);
            }
            receipt.result_available = self.results.contains_key(&binding.job_id);
            return Ok(receipt);
        }
        let count: i64 = tx.query_row("SELECT count(*) FROM relay_jobs", [], |r| r.get(0))?;
        let queued: i64 = tx.query_row(
            "SELECT count(*) FROM relay_jobs WHERE account=?1 AND state='queued'",
            [&account.account_id],
            |r| r.get(0),
        )?;
        if usize::try_from(count).unwrap_or(usize::MAX) >= self.limits.job_records
            || queued >= i64::from(self.limits.queued_jobs)
            || self.memory_bytes() + submission.payload.bytes() > self.limits.memory_bytes
        {
            return Err(Error::Full);
        }
        tx.execute("INSERT INTO relay_jobs(id,account,node,binding,digest,state,deadline) VALUES(?1,?2,?3,?4,?5,'queued',?6)",params![binding.job_id,account.account_id,binding.node_id,json(binding)?,digest,binding.expires_at])?;
        if let Some(policy) = policy {
            tx.execute(
                "UPDATE relay_jobs SET delivery_policy=?1,delivery_until=?2 WHERE id=?3",
                params![json(policy)?, policy.policy.until_unix_s, binding.job_id],
            )?;
        }
        audit(&tx, now, "job.queued", &account.account_id, &binding.job_id)?;
        let receipt = lookup(&tx, &binding.job_id)?.ok_or(Error::Denied)?;
        tx.commit()?;
        self.pending.insert(binding.job_id.clone(), submission);
        Ok(receipt)
    }
    /// Execute one action-bound fresh node request. Every mutation rechecks authority.
    pub fn node(
        &mut self,
        authority: &mut Authority,
        message: &NodeMessage,
        now: i64,
    ) -> Result<NodeReply> {
        self.cleanup(authority, now)?;
        message.validate_action()?;
        let cert = authority.authenticate_node(&message.proof, &message.signature, now)?;
        match &message.action {
            NodeAction::Recovery { request } => self.recover(authority, &cert, request, now),
            NodeAction::Stream { job_id, frame } => {
                self.stream(authority, &cert, job_id, frame, now)
            }
            NodeAction::Stopped { job_id } => Self::stopped(authority, &cert, job_id, now),
            NodeAction::Poll => self.poll(authority, &cert, now),
            NodeAction::Inspect { job_id } => Self::permit(authority, &cert, job_id, now),
            NodeAction::Complete { job_id, payload } => {
                self.complete(authority, &cert, job_id, Some(payload), now)
            }
            NodeAction::Fail { job_id } => self.complete(authority, &cert, job_id, None, now),
        }
    }
    /// Owning account lookup; no caller-selected tenant scope is trusted.
    pub fn view(
        &mut self,
        authority: &mut Authority,
        account: &Account,
        id: &str,
        now: i64,
    ) -> Result<JobView> {
        self.cleanup(authority, now)?;
        let receipt = lookup(&authority.db, id)?.ok_or(Error::Denied)?;
        if receipt.binding.account_id != account.account_id
            || receipt.binding.tenant_id != account.tenant_id
        {
            return Err(Error::Denied);
        }
        let mut cancelled = false;
        if let Some(policy) = Self::delivery_policy(&authority.db, id)? {
            if now >= policy.policy.until_unix_s {
                return Err(Error::Denied);
            }
            let tx = authority.db.transaction()?;
            let cert = owned_node(
                &tx,
                &receipt.binding.node_id,
                Some(account),
                &authority.key.verifying_key().to_bytes(),
                &authority.audience,
                now,
            )?;
            authorize_binding(&cert, &receipt.binding)?;
            cancelled =
                tx.query_row("SELECT cancelled FROM relay_jobs WHERE id=?1", [id], |r| {
                    r.get::<_, bool>(0)
                })?;
        }
        let result = if cancelled {
            None
        } else {
            self.results.get(id).cloned()
        };
        let mut receipt = receipt;
        receipt.result_available = result.is_some();
        let frames = self.frames.get(id).cloned().unwrap_or_default();
        Ok(JobView {
            receipt,
            result,
            frames,
        })
    }
    /// Cancel undispatched work; dispatched work waits for a signed worker stop acknowledgement.
    pub fn cancel(
        &mut self,
        authority: &mut Authority,
        account: &Account,
        id: &str,
        now: i64,
    ) -> Result<JobReceipt> {
        self.cleanup(authority, now)?;
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let receipt = lookup(&tx, id)?.ok_or(Error::Denied)?;
        if receipt.binding.account_id != account.account_id
            || receipt.binding.tenant_id != account.tenant_id
        {
            return Err(Error::Denied);
        }
        tx.execute("UPDATE relay_jobs SET cancelled=1 WHERE id=?1", [id])?;
        let durable = Self::delivery_policy(&tx, id)?.is_some();
        let state = match receipt.state.as_str() {
            "completed" | "unresolved" if durable => "cancelled",
            "queued" | "offered" => "cancelled",
            "dispatched" => "cancel_requested",
            other => other,
        };
        if state != receipt.state {
            tx.execute(
                "UPDATE relay_jobs SET state=?1 WHERE id=?2",
                params![state, id],
            )?;
            audit(&tx, now, "job.cancelled", &account.account_id, id)?;
        }
        let receipt = lookup(&tx, id)?.ok_or(Error::Denied)?;
        tx.commit()?;
        self.pending.remove(id);
        Ok(receipt)
    }
    pub(crate) fn cleanup(&mut self, authority: &mut Authority, now: i64) -> Result<()> {
        authority.db.execute("UPDATE relay_jobs SET state=CASE WHEN state IN ('dispatched','cancel_requested') THEN 'unresolved' ELSE 'expired' END WHERE deadline<=?1 AND state IN ('queued','offered','dispatched','cancel_requested')",[now])?;
        authority.db.execute("UPDATE relay_jobs SET state=CASE WHEN state IN ('dispatched','cancel_requested') THEN 'unresolved' ELSE 'cancelled' END WHERE state IN ('queued','offered','dispatched','cancel_requested') AND node IN (SELECT id FROM nodes WHERE revoked=1)",[])?;
        let mut stmt = authority
            .db
            .prepare("SELECT id,state FROM relay_jobs WHERE deadline>?1 OR delivery_until>?1")?;
        let valid = stmt
            .query_map([now], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<BTreeMap<_, _>, _>>()?;
        self.pending
            .retain(|id, _| valid.get(id).is_some_and(|s| s == "queued"));
        self.results
            .retain(|id, _| valid.get(id).is_some_and(|s| s == "completed"));
        self.frames.retain(|id, _| valid.contains_key(id));
        self.terminals.retain(|id, _| valid.contains_key(id));
        Ok(())
    }
    pub(crate) fn memory_bytes(&self) -> usize {
        self.frames
            .values()
            .flatten()
            .map(|f| f.payload.bytes())
            .sum::<usize>()
            + self
                .pending
                .values()
                .map(|p| p.payload.bytes())
                .sum::<usize>()
            + self
                .results
                .values()
                .map(SealedPayload::bytes)
                .sum::<usize>()
    }
    fn poll(
        &mut self,
        authority: &mut Authority,
        cert: &NodeCertificate,
        now: i64,
    ) -> Result<NodeReply> {
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        owned_node(
            &tx,
            &cert.node_id,
            None,
            &authority.key.verifying_key().to_bytes(),
            &authority.audience,
            now,
        )?;
        let active:i64=tx.query_row("SELECT count(*) FROM relay_jobs WHERE account=?1 AND state IN ('offered','dispatched','cancel_requested')",[&cert.claim.account_id],|r|r.get(0))?;
        if active >= i64::from(self.limits.active_jobs) {
            return Ok(empty());
        }
        let id: Option<String> = tx
            .query_row(
                "SELECT id FROM relay_jobs WHERE node=?1 AND state='queued' ORDER BY rowid LIMIT 1",
                [&cert.node_id],
                |r| r.get(0),
            )
            .optional()?;
        let Some(id) = id else {
            return Ok(empty());
        };
        let submission = self.pending.get(&id).cloned().ok_or(Error::Denied)?;
        authorize_binding(cert, &submission.binding)?;
        let mut dispatch = Dispatch {
            submission,
            signature: Vec::new(),
        };
        dispatch.signature = authority.key.sign(&dispatch.bytes()?).to_bytes().to_vec();
        tx.execute("UPDATE relay_jobs SET state='offered' WHERE id=?1", [&id])?;
        audit(&tx, now, "job.offered", &cert.claim.account_id, &id)?;
        tx.commit()?;
        self.pending.remove(&id);
        let delivery = Self::delivery_policy(&authority.db, &id)?;
        Ok(NodeReply {
            delivery,
            recovery: None,
            dispatch: Some(dispatch),
            receipt: None,
            permit: None,
        })
    }
    fn permit(
        authority: &mut Authority,
        cert: &NodeCertificate,
        id: &str,
        now: i64,
    ) -> Result<NodeReply> {
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        owned_node(
            &tx,
            &cert.node_id,
            None,
            &authority.key.verifying_key().to_bytes(),
            &authority.audience,
            now,
        )?;
        let receipt = node_receipt(&tx, cert, id)?;
        if !matches!(receipt.state.as_str(), "offered" | "dispatched") {
            return Ok(NodeReply {
                delivery: None,
                recovery: None,
                dispatch: None,
                receipt: Some(receipt),
                permit: None,
            });
        }
        if receipt.state == "offered" {
            tx.execute(
                "UPDATE relay_jobs SET state='dispatched',permit_issued=?1 WHERE id=?2",
                params![now, id],
            )?;
            audit(&tx, now, "job.dispatched", &cert.claim.account_id, id)?;
        }
        // Human approval may outlive a five-second tool permit. Refresh only
        // its timestamp after current ownership/state checks; worker admission
        // and local approval intent still enforce single execution.
        if receipt.state == "dispatched" && receipt.binding.capability == "tool.execute" {
            tx.execute(
                "UPDATE relay_jobs SET permit_issued=?1 WHERE id=?2",
                params![now, id],
            )?;
        }
        let issued_at: i64 = tx.query_row(
            "SELECT permit_issued FROM relay_jobs WHERE id=?1",
            [id],
            |r| r.get(0),
        )?;
        let mut permit = DispatchPermit {
            binding: receipt.binding.clone(),
            submission_digest: receipt.submission_digest,
            issued_at,
            expires_at: (issued_at + 5).min(receipt.binding.expires_at),
            signature: Vec::new(),
        };
        permit.signature = authority.key.sign(&permit.bytes()?).to_bytes().to_vec();
        let receipt = lookup(&tx, id)?.ok_or(Error::Denied)?;
        tx.commit()?;
        Ok(NodeReply {
            delivery: None,
            recovery: None,
            dispatch: None,
            receipt: Some(receipt),
            permit: Some(permit),
        })
    }
    fn complete(
        &mut self,
        authority: &mut Authority,
        cert: &NodeCertificate,
        id: &str,
        payload: Option<&SealedPayload>,
        now: i64,
    ) -> Result<NodeReply> {
        if let Some(p) = payload {
            p.validate()?;
        }
        let tx = authority
            .db
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        owned_node(
            &tx,
            &cert.node_id,
            None,
            &authority.key.verifying_key().to_bytes(),
            &authority.audience,
            now,
        )?;
        let mut receipt = node_receipt(&tx, cert, id)?;
        let digest = payload.map(json).transpose()?.map(|p| hash(&p));
        if let Some(p) = payload
            && p.ciphertext.len() > receipt.binding.max_result_bytes + 16
        {
            return Err(Error::Invalid);
        }
        if receipt.state == "completed" {
            let previous: Option<String> = tx.query_row(
                "SELECT result_digest FROM relay_jobs WHERE id=?1",
                [id],
                |r| r.get(0),
            )?;
            if digest != previous {
                return Err(Error::Denied);
            }
            receipt.result_available = self.results.contains_key(id);
            return Ok(NodeReply {
                delivery: None,
                recovery: None,
                dispatch: None,
                receipt: Some(receipt),
                permit: None,
            });
        }
        if payload.is_some_and(|p| self.memory_bytes() + p.bytes() > self.limits.memory_bytes) {
            return Err(Error::Full);
        }
        if receipt.state != "dispatched" {
            return Err(Error::Denied);
        }
        let state = if payload.is_some() {
            "completed"
        } else {
            "unresolved"
        };
        tx.execute(
            "UPDATE relay_jobs SET state=?1,result_digest=?2 WHERE id=?3",
            params![state, digest, id],
        )?;
        audit(&tx, now, "job.finished", &cert.claim.account_id, id)?;
        let mut receipt = lookup(&tx, id)?.ok_or(Error::Denied)?;
        tx.commit()?;
        if let Some(p) = payload {
            self.results.insert(id.into(), p.clone());
            receipt.result_available = true;
        }
        Ok(NodeReply {
            delivery: None,
            recovery: None,
            dispatch: None,
            receipt: Some(receipt),
            permit: None,
        })
    }
}
fn empty() -> NodeReply {
    NodeReply {
        delivery: None,
        recovery: None,
        dispatch: None,
        receipt: None,
        permit: None,
    }
}
fn validate(s: &Submission, a: &Account, audience: &str, now: i64) -> Result<()> {
    let b = &s.binding;
    s.payload.validate()?;
    if b.protocol != "abi.relay-job.v1"
        || b.audience != audience
        || b.account_id != a.account_id
        || b.tenant_id != a.tenant_id
        || !matches!(b.capability.as_str(), "model.infer" | "tool.execute")
        || b.node_id.len() > 64
        || uuid::Uuid::parse_str(&b.job_id).map_or(true, |id| id.to_string() != b.job_id)
        || b.submitted_at > now
        || b.submitted_at < now - 300
        || b.expires_at <= now
        || b.expires_at > b.submitted_at + 300
        || !(4096..=65536).contains(&b.max_result_bytes)
        || b.reply_key == [0; 32]
    {
        return Err(Error::Denied);
    }
    Ok(())
}
pub(crate) fn owned_node(
    tx: &Transaction<'_>,
    node: &str,
    account: Option<&Account>,
    key: &[u8; 32],
    audience: &str,
    now: i64,
) -> Result<NodeCertificate> {
    let encoded:String=tx.query_row("SELECT n.certificate FROM nodes n JOIN accounts a ON n.account=a.id WHERE n.id=?1 AND n.revoked=0",[node],|r|r.get(0)).optional()?.ok_or(Error::Denied)?;
    let cert: NodeCertificate = decode(&encoded)?;
    cert.verify(key, audience, now)?;
    if !cert
        .claim
        .capabilities
        .iter()
        .any(|s| matches!(s.as_str(), "model.infer" | "tool.execute"))
        || cert.claim.encryption_public_key.is_none()
        || account.is_some_and(|a| {
            a.account_id != cert.claim.account_id || a.tenant_id != cert.claim.tenant_id
        })
    {
        return Err(Error::Denied);
    }
    Ok(cert)
}
pub(crate) fn lookup(db: &rusqlite::Connection, id: &str) -> Result<Option<JobReceipt>> {
    let row: Option<(String, String, String)> = db
        .query_row(
            "SELECT binding,digest,state FROM relay_jobs WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    row.map(|(binding, submission_digest, state)| {
        Ok(JobReceipt {
            binding: decode(&binding)?,
            submission_digest,
            state,
            result_available: false,
        })
    })
    .transpose()
}
pub(crate) fn node_receipt(
    tx: &Transaction<'_>,
    cert: &NodeCertificate,
    id: &str,
) -> Result<JobReceipt> {
    let r = lookup(tx, id)?.ok_or(Error::Denied)?;
    if r.binding.node_id != cert.node_id
        || r.binding.account_id != cert.claim.account_id
        || r.binding.tenant_id != cert.claim.tenant_id
    {
        return Err(Error::Denied);
    }
    authorize_binding(cert, &r.binding)?;
    Ok(r)
}
pub(crate) fn authorize_binding(cert: &NodeCertificate, binding: &crate::JobBinding) -> Result<()> {
    if binding.node_id != cert.node_id
        || binding.account_id != cert.claim.account_id
        || binding.tenant_id != cert.claim.tenant_id
        || !cert.claim.capabilities.contains(&binding.capability)
    {
        return Err(Error::Denied);
    }
    Ok(())
}
#[cfg(all(test, unix))]
mod tests;
