//! Durable write-before-send bridge. Reconciliation has no provider argument.
use super::RelayRuntime;
use crate::{Dispatch, DispatchPermit, Error, Result, authority::hash, enrollment::json};
use crate::{
    FileRelayOutbox, OutboxPacket, PacketId, RecordedOutcome, SignedDeliveryPolicy,
    TerminalManifest, TerminalPacket,
};
use abi_agent_runtime::ModelProvider;
use abi_worker::JobControlStore;
use abi_worker::{CancellationToken, RecordedExecution};
use sha2::{Digest as _, Sha256};
/// User-node execution plus its exclusive encrypted transport journal.
pub struct DurableRelayRuntime<S: JobControlStore> {
    pub(crate) runtime: RelayRuntime<S>,
    /// Exact ciphertext journal; never hosted on the public authority.
    pub outbox: FileRelayOutbox,
    pub(crate) unresolved_live: bool,
}
impl<S: JobControlStore> DurableRelayRuntime<S> {
    /// Bind existing stores; invalid history refuses before any execution.
    pub fn new(runtime: RelayRuntime<S>, outbox: FileRelayOutbox) -> Result<Self> {
        let value = Self {
            runtime,
            outbox,
            unresolved_live: false,
        };
        for entry in value.outbox.entries(i64::MIN)? {
            value.validate_entry(&entry)?;
            Self::validate_recorded(&entry, &value.runtime.recorded_dispatch(&entry.dispatch)?)?;
        }
        Ok(value)
    }
    /// Validate and reconcile prior execution facts, never admitting work.
    pub fn reconcile_local(&mut self, now: i64) -> Result<Vec<TerminalManifest>> {
        if self.unresolved_live {
            return Err(Error::Denied);
        }
        self.outbox.expire(now)?;
        for e in self.outbox.entries(now)? {
            self.validate_entry(&e)?;
            let recorded = self.runtime.recorded_dispatch(&e.dispatch)?;
            Self::validate_recorded(&e, &recorded)?;
            // Tool proposals await an explicit local human decision. Reconnect
            // must neither abandon an unexpired proposal nor execute it.
            if e.dispatch.submission.binding.capability == "tool.execute"
                && matches!(recorded, RecordedExecution::Missing)
                && !e.cancelled
                && !e.abandoned
                && e.manifest.is_none()
                && now < e.dispatch.submission.binding.expires_at
            {
                continue;
            }
            match recorded {
                RecordedExecution::Admitted(_) => {
                    self.runtime.recover_dispatch(&e.dispatch, now)?;
                    self.outbox
                        .mark_unresolved(&e.dispatch.submission.binding.job_id, now)?;
                }
                RecordedExecution::Missing => self
                    .outbox
                    .mark_unresolved(&e.dispatch.submission.binding.job_id, now)?,
                _ => {}
            }
            self.finish_local(&e.dispatch, now)?;
        }
        self.outbox.pending(now)
    }
    fn validate_entry(&self, e: &crate::OutboxEntry) -> Result<()> {
        self.runtime.envelope_for_dispatch(&e.dispatch)?;
        e.policy.verify(
            &self.runtime.authority_key,
            &e.dispatch.submission.binding,
            &hash(&json(&e.dispatch.submission)?),
        )?;
        for p in &e.packets {
            p.verify(
                &e.dispatch.submission.binding,
                &self.runtime.certificate.claim.public_key,
            )?;
        }
        if let Some(m) = &e.manifest {
            m.verify(&self.runtime.certificate.claim.public_key)?;
        }
        Ok(())
    }
    fn validate_recorded(e: &crate::OutboxEntry, recorded: &RecordedExecution) -> Result<()> {
        let terminal = e
            .packets
            .iter()
            .any(|p| matches!(p, OutboxPacket::Terminal { .. }));
        let outcome = e.manifest.as_ref().map(|m| m.outcome.clone());
        let valid = match recorded {
            RecordedExecution::Missing => {
                e.packets.is_empty() && outcome.is_none_or(|o| o == RecordedOutcome::Abandoned)
            }
            RecordedExecution::Admitted(_) | RecordedExecution::Unresolved => {
                !terminal && outcome.is_none_or(|o| o == RecordedOutcome::Unresolved)
            }
            RecordedExecution::Cancelled => {
                !terminal && outcome.is_none_or(|o| o == RecordedOutcome::Cancelled)
            }
            RecordedExecution::Completed(result) => {
                result.verify_digest()
                    && result.job_id.to_string() == e.dispatch.submission.binding.job_id
                    && result.chunks.len() >= e.plaintext_digests.len()
                    && result
                        .chunks
                        .iter()
                        .zip(&e.plaintext_digests)
                        .all(|(c, h)| chunk_digest(&c.bytes) == *h)
                    && outcome.is_none_or(|o| {
                        matches!(o, RecordedOutcome::Completed | RecordedOutcome::Cancelled)
                    })
            }
        };
        if valid { Ok(()) } else { Err(Error::Denied) }
    }
    /// Execute a new dispatch once; sync each artifact before notifying its uploader.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_durable(
        &mut self,
        dispatch: &Dispatch,
        policy: &SignedDeliveryPolicy,
        permit: &DispatchPermit,
        provider: &dyn ModelProvider,
        mut clock: impl FnMut() -> i64,
        cancel: &CancellationToken,
        mut emit: impl FnMut(PacketId) -> Result<()> + Send,
    ) -> Result<()> {
        if self
            .outbox
            .entry(&dispatch.submission.binding.job_id)?
            .is_some()
        {
            return Err(Error::Denied);
        }
        self.execute_durable_packets(
            dispatch,
            policy,
            permit,
            provider,
            &mut clock,
            cancel,
            |id, _| emit(id),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_durable_packets(
        &mut self,
        dispatch: &Dispatch,
        policy: &SignedDeliveryPolicy,
        permit: &DispatchPermit,
        provider: &dyn ModelProvider,
        mut clock: impl FnMut() -> i64,
        cancel: &CancellationToken,
        mut emit: impl FnMut(PacketId, OutboxPacket) -> Result<()> + Send,
    ) -> Result<()> {
        if self.unresolved_live {
            return Err(Error::Denied);
        }
        let b = &dispatch.submission.binding;
        policy.verify(
            &self.runtime.authority_key,
            b,
            &hash(&json(&dispatch.submission)?),
        )?;
        if let Some(entry) = self.outbox.entry(&b.job_id)?
            && (entry.manifest.is_some()
                || entry.abandoned
                || !entry.packets.is_empty()
                || !matches!(
                    self.runtime.recorded_dispatch(dispatch)?,
                    RecordedExecution::Missing
                ))
        {
            return Err(Error::Denied);
        }
        self.outbox.prepare(dispatch, policy)?;
        let outbox = &mut self.outbox;
        let result = self.runtime.infer_dispatch(
            dispatch,
            permit,
            provider,
            &mut clock,
            cancel,
            |chunk, frame| {
                outbox.append_frame(&b.job_id, &frame, &chunk_digest(&chunk.bytes))?;
                let packet = OutboxPacket::Delta { frame };
                let id = packet.id(b)?;
                if emit(id, packet).is_err() {
                    cancel.cancel();
                }
                Ok(())
            },
        );
        self.unresolved_live = result.is_err();
        if outbox.poisoned() {
            return Err(Error::Denied);
        }
        // A refused admission remains prepared; reconciliation never dispatches it.
        self.finish_local(dispatch, clock())?;
        result.map(|_| ())
    }
    pub(crate) fn verify_grant(
        &self,
        grant: &crate::ReplayGrant,
        manifest: &TerminalManifest,
        now: i64,
    ) -> Result<()> {
        grant.verify(&self.runtime.authority_key, now)?;
        if grant.binding != manifest.binding || grant.manifest_digest != manifest.digest()? {
            return Err(Error::Denied);
        }
        let entry = self
            .outbox
            .entry(&manifest.binding.job_id)?
            .ok_or(Error::Denied)?;
        if grant.policy != entry.policy.policy
            || grant
                .packets
                .iter()
                .any(|p| !manifest.packets().contains(p))
        {
            return Err(Error::Denied);
        }
        Ok(())
    }
    pub(super) fn finish_local(&mut self, dispatch: &Dispatch, now: i64) -> Result<()> {
        let b = &dispatch.submission.binding;
        let mut e = self.outbox.entry(&b.job_id)?.ok_or(Error::Denied)?;
        Self::validate_recorded(&e, &self.runtime.recorded_dispatch(dispatch)?)?;
        if e.manifest.is_some() {
            return Ok(());
        }
        self.validate_entry(&e)?;
        let recorded = self.runtime.recorded_dispatch(dispatch)?;
        let (mut outcome, mut completed_at) = (RecordedOutcome::Unresolved, None);
        if let RecordedExecution::Completed(result) = &recorded {
            if !result.verify_digest()
                || result.job_id.to_string() != b.job_id
                || result.chunks.len() < e.plaintext_digests.len()
                || result
                    .chunks
                    .iter()
                    .zip(&e.plaintext_digests)
                    .any(|(c, h)| chunk_digest(&c.bytes) != *h)
            {
                return Err(Error::Denied);
            }
            for chunk in result.chunks.iter().skip(e.plaintext_digests.len()) {
                let frame =
                    crate::StreamFrame::seal(b, chunk.sequence, &chunk.bytes, &self.runtime.key)?;
                self.outbox
                    .append_frame(&b.job_id, &frame, &chunk_digest(&chunk.bytes))?;
            }
            e = self.outbox.entry(&b.job_id)?.ok_or(Error::Denied)?;
            completed_at = Some(result.completed_at_unix_ms / 1000);
            if e.cancelled {
                outcome = RecordedOutcome::Cancelled;
            } else {
                let existing = e
                    .packets
                    .iter()
                    .any(|p| matches!(p, OutboxPacket::Terminal { .. }));
                if !existing {
                    let bytes = serde_json::to_vec(result).map_err(|_| Error::Invalid)?;
                    let packet = TerminalPacket::seal(b, &bytes, &self.runtime.key)?;
                    self.outbox
                        .append(&b.job_id, &OutboxPacket::Terminal { packet })?;
                    e = self.outbox.entry(&b.job_id)?.ok_or(Error::Denied)?;
                }
                outcome = RecordedOutcome::Completed;
            }
        } else if matches!(recorded, RecordedExecution::Cancelled) {
            outcome = RecordedOutcome::Cancelled;
        } else if matches!(recorded, RecordedExecution::Missing) {
            outcome = RecordedOutcome::Abandoned;
            self.outbox.mark_unresolved(&b.job_id, now)?;
        }
        let ids = e
            .packets
            .iter()
            .map(|p| p.id(b))
            .collect::<Result<Vec<_>>>()?;
        let m = TerminalManifest {
            binding: b.clone(),
            submission_digest: e.policy.policy.submission_digest,
            frame_digests: ids
                .iter()
                .filter(|p| p.kind == crate::PacketKind::Delta)
                .map(|p| p.digest.clone())
                .collect(),
            terminal_digest: ids
                .iter()
                .find(|p| p.kind == crate::PacketKind::Terminal)
                .map(|p| p.digest.clone()),
            outcome,
            completed_at,
            signature: vec![],
        }
        .sign(&self.runtime.key)?;
        self.outbox.set_manifest(&b.job_id, &m)
    }
}

fn chunk_digest(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut text = String::with_capacity(64);
    for byte in Sha256::digest(bytes) {
        write!(&mut text, "{byte:02x}").expect("String writes succeed");
    }
    text
}
