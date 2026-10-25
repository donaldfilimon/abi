//! Reconnect republishes saved packets; no recovery function accepts a provider.
use crate::{
    DurableRelayRuntime, Error, JobReceipt, NodeAction, OutboxPacket, RecoveryAction,
    RecoveryReply, RecoveryRequest, RecoveryStatus, RelayClient, Result, service::now,
};
use abi_agent_runtime::{CancellationToken, ModelProvider};
use abi_worker::JobControlStore;
use std::{
    sync::{Arc, mpsc},
    time::Duration,
};
/// Metadata-only reconnect progress.
#[derive(Default, Debug)]
pub struct RecoverySummary {
    /// Jobs whose authorized packet set is cached in this epoch.
    pub restored: usize,
    /// Jobs awaiting bounded redelivery or capacity.
    pub pending: usize,
    /// Expired/unavailable jobs.
    pub unavailable: usize,
    /// Refused or contradictory jobs; fresh execution must stop.
    pub refused: usize,
}
struct CancelOnDrop(CancellationToken);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
impl RelayClient {
    async fn recovery_exchange(&self, action: RecoveryAction, time: i64) -> Result<RecoveryReply> {
        self.exchange(
            NodeAction::Recovery {
                request: RecoveryRequest {
                    protocol: "abi.relay-recovery.v1".into(),
                    action,
                },
            },
            time,
        )
        .await?
        .recovery
        .ok_or(Error::Denied)
    }
    /// Reconcile immutable saved outcomes and republish at most sixteen artifacts.
    /// No inference admission or provider call exists on this path.
    pub async fn recover_pending<S: JobControlStore + Send + 'static>(
        &self,
        runtime: &mut DurableRelayRuntime<S>,
        time: i64,
    ) -> Result<RecoverySummary> {
        use std::sync::atomic::Ordering;
        let manifests = runtime.reconcile_local(time)?;
        let mut summary = RecoverySummary::default();
        if manifests.is_empty() {
            return Ok(summary);
        }
        let start = self.recovery_cursor.load(Ordering::Relaxed) % manifests.len();
        let mut budget = 16;
        let mut current_epoch = None::<String>;
        for offset in 0..manifests.len() {
            let index = (start + offset) % manifests.len();
            let manifest = &manifests[index];
            let id = &manifest.binding.job_id;
            let digest = manifest.digest()?;
            if offset > 0
                && current_epoch.as_ref().is_some_and(|epoch| {
                    self.recovered
                        .lock()
                        .is_ok_and(|seen| seen.get(id) == Some(&(epoch.clone(), digest.clone())))
                })
            {
                summary.restored += 1;
                continue;
            }
            if budget == 0 {
                summary.pending += 1;
                continue;
            }
            budget -= 1;
            let reply = self
                .recovery_exchange(
                    RecoveryAction::Inspect {
                        job_id: id.clone(),
                        manifest: manifest.clone(),
                    },
                    now()?,
                )
                .await?;
            self.recovery_cursor.store(index + 1, Ordering::Relaxed);
            if current_epoch.as_ref() != Some(&reply.epoch) {
                let mut seen = self.recovered.lock().map_err(|_| Error::Denied)?;
                seen.retain(|_, (epoch, _)| *epoch == reply.epoch);
                current_epoch = Some(reply.epoch.clone());
            }
            if matches!(
                reply.status,
                RecoveryStatus::Refused | RecoveryStatus::Contradictory
            ) {
                summary.refused += 1;
                continue;
            }
            let Some(grant) = reply.grant.clone() else {
                summary.unavailable += 1;
                continue;
            };
            runtime.verify_grant(&grant, manifest, now()?)?;
            if reply.cancelled {
                runtime.outbox.observe_cancel(id)?;
            }
            let (pending, refused) = self
                .publish_missing(runtime, id, &grant, &reply.cached, &mut budget)
                .await?;
            summary.refused += usize::from(refused);
            if pending {
                summary.pending += 1;
            } else {
                summary.restored += 1;
                runtime.outbox.record_ack(&reply)?;
                self.recovered
                    .lock()
                    .map_err(|_| Error::Denied)?
                    .insert(id.clone(), (reply.epoch, digest));
            }
        }
        Ok(summary)
    }
    async fn publish_missing<S: JobControlStore + Send + 'static>(
        &self,
        runtime: &mut DurableRelayRuntime<S>,
        id: &str,
        grant: &crate::ReplayGrant,
        cached: &[crate::PacketId],
        budget: &mut usize,
    ) -> Result<(bool, bool)> {
        for packet_id in grant
            .packets
            .iter()
            .filter(|packet| !cached.contains(packet))
        {
            if *budget == 0 {
                return Ok((true, false));
            }
            *budget -= 1;
            let packet = runtime
                .outbox
                .packet(id, packet_id.clone())?
                .ok_or(Error::Denied)?;
            let ack = self
                .recovery_exchange(
                    RecoveryAction::Publish {
                        grant: grant.clone(),
                        packet,
                    },
                    now()?,
                )
                .await?;
            match ack.status {
                RecoveryStatus::Cached | RecoveryStatus::Recorded => {
                    runtime.outbox.record_ack(&ack)?;
                }
                RecoveryStatus::CapacityFull
                | RecoveryStatus::Expired
                | RecoveryStatus::Missing => {
                    return Ok((true, false));
                }
                _ => return Ok((true, true)),
            }
        }
        Ok((false, false))
    }
    /// Recover old output, then execute one fresh durable dispatch with live encrypted deltas.
    pub async fn run_durable_once<S: JobControlStore + Send + 'static>(
        &self,
        mut runtime: DurableRelayRuntime<S>,
        provider: Arc<dyn ModelProvider>,
    ) -> (DurableRelayRuntime<S>, Result<Option<JobReceipt>>) {
        let recovery = match now() {
            Ok(t) => self.recover_pending(&mut runtime, t).await,
            Err(e) => Err(e),
        };
        match recovery {
            Ok(s) if s.refused == 0 && s.pending == 0 => {}
            Ok(_) => return (runtime, Err(Error::Unavailable)),
            Err(e) => return (runtime, Err(e)),
        }
        let preflight = self.durable_preflight().await;
        let (dispatch, policy, reply) = match preflight {
            Ok(Some(p)) => p,
            Ok(None) => return (runtime, Ok(None)),
            Err(e) => return (runtime, Err(e)),
        };
        let Some(permit) = reply.permit else {
            return (runtime, Ok(reply.receipt));
        };
        let id = dispatch.submission.binding.job_id.clone();
        if let Err(e) = runtime.outbox.prepare(&dispatch, &policy) {
            return (runtime, Err(e));
        }
        let recorder = match runtime.outbox.cancellation_recorder(&id) {
            Ok(r) => r,
            Err(e) => return (runtime, Err(e)),
        };
        let cancel = CancellationToken::new();
        let _guard = CancelOnDrop(cancel.clone());
        let worker_cancel = cancel.clone();
        let (send, mut receive) =
            tokio::sync::mpsc::channel::<(OutboxPacket, mpsc::SyncSender<()>)>(1);
        let mut task = spawn_execution(
            runtime,
            dispatch,
            policy,
            permit,
            provider,
            worker_cancel,
            send,
        );
        let mut recorder = Some(recorder);
        let mut interval = tokio::time::interval(Duration::from_millis(250));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut frames_open = true;
        let mut persistence_failed = false;
        let finished = loop {
            tokio::select! {
                done=&mut task=>break done,
                item=receive.recv(),if frames_open=>{
                    if let Some((OutboxPacket::Delta{frame},ack))=item {
                        let response=self.exchange(NodeAction::Stream{job_id:id.clone(),frame},now().unwrap_or(i64::MAX/1000)).await;
                        if response.as_ref().is_ok_and(|r|r.receipt.as_ref().is_some_and(|r|matches!(r.state.as_str(),"cancel_requested"|"cancelled"))) {
                            persistence_failed |= !persist_cancel(&mut recorder).await;
                        }
                        if !response.as_ref().is_ok_and(|r|r.receipt.as_ref().is_some_and(|r|r.state=="dispatched")){cancel.cancel();}
                        let _=ack.send(());
                    }else{frames_open=false;}
                },
                _=interval.tick(),if !cancel.is_cancelled()=>{
                    match self.exchange(NodeAction::Inspect{job_id:id.clone()},now().unwrap_or(i64::MAX/1000)).await {
                        Ok(r) if r.receipt.as_ref().is_some_and(|r|r.state=="dispatched")=>{},
                        Ok(r)=>{
                            if r.receipt.as_ref().is_some_and(|r|matches!(r.state.as_str(),"cancel_requested"|"cancelled")) {
                                persistence_failed |= !persist_cancel(&mut recorder).await;
                            }
                            cancel.cancel();
                        },
                        Err(_)=>cancel.cancel(),
                    }
                }
            }
        };
        let (mut runtime, result) = match finished {
            Ok(value) => value,
            Err(error) => std::panic::resume_unwind(Box::new(error)),
        };
        drop(recorder);
        if persistence_failed {
            runtime.unresolved_live = true;
            return (runtime, Err(Error::Denied));
        }
        if let Err(e) = result {
            return (runtime, Err(e));
        }
        let outcome = self.finish_durable(&mut runtime, &id).await;
        (runtime, outcome)
    }
    async fn durable_preflight(
        &self,
    ) -> Result<
        Option<(
            crate::Dispatch,
            crate::SignedDeliveryPolicy,
            crate::NodeReply,
        )>,
    > {
        let reply = self.exchange(NodeAction::Poll, now()?).await?;
        let Some(dispatch) = reply.dispatch else {
            return Ok(None);
        };
        let policy = reply.delivery.ok_or(Error::Denied)?;
        let reply = self
            .exchange(
                NodeAction::Inspect {
                    job_id: dispatch.submission.binding.job_id.clone(),
                },
                now()?,
            )
            .await?;
        Ok(Some((dispatch, policy, reply)))
    }
    async fn finish_durable<S: JobControlStore + Send + 'static>(
        &self,
        runtime: &mut DurableRelayRuntime<S>,
        id: &str,
    ) -> Result<Option<JobReceipt>> {
        let summary = self.recover_pending(runtime, now()?).await?;
        if summary.refused > 0 {
            return Err(Error::Denied);
        }
        if summary.pending > 0 {
            return Err(Error::Unavailable);
        }
        Ok(self
            .exchange(NodeAction::Inspect { job_id: id.into() }, now()?)
            .await?
            .receipt)
    }
}
/// Bounded exponential reconnect delay with OS-backed jitter; never retries inference.
#[must_use]
pub fn reconnect_delay(attempt: u32) -> Duration {
    use rand::Rng as _;
    Duration::from_millis(
        (250u64.saturating_mul(1u64 << attempt.min(5))).min(5000)
            + rand::rng().random_range(0..=250),
    )
}

async fn persist_cancel(recorder: &mut Option<crate::delivery_outbox::CancelRecorder>) -> bool {
    match recorder.take() {
        Some(mut rec) => tokio::task::spawn_blocking(move || rec.observe())
            .await
            .is_ok_and(|r| r.is_ok()),
        None => true,
    }
}

type PacketSender = tokio::sync::mpsc::Sender<(OutboxPacket, mpsc::SyncSender<()>)>;
type DurableOutcome<S> = (DurableRelayRuntime<S>, Result<()>);
fn spawn_execution<S: JobControlStore + Send + 'static>(
    mut runtime: DurableRelayRuntime<S>,
    dispatch: crate::Dispatch,
    policy: crate::SignedDeliveryPolicy,
    permit: crate::DispatchPermit,
    provider: Arc<dyn ModelProvider>,
    worker_cancel: CancellationToken,
    send: PacketSender,
) -> tokio::task::JoinHandle<DurableOutcome<S>> {
    tokio::task::spawn_blocking(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.execute_durable_packets(
                &dispatch,
                &policy,
                &permit,
                provider.as_ref(),
                || now().unwrap_or(i64::MAX / 1000),
                &worker_cancel,
                |_, packet| {
                    let (ack, wait) = mpsc::sync_channel(1);
                    send.blocking_send((packet, ack))
                        .map_err(|_| Error::Unavailable)?;
                    wait.recv_timeout(Duration::from_secs(16))
                        .map_err(|_| Error::Unavailable)
                },
            )
        }))
        .unwrap_or(Err(Error::Denied));
        (runtime, result)
    })
}
