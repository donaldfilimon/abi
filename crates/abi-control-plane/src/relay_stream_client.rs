//! Bounded bridge between synchronous user-node inference and outbound async relay IO.
use crate::{
    Error, JobReceipt, NodeAction, RelayClient, RelayRuntime, Result, StreamFrame, service::now,
};
use abi_agent_runtime::{CancellationToken, ModelProvider};
use abi_worker::JobControlStore;
use std::{
    sync::{Arc, mpsc},
    time::Duration,
};
struct CancelOnDrop(CancellationToken);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}
impl RelayClient {
    /// Execute one job with live encrypted frame delivery and concurrent cancellation polling.
    /// The owned runtime is returned for subsequent polls. Ambiguous IO never retries inference.
    /// Cancellation is cooperative; noncooperative providers require OS process isolation.
    pub async fn run_streaming_once<S: JobControlStore + Send + 'static>(
        &self,
        mut runtime: RelayRuntime<S>,
        provider: Arc<dyn ModelProvider>,
    ) -> (RelayRuntime<S>, Result<Option<JobReceipt>>) {
        let preflight = self.stream_preflight().await;
        let (dispatch, reply) = match preflight {
            Ok(Some(p)) => p,
            Ok(None) => return (runtime, Ok(None)),
            Err(e) => return (runtime, Err(e)),
        };
        let Some(permit) = reply.permit else {
            return (runtime, Ok(reply.receipt));
        };
        let id = dispatch.submission.binding.job_id.clone();
        let cancel = CancellationToken::new();
        let _guard = CancelOnDrop(cancel.clone());
        let worker_cancel = cancel.clone();
        let (send, mut receive) =
            tokio::sync::mpsc::channel::<(StreamFrame, mpsc::SyncSender<bool>)>(1);
        let mut task = tokio::task::spawn_blocking(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.execute_streaming(
                    &dispatch,
                    &permit,
                    provider.as_ref(),
                    || now().unwrap_or(i64::MAX / 1000),
                    &worker_cancel,
                    |frame| {
                        let (ack, wait) = mpsc::sync_channel(1);
                        send.blocking_send((frame, ack))
                            .map_err(|_| Error::Denied)?;
                        if wait.recv_timeout(Duration::from_secs(16)).unwrap_or(false) {
                            Ok(())
                        } else {
                            Err(Error::Denied)
                        }
                    },
                )
            }))
            .unwrap_or(Err(Error::Denied));
            (runtime, result)
        });
        let mut interval = tokio::time::interval(Duration::from_millis(250));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut stopped = false;
        let mut transport_failed = false;
        let mut frames_open = true;
        let completed = loop {
            tokio::select! {
                result=&mut task => break result,
                frame=receive.recv(),if frames_open => {
                    if let Some((frame,ack))=frame {
                        let response=self.exchange(NodeAction::Stream {job_id:id.clone(),frame},now().unwrap_or(i64::MAX/1000)).await;
                        let accepted=response.as_ref().is_ok_and(|r|r.receipt.as_ref().is_some_and(|r|r.state=="dispatched"));
                        if !accepted {cancel.cancel();transport_failed=true;}
                        let _=ack.send(accepted);
                    } else {frames_open=false;}
                },
                _=interval.tick(),if !cancel.is_cancelled()=>{
                    match self.exchange(NodeAction::Inspect {job_id:id.clone()},now().unwrap_or(i64::MAX/1000)).await {
                        Ok(reply) if reply.receipt.as_ref().is_some_and(|r|r.state=="dispatched")=>{},
                        Ok(reply)=>{stopped=reply.receipt.as_ref().is_some_and(|r|r.state=="cancel_requested");cancel.cancel();},
                        Err(_)=>{transport_failed=true;cancel.cancel();},
                    }
                }
            }
        };
        // Provider panics were contained above, leaving admission unresolved.
        // A panic outside that boundary cannot return the moved runtime.
        let (runtime, result) = match completed {
            Ok(r) => r,
            Err(e) => std::panic::resume_unwind(Box::new(e)),
        };
        let outcome = self
            .finish_stream(
                &id,
                stopped,
                transport_failed,
                cancel.is_cancelled(),
                result,
            )
            .await;
        (runtime, outcome)
    }
    async fn stream_preflight(&self) -> Result<Option<(crate::Dispatch, crate::NodeReply)>> {
        let reply = self.exchange(NodeAction::Poll, now()?).await?;
        let Some(dispatch) = reply.dispatch else {
            return Ok(None);
        };
        let job_id = dispatch.submission.binding.job_id.clone();
        let reply = self
            .exchange(NodeAction::Inspect { job_id }, now()?)
            .await?;
        Ok(Some((dispatch, reply)))
    }
    async fn finish_stream(
        &self,
        id: &str,
        mut stopped: bool,
        transport_failed: bool,
        cancelled: bool,
        result: Result<Option<crate::SealedPayload>>,
    ) -> Result<Option<JobReceipt>> {
        // A stream rejection can race the periodic inspection. Reconcile only
        // after the provider has returned, so Stopped never claims an idle live task.
        if cancelled && !stopped {
            stopped = self
                .exchange(
                    NodeAction::Inspect {
                        job_id: id.to_owned(),
                    },
                    now().unwrap_or(i64::MAX / 1000),
                )
                .await
                .is_ok_and(|r| r.receipt.is_some_and(|r| r.state == "cancel_requested"));
        }
        let action = if stopped {
            NodeAction::Stopped {
                job_id: id.to_owned(),
            }
        } else if !transport_failed {
            match result {
                Ok(Some(payload)) => NodeAction::Complete {
                    job_id: id.to_owned(),
                    payload,
                },
                _ => NodeAction::Fail {
                    job_id: id.to_owned(),
                },
            }
        } else {
            NodeAction::Fail {
                job_id: id.to_owned(),
            }
        };
        let outcome = match now() {
            Ok(time) => self.exchange(action, time).await.map(|r| r.receipt),
            Err(e) => Err(e),
        };
        if outcome.is_err() {
            match now() {
                Ok(time) => match self
                    .exchange(
                        NodeAction::Inspect {
                            job_id: id.to_owned(),
                        },
                        time,
                    )
                    .await
                {
                    Ok(reply)
                        if reply
                            .receipt
                            .as_ref()
                            .is_some_and(|r| r.state == "cancel_requested") =>
                    {
                        self.exchange(
                            NodeAction::Stopped {
                                job_id: id.to_owned(),
                            },
                            now().unwrap_or(i64::MAX / 1000),
                        )
                        .await
                        .map(|r| r.receipt)
                    }
                    _ => outcome,
                },
                Err(_) => outcome,
            }
        } else {
            outcome
        }
    }
}
