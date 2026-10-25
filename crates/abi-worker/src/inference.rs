//! Bounded inference execution after authenticated durable admission.

use crate::{
    Admission, CancellationToken, JobControlStore, JobEnvelope, JobLease, JobResult, JobStatus,
    ResultChunk, WorkerError, WorkerManager,
};
use abi_agent_runtime::{
    EventSink, ModelEvent, ModelProvider, ModelRequest, RunBudget, StopReason, run_provider,
};
use serde::Deserialize;
use std::time::Duration;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InferenceInput {
    prompt: String,
    tokens: u64,
}

/// The result of inference submission; only a fresh admission invokes a provider.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InferenceOutcome {
    /// Newly executed, durably completed inference.
    Completed(JobResult),
    /// Exact retry returned the previously committed result.
    Cached(JobResult),
    /// A prior admission exists; no provider was invoked or automatically resumed.
    InProgress(JobLease),
}

impl<S: JobControlStore> WorkerManager<S> {
    /// Execute one inference with a startup-bound provider and trusted clock.
    ///
    /// The caller must bind the provider to the immutable capability revision
    /// at startup. That revision is the exact model identifier sent to it.
    /// Only `model.infer` / `infer` accepts `{"prompt": string, "tokens": u64}`.
    /// This path offers no tools and never executes provider tool requests.
    /// Limits and cancellation are cooperative at provider checkpoints; an
    /// untrusted or noncooperative provider requires process isolation.
    /// The trusted clock is re-read before dispatch and completion. Late or
    /// uncertain completion is not retried. Only `Admission::Leased` dispatches.
    pub fn infer(
        &mut self,
        envelope: &JobEnvelope,
        provider: &dyn ModelProvider,
        clock: impl FnMut() -> i64,
    ) -> Result<InferenceOutcome, WorkerError> {
        self.infer_with_authority(envelope, provider, clock, |_| Ok(()))
    }

    /// Execute with a fresh authority guard after durable admission and before provider dispatch.
    /// A refusal leaves the durable admission unresolved; retry never repeats execution.
    /// Cached results and existing leases do not invoke the guard or provider.
    pub fn infer_with_authority(
        &mut self,
        envelope: &JobEnvelope,
        provider: &dyn ModelProvider,
        mut clock: impl FnMut() -> i64,
        mut authority: impl FnMut(i64) -> Result<(), WorkerError>,
    ) -> Result<InferenceOutcome, WorkerError> {
        self.infer_observed(
            envelope,
            provider,
            &mut clock,
            &mut authority,
            &CancellationToken::new(),
            |_| Ok(()),
        )
    }

    /// Execute with caller cancellation and a bounded synchronous chunk observer.
    /// Observer refusal cancels execution; callers must provide their own bounded transport.
    pub fn infer_observed(
        &mut self,
        envelope: &JobEnvelope,
        provider: &dyn ModelProvider,
        mut clock: impl FnMut() -> i64,
        mut authority: impl FnMut(i64) -> Result<(), WorkerError>,
        cancel: &CancellationToken,
        mut observer: impl FnMut(&ResultChunk) -> Result<(), WorkerError> + Send,
    ) -> Result<InferenceOutcome, WorkerError> {
        let admitted_at = clock();
        let lease = match self.submit(envelope, admitted_at)? {
            Admission::Leased(lease) => lease,
            Admission::ExistingLease(lease) => return Ok(InferenceOutcome::InProgress(lease)),
            Admission::Cached(result) => return Ok(InferenceOutcome::Cached(result)),
        };
        let started_at = clock();
        let expires = lease.expires_at_unix_ms.min(envelope.deadline_unix_ms);
        if started_at < admitted_at || started_at >= expires {
            return Err(WorkerError::Deadline(
                "lease expired or clock regressed before dispatch".into(),
            ));
        }
        authority(started_at)?;
        // Authority lookup can block. Its admission-time timestamp does not
        // authorize dispatch after the lease expires or the clock regresses.
        let dispatch_at = clock();
        if dispatch_at < started_at || dispatch_at >= expires {
            return Err(WorkerError::Deadline(
                "lease expired or clock regressed during authority check".into(),
            ));
        }
        let limits = self.execution_limits();
        // Caller cancellation is cooperative and failure-terminal in local history.
        // It never fabricates a signed worker cancellation record.
        let mut sink = BoundedOutput {
            model: &envelope.request.capability_revision,
            chunks: Vec::new(),
            bytes: 0,
            max_bytes: limits
                .result_bytes
                .min(usize::try_from(envelope.request.max_result_bytes).unwrap_or(usize::MAX)),
            max_chunks: limits.stream_chunks.min(envelope.request.max_stream_chunks),
            refused: false,
            cancel: cancel.clone(),
            observer: &mut observer,
        };
        let input = serde_json::from_value::<InferenceInput>(envelope.request.input.clone());
        let status = match input {
            Ok(input)
                if envelope.request.capability == "model.infer"
                    && envelope.request.operation == "infer"
                    && input.tokens > 0 =>
            {
                let request = ModelRequest::new(&envelope.request.capability_revision)
                    .with_user(input.prompt)
                    .with_max_output_tokens(input.tokens);
                let budget = RunBudget::unlimited()
                    .with_max_events(u64::from(sink.max_chunks) + 2)
                    .with_max_output_tokens(input.tokens)
                    .with_max_duration(Duration::from_millis(
                        u64::try_from(expires.saturating_sub(dispatch_at)).unwrap_or(0),
                    ));
                let report = (!cancel.is_cancelled())
                    .then(|| run_provider(provider, &request, &mut sink, cancel, budget));
                if !sink.refused
                    && report.is_some_and(|r| {
                        r.is_ok_and(|r| {
                            r.stop == StopReason::Completed && r.usage.output_tokens <= input.tokens
                        })
                    })
                {
                    JobStatus::Succeeded
                } else {
                    JobStatus::Failed
                }
            }
            _ => JobStatus::Failed,
        };
        let completed_at = clock();
        if completed_at < dispatch_at {
            return Err(WorkerError::Deadline(
                "clock regressed during inference".into(),
            ));
        }
        // Errors and provider diagnostics are never copied into durable result bytes.
        let result = self.complete(&lease, status, sink.chunks, completed_at)?;
        Ok(InferenceOutcome::Completed(result))
    }
}

struct BoundedOutput<'a, F> {
    model: &'a str,
    chunks: Vec<ResultChunk>,
    bytes: usize,
    max_bytes: usize,
    max_chunks: u32,
    refused: bool,
    cancel: CancellationToken,
    observer: &'a mut F,
}

impl<F: FnMut(&ResultChunk) -> Result<(), WorkerError> + Send> EventSink for BoundedOutput<'_, F> {
    fn emit(&mut self, event: &ModelEvent) {
        if self.refused || self.cancel.is_cancelled() {
            return;
        }
        match event {
            ModelEvent::Started { model } if model == self.model => {}
            ModelEvent::Finished { .. } => {}
            ModelEvent::TextDelta { text } if text.is_empty() => {}
            ModelEvent::TextDelta { text }
                if text.len() <= self.max_bytes.saturating_sub(self.bytes)
                    && self.chunks.len() < self.max_chunks as usize =>
            {
                let sequence = u32::try_from(self.chunks.len()).unwrap_or(u32::MAX);
                self.bytes += text.len();
                let chunk = ResultChunk {
                    sequence,
                    bytes: text.as_bytes().to_vec(),
                };
                if (self.observer)(&chunk).is_err() {
                    self.refused = true;
                    self.cancel.cancel();
                    return;
                }
                self.chunks.push(chunk);
            }
            _ => {
                self.refused = true;
                self.cancel.cancel();
            }
        }
    }
}
