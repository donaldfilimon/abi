//! Driving one run: the checkpoint where cancellation, budgets, and bounded
//! capture are enforced.

use crate::{
    cancel::CancellationToken,
    capture::{CapturedText, MAX_CAPTURED_BYTES},
    error::RuntimeError,
    event::{EventSink, ModelEvent, StopReason},
    provider::ModelProvider,
    request::ModelRequest,
    usage::{RunBudget, Usage},
};
use std::time::Instant;

/// Whether a provider should keep going after delivering an event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Keep producing events.
    Continue,
    /// Stop; the run has been cancelled or has exhausted its budget.
    Stop,
}

impl Flow {
    /// Whether the provider must stop.
    #[must_use]
    pub fn is_stop(self) -> bool {
        matches!(self, Self::Stop)
    }
}

/// What one run produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunReport {
    /// Why the run ended.
    pub stop: StopReason,
    /// Provider-reported token counts plus measured wall time.
    pub usage: Usage,
    /// Events delivered to the sink, excluding the terminal event.
    ///
    /// A provider's leading [`ModelEvent::Started`] is one of them, so this is
    /// not a count of content events.
    pub events: u64,
    /// Bounded copy of the streamed assistant text.
    pub output: CapturedText,
}

impl RunReport {
    /// Whether the provider ran to its own end.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.stop.is_complete()
    }

    /// Retained assistant text. May be truncated; see [`CapturedText::truncated`].
    #[must_use]
    pub fn text(&self) -> &str {
        self.output.text()
    }
}

/// The handle a provider writes through for the duration of one run.
///
/// Every event passes through [`RunContext::checkpoint`], which is the single
/// checkpoint for cancellation and budgets. Providers that perform a bounded
/// blocking step may call it directly from their polling loop without emitting
/// a synthetic event:
///
/// 1. **Cancellation** — the token is read at each checkpoint, including before
///    each event. Work already inside a provider step is not interrupted unless
///    that provider adds bounded polling checkpoints.
/// 2. **Budget** — [`RunBudget`] ceilings are evaluated at the same checkpoints.
/// 3. **Bounded capture** — text is copied into a [`CapturedText`] buffer that
///    stops at [`MAX_CAPTURED_BYTES`], while the event itself is still
///    delivered to the sink in full.
///
/// Once stopped, the context stays stopped: further events are dropped rather
/// than delivered, so a provider that ignores a [`Flow::Stop`] cannot leak
/// output past the boundary.
pub struct RunContext<'a> {
    sink: &'a mut dyn EventSink,
    cancel: &'a CancellationToken,
    budget: RunBudget,
    started: Instant,
    events: u64,
    input_tokens: u64,
    output_tokens: u64,
    capture: CapturedText,
    stop: Option<StopReason>,
}

impl<'a> RunContext<'a> {
    /// Begin a run that writes to `sink`, watching `cancel` and `budget`.
    ///
    /// Capture is bounded by [`MAX_CAPTURED_BYTES`].
    pub fn new(
        sink: &'a mut dyn EventSink,
        cancel: &'a CancellationToken,
        budget: RunBudget,
    ) -> Self {
        Self::with_capture_limit(sink, cancel, budget, MAX_CAPTURED_BYTES)
    }

    /// Begin a run with an explicit capture ceiling.
    pub fn with_capture_limit(
        sink: &'a mut dyn EventSink,
        cancel: &'a CancellationToken,
        budget: RunBudget,
        capture_limit: usize,
    ) -> Self {
        Self {
            sink,
            cancel,
            budget,
            started: Instant::now(),
            events: 0,
            input_tokens: 0,
            output_tokens: 0,
            capture: CapturedText::with_limit(capture_limit),
            stop: None,
        }
    }

    /// Deliver one event, or stop the run.
    ///
    /// Returns [`Flow::Stop`] when the run has been cancelled or has reached a
    /// budget ceiling; in that case `event` is **not** delivered.
    pub fn emit(&mut self, event: &ModelEvent) -> Flow {
        if self.checkpoint().is_stop() {
            return Flow::Stop;
        }
        if let Some(text) = event.as_text() {
            self.capture.push_str(text);
        }
        self.sink.emit(event);
        self.events = self.events.saturating_add(1);
        Flow::Continue
    }

    /// Check cancellation and run budgets without delivering an event.
    ///
    /// Process-backed and network-backed providers should call this from their
    /// bounded polling loops. Once it returns [`Flow::Stop`], the context stays
    /// stopped and the provider must return promptly. This is an observation
    /// boundary only: it does not expose the cancellation token or let a
    /// provider manufacture a cancellation request.
    pub fn checkpoint(&mut self) -> Flow {
        if self.stop.is_some() {
            return Flow::Stop;
        }
        if self.cancel.is_cancelled() {
            self.stop = Some(StopReason::Cancelled);
            return Flow::Stop;
        }
        if let Some(limit) =
            self.budget
                .exceeded(self.events, self.output_tokens, self.started.elapsed())
        {
            self.stop = Some(StopReason::BudgetExhausted(limit));
            return Flow::Stop;
        }
        Flow::Continue
    }

    /// Record prompt tokens as reported by the provider.
    pub fn record_input_tokens(&mut self, tokens: u64) {
        self.input_tokens = self.input_tokens.saturating_add(tokens);
    }

    /// Record completion tokens as reported by the provider.
    ///
    /// The output-token budget is evaluated against this running total at the
    /// next event boundary.
    pub fn record_output_tokens(&mut self, tokens: u64) {
        self.output_tokens = self.output_tokens.saturating_add(tokens);
    }

    /// Events delivered so far, excluding the terminal event.
    #[must_use]
    pub fn events(&self) -> u64 {
        self.events
    }

    /// Whether the run has already stopped.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.stop.is_some()
    }

    /// Budget in force for this run.
    #[must_use]
    pub fn budget(&self) -> RunBudget {
        self.budget
    }

    /// Bounded capture accumulated so far.
    #[must_use]
    pub fn captured(&self) -> &CapturedText {
        &self.capture
    }

    /// End the run, delivering the terminal [`ModelEvent::Finished`].
    #[must_use]
    pub fn finish(self) -> RunReport {
        let stop = self.stop.unwrap_or(StopReason::Completed);
        self.close(stop)
    }

    /// End a run whose provider returned an error.
    ///
    /// The terminal event still reaches the sink, so a consumer never sees a
    /// stream that simply stops.
    #[must_use]
    pub fn finish_failed(self) -> RunReport {
        self.close(StopReason::Failed)
    }

    fn close(self, stop: StopReason) -> RunReport {
        self.sink.emit(&ModelEvent::Finished { stop });
        RunReport {
            stop,
            usage: Usage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                duration: self.started.elapsed(),
            },
            events: self.events,
            output: self.capture,
        }
    }
}

/// Drive `provider` for one request and return what it produced.
///
/// This is the canonical entry point: it builds the [`RunContext`], hands it to
/// the provider, and closes it either way, so the terminal event is delivered
/// even when the provider fails.
pub fn run_provider(
    provider: &dyn ModelProvider,
    request: &ModelRequest,
    sink: &mut dyn EventSink,
    cancel: &CancellationToken,
    budget: RunBudget,
) -> Result<RunReport, RuntimeError> {
    let mut run = RunContext::new(sink, cancel, budget);
    match provider.run(request, &mut run) {
        Ok(()) => Ok(run.finish()),
        Err(error) => {
            let _report = run.finish_failed();
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Flow, RunContext, run_provider};
    use crate::{
        cancel::CancellationToken,
        event::{CollectingSink, ModelEvent, StopReason},
        provider::ModelProvider,
        request::ModelRequest,
        testing::ScriptedProvider,
        usage::{BudgetLimit, RunBudget},
    };

    #[test]
    fn emit_delivers_and_counts() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let mut run = RunContext::new(&mut sink, &cancel, RunBudget::unlimited());
        assert_eq!(run.emit(&ModelEvent::text("a")), Flow::Continue);
        run.record_input_tokens(4);
        run.record_output_tokens(1);
        assert_eq!(run.events(), 1);
        assert!(!run.is_stopped());
        assert_eq!(run.captured().text(), "a");
        assert_eq!(run.budget(), RunBudget::unlimited());

        let report = run.finish();
        assert_eq!(report.stop, StopReason::Completed);
        assert!(report.is_complete());
        assert_eq!(report.events, 1);
        assert_eq!(report.usage.input_tokens, 4);
        assert_eq!(report.usage.output_tokens, 1);
        assert_eq!(report.text(), "a");
        // The terminal event is delivered but not counted in `events`.
        assert_eq!(sink.kinds(), vec!["text_delta", "finished"]);
    }

    #[test]
    fn a_stopped_context_drops_later_events() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let mut run = RunContext::new(&mut sink, &cancel, RunBudget::unlimited());
        cancel.cancel();
        assert!(run.emit(&ModelEvent::text("a")).is_stop());
        assert!(run.emit(&ModelEvent::text("b")).is_stop());
        let report = run.finish();
        assert_eq!(report.stop, StopReason::Cancelled);
        assert_eq!(report.events, 0);
        assert_eq!(sink.kinds(), vec!["finished"]);
    }

    #[test]
    fn checkpoint_observes_cancellation_without_emitting_an_event() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let mut run = RunContext::new(&mut sink, &cancel, RunBudget::unlimited());
        assert_eq!(run.checkpoint(), Flow::Continue);
        cancel.cancel();
        assert_eq!(run.checkpoint(), Flow::Stop);
        assert_eq!(run.checkpoint(), Flow::Stop);
        let report = run.finish();
        assert_eq!(report.stop, StopReason::Cancelled);
        assert_eq!(report.events, 0);
        assert_eq!(sink.kinds(), vec!["finished"]);
    }

    #[test]
    fn checkpoint_observes_an_event_budget_without_counting_itself() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let mut run = RunContext::new(
            &mut sink,
            &cancel,
            RunBudget::unlimited().with_max_events(1),
        );
        assert_eq!(run.checkpoint(), Flow::Continue);
        assert_eq!(run.emit(&ModelEvent::text("one")), Flow::Continue);
        assert_eq!(run.checkpoint(), Flow::Stop);
        let report = run.finish();
        assert_eq!(
            report.stop,
            StopReason::BudgetExhausted(BudgetLimit::Events)
        );
        assert_eq!(report.events, 1);
        assert_eq!(sink.text(), "one");
    }

    #[test]
    fn capture_limit_bounds_the_report_but_not_the_sink() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let mut run = RunContext::with_capture_limit(&mut sink, &cancel, RunBudget::unlimited(), 3);
        assert_eq!(run.emit(&ModelEvent::text("abcdef")), Flow::Continue);
        let report = run.finish();
        assert_eq!(report.text(), "abc");
        assert!(report.output.truncated());
        assert_eq!(report.output.dropped_bytes(), 3);
        // The sink still saw the whole chunk.
        assert_eq!(sink.text(), "abcdef");
    }

    #[test]
    fn output_token_budget_stops_at_the_next_boundary() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let mut run = RunContext::new(
            &mut sink,
            &cancel,
            RunBudget::unlimited().with_max_output_tokens(2),
        );
        assert_eq!(run.emit(&ModelEvent::text("a")), Flow::Continue);
        run.record_output_tokens(2);
        assert!(run.emit(&ModelEvent::text("b")).is_stop());
        let report = run.finish();
        assert_eq!(
            report.stop,
            StopReason::BudgetExhausted(BudgetLimit::OutputTokens)
        );
        assert_eq!(report.events, 1);
    }

    #[test]
    fn a_failing_provider_still_closes_the_stream() {
        struct FailingProvider;
        impl ModelProvider for FailingProvider {
            fn id(&self) -> &'static str {
                "failing"
            }
            fn run(
                &self,
                _request: &ModelRequest,
                run: &mut RunContext<'_>,
            ) -> Result<(), crate::error::RuntimeError> {
                run.emit(&ModelEvent::text("partial"));
                Err(crate::error::RuntimeError::Provider {
                    provider: "failing".to_owned(),
                    message: "no transport".to_owned(),
                })
            }
        }

        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let error = run_provider(
            &FailingProvider,
            &ModelRequest::new("m"),
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect_err("provider was expected to fail");
        assert!(error.to_string().contains("no transport"));
        assert_eq!(sink.kinds(), vec!["text_delta", "finished"]);
        assert_eq!(
            sink.events().last(),
            Some(&ModelEvent::Finished {
                stop: StopReason::Failed
            })
        );
    }

    #[test]
    fn run_provider_drives_a_boxed_provider() {
        let provider: Box<dyn ModelProvider> =
            Box::new(ScriptedProvider::new("scripted").with_text("hi"));
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let report = run_provider(
            provider.as_ref(),
            &ModelRequest::new("m").with_user("hello"),
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect("scripted provider runs");
        assert_eq!(report.text(), "hi");
    }
}
