//! Deterministic providers for testing downstream code without a network.
//!
//! Neither provider here is a model. They exist so that a host's streaming,
//! cancellation, tool, and metering paths can be exercised end to end with a
//! fully predictable event sequence and no external dependency.

use crate::{
    error::RuntimeError,
    event::ModelEvent,
    request::ModelRequest,
    run::RunContext,
    tool::{ToolCall, ToolResult},
};

/// Provider identifier reported by [`EchoProvider`].
pub const ECHO_PROVIDER_ID: &str = "echo";

/// A provider that streams the last user message back, one word per event.
///
/// # Token counting is not tokenization
///
/// The counts this provider reports are **whitespace-separated word counts**.
/// It contains no tokenizer, and the numbers must not be compared against any
/// real model's accounting. They exist so that metering code has something
/// deterministic to assert against.
#[derive(Clone, Debug, Default)]
pub struct EchoProvider;

impl EchoProvider {
    /// Build the provider.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl crate::provider::ModelProvider for EchoProvider {
    fn id(&self) -> &'static str {
        ECHO_PROVIDER_ID
    }

    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        let echoed = request
            .last_user_message()
            .ok_or(RuntimeError::EmptyRequest)?
            .content
            .clone();

        let prompt_words: u64 = request
            .system()
            .map(word_count)
            .unwrap_or_default()
            .saturating_add(
                request
                    .messages()
                    .iter()
                    .map(|message| word_count(&message.content))
                    .fold(0, u64::saturating_add),
            );
        run.record_input_tokens(prompt_words);

        if run.emit(&ModelEvent::started(request.model())).is_stop() {
            return Ok(());
        }

        let words: Vec<&str> = echoed.split_whitespace().collect();
        for (index, word) in words.iter().enumerate() {
            let chunk = if index + 1 == words.len() {
                (*word).to_owned()
            } else {
                format!("{word} ")
            };
            if run.emit(&ModelEvent::text(chunk)).is_stop() {
                return Ok(());
            }
            run.record_output_tokens(1);
        }
        Ok(())
    }
}

/// A provider that replays a fixed event script.
///
/// The script is delivered in order after a [`ModelEvent::Started`] event.
/// Delivery stops as soon as the [`RunContext`] says to stop, so the number of
/// events a sink observes is exactly the number the run allowed.
///
/// Usage is *declared*, not derived: the input count is recorded once at the
/// start, and the per-event output count is recorded only for events that were
/// actually delivered.
#[derive(Clone, Debug, Default)]
pub struct ScriptedProvider {
    id: String,
    script: Vec<ModelEvent>,
    input_tokens: u64,
    output_tokens_per_event: u64,
    failure: Option<RuntimeError>,
}

impl ScriptedProvider {
    /// Build an empty script under the identifier `id`.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// Append one scripted event.
    #[must_use]
    pub fn with_event(mut self, event: ModelEvent) -> Self {
        self.script.push(event);
        self
    }

    /// Append a text chunk.
    #[must_use]
    pub fn with_text(self, text: impl Into<String>) -> Self {
        self.with_event(ModelEvent::text(text))
    }

    /// Append a structured tool call.
    #[must_use]
    pub fn with_tool_call(self, call: ToolCall) -> Self {
        self.with_event(ModelEvent::ToolCall(call))
    }

    /// Append a tool result.
    #[must_use]
    pub fn with_tool_result(self, result: ToolResult) -> Self {
        self.with_event(ModelEvent::ToolResult(result))
    }

    /// Declare the prompt-token count recorded at the start of the run.
    #[must_use]
    pub fn with_input_tokens(mut self, tokens: u64) -> Self {
        self.input_tokens = tokens;
        self
    }

    /// Declare the output-token count recorded per delivered scripted event.
    #[must_use]
    pub fn with_output_tokens_per_event(mut self, tokens: u64) -> Self {
        self.output_tokens_per_event = tokens;
        self
    }

    /// Fail with `error` after the whole script has been delivered.
    ///
    /// A run that was cancelled or ran out of budget mid-script returns
    /// `Ok(())` instead: a stopped run is not also a provider failure.
    #[must_use]
    pub fn with_failure(mut self, error: RuntimeError) -> Self {
        self.failure = Some(error);
        self
    }

    /// Number of scripted events, excluding the leading `Started` event.
    #[must_use]
    pub fn script_len(&self) -> usize {
        self.script.len()
    }
}

impl crate::provider::ModelProvider for ScriptedProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn run(&self, request: &ModelRequest, run: &mut RunContext<'_>) -> Result<(), RuntimeError> {
        run.record_input_tokens(self.input_tokens);
        if run.emit(&ModelEvent::started(request.model())).is_stop() {
            return Ok(());
        }
        for event in &self.script {
            if run.emit(event).is_stop() {
                return Ok(());
            }
            run.record_output_tokens(self.output_tokens_per_event);
        }
        match &self.failure {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }
}

/// Whitespace-separated word count, widened without a lossy cast.
fn word_count(text: &str) -> u64 {
    u64::try_from(text.split_whitespace().count()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::{ECHO_PROVIDER_ID, EchoProvider, ScriptedProvider, word_count};
    use crate::{
        cancel::CancellationToken,
        error::RuntimeError,
        event::{CollectingSink, ModelEvent, StopReason},
        provider::ModelProvider,
        request::ModelRequest,
        run::run_provider,
        tool::{ToolCall, ToolResult},
        usage::{BudgetLimit, RunBudget},
    };

    fn echo(prompt: &str) -> (CollectingSink, crate::run::RunReport) {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let report = run_provider(
            &EchoProvider::new(),
            &ModelRequest::new("test-model").with_user(prompt),
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect("echo provider runs");
        (sink, report)
    }

    #[test]
    fn echo_streams_one_event_per_word() {
        let (sink, report) = echo("alpha beta gamma");
        assert_eq!(
            sink.kinds(),
            vec![
                "started",
                "text_delta",
                "text_delta",
                "text_delta",
                "finished"
            ]
        );
        assert_eq!(sink.text(), "alpha beta gamma");
        assert_eq!(report.text(), "alpha beta gamma");
        assert_eq!(report.stop, StopReason::Completed);
        assert_eq!(report.events, 4);
        assert_eq!(report.usage.output_tokens, 3);
        assert_eq!(report.usage.input_tokens, 3);
        assert_eq!(EchoProvider::new().id(), ECHO_PROVIDER_ID);
    }

    #[test]
    fn echo_is_deterministic_across_runs() {
        let (first_sink, first) = echo("same input here");
        let (second_sink, second) = echo("same input here");
        assert_eq!(first_sink.events(), second_sink.events());
        assert_eq!(first.text(), second.text());
        assert_eq!(first.usage.input_tokens, second.usage.input_tokens);
        assert_eq!(first.usage.output_tokens, second.usage.output_tokens);
    }

    #[test]
    fn echo_counts_the_system_instruction_as_prompt_words() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let report = run_provider(
            &EchoProvider::new(),
            &ModelRequest::new("m")
                .with_system("be terse")
                .with_user("hi"),
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect("echo provider runs");
        assert_eq!(report.usage.input_tokens, 3);
    }

    #[test]
    fn echo_refuses_a_request_with_no_user_message() {
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let error = run_provider(
            &EchoProvider::new(),
            &ModelRequest::new("m"),
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect_err("an empty request has nothing to echo");
        assert_eq!(error, RuntimeError::EmptyRequest);
        assert_eq!(sink.kinds(), vec!["finished"]);
    }

    #[test]
    fn scripted_replays_in_order_and_counts_declared_usage() {
        let provider = ScriptedProvider::new("scripted")
            .with_text("hello ")
            .with_tool_call(ToolCall::new("c1", "read_file", "{}"))
            .with_tool_result(ToolResult::ok("c1", "contents"))
            .with_text("done")
            .with_input_tokens(11)
            .with_output_tokens_per_event(2);
        assert_eq!(provider.script_len(), 4);

        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let report = run_provider(
            &provider,
            &ModelRequest::new("m"),
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect("scripted provider runs");

        assert_eq!(
            sink.kinds(),
            vec![
                "started",
                "text_delta",
                "tool_call",
                "tool_result",
                "text_delta",
                "finished"
            ]
        );
        assert_eq!(report.text(), "hello done");
        assert_eq!(report.usage.input_tokens, 11);
        assert_eq!(report.usage.output_tokens, 8);
        assert_eq!(report.events, 5);
    }

    #[test]
    fn scripted_output_tokens_stop_the_run_at_the_declared_ceiling() {
        let provider = ScriptedProvider::new("scripted")
            .with_text("a")
            .with_text("b")
            .with_text("c")
            .with_output_tokens_per_event(2);

        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let report = run_provider(
            &provider,
            &ModelRequest::new("m"),
            &mut sink,
            &cancel,
            RunBudget::unlimited().with_max_output_tokens(3),
        )
        .expect("scripted provider runs");

        assert_eq!(
            report.stop,
            StopReason::BudgetExhausted(BudgetLimit::OutputTokens)
        );
        assert_eq!(report.text(), "ab");
        assert_eq!(report.usage.output_tokens, 4);
    }

    #[test]
    fn scripted_can_declare_a_failure_after_its_script() {
        let provider = ScriptedProvider::new("scripted")
            .with_text("partial")
            .with_failure(RuntimeError::Provider {
                provider: "scripted".to_owned(),
                message: "declared failure".to_owned(),
            });
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        let error = run_provider(
            &provider,
            &ModelRequest::new("m"),
            &mut sink,
            &cancel,
            RunBudget::unlimited(),
        )
        .expect_err("the script declares a failure");
        assert!(error.to_string().contains("declared failure"));
        assert_eq!(
            sink.events().last(),
            Some(&ModelEvent::Finished {
                stop: StopReason::Failed
            })
        );
    }

    #[test]
    fn a_scripted_run_stopped_early_is_not_also_a_failure() {
        let provider = ScriptedProvider::new("scripted")
            .with_text("a")
            .with_text("b")
            .with_failure(RuntimeError::Provider {
                provider: "scripted".to_owned(),
                message: "never reached".to_owned(),
            });
        let mut sink = CollectingSink::new();
        let cancel = CancellationToken::new();
        // The leading `Started` event spends one unit of the ceiling, so only
        // one text event is delivered before the run stops.
        let report = run_provider(
            &provider,
            &ModelRequest::new("m"),
            &mut sink,
            &cancel,
            RunBudget::unlimited().with_max_events(2),
        )
        .expect("a stopped run is not a provider failure");
        assert_eq!(
            report.stop,
            StopReason::BudgetExhausted(BudgetLimit::Events)
        );
        assert_eq!(report.events, 2);
        assert_eq!(report.text(), "a");
    }

    #[test]
    fn word_count_handles_padding_and_emptiness() {
        assert_eq!(word_count("  a   b  "), 2);
        assert_eq!(word_count(""), 0);
    }
}
