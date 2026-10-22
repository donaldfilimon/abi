//! Provider/tool continuation loop.

use crate::{
    HostBudget, HostBudgetLimit, HostError, HostRunReport, HostStopReason, ToolExecutionContext,
    ToolExecutor, ToolOutput,
    schema::{ToolSchemas, bounded},
};
use abi_agent_runtime::{
    AuditSink, CancellationToken, CapturedText, EventSink, ExecutionPolicy, Message, ModelEvent,
    ModelProvider, ModelRequest, PolicyDecision, Role, RunBudget, StopReason, ToolCall,
    ToolRegistry, ToolResult, ToolStatus, Usage, authorize_and_audit, run_provider,
};
use serde_json::json;
use std::{collections::HashSet, time::Instant};

/// Internal governed boundary sharing the legacy bounded continuation driver.
pub(crate) trait CallHook {
    fn rejected(&mut self, error: &HostError);
    fn call(
        &mut self,
        call: &ToolCall,
        context: ToolExecutionContext<'_>,
        limit: usize,
    ) -> Option<ToolResult>;
}

/// Startup-owned composition of provider, tools, policy, audit, and execution.
pub struct AgentHost<'a> {
    provider: &'a dyn ModelProvider,
    schemas: ToolSchemas,
    policy: &'a dyn ExecutionPolicy,
    audit: &'a dyn AuditSink,
    executor: &'a dyn ToolExecutor,
}

impl<'a> AgentHost<'a> {
    /// Compile the registry's schemas and assemble a host.
    ///
    /// Duplicate names and invalid schemas fail here, before a model can run or
    /// a policy can authorize anything.
    pub fn new(
        provider: &'a dyn ModelProvider,
        registry: &dyn ToolRegistry,
        policy: &'a dyn ExecutionPolicy,
        audit: &'a dyn AuditSink,
        executor: &'a dyn ToolExecutor,
    ) -> Result<Self, HostError> {
        Ok(Self {
            provider,
            schemas: ToolSchemas::compile(registry.specs())?,
            policy,
            audit,
            executor,
        })
    }

    /// Run the provider/tool loop under one shared cancellation token and
    /// finite host budget.
    pub fn run(
        &self,
        request: &ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        budget: HostBudget,
    ) -> Result<HostRunReport, HostError> {
        self.run_with_hook(request, sink, cancellation, budget, &mut None)
    }

    pub(crate) fn run_with_hook(
        &self,
        request: &ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        budget: HostBudget,
        hook: &mut Option<&mut dyn CallHook>,
    ) -> Result<HostRunReport, HostError> {
        let started = Instant::now();
        let deadline = started
            .checked_add(budget.max_duration)
            .ok_or(HostError::BudgetExceeded(HostBudgetLimit::Deadline))
            .inspect_err(|error| {
                if let Some(hook) = hook.as_deref_mut() {
                    hook.rejected(error);
                }
            })?;
        let mut state = RunState::new(budget.max_output_bytes);
        let mut continuation = rebuild_request(request, self.schemas.specs());
        let result = self.drive(
            &mut continuation,
            sink,
            cancellation,
            budget,
            deadline,
            &mut state,
            hook,
        );

        match result {
            Ok(stop) => {
                sink.emit(&ModelEvent::Finished {
                    stop: runtime_stop(stop),
                });
                Ok(state.report(stop, started.elapsed()))
            }
            Err(error) => {
                if let Some(hook) = hook.as_deref_mut() {
                    hook.rejected(&error);
                }
                sink.emit(&ModelEvent::Finished {
                    stop: StopReason::Failed,
                });
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn drive(
        &self,
        request: &mut ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        budget: HostBudget,
        deadline: Instant,
        state: &mut RunState,
        hook: &mut Option<&mut dyn CallHook>,
    ) -> Result<HostStopReason, HostError> {
        loop {
            if !check_boundary(cancellation, deadline)? {
                return Ok(HostStopReason::Cancelled);
            }
            admit(
                state.provider_runs,
                budget.max_provider_runs,
                HostBudgetLimit::ProviderRuns,
            )?;
            state.provider_runs = state.provider_runs.saturating_add(1);

            let mut turn_sink = TurnSink::new(budget.max_event_bytes);
            let provider_budget = remaining_provider_budget(budget, deadline, state);
            let provider_report = run_provider(
                self.provider,
                request,
                &mut turn_sink,
                cancellation,
                provider_budget,
            )
            .map_err(HostError::Provider)?;
            turn_sink.finish()?;
            validate_terminal_sequence(turn_sink.events())?;
            state.usage = state.usage.merged(provider_report.usage);
            let provider_stop = provider_admission_stop(
                provider_report.stop,
                state.usage.output_tokens,
                budget.max_output_tokens,
            );

            let mut calls_this_turn = 0_u64;
            for event in turn_sink.events() {
                if !check_boundary(cancellation, deadline)? {
                    return Ok(HostStopReason::Cancelled);
                }
                match event {
                    ModelEvent::Finished { .. } => {}
                    ModelEvent::ToolResult(result) => {
                        return Err(HostError::ProviderToolResult {
                            call_id: result.call_id.clone(),
                        });
                    }
                    ModelEvent::ToolCall(call) => {
                        // A completed turn's usage/stop is known before buffered
                        // calls are admitted. Retain prior text, but never
                        // authorize or execute new work after that boundary.
                        if let Some(stop) = provider_stop {
                            return Ok(stop);
                        }
                        if calls_this_turn == 0 {
                            admit(
                                state.tool_rounds,
                                budget.max_tool_rounds,
                                HostBudgetLimit::ToolRounds,
                            )?;
                            state.tool_rounds = state.tool_rounds.saturating_add(1);
                        }
                        calls_this_turn = calls_this_turn.saturating_add(1);
                        if !self.handle_call(
                            call,
                            request,
                            sink,
                            cancellation,
                            budget,
                            deadline,
                            state,
                            hook,
                        )? {
                            return Ok(HostStopReason::Cancelled);
                        }
                    }
                    ModelEvent::TextDelta { text } => {
                        let next = state.output_bytes.saturating_add(text.len());
                        if next > budget.max_output_bytes {
                            return Ok(HostStopReason::BudgetExhausted(
                                HostBudgetLimit::OutputBytes,
                            ));
                        }
                        state.output_bytes = next;
                        state.output.push_str(text);
                        emit_bounded(sink, event, state, budget.max_events)?;
                    }
                    _ => emit_bounded(sink, event, state, budget.max_events)?,
                }
            }

            if cancellation.is_cancelled() {
                return Ok(HostStopReason::Cancelled);
            }
            if let Some(stop) = provider_stop {
                return Ok(stop);
            }
            if calls_this_turn == 0 {
                return Ok(HostStopReason::Completed);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_call(
        &self,
        call: &ToolCall,
        request: &mut ModelRequest,
        sink: &mut dyn EventSink,
        cancellation: &CancellationToken,
        budget: HostBudget,
        deadline: Instant,
        state: &mut RunState,
        hook: &mut Option<&mut dyn CallHook>,
    ) -> Result<bool, HostError> {
        if !check_boundary(cancellation, deadline)? {
            return Ok(true);
        }
        admit(
            state.tool_calls,
            budget.max_tool_calls,
            HostBudgetLimit::ToolCalls,
        )?;
        if !state.call_ids.insert(call.id.clone()) {
            return Err(HostError::DuplicateCallId {
                call_id: call.id.clone(),
            });
        }
        // Governed hook performs its own staged schema admission.
        let tool = if hook.is_none() {
            Some(self.schemas.validate(call)?)
        } else {
            None
        };
        admit_many(state.events, budget.max_events, 2, HostBudgetLimit::Events)?;
        emit_bounded(
            sink,
            &ModelEvent::ToolCall(call.clone()),
            state,
            budget.max_events,
        )?;
        state.tool_calls = state.tool_calls.saturating_add(1);
        if !check_boundary(cancellation, deadline)? {
            return Ok(true);
        }

        let result = if let Some(hook) = hook.as_deref_mut() {
            let Some(result) = hook.call(
                call,
                ToolExecutionContext {
                    cancellation,
                    deadline,
                },
                budget.max_tool_result_bytes,
            ) else {
                return Ok(false);
            };
            result
        } else {
            let tool = tool.expect("legacy call validated");
            let decision = authorize_and_audit(self.policy, self.audit, call, Some(&tool.spec));
            if !check_boundary(cancellation, deadline)? {
                return Ok(true);
            }
            match decision {
                PolicyDecision::Allow => {
                    let output = self.executor.execute(
                        call,
                        &tool.spec,
                        ToolExecutionContext {
                            cancellation,
                            deadline,
                        },
                    );
                    if !check_boundary(cancellation, deadline)? {
                        return Ok(true);
                    }
                    output_to_result(call, output, budget.max_tool_result_bytes)?
                }
                PolicyDecision::RequireConfirmation { reason } => ToolResult::with_limit(
                    call.id.clone(),
                    ToolStatus::Denied,
                    &format!("confirmation required: {reason}"),
                    budget.max_tool_result_bytes,
                ),
                PolicyDecision::Deny { reason } => ToolResult::with_limit(
                    call.id.clone(),
                    ToolStatus::Denied,
                    &reason,
                    budget.max_tool_result_bytes,
                ),
            }
        };
        emit_bounded(
            sink,
            &ModelEvent::ToolResult(result.clone()),
            state,
            budget.max_events,
        )?;
        *request = request
            .clone()
            .with_message(tool_result_message(call, &result));
        state.tool_results.push(result);
        Ok(true)
    }
}

struct RunState {
    usage: Usage,
    events: u64,
    output_bytes: usize,
    tool_calls: u64,
    tool_rounds: u64,
    provider_runs: u64,
    output: CapturedText,
    tool_results: Vec<ToolResult>,
    call_ids: HashSet<String>,
}

struct TurnSink {
    events: Vec<ModelEvent>,
    max_event_bytes: usize,
    rejected: Option<HostError>,
}

impl TurnSink {
    fn new(max_event_bytes: usize) -> Self {
        Self {
            events: Vec::new(),
            max_event_bytes,
            rejected: None,
        }
    }

    fn events(&self) -> &[ModelEvent] {
        &self.events
    }

    fn finish(&mut self) -> Result<(), HostError> {
        match self.rejected.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

impl EventSink for TurnSink {
    fn emit(&mut self, event: &ModelEvent) {
        if self.rejected.is_some() {
            return;
        }
        if event_bytes(event) > self.max_event_bytes {
            self.rejected = Some(HostError::BudgetExceeded(HostBudgetLimit::EventBytes));
            return;
        }
        // A provider cannot attest to a tool's outcome. Reject the entire
        // buffered turn before an earlier call in that turn can execute.
        if let ModelEvent::ToolResult(result) = event {
            self.rejected = Some(HostError::ProviderToolResult {
                call_id: result.call_id.clone(),
            });
            return;
        }
        self.events.push(event.clone());
    }
}

impl RunState {
    fn new(output_limit: usize) -> Self {
        Self {
            usage: Usage::zero(),
            events: 0,
            output_bytes: 0,
            tool_calls: 0,
            tool_rounds: 0,
            provider_runs: 0,
            output: CapturedText::with_limit(output_limit),
            tool_results: Vec::new(),
            call_ids: HashSet::new(),
        }
    }

    fn report(&self, stop: HostStopReason, elapsed: std::time::Duration) -> HostRunReport {
        let mut usage = self.usage;
        usage.duration = elapsed;
        HostRunReport {
            stop,
            usage,
            events: self.events,
            tool_calls: self.tool_calls,
            provider_runs: self.provider_runs,
            output: self.output.clone(),
            tool_results: self.tool_results.clone(),
        }
    }
}

fn rebuild_request(request: &ModelRequest, specs: &[abi_agent_runtime::ToolSpec]) -> ModelRequest {
    let mut rebuilt = ModelRequest::new(request.model());
    if let Some(system) = request.system() {
        rebuilt = rebuilt.with_system(system);
    }
    for message in request.messages() {
        rebuilt = rebuilt.with_message(message.clone());
    }
    for spec in specs {
        rebuilt = rebuilt.with_tool(spec.clone());
    }
    if let Some(tokens) = request.max_output_tokens() {
        rebuilt = rebuilt.with_max_output_tokens(tokens);
    }
    rebuilt
}

fn tool_result_message(call: &ToolCall, result: &ToolResult) -> Message {
    Message::new(
        Role::Tool,
        json!({
            "call_id": call.id,
            "tool": call.name,
            "status": result.status.as_str(),
            "payload": result.payload.text(),
        })
        .to_string(),
    )
}

fn output_to_result(
    call: &ToolCall,
    output: Result<ToolOutput, crate::ToolExecutionError>,
    limit: usize,
) -> Result<ToolResult, HostError> {
    let (status, payload) = match output {
        Ok(output) => (output.status(), output.payload().to_owned()),
        Err(error) => (ToolStatus::Error, bounded(error.to_string())),
    };
    if payload.len() > limit {
        return Err(HostError::ToolResultTooLarge {
            call_id: call.id.clone(),
            bytes: payload.len(),
            limit,
        });
    }
    Ok(ToolResult::with_limit(
        call.id.clone(),
        status,
        &payload,
        limit,
    ))
}

fn validate_terminal_sequence(events: &[ModelEvent]) -> Result<(), HostError> {
    let Some((last, prefix)) = events.split_last() else {
        return Err(HostError::InvalidTerminalSequence);
    };
    if !last.is_terminal() {
        return Err(HostError::InvalidTerminalSequence);
    }
    let mut saw_terminal = false;
    for event in prefix {
        if saw_terminal {
            return Err(HostError::PostTerminalEvent {
                kind: event.kind().to_owned(),
            });
        }
        saw_terminal = event.is_terminal();
    }
    if saw_terminal {
        return Err(HostError::PostTerminalEvent {
            kind: last.kind().to_owned(),
        });
    }
    Ok(())
}

fn emit_bounded(
    sink: &mut dyn EventSink,
    event: &ModelEvent,
    state: &mut RunState,
    max_events: u64,
) -> Result<(), HostError> {
    admit(state.events, max_events, HostBudgetLimit::Events)?;
    sink.emit(event);
    state.events = state.events.saturating_add(1);
    Ok(())
}

fn admit(current: u64, maximum: u64, limit: HostBudgetLimit) -> Result<(), HostError> {
    if current >= maximum {
        Err(HostError::BudgetExceeded(limit))
    } else {
        Ok(())
    }
}

fn admit_many(
    current: u64,
    maximum: u64,
    needed: u64,
    limit: HostBudgetLimit,
) -> Result<(), HostError> {
    if maximum.saturating_sub(current) < needed {
        Err(HostError::BudgetExceeded(limit))
    } else {
        Ok(())
    }
}

fn event_bytes(event: &ModelEvent) -> usize {
    match event {
        ModelEvent::Started { model } => model.len(),
        ModelEvent::TextDelta { text } => text.len(),
        ModelEvent::ToolCall(call) => call
            .id
            .len()
            .saturating_add(call.name.len())
            .saturating_add(call.input.len()),
        ModelEvent::ToolResult(result) => result
            .call_id
            .len()
            .saturating_add(result.payload.text().len()),
        ModelEvent::Finished { .. } => 0,
        _ => usize::MAX,
    }
}

fn check_deadline(deadline: Instant) -> Result<(), HostError> {
    if Instant::now() >= deadline {
        Err(HostError::BudgetExceeded(HostBudgetLimit::Deadline))
    } else {
        Ok(())
    }
}

// A callback can cancel the shared token before returning. Observe that before
// admitting more work, with the same cancellation-first precedence as run entry.
fn check_boundary(cancellation: &CancellationToken, deadline: Instant) -> Result<bool, HostError> {
    if cancellation.is_cancelled() {
        return Ok(false);
    }
    check_deadline(deadline)?;
    Ok(true)
}

fn remaining_provider_budget(budget: HostBudget, deadline: Instant, state: &RunState) -> RunBudget {
    RunBudget::unlimited()
        .with_max_events(budget.max_events.saturating_sub(state.events))
        .with_max_output_tokens(
            budget
                .max_output_tokens
                .saturating_sub(state.usage.output_tokens),
        )
        .with_max_duration(deadline.saturating_duration_since(Instant::now()))
}

fn provider_admission_stop(
    stop: StopReason,
    output_tokens: u64,
    max_output_tokens: u64,
) -> Option<HostStopReason> {
    match stop {
        StopReason::Cancelled => Some(HostStopReason::Cancelled),
        StopReason::BudgetExhausted(limit) => Some(HostStopReason::BudgetExhausted(match limit {
            abi_agent_runtime::BudgetLimit::Events => HostBudgetLimit::Events,
            abi_agent_runtime::BudgetLimit::OutputTokens => HostBudgetLimit::OutputTokens,
            abi_agent_runtime::BudgetLimit::Duration => HostBudgetLimit::Deadline,
        })),
        _ if output_tokens >= max_output_tokens => Some(HostStopReason::BudgetExhausted(
            HostBudgetLimit::OutputTokens,
        )),
        _ => None,
    }
}

fn runtime_stop(stop: HostStopReason) -> StopReason {
    match stop {
        HostStopReason::Completed => StopReason::Completed,
        HostStopReason::Cancelled => StopReason::Cancelled,
        HostStopReason::BudgetExhausted(limit) => StopReason::BudgetExhausted(match limit {
            HostBudgetLimit::OutputTokens => abi_agent_runtime::BudgetLimit::OutputTokens,
            HostBudgetLimit::Deadline => abi_agent_runtime::BudgetLimit::Duration,
            HostBudgetLimit::Events
            | HostBudgetLimit::EventBytes
            | HostBudgetLimit::OutputBytes
            | HostBudgetLimit::ToolCalls
            | HostBudgetLimit::ToolRounds
            | HostBudgetLimit::ProviderRuns => abi_agent_runtime::BudgetLimit::Events,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{HostBudgetLimit, HostError, check_boundary};
    use abi_agent_runtime::CancellationToken;
    use std::time::Instant;

    #[test]
    fn observed_cancellation_takes_precedence_over_expired_deadline() {
        let deadline = Instant::now();
        let cancellation = CancellationToken::new();
        assert_eq!(
            check_boundary(&cancellation, deadline),
            Err(HostError::BudgetExceeded(HostBudgetLimit::Deadline))
        );
        cancellation.cancel();
        assert_eq!(check_boundary(&cancellation, deadline), Ok(false));
    }
}
