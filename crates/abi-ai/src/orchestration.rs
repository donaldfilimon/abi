//! Multi-agent orchestration: custom worker specs and their aggregated results.
//!
//! Ported from `src/features/ai/orchestration.zig` (recovered via
//! `git show 34c35d5^:src/features/ai/orchestration.zig`, the commit before the
//! Rust rewrite landed) plus `runAgent` from `src/features/ai/mod.zig`.
//!
//! This module is **pure**: no scheduler, no store, no I/O. The Zig original
//! threaded a `*Scheduler` and a `submitFn` through every entry point so worker
//! tasks could be queued, but each task body was itself pure — `runAgent` routed
//! the persona, audited, and formatted a string. Keeping the scheduler in
//! `abi-cli` (where the CLI already owns one) and leaving the worker body here
//! preserves the observable output while keeping this crate free of an
//! `abi-core` dependency.

use std::fmt::Write as _;

use crate::constitution::validate;
use crate::identity::{AgentProfile, profile_contract};
use crate::router::{analyze_sentiment, select_best_profile};

/// A capability a worker is hinted to use.
///
/// Deliberately a closed set, matching Zig's `AgentToolHint`: an unrecognized
/// hint in a `--workers` spec is a usage error rather than a silently ignored
/// string, so a typo cannot quietly produce a worker with no tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentToolHint {
    /// Planning only.
    Plan,
    /// Read-only exploration.
    Explore,
    /// Browser orchestration.
    Browser,
}

impl AgentToolHint {
    /// The hint's canonical label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Plan => "plan",
            Self::Explore => "explore",
            Self::Browser => "browser",
        }
    }

    /// Parse a hint label. `None` for anything outside the closed set.
    #[must_use]
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "plan" => Some(Self::Plan),
            "explore" => Some(Self::Explore),
            "browser" => Some(Self::Browser),
            _ => None,
        }
    }
}

/// One worker in a spawn batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentWorkerSpec {
    /// Worker name, used in the task name and the result banner.
    pub name: String,
    /// Free-text instructions echoed into the worker's output.
    pub instructions: String,
    /// Whether the worker only plans. Zig defaulted this to `true`.
    pub dry_run: bool,
    /// Force a persona instead of routing by sentiment.
    pub profile_override: Option<AgentProfile>,
    /// Declared capabilities.
    pub tool_hints: Vec<AgentToolHint>,
}

impl AgentWorkerSpec {
    /// A dry-run worker with no forced persona and no hints.
    #[must_use]
    pub fn new(name: impl Into<String>, instructions: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            instructions: instructions.into(),
            dry_run: true,
            profile_override: None,
            tool_hints: Vec::new(),
        }
    }

    /// Set the declared capabilities.
    #[must_use]
    pub fn with_hints(mut self, hints: impl IntoIterator<Item = AgentToolHint>) -> Self {
        self.tool_hints = hints.into_iter().collect();
        self
    }

    /// Force a persona.
    #[must_use]
    pub fn with_profile(mut self, profile: AgentProfile) -> Self {
        self.profile_override = Some(profile);
        self
    }
}

/// Why a `--workers` spec was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerSpecError {
    /// The spec was empty, had no usable worker segment, a segment was missing
    /// its `name` or `instructions` field, or the batch exceeded
    /// [`MAX_WORKER_COUNT`].
    ///
    /// Zig collapsed all of these into one `error.InvalidWorkerSpec`, and both
    /// map to the same CLI usage error, so they stay collapsed here.
    InvalidWorkerSpec,
    /// A hint was not one of `plan`, `explore`, `browser`.
    InvalidToolHint {
        /// The rejected label.
        value: String,
    },
}

impl std::fmt::Display for WorkerSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWorkerSpec => f.write_str("invalid worker spec"),
            Self::InvalidToolHint { value } => write!(f, "unknown tool hint {value:?}"),
        }
    }
}

impl std::error::Error for WorkerSpecError {}

/// Largest worker fan-out a single spawn accepts.
///
/// A hard cap, matching Zig: without it a pathological `--workers` string could
/// queue unbounded work from one command line.
pub const MAX_WORKER_COUNT: usize = 32;

/// Render tool hints for display, or `"none"` when there are none.
#[must_use]
pub fn format_tool_hints(hints: &[AgentToolHint]) -> String {
    if hints.is_empty() {
        return "none".to_string();
    }
    hints
        .iter()
        .map(|hint| hint.label())
        .collect::<Vec<_>>()
        .join(",")
}

/// Parse a comma-separated hint list. An empty list is valid (no hints).
///
/// Blank entries are skipped rather than rejected, so `plan,,explore` parses —
/// matching Zig's `if (trimmed.len == 0) continue`.
fn parse_tool_hints_csv(csv: &str) -> Result<Vec<AgentToolHint>, WorkerSpecError> {
    let mut hints = Vec::new();
    for part in csv.split(',') {
        let trimmed = part.trim_matches([' ', '\t']);
        if trimmed.is_empty() {
            continue;
        }
        let hint =
            AgentToolHint::parse(trimmed).ok_or_else(|| WorkerSpecError::InvalidToolHint {
                value: trimmed.to_string(),
            })?;
        hints.push(hint);
    }
    Ok(hints)
}

/// Parse a `--workers` spec into worker definitions.
///
/// Grammar, verbatim from Zig: workers are separated by `;`, and each worker is
/// `name|instructions|hints` with `|` between fields and `,` between hints. The
/// hints field is optional; `name` and `instructions` are required and must be
/// non-empty after trimming. Fields past the third are ignored, which is what
/// Zig's three sequential `fields.next()` calls did.
///
/// Empty `;` segments are skipped, so a trailing `;` is tolerated — but a spec
/// that yields no workers at all is an error, not an empty batch.
pub fn parse_worker_specs(spec_text: &str) -> Result<Vec<AgentWorkerSpec>, WorkerSpecError> {
    if spec_text.is_empty() {
        return Err(WorkerSpecError::InvalidWorkerSpec);
    }

    let mut workers = Vec::new();
    for segment in spec_text.split(';') {
        let trimmed = segment.trim_matches([' ', '\t']);
        if trimmed.is_empty() {
            continue;
        }

        let mut fields = trimmed.split('|');
        let name = fields
            .next()
            .ok_or(WorkerSpecError::InvalidWorkerSpec)?
            .trim_matches([' ', '\t']);
        let instructions = fields
            .next()
            .ok_or(WorkerSpecError::InvalidWorkerSpec)?
            .trim_matches([' ', '\t']);
        let hints_csv = fields.next().unwrap_or("").trim_matches([' ', '\t']);

        if name.is_empty() || instructions.is_empty() {
            return Err(WorkerSpecError::InvalidWorkerSpec);
        }

        workers.push(
            AgentWorkerSpec::new(name, instructions).with_hints(parse_tool_hints_csv(hints_csv)?),
        );
    }

    if workers.is_empty() || workers.len() > MAX_WORKER_COUNT {
        return Err(WorkerSpecError::InvalidWorkerSpec);
    }
    Ok(workers)
}

/// The worker `abi agent spawn` uses when `--workers` is absent.
#[must_use]
pub fn default_spawn_spec() -> AgentWorkerSpec {
    AgentWorkerSpec::new("smart-agent", "General-purpose planning and exploration.")
        .with_hints([AgentToolHint::Plan, AgentToolHint::Explore])
}

/// The fixed three-persona roster behind `abi agent multi`.
///
/// Instructions come from each persona's identity contract rather than being
/// restated here, so the roster cannot drift from the contracts.
#[must_use]
pub fn default_trio_specs() -> Vec<AgentWorkerSpec> {
    [
        (
            AgentProfile::Abbey,
            vec![AgentToolHint::Explore, AgentToolHint::Plan],
        ),
        (AgentProfile::Aviva, vec![AgentToolHint::Plan]),
        (AgentProfile::Abi, vec![AgentToolHint::Plan]),
    ]
    .into_iter()
    .map(|(profile, hints)| {
        AgentWorkerSpec::new(profile.label(), profile_contract(profile).description)
            .with_profile(profile)
            .with_hints(hints)
    })
    .collect()
}

/// One worker's rendered result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentResult {
    /// The formatted worker report.
    pub output: String,
    /// Whether a human must review before acting.
    pub requires_review: bool,
}

/// Run one worker: route a persona, audit, and format the report.
///
/// Ported from Zig's `runAgent`. `requires_review` is true when the worker is
/// not a dry run **or** the constitutional audit failed — so a governance
/// failure escalates even a dry run, which is the conservative direction.
///
/// Returns `None` for an empty name, empty instructions, or empty input, matching
/// Zig's `error.InvalidAgentConfig` guard.
#[must_use]
pub fn run_agent(spec: &AgentWorkerSpec, input: &str) -> Option<AgentResult> {
    if spec.name.is_empty() || spec.instructions.is_empty() || input.is_empty() {
        return None;
    }

    let mode = if spec.dry_run {
        "dry-run"
    } else {
        "review-required"
    };
    let selected = spec
        .profile_override
        .unwrap_or_else(|| select_best_profile(analyze_sentiment(input)));
    let response = crate::route_to_profile(selected, input);
    let audit = validate(&response);
    let requires_review = !spec.dry_run || !audit.passed;

    let mut output = String::new();
    let _ = writeln!(output, "agent={}", spec.name);
    let _ = writeln!(output, "mode={mode}");
    let _ = writeln!(output, "selected_profile={}", selected.label());
    let _ = writeln!(
        output,
        "review_required={}",
        if requires_review { "true" } else { "false" }
    );
    let _ = writeln!(output, "tool_hints={}", format_tool_hints(&spec.tool_hints));
    let _ = writeln!(output, "instructions={}", spec.instructions);
    // No trailing newline: Zig's format string ended at `response={s}`, and the
    // aggregator adds the separator.
    let _ = write!(output, "response={response}");

    Some(AgentResult {
        output,
        requires_review,
    })
}

/// A worker's result paired with its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedAgentResult {
    /// The worker's name.
    pub name: String,
    /// Its result.
    pub result: AgentResult,
}

/// The aggregated outcome of a custom multi-agent batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomMultiAgentResult {
    /// Per-worker results, in spec order.
    pub results: Vec<NamedAgentResult>,
    /// The rendered banner block.
    pub aggregated: String,
}

/// Run every worker in `specs` against `input` and aggregate the reports.
///
/// The banner shape is contract — `tools/contract_cli/complete_through_wdbx.sh`
/// and `agent_orchestration.sh` both assert on it:
///
/// ```text
/// === CUSTOM MULTI-AGENT RESULTS ===
///
/// [name]
/// <worker report>
///
/// === END ===
/// ```
///
/// Returns `None` if `specs` is empty, exceeds [`MAX_WORKER_COUNT`], or any
/// worker fails [`run_agent`]'s guard.
#[must_use]
pub fn run_custom_multi_agent(
    specs: &[AgentWorkerSpec],
    input: &str,
) -> Option<CustomMultiAgentResult> {
    if specs.is_empty() || specs.len() > MAX_WORKER_COUNT {
        return None;
    }

    let mut results = Vec::with_capacity(specs.len());
    let mut aggregated = String::from("=== CUSTOM MULTI-AGENT RESULTS ===\n");
    for spec in specs {
        let result = run_agent(spec, input)?;
        let _ = write!(aggregated, "\n[{}]\n{}\n", spec.name, result.output);
        results.push(NamedAgentResult {
            name: spec.name.clone(),
            result,
        });
    }
    aggregated.push_str("\n=== END ===");

    Some(CustomMultiAgentResult {
        results,
        aggregated,
    })
}

/// A local browser-orchestration plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrowserOrchestrationPlan {
    /// The rendered plan.
    pub output: String,
    /// Always true: a browser task is never auto-approved.
    pub requires_review: bool,
    /// Whether the caller asked to execute rather than plan.
    pub execute_requested: bool,
}

/// Plan a browser task locally, without launching a browser.
///
/// Ported from `src/features/ai/browser_plan.zig`. This records a structured task
/// spec and nothing else — **no embedded headless browser**, and no navigation or
/// credential access. Real automation is delegated to an external MCP Playwright
/// peer, which is why `embedded_browser=false` and `delegation_hint` are emitted
/// as explicit fields rather than left implied: they are the honesty boundary,
/// and `tools/contract_cli/` asserts on them.
///
/// `requires_review` is unconditionally true. `execute_confirmed` only changes
/// the reported mode and the closing step's wording — it never makes this
/// function navigate anywhere.
///
/// Returns `None` for an empty task, matching Zig's `error.InvalidAgentConfig`.
#[must_use]
pub fn plan_browser_orchestration(
    task: &str,
    url: Option<&str>,
    execute_confirmed: bool,
) -> Option<BrowserOrchestrationPlan> {
    if task.is_empty() {
        return None;
    }

    let mode = if execute_confirmed {
        "execute-requested"
    } else {
        "dry-run"
    };
    // Emitted only when a URL was supplied — Zig wrote an empty string
    // otherwise rather than inventing a target.
    let url_line = url.map_or_else(String::new, |value| format!("target_url={value}\n"));
    let closing = if execute_confirmed {
        "execute path requires explicit --confirm and an external browser MCP; ABI does not launch a headless browser in-process"
    } else {
        "dry-run only — no navigation or credential access"
    };

    let mut output = String::new();
    let _ = writeln!(output, "orchestration=browser-local");
    let _ = writeln!(output, "mode={mode}");
    let _ = writeln!(output, "review_required=true");
    let _ = writeln!(output, "embedded_browser=false");
    let _ = writeln!(output, "delegation_hint=external-mcp-playwright");
    let _ = writeln!(output, "policy=loopback-credentials-user-consent");
    let _ = writeln!(output, "tool_hints_enforced=false");
    output.push_str(&url_line);
    let _ = writeln!(output, "task={task}");
    let _ = writeln!(output, "steps:");
    let _ = writeln!(output, "  1. record structured browser task spec locally");
    let _ = writeln!(
        output,
        "  2. record tool_hints in agent output (constitution does not consume hints today)"
    );
    let _ = writeln!(
        output,
        "  3. recommended next step: delegate to external MCP Playwright peer (not performed by ABI)"
    );
    let _ = writeln!(output, "  4. {closing}");

    Some(BrowserOrchestrationPlan {
        output,
        requires_review: true,
        execute_requested: execute_confirmed,
    })
}

/// The single worker `abi agent browser` runs alongside its plan.
#[must_use]
pub fn browser_planner_spec() -> AgentWorkerSpec {
    AgentWorkerSpec::new(
        "browser-planner",
        "Plan safe browser steps; never access credentials.",
    )
    .with_hints([AgentToolHint::Browser, AgentToolHint::Plan])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_zig_reference_spec() {
        // Verbatim from Zig's "parse worker specs and tool hints" test.
        let specs =
            parse_worker_specs("scout|Explore pages|explore,browser;lead|Plan rollout|plan")
                .expect("valid spec");
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "scout");
        assert_eq!(specs[0].tool_hints.len(), 2);
        assert_eq!(specs[0].tool_hints[1], AgentToolHint::Browser);
        assert_eq!(specs[1].name, "lead");
        assert_eq!(specs[1].tool_hints, vec![AgentToolHint::Plan]);
    }

    #[test]
    fn a_spec_without_hints_is_valid() {
        // Verbatim from Zig's "parse worker specs without hints frees safely".
        let specs = parse_worker_specs("scout|Explore pages").expect("valid spec");
        assert_eq!(specs.len(), 1);
        assert!(specs[0].tool_hints.is_empty());
        assert_eq!(format_tool_hints(&specs[0].tool_hints), "none");
    }

    #[test]
    fn rejects_fan_out_above_the_cap() {
        // Verbatim from Zig's max_worker_count test.
        let over = (0..=MAX_WORKER_COUNT)
            .map(|i| format!("w{i}|instructions"))
            .collect::<Vec<_>>()
            .join(";");
        assert_eq!(
            parse_worker_specs(&over),
            Err(WorkerSpecError::InvalidWorkerSpec)
        );

        let at_cap = (0..MAX_WORKER_COUNT)
            .map(|i| format!("w{i}|instructions"))
            .collect::<Vec<_>>()
            .join(";");
        assert_eq!(
            parse_worker_specs(&at_cap)
                .expect("the cap itself is allowed")
                .len(),
            MAX_WORKER_COUNT
        );
    }

    #[test]
    fn rejects_malformed_specs() {
        for bad in [
            "",              // empty
            ";",             // no usable segment
            "   ",           // whitespace only
            "scout",         // missing instructions
            "|instructions", // empty name
            "scout|",        // empty instructions
            "scout| ",       // whitespace-only instructions
        ] {
            assert_eq!(
                parse_worker_specs(bad),
                Err(WorkerSpecError::InvalidWorkerSpec),
                "{bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn rejects_an_unknown_tool_hint_by_name() {
        // A typo must be a usage error, not a silently hint-less worker.
        assert_eq!(
            parse_worker_specs("scout|Explore|explor"),
            Err(WorkerSpecError::InvalidToolHint {
                value: "explor".to_string()
            })
        );
    }

    #[test]
    fn tolerates_a_trailing_separator_and_blank_hints() {
        let specs = parse_worker_specs("scout|Explore|plan,,explore;").expect("valid spec");
        assert_eq!(specs.len(), 1);
        assert_eq!(
            specs[0].tool_hints,
            vec![AgentToolHint::Plan, AgentToolHint::Explore]
        );
    }

    #[test]
    fn extra_fields_past_hints_are_ignored() {
        // Zig read exactly three fields and dropped the rest.
        let specs = parse_worker_specs("scout|Explore|plan|ignored|also").expect("valid spec");
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].tool_hints, vec![AgentToolHint::Plan]);
    }

    #[test]
    fn run_agent_renders_the_zig_field_order() {
        let spec =
            AgentWorkerSpec::new("scout", "Explore safely").with_hints([AgentToolHint::Explore]);
        let result = run_agent(&spec, "inspect docs").expect("valid worker");
        let lines: Vec<&str> = result.result_lines();
        assert_eq!(lines[0], "agent=scout");
        assert_eq!(lines[1], "mode=dry-run");
        assert!(lines[2].starts_with("selected_profile="));
        assert_eq!(lines[3], "review_required=false");
        assert_eq!(lines[4], "tool_hints=explore");
        assert_eq!(lines[5], "instructions=Explore safely");
        assert!(lines[6].starts_with("response="));
        assert!(!result.requires_review);
    }

    #[test]
    fn a_non_dry_run_worker_requires_review() {
        let mut spec = AgentWorkerSpec::new("doer", "Act now");
        spec.dry_run = false;
        let result = run_agent(&spec, "ship it").expect("valid worker");
        assert!(result.requires_review);
        assert!(result.output.contains("mode=review-required"));
        assert!(result.output.contains("review_required=true"));
    }

    #[test]
    fn a_profile_override_bypasses_sentiment_routing() {
        // "execute deploy run" routes to Aviva by sentiment; the override wins.
        let spec = AgentWorkerSpec::new("forced", "Be warm").with_profile(AgentProfile::Abbey);
        let result = run_agent(&spec, "execute deploy run the build").expect("valid worker");
        assert!(result.output.contains("selected_profile=abbey"));
    }

    #[test]
    fn run_agent_rejects_empty_config_or_input() {
        assert!(run_agent(&AgentWorkerSpec::new("", "x"), "in").is_none());
        assert!(run_agent(&AgentWorkerSpec::new("n", ""), "in").is_none());
        assert!(run_agent(&AgentWorkerSpec::new("n", "x"), "").is_none());
    }

    #[test]
    fn aggregate_matches_the_asserted_banner_shape() {
        let specs = parse_worker_specs("scout|Explore safely|explore").expect("valid spec");
        let batch = run_custom_multi_agent(&specs, "inspect docs").expect("valid batch");
        assert!(
            batch
                .aggregated
                .starts_with("=== CUSTOM MULTI-AGENT RESULTS ===\n")
        );
        assert!(batch.aggregated.contains("\n[scout]\n"));
        assert!(batch.aggregated.ends_with("\n=== END ==="));
        assert_eq!(batch.results.len(), 1);
        assert_eq!(batch.results[0].name, "scout");
    }

    #[test]
    fn aggregate_preserves_spec_order_across_workers() {
        let specs = parse_worker_specs("first|One;second|Two;third|Three").expect("valid spec");
        let batch = run_custom_multi_agent(&specs, "task").expect("valid batch");
        let names: Vec<&str> = batch.results.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["first", "second", "third"]);
        let first = batch.aggregated.find("[first]").expect("present");
        let second = batch.aggregated.find("[second]").expect("present");
        let third = batch.aggregated.find("[third]").expect("present");
        assert!(first < second && second < third);
    }

    #[test]
    fn aggregate_rejects_an_empty_or_oversized_batch() {
        assert!(run_custom_multi_agent(&[], "task").is_none());
        let too_many: Vec<AgentWorkerSpec> = (0..=MAX_WORKER_COUNT)
            .map(|i| AgentWorkerSpec::new(format!("w{i}"), "x"))
            .collect();
        assert!(run_custom_multi_agent(&too_many, "task").is_none());
    }

    #[test]
    fn the_default_spawn_worker_matches_zig() {
        let spec = default_spawn_spec();
        assert_eq!(spec.name, "smart-agent");
        assert_eq!(
            spec.instructions,
            "General-purpose planning and exploration."
        );
        assert_eq!(
            spec.tool_hints,
            vec![AgentToolHint::Plan, AgentToolHint::Explore]
        );
    }

    #[test]
    fn the_trio_roster_draws_instructions_from_identity_contracts() {
        let trio = default_trio_specs();
        assert_eq!(trio.len(), 3);
        for (spec, profile) in trio.iter().zip(crate::identity::KNOWN_PROFILES) {
            assert_eq!(spec.name, profile.label());
            assert_eq!(spec.profile_override, Some(profile));
            assert_eq!(spec.instructions, profile_contract(profile).description);
        }
    }

    #[test]
    fn tool_hint_labels_round_trip() {
        for hint in [
            AgentToolHint::Plan,
            AgentToolHint::Explore,
            AgentToolHint::Browser,
        ] {
            assert_eq!(AgentToolHint::parse(hint.label()), Some(hint));
        }
        assert_eq!(AgentToolHint::parse("nope"), None);
        assert_eq!(format_tool_hints(&[]), "none");
        assert_eq!(
            format_tool_hints(&[AgentToolHint::Plan, AgentToolHint::Browser]),
            "plan,browser"
        );
    }

    impl AgentResult {
        /// Test helper: the report split into lines.
        fn result_lines(&self) -> Vec<&str> {
            self.output.lines().collect()
        }
    }
}
