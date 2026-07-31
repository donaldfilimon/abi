//! Deterministic, explicitly bounded multiway string rewriting.
//!
//! This is a reference simulator for finite rule-space slices. It does not
//! model a complete ruliad or establish claims about fundamental physics.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

/// Frozen canonical-export format identifier.
pub const MULTIWAY_FORMAT_VERSION: &str = "abi-multiway-v1";
/// Hard ceiling on candidate rules.
pub const MAX_MULTIWAY_RULES: usize = 256;

const STATE_OVERHEAD_BYTES: u64 = 112;
const EVENT_OVERHEAD_BYTES: u64 = 28;

/// One exact byte-string rewrite rule.
#[derive(Debug, Clone, PartialEq)]
pub struct Rule {
    /// Matched byte sequence. Empty matches are rejected.
    pub lhs: Vec<u8>,
    /// Replacement byte sequence; empty means deletion.
    pub rhs: Vec<u8>,
    /// Metadata retained for future weighted traversal.
    pub weight: f64,
    /// Optional rule family metadata.
    pub family: Option<String>,
}

impl Rule {
    /// Construct an unweighted rule.
    #[must_use]
    pub fn new(lhs: impl Into<Vec<u8>>, rhs: impl Into<Vec<u8>>) -> Self {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
            weight: 1.0,
            family: None,
        }
    }

    /// Stable `lhs->rhs` bytes.
    #[must_use]
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.lhs.len() + self.rhs.len() + 2);
        bytes.extend_from_slice(&self.lhs);
        bytes.extend_from_slice(b"->");
        bytes.extend_from_slice(&self.rhs);
        bytes
    }

    /// SHA-256 of the canonical rule bytes.
    #[must_use]
    pub fn content_hash(&self) -> [u8; 32] {
        sha256(&self.canonical_bytes())
    }
}

/// Parse failure for a textual rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseRuleError {
    /// No `->` delimiter was present.
    MissingArrow,
    /// The trimmed left-hand side was empty.
    EmptyLhs,
}

impl fmt::Display for ParseRuleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingArrow => formatter.write_str("rule is missing '->'"),
            Self::EmptyLhs => formatter.write_str("rule left-hand side must not be empty"),
        }
    }
}

impl std::error::Error for ParseRuleError {}

/// Parse `lhs->rhs`, trimming ASCII spaces and tabs around each side.
pub fn parse_rule(text: &str) -> Result<Rule, ParseRuleError> {
    let (lhs, rhs) = text.split_once("->").ok_or(ParseRuleError::MissingArrow)?;
    let lhs = lhs.trim_matches([' ', '\t']);
    if lhs.is_empty() {
        return Err(ParseRuleError::EmptyLhs);
    }
    Ok(Rule::new(
        lhs.as_bytes(),
        rhs.trim_matches([' ', '\t']).as_bytes(),
    ))
}

/// The only currently supported traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Traversal {
    /// Expand all states at depth N before depth N+1.
    BreadthFirst,
}

/// The only currently supported canonicalization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Canonicalization {
    /// State identity is its exact payload.
    ExactString,
}

/// The only currently supported state deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupPolicy {
    /// SHA-256 content identity.
    ByCanonicalHash,
}

/// Complete bounded experiment configuration.
#[derive(Debug, Clone)]
pub struct MultiwayConfig {
    /// Initial payloads, deduplicated in listed order.
    pub initial: Vec<Vec<u8>>,
    /// Rules, applied in listed order.
    pub rules: Vec<Rule>,
    /// Maximum expanded depth.
    pub max_depth: u32,
    /// Maximum unique states.
    pub max_states: u32,
    /// Maximum rule-application events.
    pub max_events: u32,
    /// Maximum payload bytes, checked before replacement allocation.
    pub max_payload: u32,
    /// Wall-clock limit; zero disables it.
    pub max_duration_ms: u64,
    /// Approximate payload-plus-node budget; zero disables it.
    pub max_memory_bytes: u64,
    /// Frozen traversal selection.
    pub traversal: Traversal,
    /// Frozen canonicalization selection.
    pub canonicalization: Canonicalization,
    /// Frozen deduplication selection.
    pub dedup: DedupPolicy,
    /// Recorded for reproducibility; currently unused by deterministic BFS.
    pub seed: u64,
    /// Recorded for reproducibility; expansion remains single-threaded.
    pub workers: u32,
}

impl MultiwayConfig {
    /// Construct a configuration with the Zig oracle defaults.
    #[must_use]
    pub fn new(initial: Vec<Vec<u8>>, rules: Vec<Rule>) -> Self {
        Self {
            initial,
            rules,
            max_depth: 5,
            max_states: 10_000,
            max_events: 100_000,
            max_payload: 4_096,
            max_duration_ms: 0,
            max_memory_bytes: 0,
            traversal: Traversal::BreadthFirst,
            canonicalization: Canonicalization::ExactString,
            dedup: DedupPolicy::ByCanonicalHash,
            seed: 0,
            workers: 1,
        }
    }
}

/// Invalid experiment configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    /// No initial states were supplied.
    NoInitialStates,
    /// No rules were supplied.
    NoRules,
    /// More than [`MAX_MULTIWAY_RULES`] rules were supplied.
    TooManyRules,
    /// A rule has an empty left-hand side.
    EmptyLhs,
    /// An initial payload exceeds `max_payload`.
    InitialPayloadTooLarge,
    /// A mandatory hard bound is zero.
    ZeroBound,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ConfigError {}

/// Validate all hard preconditions.
pub fn validate_config(config: &MultiwayConfig) -> Result<(), ConfigError> {
    if config.initial.is_empty() {
        return Err(ConfigError::NoInitialStates);
    }
    if config.rules.is_empty() {
        return Err(ConfigError::NoRules);
    }
    if config.rules.len() > MAX_MULTIWAY_RULES {
        return Err(ConfigError::TooManyRules);
    }
    if config.rules.iter().any(|rule| rule.lhs.is_empty()) {
        return Err(ConfigError::EmptyLhs);
    }
    if config.max_depth == 0
        || config.max_states == 0
        || config.max_events == 0
        || config.max_payload == 0
    {
        return Err(ConfigError::ZeroBound);
    }
    let max_payload = usize::try_from(config.max_payload).unwrap_or(usize::MAX);
    if config
        .initial
        .iter()
        .any(|payload| payload.len() > max_payload)
    {
        return Err(ConfigError::InitialPayloadTooLarge);
    }
    Ok(())
}

/// Why evolution stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// No undiscovered state remained.
    FrontierExhausted,
    /// The configured depth was reached.
    MaxDepth,
    /// The next atomic expansion would exceed the state cap.
    MaxStates,
    /// The next atomic expansion would exceed the event cap.
    MaxEvents,
    /// A replacement would exceed the payload cap.
    PayloadLimit,
    /// The wall-clock budget elapsed.
    Deadline,
    /// The caller cancellation flag was set.
    Cancelled,
    /// The configured approximate memory budget would be exceeded.
    AllocationFailure,
    /// Configuration validation failed.
    InvalidRule,
    /// Internal state was inconsistent.
    InvariantFailure,
}

impl Termination {
    /// Frozen snake-case label used by canonical exports.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::FrontierExhausted => "frontier_exhausted",
            Self::MaxDepth => "max_depth",
            Self::MaxStates => "max_states",
            Self::MaxEvents => "max_events",
            Self::PayloadLimit => "payload_limit",
            Self::Deadline => "deadline",
            Self::Cancelled => "cancelled",
            Self::AllocationFailure => "allocation_failure",
            Self::InvalidRule => "invalid_rule",
            Self::InvariantFailure => "invariant_failure",
        }
    }
}

/// One canonical unique state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiwayState {
    /// Exact state payload.
    pub payload: Vec<u8>,
    /// SHA-256 payload identity.
    pub hash: [u8; 32],
    /// Minimum BFS discovery depth.
    pub depth: u32,
    /// Deterministic creation index.
    pub sequence: u32,
    /// First event that produced this state.
    pub first_event: Option<u32>,
}

/// One rule application. Events are never deduplicated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MultiwayEvent {
    /// Deterministic event index.
    pub id: u32,
    /// Source state index.
    pub source: u32,
    /// Destination state index.
    pub destination: u32,
    /// Rule index.
    pub rule: u32,
    /// Byte offset in the source payload.
    pub position: u32,
    /// Destination depth.
    pub depth: u32,
    /// Candidate index within this source expansion.
    pub local: u32,
}

/// A bounded experiment result, including a resumable cursor.
#[derive(Debug, Clone)]
pub struct MultiwayResult {
    /// Unique states in deterministic discovery order.
    pub states: Vec<MultiwayState>,
    /// All rule applications in deterministic commit order.
    pub events: Vec<MultiwayEvent>,
    /// Unique states first discovered per depth.
    pub states_per_depth: Vec<u32>,
    /// Events committed per destination depth.
    pub events_per_depth: Vec<u32>,
    /// Typed stop reason.
    pub termination: Termination,
    /// True only for exhaustive frontier termination.
    pub complete: bool,
    /// Depth currently represented by `frontier`.
    pub resume_depth: u32,
    /// Current BFS frontier.
    pub frontier: Vec<u32>,
    /// Newly discovered next-depth states.
    pub next_frontier: Vec<u32>,
    /// Next unexpanded frontier offset.
    pub cursor: u32,
    /// Time spent evolving; excluded from canonical output.
    pub elapsed: Duration,
}

impl Default for MultiwayResult {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            events: Vec::new(),
            states_per_depth: Vec::new(),
            events_per_depth: Vec::new(),
            termination: Termination::InvariantFailure,
            complete: false,
            resume_depth: 0,
            frontier: Vec::new(),
            next_frontier: Vec::new(),
            cursor: 0,
            elapsed: Duration::ZERO,
        }
    }
}

impl MultiwayResult {
    /// Find a state by canonical content hash.
    #[must_use]
    pub fn find_state(&self, hash: [u8; 32]) -> Option<u32> {
        self.states
            .iter()
            .find(|state| state.hash == hash)
            .map(|state| state.sequence)
    }
}

#[derive(Debug)]
struct Candidate {
    rule: u32,
    position: u32,
    payload: Vec<u8>,
    hash: [u8; 32],
}

struct Engine<'config> {
    config: &'config MultiwayConfig,
    result: MultiwayResult,
    index_by_hash: HashMap<[u8; 32], u32>,
    approximate_bytes: u64,
    cancel: Option<&'config AtomicBool>,
    deadline: Option<Instant>,
}

impl Engine<'_> {
    fn bump_depth_counters(&mut self, depth: u32) {
        let required = usize::try_from(depth)
            .unwrap_or(usize::MAX)
            .saturating_add(1);
        self.result.states_per_depth.resize(required, 0);
        self.result.events_per_depth.resize(required, 0);
    }

    fn seed(&mut self) -> Option<Termination> {
        self.bump_depth_counters(0);
        for payload in &self.config.initial {
            let hash = hash_payload(payload);
            if self.index_by_hash.contains_key(&hash) {
                continue;
            }
            if self.result.states.len() >= self.config.max_states as usize {
                return Some(Termination::MaxStates);
            }
            let sequence = u32::try_from(self.result.states.len()).ok()?;
            self.index_by_hash.insert(hash, sequence);
            self.result.states.push(MultiwayState {
                payload: payload.clone(),
                hash,
                depth: 0,
                sequence,
                first_event: None,
            });
            self.result.frontier.push(sequence);
            self.result.states_per_depth[0] += 1;
            self.approximate_bytes = self
                .approximate_bytes
                .saturating_add(payload.len() as u64)
                .saturating_add(STATE_OVERHEAD_BYTES);
        }
        None
    }

    fn generate_candidates(&self, source: &[u8]) -> Result<Vec<Candidate>, Termination> {
        let mut candidates = Vec::new();
        for (rule_index, rule) in self.config.rules.iter().enumerate() {
            let mut start = 0;
            while start <= source.len().saturating_sub(rule.lhs.len()) {
                let Some(relative) = source[start..]
                    .windows(rule.lhs.len())
                    .position(|window| window == rule.lhs)
                else {
                    break;
                };
                let position = start + relative;
                let destination_len = source.len() - rule.lhs.len() + rule.rhs.len();
                if destination_len > self.config.max_payload as usize {
                    return Err(Termination::PayloadLimit);
                }
                let mut payload = Vec::with_capacity(destination_len);
                payload.extend_from_slice(&source[..position]);
                payload.extend_from_slice(&rule.rhs);
                payload.extend_from_slice(&source[position + rule.lhs.len()..]);
                candidates.push(Candidate {
                    rule: u32::try_from(rule_index).map_err(|_| Termination::InvariantFailure)?,
                    position: u32::try_from(position).map_err(|_| Termination::InvariantFailure)?,
                    hash: hash_payload(&payload),
                    payload,
                });
                start = position.saturating_add(1);
            }
        }
        Ok(candidates)
    }

    fn expand_one(&mut self, source_index: u32, child_depth: u32) -> Option<Termination> {
        let source = self
            .result
            .states
            .get(source_index as usize)?
            .payload
            .clone();
        let candidates = match self.generate_candidates(&source) {
            Ok(candidates) => candidates,
            Err(termination) => return Some(termination),
        };

        if self.result.events.len().saturating_add(candidates.len())
            > self.config.max_events as usize
        {
            return Some(Termination::MaxEvents);
        }

        let mut batch_hashes = HashSet::new();
        let mut new_unique = 0usize;
        let mut payload_bytes = 0u64;
        for candidate in &candidates {
            if self.index_by_hash.contains_key(&candidate.hash)
                || !batch_hashes.insert(candidate.hash)
            {
                continue;
            }
            new_unique += 1;
            payload_bytes = payload_bytes.saturating_add(candidate.payload.len() as u64);
        }
        if self.result.states.len().saturating_add(new_unique) > self.config.max_states as usize {
            return Some(Termination::MaxStates);
        }
        if self.config.max_memory_bytes != 0 {
            let projected = self
                .approximate_bytes
                .saturating_add(payload_bytes)
                .saturating_add((new_unique as u64).saturating_mul(STATE_OVERHEAD_BYTES))
                .saturating_add((candidates.len() as u64).saturating_mul(EVENT_OVERHEAD_BYTES));
            if projected > self.config.max_memory_bytes {
                return Some(Termination::AllocationFailure);
            }
        }

        self.bump_depth_counters(child_depth);
        for (local, candidate) in candidates.into_iter().enumerate() {
            let event_id = u32::try_from(self.result.events.len()).ok()?;
            let destination = if let Some(existing) = self.index_by_hash.get(&candidate.hash) {
                *existing
            } else {
                let destination = u32::try_from(self.result.states.len()).ok()?;
                self.index_by_hash.insert(candidate.hash, destination);
                self.result.states.push(MultiwayState {
                    payload: candidate.payload.clone(),
                    hash: candidate.hash,
                    depth: child_depth,
                    sequence: destination,
                    first_event: Some(event_id),
                });
                self.result.next_frontier.push(destination);
                self.result.states_per_depth[child_depth as usize] += 1;
                self.approximate_bytes = self
                    .approximate_bytes
                    .saturating_add(candidate.payload.len() as u64)
                    .saturating_add(STATE_OVERHEAD_BYTES);
                destination
            };
            self.result.events.push(MultiwayEvent {
                id: event_id,
                source: source_index,
                destination,
                rule: candidate.rule,
                position: candidate.position,
                depth: child_depth,
                local: u32::try_from(local).ok()?,
            });
            self.result.events_per_depth[child_depth as usize] += 1;
            self.approximate_bytes = self.approximate_bytes.saturating_add(EVENT_OVERHEAD_BYTES);
        }
        None
    }

    /// Drain the current frontier at `child_depth`, one candidate at a time.
    ///
    /// Returns `Some(termination)` on cancellation, deadline expiry, a
    /// corrupt cursor, or whatever [`Engine::expand_one`] reports; `None`
    /// once every frontier entry at this depth has been expanded.
    fn process_frontier(&mut self, child_depth: u32) -> Option<Termination> {
        while usize::try_from(self.result.cursor).unwrap_or(usize::MAX) < self.result.frontier.len()
        {
            if self.cancel.is_some_and(|flag| flag.load(Ordering::Acquire)) {
                return Some(Termination::Cancelled);
            }
            if self
                .deadline
                .is_some_and(|deadline| Instant::now() >= deadline)
            {
                return Some(Termination::Deadline);
            }
            let Some(source) = self
                .result
                .frontier
                .get(self.result.cursor as usize)
                .copied()
            else {
                return Some(Termination::InvariantFailure);
            };
            if let Some(termination) = self.expand_one(source, child_depth) {
                return Some(termination);
            }
            self.result.cursor += 1;
        }
        None
    }

    fn evolve(&mut self) {
        let started = Instant::now();
        loop {
            if self.result.frontier.is_empty() {
                self.result.termination = Termination::FrontierExhausted;
                self.result.complete = true;
                break;
            }
            if self.result.resume_depth >= self.config.max_depth {
                self.result.termination = Termination::MaxDepth;
                break;
            }
            let child_depth = self.result.resume_depth + 1;
            if let Some(termination) = self.process_frontier(child_depth) {
                self.result.termination = termination;
                self.result.elapsed += started.elapsed();
                return;
            }
            self.result.frontier = std::mem::take(&mut self.result.next_frontier);
            self.result.cursor = 0;
            self.result.resume_depth += 1;
        }
        self.result.elapsed += started.elapsed();
    }
}

/// Run a bounded deterministic experiment.
///
/// Invalid configurations return a valid empty result with
/// [`Termination::InvalidRule`], matching the frozen Zig API.
#[must_use]
pub fn run_multiway(config: &MultiwayConfig, cancel: Option<&AtomicBool>) -> MultiwayResult {
    if validate_config(config).is_err() {
        return MultiwayResult {
            termination: Termination::InvalidRule,
            ..MultiwayResult::default()
        };
    }
    let deadline = (config.max_duration_ms != 0)
        .then(|| Instant::now() + Duration::from_millis(config.max_duration_ms));
    let mut engine = Engine {
        config,
        result: MultiwayResult::default(),
        index_by_hash: HashMap::new(),
        approximate_bytes: 0,
        cancel,
        deadline,
    };
    if let Some(termination) = engine.seed() {
        engine.result.termination = termination;
    } else {
        engine.evolve();
    }
    engine.result
}

/// SHA-256 content identity for a state payload.
#[must_use]
pub fn hash_payload(payload: &[u8]) -> [u8; 32] {
    sha256(payload)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// One token-lineage event dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalEdge {
    /// Event that produced a consumed byte.
    pub parent: u32,
    /// Event that consumed the byte.
    pub child: u32,
}

/// Canonical export or resume failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiwayExportError {
    /// A payload or rule is not valid UTF-8 JSON text.
    NonUtf8,
    /// The document is not a complete canonical multiway export.
    MalformedExport,
    /// The format identifier is not supported.
    UnsupportedFormat,
    /// A completed experiment has no resume cursor.
    AlreadyComplete,
    /// Initial states or ordered rules do not match.
    ConfigMismatch,
}

impl fmt::Display for MultiwayExportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MultiwayExportError {}

/// Derive event dependencies from first-writer token lineage.
#[must_use]
pub fn build_causal_edges(config: &MultiwayConfig, result: &MultiwayResult) -> Vec<CausalEdge> {
    let mut lineage: Vec<Vec<Option<u32>>> = result
        .states
        .iter()
        .map(|state| vec![None; state.payload.len()])
        .collect();
    let mut edges = Vec::new();

    for event in &result.events {
        let (Some(rule), Some(source_state), Some(destination_state)) = (
            config.rules.get(event.rule as usize),
            result.states.get(event.source as usize),
            result.states.get(event.destination as usize),
        ) else {
            continue;
        };
        let position = event.position as usize;
        if position.saturating_add(rule.lhs.len()) > source_state.payload.len() {
            continue;
        }
        let source_lineage = lineage[event.source as usize].clone();
        let mut parents = HashSet::new();
        for parent in source_lineage[position..position + rule.lhs.len()]
            .iter()
            .flatten()
        {
            if parents.insert(*parent) {
                edges.push(CausalEdge {
                    parent: *parent,
                    child: event.id,
                });
            }
        }

        if destination_state.first_event == Some(event.id) {
            let expected = source_lineage.len() - rule.lhs.len() + rule.rhs.len();
            if expected != destination_state.payload.len() {
                continue;
            }
            let mut destination_lineage = Vec::with_capacity(expected);
            destination_lineage.extend_from_slice(&source_lineage[..position]);
            destination_lineage.extend(std::iter::repeat_n(Some(event.id), rule.rhs.len()));
            destination_lineage.extend_from_slice(&source_lineage[position + rule.lhs.len()..]);
            lineage[event.destination as usize] = destination_lineage;
        }
    }
    edges
}

/// SHA-256 identity of the canonical configuration JSON.
pub fn multiway_config_hash(config: &MultiwayConfig) -> Result<[u8; 32], MultiwayExportError> {
    Ok(hash_payload(canonical_config_json(config)?.as_bytes()))
}

/// Lowercase SHA-256 of a canonical export document.
#[must_use]
pub fn multiway_export_hash_hex(export: &[u8]) -> String {
    hex_hash(hash_payload(export))
}

/// Serialize a deterministic canonical experiment document.
pub fn export_multiway_json(
    config: &MultiwayConfig,
    result: &MultiwayResult,
    metrics: &MultiwayMetrics,
) -> Result<String, MultiwayExportError> {
    let mut output = String::new();
    output.push_str("{\"format\":\"");
    output.push_str(MULTIWAY_FORMAT_VERSION);
    output.push_str("\",\"zig_version\":\"0.17.0-dev.1442+972627084\",\"config\":");
    output.push_str(&canonical_config_json(config)?);
    output.push_str(",\"config_hash\":\"");
    output.push_str(&hex_hash(multiway_config_hash(config)?));
    output.push_str("\",\"states\":[");
    for (index, state) in result.states.iter().enumerate() {
        comma(&mut output, index);
        output.push_str("{\"payload\":");
        push_json_bytes(&mut output, &state.payload)?;
        output.push_str(",\"hash\":\"");
        output.push_str(&hex_hash(state.hash));
        output.push_str("\",\"depth\":");
        output.push_str(&state.depth.to_string());
        if let Some(first_event) = state.first_event {
            output.push_str(",\"first_event\":");
            output.push_str(&first_event.to_string());
        }
        output.push('}');
    }
    output.push_str("],\"events\":[");
    for (index, event) in result.events.iter().enumerate() {
        comma(&mut output, index);
        output.push_str("{\"src\":");
        output.push_str(&event.source.to_string());
        output.push_str(",\"dst\":");
        output.push_str(&event.destination.to_string());
        output.push_str(",\"rule\":");
        output.push_str(&event.rule.to_string());
        output.push_str(",\"pos\":");
        output.push_str(&event.position.to_string());
        output.push_str(",\"depth\":");
        output.push_str(&event.depth.to_string());
        output.push_str(",\"local\":");
        output.push_str(&event.local.to_string());
        output.push('}');
    }
    output.push_str("],\"metrics\":");
    push_metrics_json(&mut output, metrics);
    output.push_str(",\"termination\":\"");
    output.push_str(result.termination.label());
    output.push_str("\",\"complete\":");
    output.push_str(if result.complete { "true" } else { "false" });
    output.push_str(",\"causal_graph\":{\"status\":\"token-lineage\",\"edges\":[");
    for (index, edge) in build_causal_edges(config, result).iter().enumerate() {
        comma(&mut output, index);
        output.push_str("{\"parent\":");
        output.push_str(&edge.parent.to_string());
        output.push_str(",\"child\":");
        output.push_str(&edge.child.to_string());
        output.push('}');
    }
    output.push_str("]}");
    if result.complete {
        output.push_str(",\"resume\":null");
    } else {
        output.push_str(",\"resume\":{\"depth\":");
        output.push_str(&result.resume_depth.to_string());
        output.push_str(",\"cursor\":");
        output.push_str(&result.cursor.to_string());
        output.push_str(",\"frontier\":[");
        push_u32_list(&mut output, &result.frontier);
        output.push_str("],\"next_frontier\":[");
        push_u32_list(&mut output, &result.next_frontier);
        output.push_str("]}");
    }
    output.push('}');
    Ok(output)
}

/// Render every state and event as a Graphviz DOT multigraph.
pub fn export_multiway_dot(
    config: &MultiwayConfig,
    result: &MultiwayResult,
) -> Result<String, MultiwayExportError> {
    let mut output = String::from(
        "digraph multiway {\n  rankdir=LR;\n  node [shape=box,fontname=\"monospace\"];\n",
    );
    for state in &result.states {
        output.push_str("  s");
        output.push_str(&state.sequence.to_string());
        output.push_str(" [label=\"");
        push_dot_bytes(&mut output, &state.payload)?;
        output.push_str("\\nd");
        output.push_str(&state.depth.to_string());
        output.push_str("\"];\n");
    }
    for event in &result.events {
        let rule = config
            .rules
            .get(event.rule as usize)
            .ok_or(MultiwayExportError::MalformedExport)?;
        output.push_str("  s");
        output.push_str(&event.source.to_string());
        output.push_str(" -> s");
        output.push_str(&event.destination.to_string());
        output.push_str(" [label=\"");
        push_dot_bytes(&mut output, &rule.lhs)?;
        output.push_str("->");
        push_dot_bytes(&mut output, &rule.rhs)?;
        output.push('@');
        output.push_str(&event.position.to_string());
        output.push_str("\"];\n");
    }
    output.push_str("}\n");
    Ok(output)
}

/// Resume a partial canonical export under new hard bounds.
pub fn resume_multiway(
    export_json: &str,
    config: &MultiwayConfig,
    cancel: Option<&AtomicBool>,
) -> Result<MultiwayResult, MultiwayExportError> {
    validate_config(config).map_err(|_| MultiwayExportError::ConfigMismatch)?;
    let root: serde_json::Value =
        serde_json::from_str(export_json).map_err(|_| MultiwayExportError::MalformedExport)?;
    let root = root
        .as_object()
        .ok_or(MultiwayExportError::MalformedExport)?;
    let format = json_string(root, "format")?;
    if format != MULTIWAY_FORMAT_VERSION {
        return Err(MultiwayExportError::UnsupportedFormat);
    }
    validate_resume_config(root, config)?;
    let resume = root
        .get("resume")
        .ok_or(MultiwayExportError::MalformedExport)?;
    if resume.is_null() {
        return Err(MultiwayExportError::AlreadyComplete);
    }
    let resume = resume
        .as_object()
        .ok_or(MultiwayExportError::MalformedExport)?;

    let state_values = json_array(root, "states")?;
    let mut result = MultiwayResult::default();
    let mut index_by_hash = HashMap::new();
    for (index, value) in state_values.iter().enumerate() {
        let state = value
            .as_object()
            .ok_or(MultiwayExportError::MalformedExport)?;
        let payload = json_string(state, "payload")?.as_bytes().to_vec();
        let depth = json_u32(state, "depth")?;
        let first_event = state.get("first_event").map(json_value_u32).transpose()?;
        let hash = hash_payload(&payload);
        let sequence = u32::try_from(index).map_err(|_| MultiwayExportError::MalformedExport)?;
        if index_by_hash.insert(hash, sequence).is_some() {
            return Err(MultiwayExportError::MalformedExport);
        }
        result.states.push(MultiwayState {
            payload,
            hash,
            depth,
            sequence,
            first_event,
        });
    }
    for (index, value) in json_array(root, "events")?.iter().enumerate() {
        let event = value
            .as_object()
            .ok_or(MultiwayExportError::MalformedExport)?;
        result.events.push(MultiwayEvent {
            id: u32::try_from(index).map_err(|_| MultiwayExportError::MalformedExport)?,
            source: json_u32(event, "src")?,
            destination: json_u32(event, "dst")?,
            rule: json_u32(event, "rule")?,
            position: json_u32(event, "pos")?,
            depth: json_u32(event, "depth")?,
            local: json_u32(event, "local")?,
        });
    }
    for state in &result.states {
        resize_counter(&mut result.states_per_depth, state.depth)?;
        result.states_per_depth[state.depth as usize] += 1;
    }
    for event in &result.events {
        resize_counter(&mut result.events_per_depth, event.depth)?;
        result.events_per_depth[event.depth as usize] += 1;
    }
    result.resume_depth = json_u32(resume, "depth")?;
    result.cursor = json_u32(resume, "cursor")?;
    result.frontier = json_u32_array(resume, "frontier", result.states.len())?;
    result.next_frontier = json_u32_array(resume, "next_frontier", result.states.len())?;
    if result.cursor as usize > result.frontier.len() {
        return Err(MultiwayExportError::MalformedExport);
    }

    let approximate_bytes = result
        .states
        .iter()
        .fold(0u64, |total, state| {
            total
                .saturating_add(state.payload.len() as u64)
                .saturating_add(STATE_OVERHEAD_BYTES)
        })
        .saturating_add((result.events.len() as u64).saturating_mul(EVENT_OVERHEAD_BYTES));
    let deadline = (config.max_duration_ms != 0)
        .then(|| Instant::now() + Duration::from_millis(config.max_duration_ms));
    let mut engine = Engine {
        config,
        result,
        index_by_hash,
        approximate_bytes,
        cancel,
        deadline,
    };
    engine.evolve();
    Ok(engine.result)
}

/// Latest persisted experiment alias.
pub const MULTIWAY_EXPORT_KEY_LATEST: &str = "multiway:experiment:latest";

/// Multiway persistence failure.
#[derive(Debug)]
pub enum MultiwayPersistError {
    /// Canonical export/config failure.
    Export(MultiwayExportError),
    /// WDBX recovery, WAL, or checkpoint failure.
    Durable(crate::DurableError),
    /// A byte payload cannot be represented by the current string-valued KV format.
    NonUtf8State,
    /// No experiment exists under the requested key.
    ExperimentNotFound,
}

impl fmt::Display for MultiwayPersistError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Export(error) => write!(formatter, "{error}"),
            Self::Durable(error) => write!(formatter, "{error}"),
            Self::NonUtf8State => {
                formatter.write_str("multiway state is not valid UTF-8 for WDBX KV storage")
            }
            Self::ExperimentNotFound => formatter.write_str("multiway experiment was not found"),
        }
    }
}

impl std::error::Error for MultiwayPersistError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Export(error) => Some(error),
            Self::Durable(error) => Some(error),
            Self::NonUtf8State | Self::ExperimentNotFound => None,
        }
    }
}

impl From<MultiwayExportError> for MultiwayPersistError {
    fn from(error: MultiwayExportError) -> Self {
        Self::Export(error)
    }
}

impl From<crate::DurableError> for MultiwayPersistError {
    fn from(error: crate::DurableError) -> Self {
        Self::Durable(error)
    }
}

/// Persist content-addressed states, canonical export aliases, and provenance.
pub fn persist_multiway(
    paths: crate::StorePaths,
    config: &MultiwayConfig,
    result: &MultiwayResult,
    export_json: &str,
) -> Result<(), MultiwayPersistError> {
    let mut store = crate::DurableStore::open(paths)?;
    for state in &result.states {
        let payload =
            std::str::from_utf8(&state.payload).map_err(|_| MultiwayPersistError::NonUtf8State)?;
        let key = format!("multiway:state:{}", hex_hash(state.hash));
        store.put(&key, payload)?;
    }

    let config_hash = hex_hash(multiway_config_hash(config)?);
    store.put(&format!("multiway:experiment:{config_hash}"), export_json)?;
    store.put(MULTIWAY_EXPORT_KEY_LATEST, export_json)?;

    let metadata = format!(
        "{{\"kind\":\"multiway_experiment\",\"config_hash\":\"{}\",\"export_hash\":\"{}\",\"states\":{},\"events\":{},\"termination\":\"{}\",\"complete\":{},\"zig_version\":\"0.17.0-dev.1442+972627084\"}}",
        config_hash,
        multiway_export_hash_hex(export_json.as_bytes()),
        result.states.len(),
        result.events.len(),
        result.termination.label(),
        result.complete,
    );
    store.add_block("multiway", 0, 0, &metadata, abi_foundation::time::unix_ms())?;
    store.checkpoint()?;
    Ok(())
}

/// Load a canonical export by config hash, or the latest alias when absent.
pub fn load_multiway_export(
    paths: crate::StorePaths,
    config_hash_hex: Option<&str>,
) -> Result<String, MultiwayPersistError> {
    let store = crate::DurableStore::open(paths)?;
    let key = config_hash_hex.map_or_else(
        || MULTIWAY_EXPORT_KEY_LATEST.to_owned(),
        |hash| format!("multiway:experiment:{hash}"),
    );
    store
        .get(&key)
        .map(str::to_owned)
        .ok_or(MultiwayPersistError::ExperimentNotFound)
}

fn canonical_config_json(config: &MultiwayConfig) -> Result<String, MultiwayExportError> {
    let mut output = String::from("{\"initial\":[");
    for (index, payload) in config.initial.iter().enumerate() {
        comma(&mut output, index);
        push_json_bytes(&mut output, payload)?;
    }
    output.push_str("],\"rules\":[");
    for (index, rule) in config.rules.iter().enumerate() {
        comma(&mut output, index);
        output.push_str("{\"id\":");
        output.push_str(&index.to_string());
        output.push_str(",\"rule\":");
        push_json_bytes(&mut output, &rule.canonical_bytes())?;
        output.push_str(",\"weight\":");
        output.push_str(&canonical_float(rule.weight));
        if let Some(family) = &rule.family {
            output.push_str(",\"family\":");
            output.push_str(
                &serde_json::to_string(family).map_err(|_| MultiwayExportError::MalformedExport)?,
            );
        }
        output.push_str(",\"hash\":\"");
        output.push_str(&hex_hash(rule.content_hash()));
        output.push_str("\"}");
    }
    output.push_str("],\"max_depth\":");
    output.push_str(&config.max_depth.to_string());
    output.push_str(",\"max_states\":");
    output.push_str(&config.max_states.to_string());
    output.push_str(",\"max_events\":");
    output.push_str(&config.max_events.to_string());
    output.push_str(",\"max_payload\":");
    output.push_str(&config.max_payload.to_string());
    output.push_str(",\"max_duration_ms\":");
    output.push_str(&config.max_duration_ms.to_string());
    output.push_str(",\"max_memory_bytes\":");
    output.push_str(&config.max_memory_bytes.to_string());
    output.push_str(
        ",\"traversal\":\"breadth_first\",\"canonicalization\":\"exact_string\",\"dedup\":\"by_canonical_hash\",\"seed\":",
    );
    output.push_str(&config.seed.to_string());
    output.push_str(",\"workers\":");
    output.push_str(&config.workers.to_string());
    output.push('}');
    Ok(output)
}

fn push_metrics_json(output: &mut String, metrics: &MultiwayMetrics) {
    output.push_str("{\"unique_states\":");
    output.push_str(&metrics.unique_states.to_string());
    output.push_str(",\"event_count\":");
    output.push_str(&metrics.event_count.to_string());
    output.push_str(",\"unique_transitions\":");
    output.push_str(&metrics.unique_transitions.to_string());
    output.push_str(",\"states_per_depth\":[");
    push_u32_list(output, &metrics.states_per_depth);
    output.push_str("],\"events_per_depth\":[");
    push_u32_list(output, &metrics.events_per_depth);
    output.push_str("],\"frontier_width_per_depth\":[");
    push_u32_list(output, &metrics.frontier_width_per_depth);
    output.push_str("],\"mean_out_degree\":");
    output.push_str(&canonical_float(metrics.mean_out_degree));
    output.push_str(",\"max_out_degree\":");
    output.push_str(&metrics.max_out_degree.to_string());
    output.push_str(",\"median_out_degree\":");
    output.push_str(&canonical_float(metrics.median_out_degree));
    output.push_str(",\"convergent_states\":");
    output.push_str(&metrics.convergent_states.to_string());
    output.push_str(",\"self_loops\":");
    output.push_str(&metrics.self_loops.to_string());
    output.push_str(",\"has_cycle\":");
    output.push_str(if metrics.has_cycle { "true" } else { "false" });
    output.push_str(",\"weakly_connected_components\":");
    output.push_str(&metrics.weakly_connected_components.to_string());
    output.push_str(",\"max_payload_bytes\":");
    output.push_str(&metrics.max_payload_bytes.to_string());
    output.push_str(",\"mean_payload_bytes\":");
    output.push_str(&canonical_float(metrics.mean_payload_bytes));
    output.push_str(",\"growth_rates\":[");
    for (index, rate) in metrics.growth_rates.iter().enumerate() {
        comma(output, index);
        output.push_str(&canonical_float(*rate));
    }
    output.push_str("],\"termination\":\"");
    output.push_str(metrics.termination.label());
    output.push_str("\",\"exhaustive\":");
    output.push_str(if metrics.exhaustive { "true" } else { "false" });
    output.push('}');
}

fn validate_resume_config(
    root: &serde_json::Map<String, serde_json::Value>,
    config: &MultiwayConfig,
) -> Result<(), MultiwayExportError> {
    let persisted = root
        .get("config")
        .and_then(serde_json::Value::as_object)
        .ok_or(MultiwayExportError::MalformedExport)?;
    let initial = json_array(persisted, "initial")?;
    if initial.len() != config.initial.len() {
        return Err(MultiwayExportError::ConfigMismatch);
    }
    for (value, expected) in initial.iter().zip(&config.initial) {
        if value.as_str().map(str::as_bytes) != Some(expected.as_slice()) {
            return Err(MultiwayExportError::ConfigMismatch);
        }
    }
    let rules = json_array(persisted, "rules")?;
    if rules.len() != config.rules.len() {
        return Err(MultiwayExportError::ConfigMismatch);
    }
    for (value, expected) in rules.iter().zip(&config.rules) {
        let object = value
            .as_object()
            .ok_or(MultiwayExportError::MalformedExport)?;
        if json_string(object, "rule")?.as_bytes() != expected.canonical_bytes() {
            return Err(MultiwayExportError::ConfigMismatch);
        }
    }
    Ok(())
}

fn json_string<'value>(
    object: &'value serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'value str, MultiwayExportError> {
    object
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or(MultiwayExportError::MalformedExport)
}

fn json_array<'value>(
    object: &'value serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<&'value Vec<serde_json::Value>, MultiwayExportError> {
    object
        .get(field)
        .and_then(serde_json::Value::as_array)
        .ok_or(MultiwayExportError::MalformedExport)
}

fn json_u32(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Result<u32, MultiwayExportError> {
    object
        .get(field)
        .ok_or(MultiwayExportError::MalformedExport)
        .and_then(json_value_u32)
}

fn json_value_u32(value: &serde_json::Value) -> Result<u32, MultiwayExportError> {
    value
        .as_u64()
        .and_then(|integer| u32::try_from(integer).ok())
        .ok_or(MultiwayExportError::MalformedExport)
}

fn json_u32_array(
    object: &serde_json::Map<String, serde_json::Value>,
    field: &str,
    state_count: usize,
) -> Result<Vec<u32>, MultiwayExportError> {
    json_array(object, field)?
        .iter()
        .map(|value| {
            let index = json_value_u32(value)?;
            if index as usize >= state_count {
                return Err(MultiwayExportError::MalformedExport);
            }
            Ok(index)
        })
        .collect()
}

fn resize_counter(counter: &mut Vec<u32>, depth: u32) -> Result<(), MultiwayExportError> {
    let required = usize::try_from(depth)
        .map_err(|_| MultiwayExportError::MalformedExport)?
        .checked_add(1)
        .ok_or(MultiwayExportError::MalformedExport)?;
    counter.resize(required.max(counter.len()), 0);
    Ok(())
}

fn push_json_bytes(output: &mut String, bytes: &[u8]) -> Result<(), MultiwayExportError> {
    let text = std::str::from_utf8(bytes).map_err(|_| MultiwayExportError::NonUtf8)?;
    output
        .push_str(&serde_json::to_string(text).map_err(|_| MultiwayExportError::MalformedExport)?);
    Ok(())
}

fn push_dot_bytes(output: &mut String, bytes: &[u8]) -> Result<(), MultiwayExportError> {
    let text = std::str::from_utf8(bytes).map_err(|_| MultiwayExportError::NonUtf8)?;
    for character in text.chars() {
        match character {
            '"' | '\\' => {
                output.push('\\');
                output.push(character);
            }
            '\n' => output.push_str("\\n"),
            other => output.push(other),
        }
    }
    Ok(())
}

fn push_u32_list(output: &mut String, values: &[u32]) {
    for (index, value) in values.iter().enumerate() {
        comma(output, index);
        output.push_str(&value.to_string());
    }
}

fn comma(output: &mut String, index: usize) {
    if index != 0 {
        output.push(',');
    }
}

fn canonical_float(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        value.to_string()
    }
}

fn hex_hash(hash: [u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(64);
    for byte in hash {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

/// Derived structural metrics over unique transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiwayMetrics {
    /// Number of unique states.
    pub unique_states: u32,
    /// Number of rule applications.
    pub event_count: u32,
    /// Number of distinct source/destination pairs.
    pub unique_transitions: u32,
    /// State discoveries by depth.
    pub states_per_depth: Vec<u32>,
    /// Events by destination depth.
    pub events_per_depth: Vec<u32>,
    /// BFS frontier width by depth.
    pub frontier_width_per_depth: Vec<u32>,
    /// Unique transitions divided by all states.
    pub mean_out_degree: f64,
    /// Largest distinct-destination out-degree.
    pub max_out_degree: u32,
    /// Median distinct-destination out-degree across all states.
    pub median_out_degree: f64,
    /// States with more than one distinct predecessor.
    pub convergent_states: u32,
    /// Unique self-loop transitions.
    pub self_loops: u32,
    /// Whether the unique-transition graph contains a cycle.
    pub has_cycle: bool,
    /// Weakly connected component count.
    pub weakly_connected_components: u32,
    /// Largest payload.
    pub max_payload_bytes: u32,
    /// Mean payload bytes.
    pub mean_payload_bytes: f64,
    /// `states[d+1] / states[d]` while the denominator is nonzero.
    pub growth_rates: Vec<f64>,
    /// Copied stop reason.
    pub termination: Termination,
    /// Whether the frontier was exhausted.
    pub exhaustive: bool,
}

/// Compute deterministic graph and payload metrics.
#[must_use]
pub fn compute_multiway_metrics(result: &MultiwayResult) -> MultiwayMetrics {
    let state_count = result.states.len();
    let transitions: HashSet<(u32, u32)> = result
        .events
        .iter()
        .map(|event| (event.source, event.destination))
        .collect();
    let mut outgoing = vec![HashSet::new(); state_count];
    let mut incoming = vec![HashSet::new(); state_count];
    for &(source, destination) in &transitions {
        if let (Some(out), Some(input)) = (
            outgoing.get_mut(source as usize),
            incoming.get_mut(destination as usize),
        ) {
            out.insert(destination);
            input.insert(source);
        }
    }

    let mut degrees: Vec<u32> = outgoing
        .iter()
        .map(|destinations| bounded_u32(destinations.len()))
        .collect();
    let max_out_degree = degrees.iter().copied().max().unwrap_or(0);
    degrees.sort_unstable();
    let median_out_degree = match degrees.len() {
        0 => 0.0,
        length if length % 2 == 1 => f64::from(degrees[length / 2]),
        length => f64::midpoint(
            f64::from(degrees[length / 2 - 1]),
            f64::from(degrees[length / 2]),
        ),
    };

    let has_cycle = graph_has_cycle(&outgoing);
    let weakly_connected_components =
        weak_component_count(state_count, transitions.iter().copied());
    let total_payload: usize = result.states.iter().map(|state| state.payload.len()).sum();
    let max_payload = result
        .states
        .iter()
        .map(|state| state.payload.len())
        .max()
        .unwrap_or(0);
    let growth_rates = result
        .states_per_depth
        .windows(2)
        .take_while(|pair| pair[0] != 0)
        .map(|pair| f64::from(pair[1]) / f64::from(pair[0]))
        .collect();

    MultiwayMetrics {
        unique_states: bounded_u32(state_count),
        event_count: bounded_u32(result.events.len()),
        unique_transitions: bounded_u32(transitions.len()),
        states_per_depth: result.states_per_depth.clone(),
        events_per_depth: result.events_per_depth.clone(),
        frontier_width_per_depth: result.states_per_depth.clone(),
        mean_out_degree: if state_count == 0 {
            0.0
        } else {
            f64::from(bounded_u32(transitions.len())) / f64::from(bounded_u32(state_count))
        },
        max_out_degree,
        median_out_degree,
        convergent_states: bounded_u32(incoming.iter().filter(|sources| sources.len() > 1).count()),
        self_loops: bounded_u32(
            transitions
                .iter()
                .filter(|(source, destination)| source == destination)
                .count(),
        ),
        has_cycle,
        weakly_connected_components,
        max_payload_bytes: bounded_u32(max_payload),
        mean_payload_bytes: if state_count == 0 {
            0.0
        } else {
            f64::from(bounded_u32(total_payload)) / f64::from(bounded_u32(state_count))
        },
        growth_rates,
        termination: result.termination,
        exhaustive: result.complete,
    }
}

fn graph_has_cycle(outgoing: &[HashSet<u32>]) -> bool {
    fn visit(node: usize, outgoing: &[HashSet<u32>], colors: &mut [u8]) -> bool {
        colors[node] = 1;
        for &child in &outgoing[node] {
            let child = child as usize;
            if colors[child] == 1 || (colors[child] == 0 && visit(child, outgoing, colors)) {
                return true;
            }
        }
        colors[node] = 2;
        false
    }

    let mut colors = vec![0; outgoing.len()];
    (0..outgoing.len()).any(|node| colors[node] == 0 && visit(node, outgoing, &mut colors))
}

fn weak_component_count(state_count: usize, transitions: impl Iterator<Item = (u32, u32)>) -> u32 {
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }

    let mut parent: Vec<usize> = (0..state_count).collect();
    for (source, destination) in transitions {
        let source_root = root(&mut parent, source as usize);
        let destination_root = root(&mut parent, destination as usize);
        if source_root != destination_root {
            parent[source_root] = destination_root;
        }
    }
    bounded_u32(
        (0..state_count)
            .filter(|&node| root(&mut parent, node) == node)
            .count(),
    )
}

fn bounded_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::{
        ConfigError, MultiwayConfig, MultiwayExportError, ParseRuleError, Rule, Termination,
        build_causal_edges, compute_multiway_metrics, export_multiway_dot, export_multiway_json,
        load_multiway_export, multiway_config_hash, parse_rule, persist_multiway, resume_multiway,
        run_multiway, validate_config,
    };

    fn config(initial: &[&str], rules: &[(&str, &str)]) -> MultiwayConfig {
        let mut config = MultiwayConfig::new(
            initial
                .iter()
                .map(|payload| payload.as_bytes().to_vec())
                .collect(),
            rules
                .iter()
                .map(|(lhs, rhs)| Rule::new(lhs.as_bytes(), rhs.as_bytes()))
                .collect(),
        );
        config.max_states = 100;
        config.max_events = 1_000;
        config.max_payload = 64;
        config
    }

    #[test]
    fn rule_parsing_hashing_and_validation_match_the_oracle() {
        assert_eq!(
            parse_rule(" A -> AB ").expect("rule"),
            Rule::new(b"A", b"AB")
        );
        assert_eq!(parse_rule("AB"), Err(ParseRuleError::MissingArrow));
        assert_eq!(parse_rule(" ->B"), Err(ParseRuleError::EmptyLhs));
        let rule = Rule::new(b"A", b"AB");
        assert_eq!(rule.canonical_bytes(), b"A->AB");
        assert_eq!(rule.content_hash(), Rule::new(b"A", b"AB").content_hash());

        let mut invalid = config(&["A"], &[("A", "B")]);
        invalid.max_depth = 0;
        assert_eq!(validate_config(&invalid), Err(ConfigError::ZeroBound));
    }

    #[test]
    fn overlapping_matches_preserve_rule_then_offset_order() {
        let mut config = config(&["AAA"], &[("AA", "B")]);
        config.max_depth = 1;
        let result = run_multiway(&config, None);
        assert_eq!(result.events.len(), 2);
        assert_eq!(result.events[0].position, 0);
        assert_eq!(result.events[1].position, 1);
        assert_eq!(
            result
                .states
                .iter()
                .map(|state| state.payload.as_slice())
                .collect::<Vec<_>>(),
            [b"AAA".as_slice(), b"BA".as_slice(), b"AB".as_slice()]
        );
    }

    #[test]
    fn duplicate_events_survive_state_deduplication() {
        let mut config = config(&["A"], &[("A", "B"), ("A", "B")]);
        config.max_depth = 1;
        let result = run_multiway(&config, None);
        let metrics = compute_multiway_metrics(&result);
        assert_eq!(result.states.len(), 2);
        assert_eq!(metrics.event_count, 2);
        assert_eq!(metrics.unique_transitions, 1);
    }

    #[test]
    fn cycles_convergence_and_empty_payloads_are_measured() {
        let cycle = run_multiway(&config(&["A"], &[("A", "B"), ("B", "A")]), None);
        let cycle_metrics = compute_multiway_metrics(&cycle);
        assert!(cycle.complete);
        assert!(cycle_metrics.has_cycle);
        assert_eq!(cycle_metrics.weakly_connected_components, 1);

        let convergence = run_multiway(&config(&["AB"], &[("A", "C"), ("B", "C")]), None);
        assert_eq!(compute_multiway_metrics(&convergence).convergent_states, 1);

        let shrinking = run_multiway(&config(&["AA"], &[("A", "")]), None);
        assert!(shrinking.complete);
        assert_eq!(shrinking.states.last().expect("empty").payload, b"");
    }

    #[test]
    fn every_hard_cap_is_atomic_and_never_exceeded() {
        let mut states = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
        states.max_depth = 10;
        states.max_states = 7;
        let state_result = run_multiway(&states, None);
        assert_eq!(state_result.termination, Termination::MaxStates);
        assert!(state_result.states.len() <= 7);

        let mut events = states.clone();
        events.max_states = 10_000;
        events.max_events = 9;
        let event_result = run_multiway(&events, None);
        assert_eq!(event_result.termination, Termination::MaxEvents);
        assert!(event_result.events.len() <= 9);

        let mut payload = config(&["A"], &[("A", "AAAA")]);
        payload.max_depth = 10;
        payload.max_payload = 8;
        assert_eq!(
            run_multiway(&payload, None).termination,
            Termination::PayloadLimit
        );

        let mut memory = config(&["A"], &[("A", "AA")]);
        memory.max_depth = 20;
        memory.max_memory_bytes = 512;
        assert_eq!(
            run_multiway(&memory, None).termination,
            Termination::AllocationFailure
        );
    }

    #[test]
    fn max_depth_and_cancellation_return_resumable_partial_state() {
        let mut depth = config(&["A"], &[("A", "AA")]);
        depth.max_depth = 2;
        let depth_result = run_multiway(&depth, None);
        assert_eq!(depth_result.termination, Termination::MaxDepth);
        assert!(!depth_result.complete);
        assert!(!depth_result.frontier.is_empty());
        assert!(
            depth_result
                .states
                .iter()
                .all(|state| state.depth <= depth.max_depth)
        );

        let cancelled = AtomicBool::new(true);
        let cancelled_result = run_multiway(&depth, Some(&cancelled));
        assert_eq!(cancelled_result.termination, Termination::Cancelled);
        assert!(!cancelled_result.complete);
    }

    #[test]
    fn repeated_runs_are_structurally_deterministic() {
        let config = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
        let first = run_multiway(&config, None);
        let second = run_multiway(&config, None);
        assert_eq!(first.states, second.states);
        assert_eq!(first.events, second.events);
        assert_eq!(first.states_per_depth, second.states_per_depth);
        assert_eq!(first.events_per_depth, second.events_per_depth);
        assert_eq!(first.termination, second.termination);
    }

    #[test]
    fn canonical_exports_are_byte_deterministic_and_dot_keeps_multiplicity() {
        let mut config = config(&["A"], &[("A", "B"), ("A", "B")]);
        config.max_depth = 1;
        let first = run_multiway(&config, None);
        let second = run_multiway(&config, None);
        let first_json = export_multiway_json(&config, &first, &compute_multiway_metrics(&first))
            .expect("first export");
        let second_json =
            export_multiway_json(&config, &second, &compute_multiway_metrics(&second))
                .expect("second export");
        assert_eq!(first_json, second_json);
        assert!(first_json.starts_with("{\"format\":\"abi-multiway-v1\""));
        assert!(first_json.contains("\"resume\":{\"depth\":1"));

        let dot = export_multiway_dot(&config, &first).expect("dot");
        assert!(dot.contains("digraph multiway"));
        assert_eq!(dot.matches("s0 -> s1").count(), 2);
    }

    #[test]
    fn interrupted_resume_matches_an_uninterrupted_canonical_export() {
        let mut full = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
        full.max_depth = 5;
        full.max_states = 500;
        full.max_events = 5_000;
        let direct = run_multiway(&full, None);
        let direct_export =
            export_multiway_json(&full, &direct, &compute_multiway_metrics(&direct))
                .expect("direct export");

        let mut shallow = full.clone();
        shallow.max_depth = 3;
        let partial = run_multiway(&shallow, None);
        assert_eq!(partial.termination, Termination::MaxDepth);
        let partial_export =
            export_multiway_json(&shallow, &partial, &compute_multiway_metrics(&partial))
                .expect("partial export");
        let resumed = resume_multiway(&partial_export, &full, None).expect("resume");
        let resumed_export =
            export_multiway_json(&full, &resumed, &compute_multiway_metrics(&resumed))
                .expect("resumed export");

        assert_eq!(resumed_export, direct_export);
    }

    #[test]
    fn resume_rejects_mismatch_malformed_and_completed_documents() {
        let mut partial_config = config(&["A"], &[("A", "AB")]);
        partial_config.max_depth = 1;
        let partial = run_multiway(&partial_config, None);
        let export = export_multiway_json(
            &partial_config,
            &partial,
            &compute_multiway_metrics(&partial),
        )
        .expect("export");
        let mismatch = config(&["A"], &[("A", "BB")]);
        assert!(matches!(
            resume_multiway(&export, &mismatch, None),
            Err(MultiwayExportError::ConfigMismatch)
        ));
        assert!(matches!(
            resume_multiway("{\"format\":\"abi-multiway-v1\"}", &partial_config, None),
            Err(MultiwayExportError::MalformedExport)
        ));
        assert!(matches!(
            resume_multiway("{\"format\":\"nope\"}", &partial_config, None),
            Err(MultiwayExportError::UnsupportedFormat)
        ));

        let complete_config = config(&["A"], &[("A", "B"), ("B", "A")]);
        let complete = run_multiway(&complete_config, None);
        let complete_export = export_multiway_json(
            &complete_config,
            &complete,
            &compute_multiway_metrics(&complete),
        )
        .expect("complete export");
        assert!(matches!(
            resume_multiway(&complete_export, &complete_config, None),
            Err(MultiwayExportError::AlreadyComplete)
        ));
    }

    #[test]
    fn token_lineage_links_producer_events_to_consumers() {
        let mut config = config(&["A"], &[("A", "AB"), ("B", "C")]);
        config.max_depth = 2;
        let result = run_multiway(&config, None);
        let edges = build_causal_edges(&config, &result);
        assert!(
            edges
                .iter()
                .any(|edge| edge.parent == 0 && edge.child > edge.parent)
        );
    }

    #[test]
    fn wdbx_persistence_round_trips_latest_and_config_hash() {
        let directory = abi_foundation::temp_path::temp_file_path("abi_multiway_persist", "store");
        std::fs::create_dir_all(&directory).expect("fixture directory");
        let paths = crate::StorePaths {
            dir: directory.clone(),
            base: "multiway".to_owned(),
        };

        let mut config = config(&["A"], &[("A", "AB"), ("A", "BA"), ("BB", "A")]);
        config.max_depth = 3;
        let result = run_multiway(&config, None);
        let export = export_multiway_json(&config, &result, &compute_multiway_metrics(&result))
            .expect("export");
        persist_multiway(paths.clone(), &config, &result, &export).expect("persist");
        assert_eq!(
            load_multiway_export(paths.clone(), None).expect("latest"),
            export
        );
        let config_hash = super::hex_hash(multiway_config_hash(&config).expect("config hash"));
        assert_eq!(
            load_multiway_export(paths.clone(), Some(&config_hash)).expect("by hash"),
            export
        );
        let reopened = crate::DurableStore::open(paths).expect("reopen");
        assert!(reopened.stats().blocks >= 1);
        assert_eq!(
            reopened.get(&format!(
                "multiway:state:{}",
                super::hex_hash(result.states[0].hash)
            )),
            Some("A")
        );

        std::fs::remove_dir_all(directory).expect("cleanup");
    }
}
