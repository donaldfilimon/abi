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

mod export;

use export::hex_hash;
pub use export::{
    CausalEdge, MultiwayExportError, build_causal_edges, export_multiway_dot, export_multiway_json,
    multiway_config_hash, multiway_export_hash_hex, resume_multiway,
};

/// Latest persisted experiment alias.
pub const MULTIWAY_EXPORT_KEY_LATEST: &str = "multiway:experiment:latest";

/// Multiway persistence failure.
#[derive(Debug)]
pub enum MultiwayPersistError {
    /// Canonical export/config failure.
    Export(MultiwayExportError),
    /// Writable version detection, migration, or commit failure.
    Versioned(crate::VersionedError),
    /// Read-only version detection or recovery failure.
    Read(crate::v2::MigrationError),
    /// A byte payload cannot be represented by the current string-valued KV format.
    NonUtf8State,
    /// No experiment exists under the requested key.
    ExperimentNotFound,
}

impl fmt::Display for MultiwayPersistError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Export(error) => write!(formatter, "{error}"),
            Self::Versioned(error) => write!(formatter, "{error}"),
            Self::Read(error) => write!(formatter, "{error}"),
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
            Self::Versioned(error) => Some(error),
            Self::Read(error) => Some(error),
            Self::NonUtf8State | Self::ExperimentNotFound => None,
        }
    }
}

impl From<MultiwayExportError> for MultiwayPersistError {
    fn from(error: MultiwayExportError) -> Self {
        Self::Export(error)
    }
}

impl From<crate::VersionedError> for MultiwayPersistError {
    fn from(error: crate::VersionedError) -> Self {
        Self::Versioned(error)
    }
}

impl From<crate::v2::MigrationError> for MultiwayPersistError {
    fn from(error: crate::v2::MigrationError) -> Self {
        Self::Read(error)
    }
}

/// Persist content-addressed states, canonical export aliases, and provenance.
pub fn persist_multiway(
    paths: crate::StorePaths,
    config: &MultiwayConfig,
    result: &MultiwayResult,
    export_json: &str,
) -> Result<(), MultiwayPersistError> {
    let mut store = crate::VersionedStore::open(paths)?;
    for state in &result.states {
        let payload =
            std::str::from_utf8(&state.payload).map_err(|_| MultiwayPersistError::NonUtf8State)?;
        let key = format!("multiway:state:{}", hex_hash(state.hash));
        store.put(&key, payload)?;
    }

    let config_hash = hex_hash(multiway_config_hash(config)?);
    store.put(&format!("multiway:experiment:{config_hash}"), export_json)?;
    store.put(MULTIWAY_EXPORT_KEY_LATEST, export_json)?;

    // `zig_version` frozen for canonical-format compatibility — see export
    // comment above; changing it breaks stored metadata comparisons.
    let metadata = format!(
        "{{\"kind\":\"multiway_experiment\",\"config_hash\":\"{}\",\"export_hash\":\"{}\",\"states\":{},\"events\":{},\"termination\":\"{}\",\"complete\":{},\"zig_version\":\"0.17.0-dev.1442+972627084\"}}",
        config_hash,
        multiway_export_hash_hex(export_json.as_bytes()),
        result.states.len(),
        result.events.len(),
        result.termination.label(),
        result.complete,
    );
    store.add_block(
        "multiway",
        crate::RecordId::new_v2(),
        crate::RecordId::new_v2(),
        &metadata,
        abi_foundation::time::unix_ms(),
    )?;
    store.compact()?;
    Ok(())
}

/// Load a canonical export by config hash, or the latest alias when absent.
pub fn load_multiway_export(
    paths: &crate::StorePaths,
    config_hash_hex: Option<&str>,
) -> Result<String, MultiwayPersistError> {
    let snapshot = crate::open_versioned_read_only(paths)?;
    let key = config_hash_hex.map_or_else(
        || MULTIWAY_EXPORT_KEY_LATEST.to_owned(),
        |hash| format!("multiway:experiment:{hash}"),
    );
    snapshot
        .preferred_value(&key)
        .map(str::to_owned)
        .ok_or(MultiwayPersistError::ExperimentNotFound)
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
mod tests;
