//! Canonical multiway export, resume, and causal-lineage support.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use super::{
    EVENT_OVERHEAD_BYTES, Engine, MULTIWAY_FORMAT_VERSION, MultiwayConfig, MultiwayEvent,
    MultiwayMetrics, MultiwayResult, MultiwayState, STATE_OVERHEAD_BYTES, hash_payload,
    validate_config,
};

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
    // `zig_version` is frozen at the value the retired Zig implementation
    // emitted: it is part of the canonical export byte stream, so changing it
    // would break export_hash/golden-fixture compatibility. Do not update.
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

pub(super) fn hex_hash(hash: [u8; 32]) -> String {
    use std::fmt::Write as _;
    let mut output = String::with_capacity(64);
    for byte in hash {
        let _ = write!(output, "{byte:02x}");
    }
    output
}
