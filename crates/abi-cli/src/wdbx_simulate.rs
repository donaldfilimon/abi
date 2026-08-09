//! Bounded multiway simulator CLI.

use std::fmt::Write as _;
use std::io::Read as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use abi_wdbx::{
    MAX_MULTIWAY_RULES, MULTIWAY_FORMAT_VERSION, MultiwayConfig,
    MultiwayConfigError as ConfigError, MultiwayExportError, MultiwayMetrics, MultiwayResult,
    MultiwayRule, compute_multiway_metrics, export_multiway_dot, export_multiway_json,
    load_multiway_export, multiway_config_hash, multiway_export_hash_hex, parse_rule,
    persist_multiway, resume_multiway, run_multiway, validate_multiway_config,
};

use crate::app::Outcome;
use crate::usage::is_help_token;
use crate::wdbx::paths_from_cli_base;

const MAX_RULE_FILE_BYTES: u64 = 1024 * 1024;
const MAX_CONFIG_FILE_BYTES: u64 = 1024 * 1024;
const MAX_RESUME_FILE_BYTES: u64 = 256 * 1024 * 1024;
static CANCELLED: AtomicBool = AtomicBool::new(false);
static CANCEL_HANDLER: OnceLock<Result<(), String>> = OnceLock::new();

const HELP: &str = "usage: abi wdbx simulate [options]\n\nRun a bounded multiway (Wolfram-style) string-rewriting experiment.\nSimulates a finite, explicitly bounded slice of rule space; it does not\nenumerate \"the ruliad\" and makes no physics claims.\n\nRules & initial states\n  --initial <STATE>        Initial state (repeatable; at least one required)\n  --rule '<LHS->RHS>'      Inline rewriting rule (repeatable)\n  --rules-file <PATH>      One rule per line; blank lines and # comments ignored\n  --config <PATH>          JSON experiment config (flags override file values)\n\nBounds (hard limits; partial results are returned when one is reached)\n  --depth <N>              Maximum depth (default 5)\n  --max-states <N>         Maximum unique states (default 10000)\n  --max-events <N>         Maximum events (default 100000)\n  --max-payload <N>        Maximum state payload bytes (default 4096)\n  --deadline-ms <N>        Wall-clock budget in ms (0 = unlimited)\n  --max-memory <N>         Approximate engine memory budget in bytes (0 = unlimited)\n\nReproducibility\n  --seed <N>               Recorded random seed (engine is deterministic)\n  --workers <N>            Recorded worker count (expansion is single-threaded)\n\nOutput & persistence\n  --format <summary|json|dot>  Output format (default summary)\n  --output <PATH>          Write json/dot export to a file instead of printing\n  --store <PATH>           Persist experiment into a WDBX checkpoint at PATH\n  --resume <PATH>          Resume from a canonical JSON export file\n  --resume-wdbx <PATH>     Resume from the latest experiment in a WDBX checkpoint\n  --dry-run                Validate configuration and print the plan, then exit\n  --quiet                  Suppress the human summary\n  --verbose                Print per-depth tables and extra detail\n\nCtrl-C cancels a running experiment; the partial result is still\nsummarized/exported with termination reason \"cancelled\".\n\nexample:\n  abi wdbx simulate --initial A --rule 'A->AB' --rule 'A->BA' --rule 'BB->A' \\\n      --depth 5 --max-states 500 --max-events 5000 --format json --output experiment.json\n";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum OutputFormat {
    #[default]
    Summary,
    Json,
    Dot,
}

#[derive(Debug, Default)]
struct Options {
    initial: Vec<Vec<u8>>,
    rules: Vec<MultiwayRule>,
    max_depth: Option<u32>,
    max_states: Option<u32>,
    max_events: Option<u32>,
    max_payload: Option<u32>,
    max_duration_ms: Option<u64>,
    max_memory_bytes: Option<u64>,
    seed: Option<u64>,
    workers: Option<u32>,
    format: OutputFormat,
    output: Option<String>,
    store: Option<String>,
    resume_file: Option<String>,
    resume_wdbx: Option<String>,
    dry_run: bool,
    quiet: bool,
    verbose: bool,
}

fn diagnostic(message: impl std::fmt::Display) -> Outcome {
    Outcome::stderr(format!("simulate: {message}\n"), 2)
}

fn failure(message: impl std::fmt::Display) -> Outcome {
    Outcome::stderr(format!("simulate: {message}\n"), 1)
}

fn value_after<'a>(args: &'a [String], index: &mut usize, flag: &str) -> Result<&'a str, Outcome> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| diagnostic(format!("{flag} requires a value")))
}

fn unsigned_after<T: std::str::FromStr>(
    args: &[String],
    index: &mut usize,
    flag: &str,
) -> Result<T, Outcome> {
    let value = value_after(args, index, flag)?;
    value.parse::<T>().map_err(|_| {
        diagnostic(format!(
            "{flag}: '{value}' is not a valid non-negative integer"
        ))
    })
}

fn push_rule(options: &mut Options, text: &str, origin: &str) -> Result<(), Outcome> {
    match parse_rule(text) {
        Ok(rule) => {
            options.rules.push(rule);
            Ok(())
        }
        Err(abi_wdbx::ParseRuleError::MissingArrow) => Err(diagnostic(format!(
            "invalid rule '{text}' ({origin}): expected 'LHS->RHS'"
        ))),
        Err(abi_wdbx::ParseRuleError::EmptyLhs) => Err(diagnostic(format!(
            "invalid rule '{text}' ({origin}): left-hand side must be non-empty"
        ))),
    }
}

fn read_limited(path: &str, maximum: u64) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|detail| detail.to_string())?;
    let mut reader = file.take(maximum.saturating_add(1));
    let mut content = String::new();
    reader
        .read_to_string(&mut content)
        .map_err(|detail| detail.to_string())?;
    if u64::try_from(content.len()).unwrap_or(u64::MAX) > maximum {
        return Err(format!("file exceeds {maximum} bytes"));
    }
    Ok(content)
}

fn load_rules_file(options: &mut Options, path: &str) -> Result<(), Outcome> {
    let content = read_limited(path, MAX_RULE_FILE_BYTES)
        .map_err(|detail| diagnostic(format!("cannot read rules file '{path}': {detail}")))?;
    for raw in content.lines() {
        let line = raw.trim_matches([' ', '\t', '\r']);
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        push_rule(options, line, path)?;
    }
    Ok(())
}

fn config_u32(
    object: &serde_json::Map<String, serde_json::Value>,
    path: &str,
    field: &str,
) -> Result<Option<u32>, Outcome> {
    let Some(value) = object.get(field) else {
        return Ok(None);
    };
    let integer = value.as_u64().ok_or_else(|| {
        diagnostic(format!(
            "config '{path}': \"{field}\" must be a non-negative integer"
        ))
    })?;
    u32::try_from(integer)
        .map(Some)
        .map_err(|_| diagnostic(format!("config '{path}': \"{field}\" too large")))
}

fn load_config_rules(
    options: &mut Options,
    object: &serde_json::Map<String, serde_json::Value>,
    path: &str,
) -> Result<(), Outcome> {
    if let Some(initial) = object.get("initial") {
        let entries = initial.as_array().ok_or_else(|| {
            diagnostic(format!(
                "config '{path}': \"initial\" must be an array of strings"
            ))
        })?;
        for entry in entries {
            let text = entry.as_str().ok_or_else(|| {
                diagnostic(format!(
                    "config '{path}': \"initial\" entries must be strings"
                ))
            })?;
            options.initial.push(text.as_bytes().to_vec());
        }
    }
    if let Some(rules) = object.get("rules") {
        let entries = rules
            .as_array()
            .ok_or_else(|| diagnostic(format!("config '{path}': \"rules\" must be an array")))?;
        for entry in entries {
            if let Some(text) = entry.as_str() {
                push_rule(options, text, path)?;
                continue;
            }
            let rule = entry.as_object().ok_or_else(|| {
                diagnostic(format!(
                    "config '{path}': \"rules\" entries must be strings or objects"
                ))
            })?;
            let text = rule
                .get("rule")
                .ok_or_else(|| {
                    diagnostic(format!("config '{path}': rule object missing \"rule\""))
                })?
                .as_str()
                .ok_or_else(|| {
                    diagnostic(format!(
                        "config '{path}': rule object \"rule\" must be a string"
                    ))
                })?;
            push_rule(options, text, path)?;
            let latest = options.rules.last_mut().expect("rule was appended");
            if let Some(weight) = rule.get("weight") {
                latest.weight = weight.as_f64().ok_or_else(|| {
                    diagnostic(format!("config '{path}': rule \"weight\" must be a number"))
                })?;
            }
            if let Some(family) = rule.get("family") {
                latest.family = Some(
                    family
                        .as_str()
                        .ok_or_else(|| {
                            diagnostic(format!("config '{path}': rule \"family\" must be a string"))
                        })?
                        .to_owned(),
                );
            }
        }
    }
    Ok(())
}

fn load_config_file(options: &mut Options, path: &str) -> Result<(), Outcome> {
    let content = read_limited(path, MAX_CONFIG_FILE_BYTES)
        .map_err(|detail| diagnostic(format!("cannot read config file '{path}': {detail}")))?;
    let document: serde_json::Value = serde_json::from_str(&content)
        .map_err(|_| diagnostic(format!("config file '{path}' is not valid JSON")))?;
    let object = document.as_object().ok_or_else(|| {
        diagnostic(format!(
            "config file '{path}': top level must be a JSON object"
        ))
    })?;
    load_config_rules(options, object, path)?;
    if let Some(value) = config_u32(object, path, "max_depth")? {
        options.max_depth = Some(value);
    }
    if let Some(value) = config_u32(object, path, "max_states")? {
        options.max_states = Some(value);
    }
    if let Some(value) = config_u32(object, path, "max_events")? {
        options.max_events = Some(value);
    }
    if let Some(value) = config_u32(object, path, "max_payload")? {
        options.max_payload = Some(value);
    }
    for (field, target) in [
        ("max_duration_ms", &mut options.max_duration_ms),
        ("max_memory_bytes", &mut options.max_memory_bytes),
        ("seed", &mut options.seed),
    ] {
        if let Some(value) = object.get(field) {
            *target = Some(value.as_u64().ok_or_else(|| {
                diagnostic(format!(
                    "config '{path}': \"{field}\" must be a non-negative integer"
                ))
            })?);
        }
    }
    if let Some(value) = object.get("workers") {
        let workers = value
            .as_u64()
            .and_then(|integer| u32::try_from(integer).ok())
            .filter(|&integer| integer != 0)
            .ok_or_else(|| diagnostic(format!("config '{path}': \"workers\" out of range")))?;
        options.workers = Some(workers);
    }
    Ok(())
}

fn parse_options(args: &[String]) -> Result<Options, Outcome> {
    let mut options = Options::default();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        match flag {
            "--initial" => {
                let value = value_after(args, &mut index, flag)?;
                options.initial.push(value.as_bytes().to_vec());
            }
            "--rule" => {
                let value = value_after(args, &mut index, flag)?;
                push_rule(&mut options, value, "--rule")?;
            }
            "--rules-file" => {
                let value = value_after(args, &mut index, flag)?;
                load_rules_file(&mut options, value)?;
            }
            "--config" => {
                let value = value_after(args, &mut index, flag)?;
                load_config_file(&mut options, value)?;
            }
            "--depth" => options.max_depth = Some(unsigned_after(args, &mut index, flag)?),
            "--max-states" => {
                options.max_states = Some(unsigned_after(args, &mut index, flag)?);
            }
            "--max-events" => {
                options.max_events = Some(unsigned_after(args, &mut index, flag)?);
            }
            "--max-payload" => {
                options.max_payload = Some(unsigned_after(args, &mut index, flag)?);
            }
            "--deadline-ms" => {
                options.max_duration_ms = Some(unsigned_after(args, &mut index, flag)?);
            }
            "--max-memory" => {
                options.max_memory_bytes = Some(unsigned_after(args, &mut index, flag)?);
            }
            "--seed" => options.seed = Some(unsigned_after(args, &mut index, flag)?),
            "--workers" => {
                let workers = unsigned_after(args, &mut index, flag)?;
                if workers == 0 {
                    return Err(diagnostic("--workers must be >= 1"));
                }
                options.workers = Some(workers);
            }
            "--format" => {
                let value = value_after(args, &mut index, flag)?;
                options.format = match value {
                    "summary" => OutputFormat::Summary,
                    "json" => OutputFormat::Json,
                    "dot" => OutputFormat::Dot,
                    _ => {
                        return Err(diagnostic(format!(
                            "--format must be summary, json, or dot (got '{value}')"
                        )));
                    }
                };
            }
            "--output" => {
                options.output = Some(value_after(args, &mut index, flag)?.to_owned());
            }
            "--store" => {
                options.store = Some(value_after(args, &mut index, flag)?.to_owned());
            }
            "--resume" => {
                options.resume_file = Some(value_after(args, &mut index, flag)?.to_owned());
            }
            "--resume-wdbx" => {
                options.resume_wdbx = Some(value_after(args, &mut index, flag)?.to_owned());
            }
            "--dry-run" => options.dry_run = true,
            "--quiet" => options.quiet = true,
            "--verbose" => options.verbose = true,
            _ => {
                return Err(diagnostic(format!(
                    "unknown flag '{flag}' (see `abi wdbx simulate --help`)"
                )));
            }
        }
        index += 1;
    }
    if options.resume_file.is_some() && options.resume_wdbx.is_some() {
        return Err(diagnostic(
            "--resume and --resume-wdbx are mutually exclusive",
        ));
    }
    Ok(options)
}

fn effective_config(options: &Options) -> MultiwayConfig {
    let mut config = MultiwayConfig::new(options.initial.clone(), options.rules.clone());
    config.max_depth = options.max_depth.unwrap_or(config.max_depth);
    config.max_states = options.max_states.unwrap_or(config.max_states);
    config.max_events = options.max_events.unwrap_or(config.max_events);
    config.max_payload = options.max_payload.unwrap_or(config.max_payload);
    config.max_duration_ms = options.max_duration_ms.unwrap_or(config.max_duration_ms);
    config.max_memory_bytes = options.max_memory_bytes.unwrap_or(config.max_memory_bytes);
    config.seed = options.seed.unwrap_or(config.seed);
    config.workers = options.workers.unwrap_or(config.workers);
    config
}

fn validate_config(config: &MultiwayConfig) -> Result<(), Outcome> {
    validate_multiway_config(config).map_err(|detail| match detail {
        ConfigError::NoInitialStates => {
            diagnostic("no initial states; pass --initial or a config file")
        }
        ConfigError::NoRules => diagnostic("no rules; pass --rule, --rules-file, or a config file"),
        ConfigError::TooManyRules => {
            diagnostic(format!("too many rules (max {MAX_MULTIWAY_RULES})"))
        }
        ConfigError::EmptyLhs => diagnostic("a rule has an empty left-hand side"),
        ConfigError::InitialPayloadTooLarge => diagnostic(format!(
            "an initial state exceeds --max-payload ({} bytes)",
            config.max_payload
        )),
        ConfigError::ZeroBound => {
            diagnostic("--depth, --max-states, --max-events, and --max-payload must all be >= 1")
        }
    })
}

fn dry_run(config: &MultiwayConfig, verbose: bool) -> Outcome {
    let mut report = format!(
        "dry-run OK: {} initial state(s), {} rule(s), depth<={}, states<={}, events<={}, payload<={}B, deadline={}ms, memory<={}B, seed={}, workers={}\n",
        config.initial.len(),
        config.rules.len(),
        config.max_depth,
        config.max_states,
        config.max_events,
        config.max_payload,
        config.max_duration_ms,
        config.max_memory_bytes,
        config.seed,
        config.workers
    );
    if verbose {
        for (index, rule) in config.rules.iter().enumerate() {
            writeln!(
                report,
                "  rule {index}: '{}' -> '{}'",
                String::from_utf8_lossy(&rule.lhs),
                String::from_utf8_lossy(&rule.rhs)
            )
            .expect("writing to a String cannot fail");
        }
    }
    Outcome::stderr(report, 0)
}

fn install_cancel_handler() -> Result<(), Outcome> {
    let result = CANCEL_HANDLER.get_or_init(|| {
        ctrlc::set_handler(|| CANCELLED.store(true, Ordering::Release))
            .map_err(|detail| detail.to_string())
    });
    result
        .as_ref()
        .map_err(|detail| failure(format!("cannot install Ctrl-C handler: {detail}")))
        .copied()
}

fn resume_error(detail: MultiwayExportError) -> Outcome {
    match detail {
        MultiwayExportError::AlreadyComplete => {
            diagnostic("experiment already ran to completion; nothing to resume")
        }
        MultiwayExportError::ConfigMismatch => diagnostic(
            "resume config mismatch: rules/initial states must match the persisted experiment (only bounds may change)",
        ),
        MultiwayExportError::UnsupportedFormat => diagnostic(format!(
            "unsupported export format (expected {MULTIWAY_FORMAT_VERSION})"
        )),
        MultiwayExportError::MalformedExport | MultiwayExportError::NonUtf8 => {
            diagnostic("malformed export document")
        }
    }
}

fn read_resume_file(path: &str) -> Result<String, Outcome> {
    read_limited(path, MAX_RESUME_FILE_BYTES)
        .map_err(|detail| diagnostic(format!("cannot read resume export '{path}': {detail}")))
}

fn execute(config: &MultiwayConfig, options: &Options) -> Result<MultiwayResult, Outcome> {
    CANCELLED.store(false, Ordering::Release);
    install_cancel_handler()?;
    if let Some(path) = options.resume_file.as_deref() {
        let export = read_resume_file(path)?;
        return resume_multiway(&export, config, Some(&CANCELLED)).map_err(resume_error);
    }
    if let Some(path) = options.resume_wdbx.as_deref() {
        let paths = paths_from_cli_base(path).map_err(|detail| {
            diagnostic(format!(
                "cannot load experiment from WDBX checkpoint '{path}': {detail}"
            ))
        })?;
        let export = load_multiway_export(&paths, None).map_err(|detail| {
            diagnostic(format!(
                "cannot load experiment from WDBX checkpoint '{path}': {detail}"
            ))
        })?;
        return resume_multiway(&export, config, Some(&CANCELLED)).map_err(resume_error);
    }
    Ok(run_multiway(config, Some(&CANCELLED)))
}

fn append_summary(
    report: &mut String,
    result: &MultiwayResult,
    metrics: &MultiwayMetrics,
    export_hash: &str,
    verbose: bool,
) {
    let elapsed_ms = result.elapsed.as_secs_f64() * 1000.0;
    let elapsed_seconds = (elapsed_ms / 1000.0).max(1e-9);
    writeln!(
        report,
        "multiway simulation\n  states (unique):        {}\n  events:                 {}\n  transitions (unique):   {}\n  termination:            {}\n  exhaustive in domain:   {}\n  mean out-degree:        {:.3} (unique transitions / unique states)\n  max out-degree:         {}\n  median out-degree:      {:.1}\n  convergent states:      {} (distinct-predecessor in-degree > 1)\n  self-loops:             {}\n  cycle present:          {}\n  weakly connected comps: {}\n  payload bytes max/mean: {}/{:.2}\n  runtime:                {:.3} ms ({:.0} states/s, {:.0} events/s)\n  export sha256:          {}",
        metrics.unique_states,
        metrics.event_count,
        metrics.unique_transitions,
        metrics.termination.label(),
        metrics.exhaustive,
        metrics.mean_out_degree,
        metrics.max_out_degree,
        metrics.median_out_degree,
        metrics.convergent_states,
        metrics.self_loops,
        metrics.has_cycle,
        metrics.weakly_connected_components,
        metrics.max_payload_bytes,
        metrics.mean_payload_bytes,
        elapsed_ms,
        f64::from(metrics.unique_states) / elapsed_seconds,
        f64::from(metrics.event_count) / elapsed_seconds,
        export_hash
    )
    .expect("writing to a String cannot fail");
    if verbose {
        report.push_str("  depth  states  events  growth\n");
        for (depth, states) in metrics.states_per_depth.iter().enumerate() {
            let events = metrics.events_per_depth.get(depth).copied().unwrap_or(0);
            let growth = depth
                .checked_sub(1)
                .and_then(|index| metrics.growth_rates.get(index))
                .copied()
                .unwrap_or(0.0);
            writeln!(
                report,
                "  {depth:>5}  {states:>6}  {events:>6}  {growth:>6.2}"
            )
            .expect("writing to a String cannot fail");
        }
    }
}

fn hex_lower(bytes: [u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        write!(output, "{byte:02x}").expect("writing to a String cannot fail");
    }
    output
}

fn write_export(path: &str, payload: &str, kind: &str) -> Result<String, String> {
    std::fs::write(path, payload).map_err(|detail| format!("cannot write '{path}': {detail}"))?;
    Ok(format!("{kind} export written to {path}\n"))
}

fn late_failure(mut report: String, message: impl std::fmt::Display) -> Outcome {
    writeln!(report, "simulate: {message}").expect("writing to a String cannot fail");
    Outcome::stderr(report, 1)
}

fn render_and_persist(
    config: &MultiwayConfig,
    result: &MultiwayResult,
    options: &Options,
) -> Outcome {
    let metrics = compute_multiway_metrics(result);
    let export_json = match export_multiway_json(config, result, &metrics) {
        Ok(export) => export,
        Err(detail) => return failure(format!("canonical export failed: {detail}")),
    };
    let export_hash = multiway_export_hash_hex(export_json.as_bytes());
    let mut report = String::new();
    if !options.quiet {
        append_summary(&mut report, result, &metrics, &export_hash, options.verbose);
    }
    match options.format {
        OutputFormat::Summary => {}
        OutputFormat::Json => {
            if let Some(path) = options.output.as_deref() {
                match write_export(path, &export_json, "canonical JSON") {
                    Ok(message) if !options.quiet => report.push_str(&message),
                    Ok(_) => {}
                    Err(detail) => return late_failure(report, detail),
                }
            } else {
                report.push_str(&export_json);
                report.push('\n');
            }
        }
        OutputFormat::Dot => {
            let dot = match export_multiway_dot(config, result) {
                Ok(dot) => dot,
                Err(detail) => return failure(format!("DOT export failed: {detail}")),
            };
            if let Some(path) = options.output.as_deref() {
                match write_export(path, &dot, "DOT") {
                    Ok(message) if !options.quiet => report.push_str(&message),
                    Ok(_) => {}
                    Err(detail) => return late_failure(report, detail),
                }
            } else {
                report.push_str(&dot);
            }
        }
    }
    if let Some(path) = options.store.as_deref() {
        let paths = match paths_from_cli_base(path) {
            Ok(paths) => paths,
            Err(detail) => {
                return late_failure(
                    report,
                    format!("WDBX persistence to '{path}' failed: {detail}"),
                );
            }
        };
        if let Err(detail) = persist_multiway(paths, config, result, &export_json) {
            return late_failure(
                report,
                format!("WDBX persistence to '{path}' failed: {detail}"),
            );
        }
        if !options.quiet {
            let config_hash = match multiway_config_hash(config) {
                Ok(hash) => hex_lower(hash),
                Err(detail) => {
                    return late_failure(report, format!("config hashing failed: {detail}"));
                }
            };
            writeln!(
                report,
                "persisted to WDBX checkpoint {path} (config {config_hash})"
            )
            .expect("writing to a String cannot fail");
        }
    }
    Outcome::stderr(report, 0)
}

/// Dispatch arguments following `abi wdbx simulate`.
pub(crate) fn run(args: &[String]) -> Outcome {
    if args.first().is_some_and(|token| is_help_token(token)) {
        return Outcome::stderr(HELP.to_owned(), 0);
    }
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(outcome) => return outcome,
    };
    let config = effective_config(&options);
    if let Err(outcome) = validate_config(&config) {
        return outcome;
    }
    if options.dry_run {
        return dry_run(&config, options.verbose);
    }
    let result = match execute(&config, &options) {
        Ok(result) => result,
        Err(outcome) => return outcome,
    };
    render_and_persist(&config, &result, &options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(ToString::to_string).collect()
    }

    fn fixture_path(name: &str) -> std::path::PathBuf {
        let sequence = NEXT_FIXTURE.fetch_add(1, AtomicOrdering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "abi_cli_simulate_{}_{}",
            std::process::id(),
            sequence
        ));
        std::fs::create_dir_all(&directory).expect("fixture directory");
        directory.join(name)
    }

    #[test]
    fn help_and_dry_run_match_the_frozen_surface() {
        assert_eq!(
            run(&strings(&["--help"])),
            Outcome::stderr(HELP.to_owned(), 0)
        );
        assert_eq!(
            run(&strings(&[
                "--initial",
                "A",
                "--rule",
                "A->AB",
                "--dry-run"
            ]))
            .stderr,
            "dry-run OK: 1 initial state(s), 1 rule(s), depth<=5, states<=10000, events<=100000, payload<=4096B, deadline=0ms, memory<=0B, seed=0, workers=1\n"
        );
    }

    #[test]
    fn malformed_rules_and_bounds_are_diagnostics() {
        for arguments in [
            vec!["--initial", "A", "--rule", "AB"],
            vec!["--initial", "A", "--rule", "->B"],
            vec!["--initial", "A"],
            vec!["--rule", "A->B"],
            vec!["--initial", "A", "--rule", "A->B", "--depth", "0"],
            vec!["--initial", "A", "--rule", "A->B", "--workers", "0"],
            vec!["--initial", "A", "--rule", "A->B", "--unknown"],
        ] {
            assert_eq!(run(&strings(&arguments)).exit_code, 2);
        }
    }

    #[test]
    fn canonical_reference_export_has_expected_shape() {
        let outcome = run(&strings(&[
            "--initial",
            "A",
            "--rule",
            "A->AB",
            "--rule",
            "A->BA",
            "--rule",
            "BB->A",
            "--depth",
            "5",
            "--max-states",
            "500",
            "--max-events",
            "5000",
            "--format",
            "json",
            "--quiet",
        ]));
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stderr.contains("\"format\":\"abi-multiway-v1\""));
        assert!(outcome.stderr.contains("\"unique_states\":31"));
        assert!(outcome.stderr.contains("\"unique_transitions\":62"));
        assert!(outcome.stderr.ends_with('\n'));
    }

    #[test]
    fn config_store_and_both_resume_modes_converge() {
        let config_path = fixture_path("config.json");
        let directory = config_path.parent().expect("fixture parent");
        let first_export = directory.join("first.json");
        let resumed_export = directory.join("resumed.json");
        let direct_export = directory.join("direct.json");
        let store = directory.join("store.jsonl");
        std::fs::write(
            &config_path,
            r#"{"initial":["A"],"rules":["A->AB","A->BA","BB->A"],"max_depth":3,"max_states":500,"max_events":5000,"max_payload":64}"#,
        )
        .expect("write config");
        let config = config_path.to_string_lossy();
        let first = first_export.to_string_lossy();
        let resumed = resumed_export.to_string_lossy();
        let direct = direct_export.to_string_lossy();
        let store = store.to_string_lossy();

        assert_eq!(
            run(&strings(&[
                "--config", &config, "--format", "json", "--output", &first, "--store", &store,
                "--quiet",
            ]))
            .exit_code,
            0
        );
        assert_eq!(
            run(&strings(&[
                "--config", &config, "--depth", "5", "--resume", &first, "--quiet",
            ]))
            .exit_code,
            0
        );
        assert_eq!(
            run(&strings(&[
                "--config",
                &config,
                "--depth",
                "5",
                "--resume-wdbx",
                &store,
                "--format",
                "json",
                "--output",
                &resumed,
                "--quiet",
            ]))
            .exit_code,
            0
        );
        assert_eq!(
            run(&strings(&[
                "--config", &config, "--depth", "5", "--format", "json", "--output", &direct,
                "--quiet",
            ]))
            .exit_code,
            0
        );
        assert_eq!(
            std::fs::read(&resumed_export).expect("resumed export"),
            std::fs::read(&direct_export).expect("direct export")
        );
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn late_persistence_failure_preserves_prior_output() {
        let output = fixture_path("export.json");
        let output_text = output.to_string_lossy();
        let outcome = run(&strings(&[
            "--initial",
            "A",
            "--rule",
            "A->AB",
            "--depth",
            "2",
            "--format",
            "json",
            "--output",
            &output_text,
            "--store",
            "/",
        ]));
        assert_eq!(outcome.exit_code, 1);
        assert!(output.is_file(), "the export succeeded before persistence");
        assert!(outcome.stderr.starts_with("multiway simulation\n"));
        assert!(outcome.stderr.contains("canonical JSON export written to "));
        assert!(
            outcome
                .stderr
                .contains("simulate: WDBX persistence to '/' failed:")
        );
        std::fs::remove_dir_all(output.parent().expect("fixture parent")).ok();
    }

    #[test]
    fn bounded_file_reader_rejects_growth_past_the_limit() {
        let rules = fixture_path("oversize-rules.txt");
        let file = std::fs::File::create(&rules).expect("rules file");
        file.set_len(MAX_RULE_FILE_BYTES + 1)
            .expect("extend rules file");
        let rules_text = rules.to_string_lossy();
        let outcome = run(&strings(&["--initial", "A", "--rules-file", &rules_text]));
        assert_eq!(outcome.exit_code, 2);
        assert!(outcome.stderr.contains("file exceeds 1048576 bytes"));
        std::fs::remove_dir_all(rules.parent().expect("fixture parent")).ok();
    }
}
