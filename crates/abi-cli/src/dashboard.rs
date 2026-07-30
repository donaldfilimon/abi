//! Diagnostics dashboard one-shot render (`abi dashboard` / `abi tui`).
//!
//! Ported from the non-interactive path of `src/cli/handlers/dashboard.zig`.
//! Interactive raw-mode refresh is **not** linked — every invocation produces
//! a one-shot stacked digest (matching Zig's non-TTY / `--once` fallback).
//! GPU fields use honest `abi-gpu` disclosure (native kernels not linked).

use std::fmt::Write as _;
use std::sync::Arc;

use abi_core::{MemoryTracker, Scheduler, TaskPriority};
use abi_plugins::PluginManager;
use serde_json::{Value, json};

use crate::app::Outcome;

const DIAG_WIDTH: usize = 68;
const LABEL_WIDTH: usize = 25;
const VALUE_WIDTH: usize = 40;
const MAX_PLUGIN_ROWS: usize = 6;
const DEFAULT_REFRESH_MS: i32 = 1000;
const MIN_REFRESH_MS: i32 = 100;
const MAX_REFRESH_MS: i32 = 60_000;

/// Pane metadata matching Zig's `DASHBOARD_PANES`.
const PANES: [(&str, &str, char); 5] = [
    ("system", "System", '1'),
    ("plugins", "Plugins", '2'),
    ("storage", "WDBX Storage", '3'),
    ("scheduler", "Scheduler", '4'),
    ("memory", "Memory", '5'),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Text,
    Json,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // mirrors Zig's DashboardOptions flag bag
struct Options {
    initial_pane: usize,
    color: bool,
    compact: bool,
    force_one_shot: bool,
    refresh_interval_ms: i32,
    format: Format,
    list_panes: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            initial_pane: 0,
            color: true,
            compact: false,
            force_one_shot: false,
            refresh_interval_ms: DEFAULT_REFRESH_MS,
            format: Format::Text,
            list_panes: false,
        }
    }
}

/// Snapshot collected for render.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // mirrors Zig's DashboardState flags
struct DashboardState {
    gpu_backend: String,
    gpu_accelerated: bool,
    gpu_linked: bool,
    plugin_count: usize,
    plugin_names: Vec<String>,
    wdbx_blocks: usize,
    wdbx_vectors: usize,
    wdbx_entries: usize,
    wdbx_spatial_records: usize,
    scheduler_source: &'static str,
    scheduler_running: usize,
    scheduler_pending: usize,
    scheduler_completed: usize,
    scheduler_failed: usize,
    memory_source: &'static str,
    memory_peak: usize,
    memory_current: usize,
    memory_leaked: usize,
    selected_pane: usize,
    refresh_interval_ms: i32,
    compact: bool,
    color: bool,
}

fn dashboard_health(ds: &DashboardState) -> &'static str {
    if ds.scheduler_failed > 0 || ds.memory_leaked > 0 {
        return "attention";
    }
    if ds.gpu_accelerated && ds.gpu_linked {
        return "nominal";
    }
    // Zig labels Metal-linked CPU-SIMD as "cpu"; we always land here until
    // native kernels are linked.
    "cpu"
}

fn pane_index_for_token(token: &str) -> Option<usize> {
    if token.len() == 1 {
        let key = token.as_bytes()[0];
        for (idx, (_, _, hotkey)) in PANES.iter().enumerate() {
            if *hotkey as u8 == key {
                return Some(idx);
            }
        }
    }
    for (idx, (name, _, _)) in PANES.iter().enumerate() {
        if token.eq_ignore_ascii_case(name) {
            return Some(idx);
        }
        if *name == "storage" && token.eq_ignore_ascii_case("wdbx") {
            return Some(idx);
        }
    }
    None
}

fn valid_refresh_interval(raw: u64) -> Option<i32> {
    let ms = i32::try_from(raw).ok()?;
    if (MIN_REFRESH_MS..=MAX_REFRESH_MS).contains(&ms) {
        Some(ms)
    } else {
        None
    }
}

fn fit(s: &str, width: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= width {
        let mut out = s.to_string();
        out.extend(std::iter::repeat_n(' ', width - chars.len()));
        out
    } else if width == 0 {
        String::new()
    } else if width == 1 {
        "~".to_string()
    } else {
        let mut out: String = chars.into_iter().take(width - 1).collect();
        out.push('~');
        out
    }
}

fn append_rule(out: &mut String, count: usize) {
    for _ in 0..count {
        out.push('─');
    }
}

fn append_border(out: &mut String, left: &str, title: &str, right: &str) {
    out.push_str(left);
    if title.is_empty() {
        append_rule(out, DIAG_WIDTH);
    } else {
        out.push(' ');
        let title_w = title.chars().count().min(DIAG_WIDTH.saturating_sub(4));
        out.push_str(&fit(title, title_w));
        out.push(' ');
        let used = title_w + 2;
        if used < DIAG_WIDTH {
            append_rule(out, DIAG_WIDTH - used);
        }
    }
    out.push_str(right);
    out.push('\n');
}

fn append_row(out: &mut String, label: &str, value: &str) {
    out.push_str("│ ");
    out.push_str(&fit(label, LABEL_WIDTH));
    out.push(' ');
    out.push_str(&fit(value, VALUE_WIDTH));
    out.push_str(" │\n");
}

fn append_metric(out: &mut String, label: &str, value: usize) {
    append_row(out, label, &value.to_string());
}

fn collect_state(options: &Options) -> DashboardState {
    let gpu = abi_gpu::detect_backend();
    let mut manager = PluginManager::new();
    manager.load_bundled();
    let plugin_names: Vec<String> = manager.list().iter().map(|p| p.name.clone()).collect();
    let plugin_count = manager.plugin_count();

    let tracker = Arc::new(MemoryTracker::new());
    let scheduler = Scheduler::new().with_memory_tracker(Arc::clone(&tracker));
    scheduler.submit("dashboard-init", TaskPriority::Normal, Box::new(|| Ok(())));
    scheduler.submit("wdbx-snapshot", TaskPriority::Low, Box::new(|| Ok(())));
    let _ = scheduler.run_all();
    let stats = scheduler.stats();
    let mem = tracker.snapshot();

    DashboardState {
        gpu_backend: gpu.backend.name().to_string(),
        gpu_accelerated: gpu.accelerated,
        gpu_linked: false, // native kernels are not linked in the Rust port
        plugin_count,
        plugin_names,
        // Ephemeral probe store — never opens the user's durable path.
        wdbx_blocks: 0,
        wdbx_vectors: 0,
        wdbx_entries: 0,
        wdbx_spatial_records: 0,
        scheduler_source: "CLI dashboard (live)",
        scheduler_running: stats.running,
        scheduler_pending: stats.pending,
        scheduler_completed: stats.completed,
        scheduler_failed: stats.failed,
        memory_source: "MemoryTracker (live)",
        memory_peak: mem.peak_usage,
        memory_current: mem.current_usage,
        // Empty probe tasks do not allocate through the tracker.
        memory_leaked: 0,
        selected_pane: options.initial_pane.min(PANES.len() - 1),
        refresh_interval_ms: options.refresh_interval_ms,
        compact: options.compact,
        color: options.color,
    }
}

fn render_text(ds: &DashboardState) -> String {
    let health = dashboard_health(ds);
    let mut out = String::new();

    // Header band (matches Zig box-drawing title).
    append_border(&mut out, "╔", "", "╗");
    out.push_str("│ ");
    out.push_str(&fit(
        "ABI Diagnostics Dashboard operational snapshot",
        DIAG_WIDTH,
    ));
    out.push_str(" │\n");
    out.push_str("│ ");
    out.push_str(&fit("health", LABEL_WIDTH));
    out.push(' ');
    out.push_str(&fit(health, VALUE_WIDTH));
    out.push_str(" │\n");
    append_border(&mut out, "╚", "", "╝");

    let visible: Vec<usize> = if ds.compact {
        vec![ds.selected_pane]
    } else {
        (0..PANES.len()).collect()
    };

    for idx in visible {
        let (_, title, _) = PANES[idx];
        append_border(&mut out, "┌", title, "┐");
        match PANES[idx].0 {
            "system" => {
                append_row(&mut out, "GPU backend", &ds.gpu_backend);
                append_row(
                    &mut out,
                    "accelerated",
                    if ds.gpu_accelerated { "yes" } else { "no" },
                );
                append_row(
                    &mut out,
                    "native linked",
                    if ds.gpu_linked { "yes" } else { "no" },
                );
            }
            "plugins" => {
                append_metric(&mut out, "Registered", ds.plugin_count);
                let shown = ds.plugin_names.len().min(MAX_PLUGIN_ROWS);
                for name in ds.plugin_names.iter().take(shown) {
                    append_row(&mut out, "plugin", name);
                }
                if ds.plugin_names.len() > shown {
                    let more = format!("+{} more registered", ds.plugin_names.len() - shown);
                    append_row(&mut out, "plugin", &more);
                }
            }
            "storage" => {
                append_row(&mut out, "scope", "ephemeral CLI probe");
                append_metric(&mut out, "Block chain", ds.wdbx_blocks);
                append_metric(&mut out, "Vectors", ds.wdbx_vectors);
                append_metric(&mut out, "KV Entries", ds.wdbx_entries);
                append_metric(&mut out, "Spatial 3D", ds.wdbx_spatial_records);
            }
            "scheduler" => {
                append_row(&mut out, "source", ds.scheduler_source);
                append_metric(&mut out, "Running", ds.scheduler_running);
                append_metric(&mut out, "Pending", ds.scheduler_pending);
                append_metric(&mut out, "Completed", ds.scheduler_completed);
                append_metric(&mut out, "Failed", ds.scheduler_failed);
            }
            "memory" => {
                append_row(&mut out, "source", ds.memory_source);
                append_metric(&mut out, "Peak bytes", ds.memory_peak);
                append_metric(&mut out, "Current bytes", ds.memory_current);
                append_metric(&mut out, "Leaked bytes", ds.memory_leaked);
            }
            _ => {}
        }
        append_border(&mut out, "└", "", "┘");
    }

    let secs = f64::from(ds.refresh_interval_ms) / 1000.0;
    // Zig interactive path says "live snapshot every 1s"; we always one-shot
    // and disclose that interactive raw-mode is not linked.
    let _ = writeln!(
        out,
        "\n[q/Esc] Quit  [r] Refresh  [h/l] Select  [1-5] Panes  one-shot snapshot (interactive not linked; {secs}s refresh is CLI metadata only)"
    );
    out
}

fn render_json(ds: &DashboardState) -> String {
    let health = dashboard_health(ds);
    let panes_meta: Vec<Value> = PANES
        .iter()
        .enumerate()
        .map(|(idx, (name, title, hotkey))| {
            let visible = !ds.compact || idx == ds.selected_pane;
            json!({
                "name": name,
                "title": title,
                "hotkey": hotkey.to_string(),
                "selected": idx == ds.selected_pane,
                "visible": visible,
            })
        })
        .collect();
    let visible_panes: Vec<&str> = if ds.compact {
        vec![PANES[ds.selected_pane].0]
    } else {
        PANES.iter().map(|(n, _, _)| *n).collect()
    };

    let doc = json!({
        "type": "abi.dashboard",
        "health": health,
        "selected_pane": PANES[ds.selected_pane].0,
        "refresh_interval_ms": ds.refresh_interval_ms,
        "layout": {
            "format": "json",
            "color": ds.color,
            "compact": ds.compact,
            "visible_panes": visible_panes,
            "panes": panes_meta,
        },
        "gpu": {
            "backend": ds.gpu_backend,
            "accelerated": ds.gpu_accelerated,
            "linked": ds.gpu_linked,
        },
        "plugins": {
            "count": ds.plugin_count,
            "names": ds.plugin_names,
        },
        "wdbx": {
            "blocks": ds.wdbx_blocks,
            "vectors": ds.wdbx_vectors,
            "kv_entries": ds.wdbx_entries,
            "spatial_records": ds.wdbx_spatial_records,
        },
        "scheduler": {
            "source": ds.scheduler_source,
            "running": ds.scheduler_running,
            "pending": ds.scheduler_pending,
            "completed": ds.scheduler_completed,
            "failed": ds.scheduler_failed,
        },
        "memory": {
            "source": ds.memory_source,
            "peak_bytes": ds.memory_peak,
            "current_bytes": ds.memory_current,
            "leaked_bytes": ds.memory_leaked,
        },
    });
    format!("{doc}\n")
}

fn render_pane_list(options: &Options) -> String {
    if options.format == Format::Json {
        let panes: Vec<Value> = PANES
            .iter()
            .enumerate()
            .map(|(idx, (name, title, hotkey))| {
                json!({
                    "name": name,
                    "title": title,
                    "hotkey": hotkey.to_string(),
                    "selected": idx == options.initial_pane,
                })
            })
            .collect();
        return format!(
            "{}\n",
            json!({
                "type": "abi.dashboard.panes",
                "selected_pane": PANES[options.initial_pane.min(PANES.len()-1)].0,
                "panes": panes,
            })
        );
    }
    let mut out = String::from("Dashboard panes:\n");
    for (idx, (name, title, hotkey)) in PANES.iter().enumerate() {
        let mark = if idx == options.initial_pane {
            '*'
        } else {
            ' '
        };
        let _ = writeln!(out, "{mark} {name} ({title}) hotkey={hotkey}");
    }
    out
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut options = Options::default();
    let mut i = 0;
    while i < args.len() {
        let tok = args[i].as_str();
        match tok {
            "--plain" | "--no-color" => options.color = false,
            "--compact" => options.compact = true,
            "--once" => options.force_one_shot = true,
            "--json" => options.format = Format::Json,
            "--list-panes" => options.list_panes = true,
            "--pane" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Err("missing --pane value".into());
                };
                let Some(idx) = pane_index_for_token(value) else {
                    return Err(format!("unknown pane '{value}'"));
                };
                options.initial_pane = idx;
            }
            "--interval" => {
                i += 1;
                let Some(value) = args.get(i) else {
                    return Err("missing --interval value".into());
                };
                let Ok(raw) = value.parse::<u64>() else {
                    return Err(format!("invalid --interval '{value}'"));
                };
                let Some(ms) = valid_refresh_interval(raw) else {
                    return Err(format!(
                        "--interval must be {MIN_REFRESH_MS}-{MAX_REFRESH_MS} ms"
                    ));
                };
                options.refresh_interval_ms = ms;
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown flag '{flag}'"));
            }
            other => {
                return Err(format!("unexpected argument '{other}'"));
            }
        }
        i += 1;
    }
    Ok(options)
}

/// Dispatch `abi dashboard` / `abi tui` (args after the command token).
pub(crate) fn run(args: &[String]) -> Outcome {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(msg) => {
            return Outcome::stderr(
                format!(
                    "error: {msg}\nusage: abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]\n"
                ),
                2,
            );
        }
    };

    if options.list_panes {
        return Outcome::stderr(render_pane_list(&options), 0);
    }

    let state = collect_state(&options);
    let text = match options.format {
        Format::Text => render_text(&state),
        Format::Json => render_json(&state),
    };
    // Dashboard prints to stderr, matching Zig's DebugWriter → std.debug.
    Outcome::stderr(text, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_panes_prints_all_five() {
        let outcome = run(&["--list-panes".to_owned()]);
        assert_eq!(outcome.exit_code, 0);
        assert!(outcome.stderr.contains("Dashboard panes:"));
        for name in ["system", "plugins", "storage", "scheduler", "memory"] {
            assert!(outcome.stderr.contains(name), "missing {name}");
        }
    }

    #[test]
    fn once_renders_all_five_panels() {
        let outcome = run(&["--once".to_owned(), "--plain".to_owned()]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stderr.contains("ABI Diagnostics Dashboard"));
        assert!(outcome.stderr.contains("System"));
        assert!(outcome.stderr.contains("Plugins"));
        assert!(outcome.stderr.contains("WDBX Storage"));
        assert!(outcome.stderr.contains("Scheduler"));
        assert!(outcome.stderr.contains("Memory"));
        assert!(outcome.stderr.contains("Registered"));
        // Honest GPU disclosure
        assert!(outcome.stderr.contains("accelerated"));
        assert!(outcome.stderr.contains("native linked"));
        assert!(!outcome.stderr.contains("native linked             yes"));
    }

    #[test]
    fn json_snapshot_is_parseable_and_claim_honest() {
        let outcome = run(&["--json".to_owned()]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        let v: Value = serde_json::from_str(outcome.stderr.trim()).expect("json");
        assert_eq!(v["type"], "abi.dashboard");
        assert_eq!(v["plugins"]["count"], 16);
        assert_eq!(v["gpu"]["linked"], false);
        assert_eq!(v["gpu"]["accelerated"], false);
        assert_eq!(v["health"], "cpu");
        assert_eq!(v["scheduler"]["completed"], 2);
        assert_eq!(v["wdbx"]["blocks"], 0);
        assert_eq!(v["wdbx"]["vectors"], 0);
    }

    #[test]
    fn compact_shows_only_selected_pane() {
        let outcome = run(&[
            "--compact".to_owned(),
            "--pane".to_owned(),
            "scheduler".to_owned(),
            "--plain".to_owned(),
        ]);
        assert_eq!(outcome.exit_code, 0, "{}", outcome.stderr);
        assert!(outcome.stderr.contains("Scheduler"));
        // Other panel titles should not appear as panel headers
        assert!(!outcome.stderr.contains("┌ Plugins"));
        assert!(!outcome.stderr.contains("┌ System"));
    }

    #[test]
    fn pane_token_aliases() {
        assert_eq!(pane_index_for_token("1"), Some(0));
        assert_eq!(pane_index_for_token("wdbx"), Some(2));
        assert_eq!(pane_index_for_token("storage"), Some(2));
        assert_eq!(pane_index_for_token("memory"), Some(4));
        assert_eq!(pane_index_for_token("nope"), None);
    }

    #[test]
    fn bad_flag_is_usage() {
        let outcome = run(&["--bogus".to_owned()]);
        assert_eq!(outcome.exit_code, 2);
    }
}
