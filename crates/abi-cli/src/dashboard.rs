//! Diagnostics dashboard (`abi dashboard` / `abi tui`).
//!
//! One-shot stacked digest for non-TTY / `--once` / `--json`. On a TTY without
//! those flags, enters a raw-mode refresh loop (q/Esc quit, r refresh, h/l and
//! 1-5 pane select). GPU fields use honest `abi-gpu` disclosure (native kernels
//! not linked).

use std::fmt::Write as _;
use std::io::{self, IsTerminal, Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    /// True when the render is for the interactive raw-mode loop footer.
    interactive: bool,
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

/// Reverse video plus bold red — how the dashboard marks the focused pane.
///
/// Kept as one constant because it is contract: `tools/run_tui_smoke.sh` asserts
/// this exact byte sequence precedes the selected pane's title.
const FOCUS_STYLE: &str = "\x1b[7m\x1b[1;31m";
/// SGR reset, closing [`FOCUS_STYLE`].
const STYLE_RESET: &str = "\x1b[0m";

/// Render a pane's top border, highlighting it when it is the focused pane.
///
/// The reset lands *before* the newline: leaving it after would let reverse video
/// bleed into the following row, which is the usual way this kind of highlight
/// goes wrong. With `color` false (`--plain` / `--no-color`) no escapes are
/// emitted at all, which is what makes those flags meaningful for the text
/// render rather than metadata that only shows up in `--json`.
fn append_pane_header(out: &mut String, title: &str, selected: bool, color: bool) {
    let mut line = String::new();
    append_border(&mut line, "┌", title, "┐");

    if selected && color {
        out.push_str(FOCUS_STYLE);
        out.push_str(line.trim_end_matches('\n'));
        out.push_str(STYLE_RESET);
        out.push('\n');
    } else {
        out.push_str(&line);
    }
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
        interactive: false,
    }
}

fn clear_screen(out: &mut dyn Write) {
    let _ = write!(out, "\x1b[2J\x1b[H");
    let _ = out.flush();
}

/// Interactive raw-mode loop for a real TTY. Returns a final one-shot frame on exit
/// (for capture) after restoring the terminal.
fn run_interactive(options: Options) -> Outcome {
    #[cfg(not(unix))]
    {
        let state = collect_state(&options);
        let text = render_text(&state);
        return Outcome::stderr(
            format!(
                "{text}\nnote: interactive raw-mode dashboard requires a Unix TTY; rendered one-shot.\n"
            ),
            0,
        );
    }

    #[cfg(unix)]
    {
        let _raw = match crate::terminal::RawMode::enter() {
            Ok(r) => r,
            Err(err) => {
                let state = collect_state(&options);
                let text = render_text(&state);
                return Outcome::stderr(
                    format!("{text}\nnote: raw-mode unavailable ({err}); one-shot fallback.\n"),
                    0,
                );
            }
        };

        let mut options = options;
        let interval =
            Duration::from_millis(u64::try_from(options.refresh_interval_ms).unwrap_or(1000));
        let mut last_paint = Instant::now();
        let mut force_paint = true;
        let mut stderr = io::stderr();
        let mut stdin = io::stdin();
        let mut buf = [0_u8; 16];

        loop {
            if force_paint || last_paint.elapsed() >= interval {
                let mut state = collect_state(&options);
                state.interactive = true;
                state.selected_pane = options.initial_pane.min(PANES.len() - 1);
                clear_screen(&mut stderr);
                let frame = render_text(&state);
                let _ = write!(stderr, "{frame}");
                let _ = stderr.flush();
                last_paint = Instant::now();
                force_paint = false;
            }

            match stdin.read(&mut buf) {
                Ok(0) => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Ok(n) => {
                    for &key in &buf[..n] {
                        match key {
                            b'q' | b'Q' | 0x1b => {
                                let mut state = collect_state(&options);
                                state.interactive = true;
                                let frame = render_text(&state);
                                return Outcome::stderr(frame, 0);
                            }
                            b'r' | b'R' => {
                                force_paint = true;
                            }
                            b'h' | b'H' => {
                                if options.initial_pane == 0 {
                                    options.initial_pane = PANES.len() - 1;
                                } else {
                                    options.initial_pane -= 1;
                                }
                                force_paint = true;
                            }
                            b'l' | b'L' => {
                                options.initial_pane = (options.initial_pane + 1) % PANES.len();
                                force_paint = true;
                            }
                            b'1'..=b'5' => {
                                options.initial_pane = usize::from(key - b'1');
                                force_paint = true;
                            }
                            _ => {}
                        }
                    }
                }
                Err(err) if err.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => {
                    let state = collect_state(&options);
                    return Outcome::stderr(render_text(&state), 0);
                }
            }
        }
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
        append_pane_header(&mut out, title, idx == ds.selected_pane, ds.color);
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
    let mode = if ds.interactive {
        format!("interactive raw-mode (refresh every {secs:.1}s)")
    } else {
        format!("one-shot snapshot ({secs:.1}s refresh is CLI metadata only)")
    };
    let _ = writeln!(
        out,
        "\n[q/Esc] Quit  [r] Refresh  [h/l] Select  [1-5] Panes  {mode}"
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

    // One-shot for JSON, --once, or non-TTY stderr (scripts / CI / pipes).
    let want_interactive = !options.force_one_shot
        && options.format == Format::Text
        && io::stderr().is_terminal()
        && io::stdin().is_terminal();

    if want_interactive {
        return run_interactive(options);
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
    fn selected_pane_is_highlighted_and_plain_suppresses_it() {
        // The focused pane must be visually marked. Zig did this and the Rust
        // port initially dropped it: `color` was stored and reported in `--json`
        // but never used to emit anything, so `--pane` had no visible effect and
        // `--plain` was a no-op for the text render.
        let colored = run(&["--pane".to_owned(), "memory".to_owned()]);
        assert_eq!(colored.exit_code, 0, "{}", colored.stderr);
        assert!(
            colored.stderr.contains(&format!("{FOCUS_STYLE}┌ Memory")),
            "focused pane must carry the highlight"
        );
        assert!(
            colored.stderr.contains(STYLE_RESET),
            "the highlight must be reset so it does not bleed"
        );
        // Unfocused panes stay unstyled.
        assert!(colored.stderr.contains("┌ System"));
        assert!(!colored.stderr.contains(&format!("{FOCUS_STYLE}┌ System")));

        let plain = run(&[
            "--pane".to_owned(),
            "memory".to_owned(),
            "--plain".to_owned(),
        ]);
        assert_eq!(plain.exit_code, 0, "{}", plain.stderr);
        assert!(
            !plain.stderr.contains(FOCUS_STYLE),
            "--plain must emit no escapes"
        );
        assert!(plain.stderr.contains("┌ Memory"));
    }

    #[test]
    fn highlight_reset_precedes_the_newline() {
        // Resetting after the newline would let reverse video bleed across the
        // following row, which is the usual way this bug shows up in a terminal
        // but is invisible to a "contains" assertion.
        let out = run(&["--pane".to_owned(), "memory".to_owned()]);
        let line = out
            .stderr
            .lines()
            .find(|l| l.contains("┌ Memory"))
            .expect("the memory pane header is rendered");
        assert!(line.starts_with(FOCUS_STYLE), "got {line:?}");
        assert!(line.ends_with(STYLE_RESET), "got {line:?}");
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
