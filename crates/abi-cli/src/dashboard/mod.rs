//! Diagnostics dashboard (`abi dashboard` / `abi tui`).
//!
//! One-shot stacked digest for non-TTY / `--once` / `--json`. On a TTY without
//! those flags, enters a raw-mode refresh loop (q/Esc quit, r refresh, h/l and
//! 1-5 pane select). GPU fields use honest `abi-gpu` disclosure (native kernels
//! not linked).

mod input;
mod layout;
mod options;
mod render;
mod state;

#[cfg(test)]
mod tests;

use std::io::{self, IsTerminal};

use crate::app::Outcome;

use input::run_interactive;
use options::parse_options;
use render::{render_json, render_pane_list, render_text};
use state::collect_state;

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
