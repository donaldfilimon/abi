//! Renderers and pane geometry for the diagnostics dashboard.
//!
//! `render_text` is the stacked digest; `render_json` is the scriptable form;
//! `render_pane_list` backs `--list-panes`. The geometry helpers map terminal
//! cells back onto the panes `render_text` lays out, so they must move with it.

use std::fmt::Write as _;

use serde_json::{Value, json};

use super::layout::{append_border, append_metric, append_pane_header, append_row, fit};
use super::state::dashboard_health;
use super::{
    DIAG_WIDTH, DashboardState, Format, LABEL_WIDTH, MAX_PLUGIN_ROWS, Options, PANES, VALUE_WIDTH,
};

pub(super) fn pane_height(ds: &DashboardState, pane: usize) -> usize {
    match PANES[pane].0 {
        "system" => 5,
        "plugins" => {
            let shown = ds.plugin_names.len().min(MAX_PLUGIN_ROWS);
            3 + shown + usize::from(ds.plugin_names.len() > shown)
        }
        "storage" | "scheduler" => 7,
        "memory" => 6,
        _ => 0,
    }
}

/// Map a one-based terminal cell to the pane occupying it in `render_text`.
/// The dashboard begins at row 1, its title band consumes four rows, and every
/// pane includes its header/content/bottom-border rows. Clicks outside the
/// 70-column dashboard or on the footer are ignored.
pub(super) fn pane_at_position(ds: &DashboardState, column: u16, row: u16) -> Option<usize> {
    let column = usize::from(column);
    let row = usize::from(row);
    if !(1..=DIAG_WIDTH + 2).contains(&column) || row < 5 {
        return None;
    }

    let mut first_row = 5;
    for pane in 0..PANES.len() {
        if ds.compact && pane != ds.selected_pane {
            continue;
        }
        let end_row = first_row + pane_height(ds, pane);
        if (first_row..end_row).contains(&row) {
            return Some(pane);
        }
        first_row = end_row;
    }
    None
}

pub(super) fn render_text(ds: &DashboardState) -> String {
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
        "\n[q/Esc] Quit  [r] Refresh  [mouse/Tab/Shift-Tab or h/l] Select  [1-5] Panes  {mode}"
    );
    out
}

pub(super) fn render_json(ds: &DashboardState) -> String {
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

pub(super) fn render_pane_list(options: &Options) -> String {
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
