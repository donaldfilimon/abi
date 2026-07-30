//! The frozen top-level command metadata table.
//!
//! Ported from `src/cli/usage.zig`. That module's own header says it is
//! deliberately std-only and isolated from the feature graph so help can be
//! rendered and contract-tested without pulling in the rest of the build; the
//! same isolation applies here — this module depends on nothing outside the
//! Rust standard library.
//!
//! **Do not add, remove, reorder, or reword a command here without updating
//! `tests/golden/help.json` and re-verifying against the Zig binary first** — this
//! table is asserted field-for-field against captured output in
//! `crates/abi-cli/tests/golden.rs`, and once the Zig source is deleted in step
//! 11, this table plus the goldens are the surviving record of the contract.

use std::fmt;

/// Grouping used to render the top-level command list, in a fixed display order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Category {
    /// Help, completion.
    Core,
    /// AI pipeline commands.
    Ai,
    /// Data commands: plugins, WDBX.
    Data,
    /// System/operational commands.
    System,
    /// Network-facing commands.
    Network,
}

impl Category {
    /// Display order for `abi help` and `abi help --json`.
    pub const ORDER: [Self; 5] = [
        Self::Core,
        Self::Ai,
        Self::Data,
        Self::System,
        Self::Network,
    ];

    /// The heading printed above this category's commands.
    #[must_use]
    pub const fn title(self) -> &'static str {
        match self {
            Self::Core => "Core",
            Self::Ai => "AI",
            Self::Data => "Data",
            Self::System => "System",
            Self::Network => "Network",
        }
    }

    /// The lowercase name used in `--json` output.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Ai => "ai",
            Self::Data => "data",
            Self::System => "system",
            Self::Network => "network",
        }
    }
}

impl fmt::Display for Category {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One top-level command's frozen metadata.
#[derive(Debug, Clone, Copy)]
pub struct Command {
    /// The command token, e.g. `"wdbx"`.
    pub name: &'static str,
    /// One-line invocation form.
    pub usage: &'static str,
    /// One-line description, shown in the top-level listing.
    pub summary: &'static str,
    /// Which group this command is listed under.
    pub category: Category,
    /// Extended prose shown by `abi help <command>`. Empty for commands with
    /// nothing more to say than their summary.
    pub details: &'static str,
    /// Example invocations shown by `abi help <command>`.
    pub examples: &'static [&'static str],
}

/// The frozen set of 13 top-level commands, in declaration order.
///
/// Declaration order is not display order — see [`Category::ORDER`] for that —
/// but it is the order `--json` emits `commands` in, and that is part of the
/// contract too.
pub const COMMANDS: &[Command] = &[
    Command {
        name: "help",
        usage: "abi help [--json|--completion <bash|zsh|fish>] [command] [subcommand]",
        summary: "Show top-level, command, subcommand, shortcut, JSON, or shell-completion help",
        category: Category::Core,
        details: "--json prints typed command/subcommand metadata plus top-level shortcut and completion-shell metadata for automation and docs tooling. --completion emits a shell completion script for bash, zsh, or fish.",
        examples: &[
            "abi help --json",
            "abi help --completion bash",
            "abi help complete",
            "abi help --tui",
            "abi help wdbx cluster",
            "abi help --json dashboard",
        ],
    },
    Command {
        name: "complete",
        usage: "abi complete [--live] [--confirm] [--learn] [--stream] [--neural] [--model <id>] <input>",
        summary: "Run completion through local ABI agent pipeline; --model selects catalog id, --live enables explicit live transport, --neural is in-process char-LM demo",
        category: Category::Ai,
        details: "--model selects a catalog id, --live enables explicit live transport, --learn runs the SEA self-learning loop, --neural selects the in-process character-level demo LM (mutually exclusive with --model; not a production LLM), and apple-fm requires --confirm. Use `--` before a prompt that starts with `-`.",
        examples: &[
            "abi complete \"summarize repository status\"",
            "abi complete -- --literal-leading-dash",
            "abi complete --learn --model claude-fable-5 \"plan next repair\"",
            "abi complete --neural \"hello\"",
        ],
    },
    Command {
        name: "train",
        usage: "abi train <input>",
        summary: "Run the AI pipeline compatibility trainer",
        category: Category::Ai,
        details: "",
        examples: &["abi train \"local routing note\""],
    },
    Command {
        name: "agent",
        usage: "abi agent <plan|train|tui|multi|spawn|browser|os> ...",
        summary: "Agent planning, training, TUI, multi-worker orchestration, browser planning, OS-control",
        category: Category::Ai,
        details: "Subcommands: plan, train, tui, multi, spawn (custom workers), browser (local orchestration), os dry-run/execute.",
        examples: &[
            "abi agent plan \"inspect WDBX persistence\"",
            "abi agent tui",
            "abi agent multi \"coordinate release\"",
            "abi agent spawn --workers \"scout|Explore pages|explore,browser\" \"task\"",
            "abi agent browser --url https://example.com \"open docs\"",
            "abi agent os dry-run ls",
        ],
    },
    Command {
        name: "backends",
        usage: "abi backends",
        summary: "Show GPU, accelerator, shader, MLIR status, feature flags, and version info",
        category: Category::System,
        details: "Reports truthful linked/native/fallback status for local compute backends, lists all build-time feature flags with their enabled/disabled status, and shows framework version and build info.",
        examples: &["abi backends"],
    },
    Command {
        name: "plugin",
        usage: "abi plugin list | run <name> [input]",
        summary: "Inspect or execute installed plugins",
        category: Category::Data,
        details: "The list path reads the generated plugin registry; run loads bundled plugins and dispatches by plugin name.",
        examples: &["abi plugin list", "abi plugin run hello \"input\""],
    },
    Command {
        name: "auth",
        usage: "abi auth <signin|logout|status>",
        summary: "Manage external-service authentication state",
        category: Category::System,
        details: "Credential helpers are local; live connectors still require explicit credentials and live transport selection.",
        examples: &["abi auth status"],
    },
    Command {
        name: "twilio",
        usage: "abi twilio simulate <input>",
        summary: "Run a local Twilio voice-agent simulation",
        category: Category::Network,
        details: "Offline simulation only; live Twilio transport requires explicit connector credentials and live mode elsewhere.",
        examples: &["abi twilio simulate \"billing question\""],
    },
    Command {
        name: "tui",
        usage: "abi tui [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]",
        summary: "Render the diagnostics dashboard",
        category: Category::System,
        details: "Launches the interactive diagnostics dashboard when stdin is a terminal; otherwise renders a one-shot dashboard frame. --pane accepts system, plugins, storage/wdbx, scheduler, memory, or 1-5. --plain/--no-color omit ANSI styling for logs. --compact renders only the selected pane. --once forces one-shot output; --interval sets the interactive refresh cadence in milliseconds. --json emits one machine-readable snapshot. --list-panes prints pane names, titles, and hotkeys without launching the dashboard.",
        examples: &[
            "abi tui",
            "abi tui --pane memory",
            "abi tui --compact --pane scheduler",
            "abi tui --plain --once",
            "abi tui --interval 250",
            "abi tui --json",
            "abi tui --list-panes",
        ],
    },
    Command {
        name: "dashboard",
        usage: "abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]",
        summary: "Render the diagnostics dashboard",
        category: Category::System,
        details: "Alias of `abi tui` for the diagnostics dashboard. --pane selects the initially highlighted diagnostics pane; --plain/--no-color omit ANSI styling. --compact renders only the selected pane. --once forces one-shot output; --interval sets the interactive refresh cadence in milliseconds. --json emits one machine-readable snapshot. --list-panes prints pane names, titles, and hotkeys without launching the dashboard.",
        examples: &[
            "abi dashboard",
            "abi dashboard --pane scheduler",
            "abi dashboard --compact --pane scheduler",
            "abi dashboard --plain --once",
            "abi dashboard --interval 250",
            "abi dashboard --json",
            "abi dashboard --list-panes",
        ],
    },
    Command {
        name: "wdbx",
        usage: "abi wdbx <db|block|query|benchmark|simulate|cluster|compute|secure|gpu|api> ...",
        summary: "Operate WDBX storage, WAL, blocks, stats, and demos",
        category: Category::Data,
        details: "Subcommands: db init|verify|compact, block insert|get, query [--limit/--json/--text/--persona], benchmark, simulate (bounded multiway rewriting experiments), cluster status|demo|serve, compute info, secure demo, gpu info, api serve. Query runs hybrid semantic\u{d7}temporal\u{d7}causal\u{d7}persona ranking over a recovered store.",
        examples: &[
            "abi wdbx db verify",
            "abi wdbx query ./store.jsonl \"local memory\" --limit 5 --json",
        ],
    },
    Command {
        name: "scheduler",
        usage: "abi scheduler status",
        summary: "Report scheduler and memory tracker status",
        category: Category::System,
        details: "Runs a one-shot scheduler status task and reports attached memory tracker counters.",
        examples: &["abi scheduler status"],
    },
    Command {
        name: "nn",
        usage: "abi nn train \"<text>\" | train --jsonl <path> [--field <name>] | sample --text \"<corpus>\" --seed <char> --n <k>",
        summary: "Run the miniature character-level demo trainer",
        category: Category::Ai,
        details: "This is a small local demo trainer, not a production/LLM/distributed trainer.",
        examples: &["abi nn sample --text \"hello\" --seed h --n 16"],
    },
];

/// A top-level shortcut that maps to a full command.
#[derive(Debug, Clone, Copy)]
pub struct Shortcut {
    /// The token, e.g. `"--tui"`.
    pub token: &'static str,
    /// The command it maps to.
    pub command: &'static str,
    /// One-line description.
    pub summary: &'static str,
}

/// The frozen set of top-level shortcuts.
pub const SHORTCUTS: &[Shortcut] = &[Shortcut {
    token: "--tui",
    command: "tui",
    summary: "Legacy top-level diagnostics dashboard shortcut; accepts the same flags as `abi tui`.",
}];

/// Resolve a shortcut token to the command name it maps to.
#[must_use]
pub fn command_name_for_shortcut(token: &str) -> Option<&'static str> {
    SHORTCUTS
        .iter()
        .find(|shortcut| shortcut.token == token)
        .map(|shortcut| shortcut.command)
}

/// Look up a command by name.
#[must_use]
pub fn find_command(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|command| command.name == name)
}

/// Whether `token` is one of the recognised ways to ask for help.
#[must_use]
pub fn is_help_token(token: &str) -> bool {
    token == "help" || token == "--help" || token == "-h"
}

/// Render `abi help`: the full top-level usage listing.
///
/// Byte-exact against `tests/golden/help.txt`, ANSI escapes included — the Zig
/// renderer does not consult `NO_COLOR`/`TERM`, and the goldens were captured
/// with both set, so this must not either.
#[must_use]
pub fn render_usage() -> String {
    let mut out =
        String::from("Usage: abi <command> [args...]\n       abi --tui [dashboard flags]\n\n");
    for category in Category::ORDER {
        out.push_str("\x1b[1m");
        out.push_str(category.title());
        out.push_str("\x1b[0m\n");
        for command in COMMANDS.iter().filter(|c| c.category == category) {
            out.push_str("  \x1b[36m");
            out.push_str(command.name);
            for _ in command.name.len()..12 {
                out.push(' ');
            }
            out.push_str("\x1b[0m ");
            out.push_str(command.summary);
            out.push('\n');
        }
        out.push('\n');
    }
    out.push_str(
        "Run `abi help [--json|--completion <shell>] <command> [subcommand]` for details.\n",
    );
    out
}

/// Render `abi help <command>`. Returns the text and the process exit code.
///
/// Exit 0 for a known command, 2 (plus the top-level usage listing) for an
/// unknown one — matching `printCommandHelp`'s contract.
#[must_use]
pub fn render_command_help(name: &str) -> (String, u8) {
    if let Some(command) = find_command(name) {
        let mut out = format!("{}\n\n{}\n", command.usage, command.summary);
        if !command.details.is_empty() {
            out.push('\n');
            out.push_str(command.details);
            out.push('\n');
        }
        if !command.examples.is_empty() {
            out.push_str("\nExamples:\n");
            for example in command.examples {
                out.push_str("  ");
                out.push_str(example);
                out.push('\n');
            }
        }
        return (out, 0);
    }
    let mut out = format!("error: unknown command '{name}'\n\n");
    out.push_str(&render_usage());
    (out, 2)
}

/// Render a usage-error line: `error: <message>`. Always exit code 2.
#[must_use]
pub fn usage_error(message: &str) -> (String, u8) {
    (format!("error: {message}\n"), 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exactly_the_thirteen_frozen_commands_are_present() {
        assert_eq!(COMMANDS.len(), 13);
        let names: Vec<&str> = COMMANDS.iter().map(|c| c.name).collect();
        assert_eq!(
            names,
            [
                "help",
                "complete",
                "train",
                "agent",
                "backends",
                "plugin",
                "auth",
                "twilio",
                "tui",
                "dashboard",
                "wdbx",
                "scheduler",
                "nn"
            ]
        );
    }

    #[test]
    fn command_names_are_unique() {
        let mut names: Vec<&str> = COMMANDS.iter().map(|c| c.name).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total);
    }

    #[test]
    fn find_command_locates_known_and_rejects_unknown() {
        let complete = find_command("complete").expect("present");
        assert!(complete.summary.contains("--model"));
        assert_eq!(complete.category.name(), "ai");
        assert!(find_command("nope").is_none());
    }

    #[test]
    fn help_token_classifier_matches_zig() {
        assert!(is_help_token("help"));
        assert!(is_help_token("--help"));
        assert!(is_help_token("-h"));
        assert!(!is_help_token("--helper"));
    }

    #[test]
    fn known_command_help_exits_zero() {
        assert_eq!(render_command_help("help").1, 0);
        assert_eq!(render_command_help("wdbx").1, 0);
    }

    #[test]
    fn unknown_command_help_exits_two_and_shows_usage() {
        let (text, code) = render_command_help("nope");
        assert_eq!(code, 2);
        assert!(text.starts_with("error: unknown command 'nope'"));
        assert!(text.contains("Usage: abi <command>"));
    }

    #[test]
    fn usage_error_always_exits_two() {
        assert_eq!(usage_error("bad arg").1, 2);
        assert_eq!(usage_error("bad arg").0, "error: bad arg\n");
    }

    #[test]
    fn tui_shortcut_resolves_to_the_tui_command() {
        assert_eq!(command_name_for_shortcut("--tui"), Some("tui"));
        assert_eq!(command_name_for_shortcut("--nope"), None);
    }

    #[test]
    fn category_order_matches_the_frozen_listing_order() {
        assert_eq!(
            Category::ORDER,
            [
                Category::Core,
                Category::Ai,
                Category::Data,
                Category::System,
                Category::Network
            ]
        );
    }
}
