//! Checks the scheduler's rendered output against fixtures captured from the
//! Zig binaries.
//!
//! These fixtures (`tests/golden/`, at the repo root) were recorded while
//! `./build.sh check` was still green. Once the Zig tree is removed they are the
//! only remaining record of the frozen output format, so this test is what makes
//! them load-bearing rather than decorative.

use abi_core::{Scheduler, TaskPriority};
use std::path::PathBuf;

fn golden(name: &str) -> String {
    // CARGO_MANIFEST_DIR is crates/abi-core; fixtures live at the repo root.
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing golden fixture {}: {e}", path.display()))
}

/// Pull the `text` field out of an MCP `tools/call` response line.
fn mcp_text(line: &str) -> String {
    let value: serde_json::Value = serde_json::from_str(line).expect("fixture line is JSON");
    value["result"]["content"][0]["text"]
        .as_str()
        .expect("tools/call result carries a text content block")
        .to_string()
}

#[test]
fn empty_scheduler_matches_the_mcp_golden() {
    let fixture = golden("mcp-scheduler-calls.jsonl");
    let mut lines = fixture.lines();

    let stats_text = mcp_text(lines.next().expect("scheduler_stats response"));
    let info_text = mcp_text(lines.next().expect("scheduler_info response"));

    // scheduler_info is documented as a compatibility alias, so the two
    // responses must be byte-identical.
    assert_eq!(
        stats_text, info_text,
        "scheduler_info must alias scheduler_stats exactly"
    );

    // The MCP server reports a fresh scheduler.
    let rendered = Scheduler::new().stats().render();
    let expected = stats_text
        .strip_prefix("scheduler ")
        .and_then(|rest| rest.strip_suffix(" source=mcp-server"))
        .expect("golden text has the documented shape");

    assert_eq!(
        rendered, expected,
        "Stats::render must reproduce the frozen MCP counter format"
    );
}

#[test]
fn one_completed_task_matches_the_cli_golden() {
    // tests/golden/scheduler-status.txt was captured after the CLI had run one
    // task, so it pins the post-execution counter values too.
    let fixture = golden("scheduler-status.txt");
    let counters = fixture
        .lines()
        .find(|line| line.starts_with("running="))
        .expect("status output has a counter line");

    let scheduler = Scheduler::new();
    scheduler.submit("one-shot", TaskPriority::Normal, Box::new(|| Ok(())));
    scheduler.run_next().expect("task runs");

    assert_eq!(
        scheduler.stats().render(),
        counters,
        "Stats::render must reproduce the frozen CLI counter format"
    );
}

#[test]
fn the_cli_golden_confirms_the_one_shot_model() {
    // Recorded from a real run: `mode=one-shot`. If this ever reads otherwise,
    // the concurrency decision in abi-core's docs is wrong and scheduler_stats
    // output shape needs revisiting.
    let fixture = golden("scheduler-status.txt");
    assert!(
        fixture.contains("mode=one-shot"),
        "scheduler-status.txt no longer reports one-shot: {fixture}"
    );
    assert!(
        fixture.contains("running=0"),
        "a synchronous scheduler always reports running=0 at rest"
    );
}
