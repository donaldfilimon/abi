//! Ralph Prompt Templates
//!
//! Prompt templates for the Ralph iterative agent loop.
//! These prompts drive the "stop hook" and "loop injection" mechanisms.

const std = @import("std");

/// The initial system prompt for Ralph is defined in personas.zig.
/// This file contains the prompts used during the iterative loop.

/// Injected when the agent attempts to stop or pause.
/// Guides the agent to verify its work and continue if necessary.
pub const LOOP_INJECTION_TEMPLATE =
    \\[SYSTEM_INJECTION]
    \\Iteration {d} complete.
    \\Task status: IN_PROGRESS.
    \\
    \\Instructions:
    \\1. Review the work done in this iteration.
    \\2. Verify if the original goal "{s}" is fully met.
    \\3. If incomplete or if errors exist, proceed to the next logical step.
    \\4. If you believe the task is complete, run verification tests.
    \\
    \\Do NOT stop unless you have verified completion with concrete evidence.
    \\Output your next step immediately.
;

/// Triggered when the agent claims completion (e.g., emits [DONE] or stops).
/// Forces a final verification step before allowing the loop to exit.
pub const STOP_HOOK_TEMPLATE =
    \\[SYSTEM_VERIFICATION]
    \\You have signaled completion.
    \\
    \\Required Verification:
    \\1. List the specific verification steps or tests you performed.
    \\2. If you have not run code to verify your changes, RESUME immediately and run them.
    \\3. If verification failed, fix the issues.
    \\4. Only if all verification passes, output the final success summary.
;

/// Helper to format the loop injection prompt.
pub fn formatLoopInjection(
    allocator: std.mem.Allocator,
    iteration: usize,
    original_goal: []const u8,
) ![]u8 {
    return std.fmt.allocPrint(allocator, LOOP_INJECTION_TEMPLATE, .{
        iteration,
        original_goal,
    });
}
