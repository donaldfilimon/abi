//! Thin hub for `abi agent` handlers.
//!
//! Re-exports leaf entrypoints consumed by `wiring.zig` / `handlers/mod.zig`.
//! Dispatch lives in `agent_dispatch.zig`; execution in sibling leaves.

const std = @import("std");

const dispatch = @import("agent_dispatch.zig");
const plan = @import("agent_plan.zig");
const train = @import("agent_train.zig");
const tui = @import("agent_tui.zig");
const multi = @import("agent_multi.zig");
const os = @import("agent_os.zig");

pub const handleAgent = dispatch.handleAgent;
pub const handleAgentPlanInput = plan.handleAgentPlanInput;
pub const handleAgentTrainProfile = train.handleAgentTrainProfile;
pub const handleAgentTuiNoArgs = tui.handleAgentTuiNoArgs;
pub const handleAgentMultiInput = multi.handleAgentMultiInput;
pub const handleAgentSpawnArgv = multi.handleAgentSpawnArgv;
pub const handleAgentBrowserArgv = multi.handleAgentBrowserArgv;
pub const handleAgentOs = os.handleAgentOs;

test {
    std.testing.refAllDecls(@This());
}
