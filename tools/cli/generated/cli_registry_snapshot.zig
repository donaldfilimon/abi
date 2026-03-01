//! Generated command registry snapshot.
//! Refresh with: `zig build refresh-cli-registry`

pub const acp = @import("../commands/acp.zig");
pub const agent = @import("../commands/agent.zig");
pub const bench = @import("../commands/bench/mod.zig");
pub const brain = @import("../commands/brain.zig");
pub const clean = @import("../commands/clean.zig");
pub const completions = @import("../commands/completions.zig");
pub const config = @import("../commands/config.zig");
pub const convert = @import("../commands/convert.zig");
pub const db = @import("../commands/db.zig");
pub const discord = @import("../commands/discord.zig");
pub const doctor = @import("../commands/doctor.zig");
pub const embed = @import("../commands/embed.zig");
pub const env = @import("../commands/env.zig");
pub const explore = @import("../commands/explore.zig");
pub const gendocs = @import("../commands/gendocs.zig");
pub const gpu = @import("../commands/gpu.zig");
pub const init = @import("../commands/init.zig");
pub const llm = @import("../commands/llm/mod.zig");
pub const lsp = @import("../commands/lsp.zig");
pub const mcp = @import("../commands/mcp.zig");
pub const model = @import("../commands/model.zig");
pub const multi_agent = @import("../commands/multi_agent.zig");
pub const network = @import("../commands/network.zig");
pub const os_agent = @import("../commands/os_agent.zig");
pub const plugins = @import("../commands/plugins.zig");
pub const profile = @import("../commands/profile.zig");
pub const ralph = @import("../commands/ralph/mod.zig");
pub const simd = @import("../commands/simd.zig");
pub const status = @import("../commands/status.zig");
pub const system_info = @import("../commands/system_info.zig");
pub const task = @import("../commands/task.zig");
pub const toolchain = @import("../commands/toolchain.zig");
pub const train = @import("../commands/train/mod.zig");
pub const ui = @import("../commands/ui/mod.zig");

pub const command_modules = .{
    acp,
    agent,
    bench,
    brain,
    clean,
    completions,
    config,
    convert,
    db,
    discord,
    doctor,
    embed,
    env,
    explore,
    gendocs,
    gpu,
    init,
    llm,
    lsp,
    mcp,
    model,
    multi_agent,
    network,
    os_agent,
    plugins,
    profile,
    ralph,
    simd,
    status,
    system_info,
    task,
    toolchain,
    train,
    ui,
};
