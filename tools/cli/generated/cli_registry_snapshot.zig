//! Generated command registry snapshot.
//! Refresh with: `zig build refresh-cli-registry`

pub const agent = @import("../commands/ai/agent.zig");
pub const brain = @import("../commands/ai/brain.zig");
pub const chat = @import("../commands/ai/chat.zig");
pub const context_agent = @import("../commands/ai/context_agent.zig");
pub const embed = @import("../commands/ai/embed.zig");
pub const llm = @import("../commands/ai/llm/mod.zig");
pub const mcp = @import("../commands/ai/mcp.zig");
pub const model = @import("../commands/ai/model.zig");
pub const multi_agent = @import("../commands/ai/multi_agent.zig");
pub const os_agent = @import("../commands/ai/os_agent.zig");
pub const ralph = @import("../commands/ai/ralph/mod.zig");
pub const train = @import("../commands/ai/train/mod.zig");
pub const config = @import("../commands/core/config.zig");
pub const discord = @import("../commands/core/discord.zig");
pub const init = @import("../commands/core/init.zig");
pub const plugins = @import("../commands/core/plugins.zig");
pub const profile = @import("../commands/core/profile.zig");
pub const ui = @import("../commands/core/ui/mod.zig");
pub const update = @import("../commands/core/update.zig");
pub const db = @import("../commands/db/db.zig");
pub const explore = @import("../commands/db/explore.zig");
pub const acp = @import("../commands/dev/acp.zig");
pub const bench = @import("../commands/dev/bench/mod.zig");
pub const clean = @import("../commands/dev/clean.zig");
pub const completions = @import("../commands/dev/completions.zig");
pub const convert = @import("../commands/dev/convert.zig");
pub const create_subagent = @import("../commands/dev/create_subagent.zig");
pub const doctor = @import("../commands/dev/doctor.zig");
pub const editor = @import("../commands/dev/editor.zig");
pub const env = @import("../commands/dev/env.zig");
pub const gendocs = @import("../commands/dev/gendocs.zig");
pub const lsp = @import("../commands/dev/lsp.zig");
pub const matrix = @import("../commands/dev/matrix.zig");
pub const status = @import("../commands/dev/status.zig");
pub const task = @import("../commands/dev/task.zig");
pub const gpu = @import("../commands/infra/gpu.zig");
pub const network = @import("../commands/infra/network.zig");
pub const simd = @import("../commands/infra/simd.zig");
pub const system_info = @import("../commands/infra/system_info.zig");

pub const command_modules = .{
    agent,
    brain,
    chat,
    context_agent,
    embed,
    llm,
    mcp,
    model,
    multi_agent,
    os_agent,
    ralph,
    train,
    config,
    discord,
    init,
    plugins,
    profile,
    ui,
    update,
    db,
    explore,
    acp,
    bench,
    clean,
    completions,
    convert,
    create_subagent,
    doctor,
    editor,
    env,
    gendocs,
    lsp,
    matrix,
    status,
    task,
    gpu,
    network,
    simd,
    system_info,
};
