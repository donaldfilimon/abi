//! Generated command registry snapshot.
//! Refresh with: `zig build refresh-cli-registry`

pub const agent = @import("../commands/ai/agent");
pub const brain = @import("../commands/ai/brain");
pub const chat = @import("../commands/ai/chat");
pub const context_agent = @import("../commands/ai/context_agent");
pub const embed = @import("../commands/ai/embed");
pub const llm = @import("../commands/ai/llm/mod.zig");
pub const mcp = @import("../commands/ai/mcp");
pub const model = @import("../commands/ai/model");
pub const multi_agent = @import("../commands/ai/multi_agent");
pub const os_agent = @import("../commands/ai/os_agent");
pub const ralph = @import("../commands/ai/ralph/mod.zig");
pub const train = @import("../commands/ai/train/mod.zig");
pub const config = @import("../commands/core/config");
pub const discord = @import("../commands/core/discord");
pub const init = @import("../commands/core/init");
pub const plugins = @import("../commands/core/plugins");
pub const profile = @import("../commands/core/profile");
pub const ui = @import("../commands/core/ui/mod.zig");
pub const update = @import("../commands/core/update");
pub const db = @import("../commands/db/db");
pub const explore = @import("../commands/db/explore");
pub const acp = @import("../commands/dev/acp");
pub const bench = @import("../commands/dev/bench/mod.zig");
pub const clean = @import("../commands/dev/clean");
pub const completions = @import("../commands/dev/completions");
pub const convert = @import("../commands/dev/convert");
pub const create_subagent = @import("../commands/dev/create_subagent");
pub const doctor = @import("../commands/dev/doctor");
pub const editor = @import("../commands/dev/editor");
pub const env = @import("../commands/dev/env");
pub const gendocs = @import("../commands/dev/gendocs");
pub const lsp = @import("../commands/dev/lsp");
pub const matrix = @import("../commands/dev/matrix");
pub const status = @import("../commands/dev/status");
pub const task = @import("../commands/dev/task");
pub const gpu = @import("../commands/infra/gpu");
pub const network = @import("../commands/infra/network");
pub const simd = @import("../commands/infra/simd");
pub const system_info = @import("../commands/infra/system_info");

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
