//! Feature module test discovery root.
//!
//! Separate test binary for discovering inline test blocks inside feature
//! module source files. Zig 0.16 restricts each source file to one module,
//! so these cannot be imported by the main test binary (which already uses
//! them via the 'abi' named module).

const build_options = @import("build_options");

test {
    if (build_options.enable_ai) {
        // Modules verified to compile cleanly in standalone context
        _ = @import("features/ai/eval/mod.zig");
        _ = @import("features/ai/rag/mod.zig");
        _ = @import("features/ai/templates/mod.zig");
        _ = @import("features/ai/orchestration/mod.zig");
        _ = @import("features/ai/documents/mod.zig");

        // Fixed: persistence.zig coercion, edit_tools I/O API, generator init signature
        _ = @import("features/ai/memory/mod.zig");
        _ = @import("features/ai/tools/mod.zig");
        _ = @import("features/ai/streaming/mod.zig");
        // _ = @import("features/ai/abbey/mod.zig");       // 13 errors (deep refactor needed)

        // Multi-agent coordination (aggregation, messaging, coordinator)
        _ = @import("features/ai/multi_agent/aggregation.zig");
        _ = @import("features/ai/multi_agent/messaging.zig");

        // AI database submodule (wdbx token dataset tests)
        _ = @import("features/ai/database/wdbx.zig");
    }

    // LLM module (152 test blocks across 47 files)
    if (@hasDecl(build_options, "enable_llm") and build_options.enable_llm) {
        _ = @import("features/ai/llm/mod.zig");
    }

    // Phase 9 feature modules (inline tests)
    if (build_options.enable_cache) _ = @import("features/cache/mod.zig");
    if (build_options.enable_gateway) _ = @import("features/gateway/mod.zig");
    if (build_options.enable_messaging) _ = @import("features/messaging/mod.zig");
    if (build_options.enable_search) _ = @import("features/search/mod.zig");
    if (build_options.enable_storage) _ = @import("features/storage/mod.zig");
    if (build_options.enable_pages) _ = @import("features/pages/mod.zig");

    // MCP/ACP service tests (types + server only — mod.zig has database dep)
    _ = @import("services/mcp/server.zig");
    _ = @import("services/mcp/types.zig");
    _ = @import("services/acp/mod.zig");
}
