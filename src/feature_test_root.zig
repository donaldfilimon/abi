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

        // These need targeted fixes before enabling:
        // _ = @import("features/ai/memory/mod.zig");     // persistence.zig: *const [2]T → []T
        // _ = @import("features/ai/tools/mod.zig");       // edit_tools.zig: Io.File.writeAll API
        // _ = @import("features/ai/streaming/mod.zig");   // generator.zig: type mismatches
        // _ = @import("features/ai/abbey/mod.zig");       // 13 errors (deep refactor needed)
    }

    // Skipped pending fix (6 errors in backward_ops, cache, generation, ops):
    // if (@hasDecl(build_options, "enable_llm") and build_options.enable_llm) {
    //     _ = @import("features/ai/llm/mod.zig");
    // }
}
