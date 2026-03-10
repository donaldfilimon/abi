//! Produces compact packets for the inference engine.

const std = @import("std");
const core = @import("../core");

pub const ContextPacket = struct {
    user_prompt: []const u8,
    system_directives: []const u8,
    memory_blocks: []const core.ids.BlockId,
    tool_results: []const []const u8,
};

pub const ContextAssembler = struct {
    allocator: std.mem.Allocator,
    max_tokens: u32,

    pub fn init(allocator: std.mem.Allocator, max_tokens: u32) ContextAssembler {
        return .{
            .allocator = allocator,
            .max_tokens = max_tokens,
        };
    }

    pub fn assemble(
        self: *ContextAssembler,
        prompt: []const u8,
        candidates: []const core.ids.BlockId,
    ) !ContextPacket {
        _ = self;
        // Gather candidates, trim to token/byte budget
        // Preserve important lineage, optionally summarize
        return ContextPacket{
            .user_prompt = prompt,
            .system_directives = "",
            .memory_blocks = candidates,
            .tool_results = &[_][]const u8{},
        };
    }
};
