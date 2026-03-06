//! Controls writes back into long-term memory.

const std = @import("std");

pub const MemoryWriteDecision = enum {
    retain,
    summarize,
    drop,
};

pub const MemoryWriter = struct {
    // TODO: apply decay curves, promote pinned memories, manage rolling summaries
};
