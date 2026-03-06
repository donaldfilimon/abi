//! Merge, dedupe, reclaim, rewrite.

const std = @import("std");

pub const CompactionJob = struct {
    pub const Status = enum { pending, running, completed, failed };

    status: Status = .pending,
    blocks_processed: usize = 0,
    bytes_reclaimed: u64 = 0,

    // FIXME: implement merge, dedupe, and rewrite logic
};
