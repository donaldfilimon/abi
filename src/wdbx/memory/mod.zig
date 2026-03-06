//! Controls writes back into long-term memory.

const std = @import("std");

pub const MemoryWriteDecision = enum {
    retain,
    summarize,
    drop,
};

pub const MemoryWriter = struct {
    decay_rate: f32 = 0.05,
    promotion_threshold: f32 = 0.8,
    summary_threshold: f32 = 0.4,

    pub fn decideWrite(self: MemoryWriter, importance: f32, age_turns: u32) MemoryWriteDecision {
        const decayed_importance = importance * std.math.pow(f32, (1.0 - self.decay_rate), @as(f32, @floatFromInt(age_turns)));
        
        if (decayed_importance >= self.promotion_threshold) {
            return .retain;
        } else if (decayed_importance >= self.summary_threshold) {
            return .summarize;
        } else {
            return .drop;
        }
    }
    
    // FIXME: implement rolling summary generation
};
