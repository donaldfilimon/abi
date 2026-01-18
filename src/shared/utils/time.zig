//! Time utilities re-export.
//!
//! This module re-exports time utilities from the main utils module
//! for backwards compatibility.

const utils = @import("../utils.zig");

pub const unixSeconds = utils.unixSeconds;
pub const nowNanoseconds = utils.nowNanoseconds;
pub const formatTimestamp = utils.formatTimestamp;
pub const parseTimestamp = utils.parseTimestamp;
