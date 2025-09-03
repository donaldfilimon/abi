//! Utilities Module
//!
//! This module provides common utilities for the WDBX database,
//! including error handling, logging, memory management, and profiling.

pub const errors = @import("errors.zig");
pub const logging = @import("logging.zig");
pub const memory = @import("memory.zig");
pub const profiling = @import("profiling.zig");

// Re-export commonly used types
pub const ErrorSet = errors.ErrorSet;
pub const Logger = logging.Logger;
pub const MemoryPool = memory.MemoryPool;
pub const Profiler = profiling.Profiler;