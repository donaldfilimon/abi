//! Shared types for the hash feature.
//!
//! Stable, portable 64-bit and 128-bit hashing utilities.
//! Both mod.zig (real implementation) and stub.zig (disabled path) import from here.

const std = @import("std");

/// Errors specific to hashing operations.
pub const HashError = error{
    FeatureDisabled,
    InvalidInput,
    OutOfMemory,
};

pub const Error = HashError;

pub const Hash64 = u64;

pub const Hash128 = struct {
    hi: u64,
    lo: u64,
};
