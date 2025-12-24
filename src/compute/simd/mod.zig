//! SIMD primitives module
//!
//! This module provides SIMD vector operations and low-level primitives for
//! high-performance compute operations.

pub const vector = @import("vector.zig");
pub const primitives = @import("primitives.zig");

const simd = @import("../../shared/simd.zig");

pub const SIMD_WIDTH = simd.SIMD_WIDTH;
pub const VectorOps = simd.VectorOps;
