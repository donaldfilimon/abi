//! Core Module - Foundation for the Abi Framework
//!
//! This module provides the foundational building blocks used throughout
//! the Abi framework, including error handling, I/O abstractions, diagnostics,
//! data structures, and type definitions.

const std = @import("std");

// Core submodules
pub const collections = @import("collections.zig");
pub const errors = @import("errors.zig");
pub const io = @import("io.zig");
pub const diagnostics = @import("diagnostics.zig");
pub const types = @import("types.zig");
pub const utils = @import("utils.zig");

// Re-export commonly used types for convenience
pub const AbiError = errors.AbiError;
pub const ErrorClass = errors.ErrorClass;
pub const ErrorContext = diagnostics.ErrorContext;

pub const Writer = io.Writer;
pub const OutputContext = io.OutputContext;
pub const TestWriter = io.TestWriter;
pub const BufferedWriter = io.BufferedWriter;

pub const Diagnostic = diagnostics.Diagnostic;
pub const DiagnosticCollector = diagnostics.DiagnosticCollector;
pub const Severity = diagnostics.Severity;
pub const SourceLocation = diagnostics.SourceLocation;

// Commonly used functions
pub const here = diagnostics.here;
pub const getMessage = errors.getMessage;
pub const isRecoverable = errors.isRecoverable;

test {
    std.testing.refAllDecls(@This());
}
