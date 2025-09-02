//! Definitions module
//! This module contains common definitions and types used across the project

const std = @import("std");

/// Placeholder for common definitions
pub const PLACEHOLDER = true;

/// Example definition type
pub const DefinitionType = enum {
    example,
    placeholder,
};

/// Example configuration struct
pub const Config = struct {
    name: []const u8 = "default",
    version: u32 = 1,
};
