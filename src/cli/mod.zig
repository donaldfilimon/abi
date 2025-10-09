<<<<<<< HEAD
pub const run = @import("comprehensive_cli.zig").main;
=======
//! Modern CLI Module
//! 
//! This module provides a clean command-line interface for the ABI framework.

const std = @import("std");

pub const commands = @import("commands/mod.zig");
pub const errors = @import("errors.zig");
pub const state = @import("state.zig");
pub const main = @import("main.zig");
>>>>>>> b17de21c4567850c62ba3b2a072d76ef36b80aa3
