//! Activation subsystem providing reusable activation primitives.

const std = @import("std");
const functions_mod = @import("functions.zig");
const utils_mod = @import("utils.zig");

pub const functions = functions_mod;
pub const utils = utils_mod;

pub const ActivationType = functions.ActivationType;
pub const ActivationConfig = functions.ActivationConfig;
pub const ActivationProcessor = functions.ActivationProcessor;
pub const ActivationRegistry = functions.ActivationRegistry;

pub const ActivationUtils = utils.ActivationUtils;
pub const ActivationConstants = utils.ActivationConstants;

test {
    std.testing.refAllDecls(@This());
}
