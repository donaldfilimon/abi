//! Activation subsystem providing reusable activation primitives.

const std = @import("std");
const functions = @import("functions.zig");
const utils = @import("utils.zig");

pub const ActivationType = functions.ActivationType;
pub const ActivationConfig = functions.ActivationConfig;
pub const ActivationProcessor = functions.ActivationProcessor;
pub const ActivationRegistry = functions.ActivationRegistry;

pub const ActivationUtils = utils.ActivationUtils;
pub const ActivationConstants = utils.ActivationConstants;

pub usingnamespace functions;

test {
    std.testing.refAllDecls(@This());
}
