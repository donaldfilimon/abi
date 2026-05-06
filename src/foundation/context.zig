const std = @import("std");

/// Generic, standardized Context implementation for feature modules.
/// Ensures consistent lifecycle management (init/deinit) and memory safety.
/// Includes validation and standardized error reporting.
pub fn FeatureContext(comptime ConfigT: type) type {
    return struct {
        allocator: std.mem.Allocator,
        config: ConfigT,

        pub fn init(allocator: std.mem.Allocator, config: ConfigT) !*@This() {
            // Validate config if it has a validate method
            if (@hasDecl(ConfigT, "validate")) {
                try ConfigT.validate(config);
            }

            const ctx = try allocator.create(@This());
            ctx.* = .{ .allocator = allocator, .config = config };
            return ctx;
        }

        pub fn deinit(self: *@This()) void {
            // Call config deinit if it has one
            if (@hasDecl(ConfigT, "deinit")) {
                ConfigT.deinit(self.config, self.allocator);
            }
            self.allocator.destroy(self);
        }

        /// Get a read-only view of the config.
        pub fn getConfig(self: *const @This()) ConfigT {
            return self.config;
        }

        /// Update config with validation.
        pub fn updateConfig(self: *@This(), new_config: ConfigT) !void {
            if (@hasDecl(ConfigT, "validate")) {
                try ConfigT.validate(new_config);
            }
            if (@hasDecl(ConfigT, "deinit")) {
                ConfigT.deinit(self.config, self.allocator);
            }
            self.config = new_config;
        }
    };
}

test {
    std.testing.refAllDecls(@This());
}

// Tests for FeatureContext validation
test "FeatureContext basic lifecycle" {
    const TestingConfig = struct {
        value: u32,
    };

    const Ctx = FeatureContext(TestingConfig);
    var ctx = try Ctx.init(std.testing.allocator, .{ .value = 42 });
    defer ctx.deinit();

    try std.testing.expectEqual(@as(u32, 42), ctx.getConfig().value);
}

test "FeatureContext with validation" {
    const ValidatedConfig = struct {
        value: u32,

        pub fn validate(config: @This()) !void {
            if (config.value == 0) return error.InvalidConfig;
        }
    };

    const Ctx = FeatureContext(ValidatedConfig);

    // Should fail with invalid config
    const result = Ctx.init(std.testing.allocator, .{ .value = 0 });
    try std.testing.expectError(error.InvalidConfig, result);

    // Should succeed with valid config
    var ctx = try Ctx.init(std.testing.allocator, .{ .value = 100 });
    defer ctx.deinit();

    try std.testing.expectEqual(@as(u32, 100), ctx.getConfig().value);
}
