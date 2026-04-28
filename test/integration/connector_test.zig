const std = @import("std");
const abi = @import("abi");

const openai_env_names = [_][]const u8{
    "ABI_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "ABI_OPENAI_BASE_URL",
    "OPENAI_BASE_URL",
    "ABI_OPENAI_MODEL",
    "OPENAI_MODEL",
};

const EnvSnapshot = struct {
    allocator: std.mem.Allocator,
    values: [openai_env_names.len]?[]u8,

    fn capture(allocator: std.mem.Allocator) !EnvSnapshot {
        var snapshot = EnvSnapshot{
            .allocator = allocator,
            .values = [_]?[]u8{null} ** openai_env_names.len,
        };
        errdefer snapshot.deinit();

        inline for (openai_env_names, 0..) |name, index| {
            snapshot.values[index] = try abi.connectors.getEnvOwned(allocator, name);
        }

        return snapshot;
    }

    fn restore(self: *EnvSnapshot) !void {
        inline for (openai_env_names, 0..) |name, index| {
            if (self.values[index]) |value| {
                try std.process.setEnvVar(name, value);
            } else {
                std.process.unsetEnvVar(name) catch {};
            }
        }
    }

    fn deinit(self: *EnvSnapshot) void {
        for (self.values) |value| {
            if (value) |owned| self.allocator.free(owned);
        }
    }
};

fn clearOpenAIEnv() void {
    inline for (openai_env_names) |name| {
        std.process.unsetEnvVar(name) catch {};
    }
}

test "connector smoke: tryLoadOpenAI loads configured environment" {
    const allocator = std.testing.allocator;
    var env = try EnvSnapshot.capture(allocator);
    defer {
        env.restore() catch {};
        env.deinit();
    }

    clearOpenAIEnv();
    try std.process.setEnvVar("OPENAI_API_KEY", "sk-1234567890abcdef1234567890abcdef");

    var config = (try abi.connectors.loaders.tryLoadOpenAI(allocator)) orelse
        return error.ExpectedOpenAIConnectorConfig;
    defer config.deinit(allocator);

    try std.testing.expectEqualStrings("https://api.openai.com/v1", config.base_url);
}

test "connector smoke: tryLoadOpenAI returns null without configured environment" {
    const allocator = std.testing.allocator;
    var env = try EnvSnapshot.capture(allocator);
    defer {
        env.restore() catch {};
        env.deinit();
    }

    clearOpenAIEnv();

    const config = try abi.connectors.loaders.tryLoadOpenAI(allocator);
    try std.testing.expect(config == null);
}
