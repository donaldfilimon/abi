//! Comprehensive tests for the logging module

const std = @import("std");
const logging = @import("abi").logging.logging;

test "Structured logging basic functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Initialize logger
    const config = logging.LoggerConfig{
        .level = .debug,
        .format = .text,
    };

    const logger = try logging.Logger.init(allocator, config);
    defer logger.deinit();

    // Test logging with fields
    try logger.info("Test message", .{ .key1 = "value1", .key2 = 42 }, @src());
    try logger.debug("Debug message", .{ .debug_field = true }, @src());
    try logger.warn("Warning message", .{ .warning_code = "WARN001" }, @src());

    try testing.expectEqual(logging.LogLevel.debug, logger.config.level);
    try testing.expectEqual(logging.OutputFormat.text, logger.config.format);
}

test "Global logging functionality" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Initialize global logger
    const config = logging.LoggerConfig{
        .level = .info,
        .format = .json,
    };

    try logging.initGlobalLogger(allocator, config);
    defer logging.deinitGlobalLogger();

    // Test global logging functions
    try logging.info("Global info message", .{ .global = true, .count = 100 }, @src());
    try logging.warn("Global warning", .{ .code = "TEST_WARN" }, @src());

    const global_logger = logging.getGlobalLogger();
    try testing.expect(global_logger != null);
    try testing.expectEqual(logging.LogLevel.info, global_logger.?.config.level);
}

test "Log level filtering" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test with INFO level
    const info_config = logging.LoggerConfig{
        .level = .info,
        .format = .text,
    };

    const info_logger = try logging.Logger.init(allocator, info_config);
    defer info_logger.deinit();

    // These should work
    try info_logger.info("Info message", .{}, @src());
    try info_logger.warn("Warn message", .{}, @src());
    try info_logger.err("Error message", .{}, @src());
    try info_logger.fatal("Fatal message", .{}, @src());

    // This should be filtered out (debug < info)
    try info_logger.debug("Debug message", .{}, @src());

    try testing.expectEqual(logging.LogLevel.info, info_logger.config.level);
}

test "JSON format logging" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const json_config = logging.LoggerConfig{
        .level = .info,
        .format = .json,
    };

    const json_logger = try logging.Logger.init(allocator, json_config);
    defer json_logger.deinit();

    // Test JSON logging
    try json_logger.info("JSON test message", .{
        .user_id = 12345,
        .action = "login",
        .success = true,
    }, @src());

    try testing.expectEqual(logging.OutputFormat.json, json_logger.config.format);
}

test "Colored format logging" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const colored_config = logging.LoggerConfig{
        .level = .debug,
        .format = .colored,
    };

    const colored_logger = try logging.Logger.init(allocator, colored_config);
    defer colored_logger.deinit();

    // Test colored logging
    try colored_logger.info("Colored info", .{ .color_test = true }, @src());
    try colored_logger.err("Colored error", .{ .error_code = 500 }, @src());

    try testing.expectEqual(logging.OutputFormat.colored, colored_logger.config.format);
}

test "Logging with various field types" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const logger = try logging.Logger.init(allocator, logging.LoggerConfig{
        .level = .info,
        .format = .text,
    });
    defer logger.deinit();

    // Test various field types
    try logger.info("Field types test", .{
        .string_field = "hello",
        .int_field = 42,
        .float_field = 3.14159,
        .bool_field = true,
        .array_field = [_]i32{ 1, 2, 3 },
    }, @src());
}

test "Multiple loggers" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Create multiple loggers with different configurations
    const logger1 = try logging.Logger.init(allocator, logging.LoggerConfig{
        .level = .debug,
        .format = .text,
    });
    defer logger1.deinit();

    const logger2 = try logging.Logger.init(allocator, logging.LoggerConfig{
        .level = .warn,
        .format = .json,
    });
    defer logger2.deinit();

    // Test that they work independently
    try logger1.debug("Logger1 debug", .{ .logger = 1 }, @src());
    try logger2.warn("Logger2 warn", .{ .logger = 2 }, @src());

    try testing.expectEqual(logging.LogLevel.debug, logger1.config.level);
    try testing.expectEqual(logging.LogLevel.warn, logger2.config.level);
    try testing.expectEqual(logging.OutputFormat.text, logger1.config.format);
    try testing.expectEqual(logging.OutputFormat.json, logger2.config.format);
}

test "Thread safety" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const logger = try logging.Logger.init(allocator, logging.LoggerConfig{
        .level = .info,
        .format = .text,
    });
    defer logger.deinit();

    // Test concurrent logging (basic test - no race conditions detected)
    var threads: [4]std.Thread = undefined;

    for (&threads, 0..) |*thread, i| {
        thread.* = try std.Thread.spawn(.{}, struct {
            fn log_worker(l: *logging.Logger, id: usize) void {
                l.info("Thread message", .{ .thread_id = id, .iteration = 1 }, @src()) catch {};
                l.info("Thread message", .{ .thread_id = id, .iteration = 2 }, @src()) catch {};
            }
        }.log_worker, .{ logger, i });
    }

    // Wait for all threads to complete
    for (threads) |thread| {
        thread.join();
    }
}

test "Logger configuration validation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Test default configuration
    const default_config = logging.LoggerConfig{};
    try testing.expectEqual(logging.LogLevel.info, default_config.level);
    try testing.expectEqual(logging.OutputFormat.colored, default_config.format);
    try testing.expect(default_config.enable_timestamps);
    try testing.expect(default_config.enable_source_info);

    // Test custom configuration
    const custom_config = logging.LoggerConfig{
        .level = .debug,
        .format = .json,
        .enable_timestamps = false,
        .enable_source_info = false,
        .buffer_size = 8192,
    };

    const logger = try logging.Logger.init(allocator, custom_config);
    defer logger.deinit();

    try testing.expectEqual(logging.LogLevel.debug, logger.config.level);
    try testing.expectEqual(logging.OutputFormat.json, logger.config.format);
    try testing.expect(!logger.config.enable_timestamps);
    try testing.expect(!logger.config.enable_source_info);
    try testing.expectEqual(@as(usize, 8192), logger.config.buffer_size);
}

test "Log level enum properties" {
    const testing = std.testing;

    // Test all log level values
    try testing.expectEqual(@as(u8, 0), @intFromEnum(logging.LogLevel.trace));
    try testing.expectEqual(@as(u8, 1), @intFromEnum(logging.LogLevel.debug));
    try testing.expectEqual(@as(u8, 2), @intFromEnum(logging.LogLevel.info));
    try testing.expectEqual(@as(u8, 3), @intFromEnum(logging.LogLevel.warn));
    try testing.expectEqual(@as(u8, 4), @intFromEnum(logging.LogLevel.err));
    try testing.expectEqual(@as(u8, 5), @intFromEnum(logging.LogLevel.fatal));

    // Test string representations
    try testing.expectEqualStrings("TRACE", logging.LogLevel.trace.toString());
    try testing.expectEqualStrings("DEBUG", logging.LogLevel.debug.toString());
    try testing.expectEqualStrings("INFO", logging.LogLevel.info.toString());
    try testing.expectEqualStrings("WARN", logging.LogLevel.warn.toString());
    try testing.expectEqualStrings("ERROR", logging.LogLevel.err.toString());
    try testing.expectEqualStrings("FATAL", logging.LogLevel.fatal.toString());
}

test "Performance logging" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const logger = try logging.Logger.init(allocator, logging.LoggerConfig{
        .level = .info,
        .format = .text,
    });
    defer logger.deinit();

    // Measure logging performance
    const start_time = std.time.nanoTimestamp();

    // Log many messages
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try logger.info("Performance test", .{
            .iteration = i,
            .timestamp = std.time.nanoTimestamp(),
        }, @src());
    }

    const end_time = std.time.nanoTimestamp();
    const total_time = end_time - start_time;
    const avg_time_per_log = total_time / 100;

    // Performance should be reasonable (< 1ms per log typically)
    try testing.expect(avg_time_per_log < 1_000_000); // Less than 1ms per log
}

test "Error handling in logging" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const logger = try logging.Logger.init(allocator, logging.LoggerConfig{
        .level = .info,
        .format = .text,
    });
    defer logger.deinit();

    // Test that logging doesn't crash with edge cases
    try logger.info("", .{}, @src()); // Empty message
    try logger.info("Very long message " ** 100, .{}, @src()); // Very long message
    try logger.info("Message with special chars: !@#$%^&*()", .{}, @src()); // Special characters
}

test "Global logger replacement" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Initialize first global logger
    try logging.initGlobalLogger(allocator, logging.LoggerConfig{
        .level = .debug,
        .format = .text,
    });

    {
        const global1 = logging.getGlobalLogger();
        try testing.expect(global1 != null);
        try testing.expectEqual(logging.LogLevel.debug, global1.?.config.level);
    }

    // Replace with new global logger
    try logging.initGlobalLogger(allocator, logging.LoggerConfig{
        .level = .warn,
        .format = .json,
    });

    {
        const global2 = logging.getGlobalLogger();
        try testing.expect(global2 != null);
        try testing.expectEqual(logging.LogLevel.warn, global2.?.config.level);
        try testing.expectEqual(logging.OutputFormat.json, global2.?.config.format);
    }

    logging.deinitGlobalLogger();
}
