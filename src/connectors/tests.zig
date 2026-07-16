const std = @import("std");

test {
    _ = @import("openai_tests.zig");
    _ = @import("anthropic_tests.zig");
    _ = @import("discord_tests.zig");
    _ = @import("twilio_tests.zig");
    _ = @import("grok_tests.zig");
    _ = @import("tests_connector.zig");
    _ = @import("http_tests.zig");
    _ = @import("json_tests.zig");
    std.testing.refAllDecls(@This());
}
