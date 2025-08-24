//! Self-Learning Discord Bot Demo

const std = @import("std");
const self_learning_bot = @import("../src/self_learning_discord_bot.zig");
const agent = @import("../src/agent.zig");

const BotEnvironment = enum { development, production, testing };

const configs = .{
    .development = .{ .debug = true, .learning_threshold = 0.5, .shards = 3 },
    .production = .{ .debug = false, .learning_threshold = 0.7, .shards = 7 },
    .testing = .{ .debug = true, .learning_threshold = 0.3, .shards = 2 },
};

const BotOptions = struct {
    env: BotEnvironment = .development,
    persona: ?agent.PersonaType = null,
    max_messages: ?u64 = null,
    help: bool = false,

    fn parse(allocator: std.mem.Allocator) !BotOptions {
        var opts = BotOptions{};
        const args = try std.process.argsAlloc(allocator);
        defer std.process.argsFree(allocator, args);

        var i: usize = 1;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "-h")) {
                opts.help = true;
            } else if (std.mem.eql(u8, args[i], "--env")) {
                i += 1;
                opts.env = std.meta.stringToEnum(BotEnvironment, args[i]) orelse return error.InvalidEnv;
            } else if (std.mem.eql(u8, args[i], "--persona")) {
                i += 1;
                opts.persona = std.meta.stringToEnum(agent.PersonaType, args[i]) orelse return error.InvalidPersona;
            } else if (std.mem.eql(u8, args[i], "--max")) {
                i += 1;
                opts.max_messages = try std.fmt.parseInt(u64, args[i], 10);
            }
        }
        return opts;
    }

    fn printHelp() void {
        std.debug.print(
            \\Discord Bot Demo
            \\Usage: zig build run-discord-bot-demo [options]
            \\  -h              Show help
            \\  --env <env>     Environment: development|production|testing
            \\  --persona <p>   Persona override
            \\  --max <n>       Max messages before shutdown
            \\
            \\Environment: DISCORD_BOT_TOKEN, OPENAI_API_KEY
            \\
        , .{});
    }
};

const Monitor = struct {
    start: i64,
    last_print: i64,
    interval: i64 = 60000,

    fn init() Monitor {
        const now = std.time.milliTimestamp();
        return .{ .start = now, .last_print = now };
    }

    fn maybeprint(self: *Monitor, bot: *self_learning_bot.SelfLearningBot) void {
        const now = std.time.milliTimestamp();
        if (now - self.last_print < self.interval) return;

        const stats = bot.getStats();
        const uptime = (now - self.start) / 1000;
        std.debug.print("📊 {}s | {} msgs | {} learned | {s}\n", .{ uptime, stats.messages_processed, stats.interactions_learned, @tagName(stats.current_persona) });
        self.last_print = now;
    }
};

var should_stop = false;
var global_bot: ?*self_learning_bot.SelfLearningBot = null;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const opts = BotOptions.parse(allocator) catch {
        BotOptions.printHelp();
        return;
    };

    if (opts.help) {
        BotOptions.printHelp();
        return;
    }

    const discord_token = std.process.getEnvVarOwned(allocator, "DISCORD_BOT_TOKEN") catch {
        std.debug.print("❌ Set DISCORD_BOT_TOKEN\n", .{});
        return;
    };
    defer allocator.free(discord_token);

    const openai_key = std.process.getEnvVarOwned(allocator, "OPENAI_API_KEY") catch null;
    defer if (openai_key) |key| allocator.free(key);

    const env_config = switch (opts.env) {
        .development => configs.development,
        .production => configs.production,
        .testing => configs.testing,
    };

    const config = self_learning_bot.BotConfig{
        .discord_token = discord_token,
        .openai_api_key = openai_key,
        .default_persona = opts.persona orelse .AdaptiveModerator,
        .debug = env_config.debug,
        .max_response_length = if (opts.env == .testing) 500 else 2000,
        .learning_threshold = env_config.learning_threshold,
        .context_limit = @intFromEnum(opts.env) + 2,
        .db_config = .{ .shard_count = env_config.shards },
    };

    std.debug.print("🚀 Starting bot | {s} | {s}\n", .{ @tagName(opts.env), @tagName(config.default_persona) });

    var bot = try self_learning_bot.SelfLearningBot.init(allocator, config);
    defer bot.deinit();
    global_bot = &bot;

    var monitor = Monitor.init();
    var count: u64 = 0;

    while (!should_stop) {
        std.time.sleep(std.time.ns_per_s);
        monitor.maybeprint(&bot);
        count += 1;

        if (opts.max_messages) |max| {
            if (count >= max) break;
        }

        // Demo persona switching
        if (count % 100 == 0) {
            const personas = [_]agent.PersonaType{ .EmpatheticAnalyst, .DirectExpert, .CreativeWriter };
            bot.switchPersona(personas[count / 100 % personas.len]);
        }
    }

    std.debug.print("🏁 Shutting down\n", .{});
    bot.stop();
}

test "options parsing" {
    const opts = BotOptions{};
    try std.testing.expectEqual(BotEnvironment.development, opts.env);
    try std.testing.expectEqual(@as(?agent.PersonaType, null), opts.persona);
}
