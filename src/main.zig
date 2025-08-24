//! Abi: High-performance AI framework

const std = @import("std");
const builtin = @import("builtin");
// build_options not available, using defaults
const root = @import("root.zig");
const core = @import("core/mod.zig");

pub const Config = struct {
    enable_gpu: bool = false,
    enable_simd: bool = true,
    enable_tracy: bool = false,
    max_memory: usize = 2 * 1024 * 1024 * 1024, // 2GB
    log_level: std.log.Level = .info,
    worker_threads: u32 = 0,
};

pub const Error = error{ EmptyText, BlacklistedWord, TextTooLong, InvalidValues, ProcessingFailed, InitializationFailed, InvalidArguments, MemoryAllocationFailed, ModeNotFound, ConfigurationError } || std.mem.Allocator.Error || std.fs.File.OpenError;

pub const Request = struct {
    text: []const u8,
    values: []const usize,

    fn validate(self: Request) Error!void {
        if (self.text.len == 0) return Error.EmptyText;
        if (self.values.len == 0) return Error.InvalidValues;
        _ = try Abbey.checkCompliance(self.text);
    }

    fn deinit(self: Request, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.values);
    }
};

pub const Response = struct {
    result: usize,
    message: []const u8,
    metadata: ?[]const u8 = null,

    fn deinit(self: Response, allocator: std.mem.Allocator) void {
        allocator.free(self.message);
        if (self.metadata) |meta| allocator.free(meta);
    }
};

pub const Abbey = struct {
    const MAX_TEXT_LENGTH = 10000;
    const BLACKLIST = [_][]const u8{ "harmful", "dangerous", "illegal", "unethical" };

    fn isCompliant(text: []const u8) bool {
        return checkCompliance(text) catch false;
    }

    fn checkCompliance(text: []const u8) Error!bool {
        if (text.len == 0) return Error.EmptyText;
        if (text.len > MAX_TEXT_LENGTH) return Error.TextTooLong;

        for (BLACKLIST) |word| {
            const found = std.mem.indexOf(u8, text, word) != null;
            if (found) return Error.BlacklistedWord;
        }
        return true;
    }
};

pub const Aviva = struct {
    fn computeSum(values: []const usize) Error!usize {
        if (values.len == 0) return Error.InvalidValues;
        var sum: usize = 0;
        for (values) |v| {
            sum = std.math.add(usize, sum, v) catch return Error.ProcessingFailed;
        }
        return sum;
    }

    fn computeAverage(values: []const usize) Error!f64 {
        if (values.len == 0) return Error.InvalidValues;
        const sum = try computeSum(values);
        return @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(values.len));
    }

    fn computeStatistics(values: []const usize) Error!struct { sum: usize, average: f64, min: usize, max: usize, count: usize } {
        if (values.len == 0) return Error.InvalidValues;

        const sum = try computeSum(values);
        const average = try computeAverage(values);
        var min = values[0];
        var max = values[0];

        for (values) |v| {
            if (v < min) min = v;
            if (v > max) max = v;
        }

        return .{ .sum = sum, .average = average, .min = min, .max = max, .count = values.len };
    }
};

pub const Abi = struct {
    fn process(allocator: std.mem.Allocator, req: Request) Error!Response {
        const logger = core.logging.framework_logger;
        logger.debug("Starting validation...", .{});
        try req.validate();
        logger.debug("Validation passed", .{});

        logger.debug("Computing statistics...", .{});
        const stats = try Aviva.computeStatistics(req.values);
        logger.debug("Statistics computed: sum={}", .{stats.sum});

        logger.debug("Allocating message...", .{});
        const message = try allocator.dupe(u8, "Processing completed successfully");
        logger.debug("Message allocated", .{});

        logger.debug("Allocating metadata...", .{});
        const metadata = try allocator.dupe(u8, "{}");
        logger.debug("Metadata allocated", .{});

        logger.debug("Returning response...", .{});
        return Response{ .result = stats.sum, .message = message, .metadata = metadata };
    }
};

pub const Mode = enum { default, tui, agent, ml, bench, web, cell };

fn modeFromString(str: []const u8) ?Mode {
    return std.meta.stringToEnum(Mode, str);
}

pub const App = struct {
    allocator: std.mem.Allocator,
    config: Config,
    gpa: std.heap.GeneralPurposeAllocator(.{}),

    fn init(config: Config) !*App {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        const allocator = gpa.allocator();

        const self = try allocator.create(App);
        self.* = .{ .allocator = allocator, .config = config, .gpa = gpa };

        return self;
    }

    fn deinit(self: *App) void {
        const allocator = self.allocator;
        _ = self.gpa.deinit();
        allocator.destroy(self);
    }

    fn run(self: *App, mode: Mode) !void {
        const logger = core.logging.framework_logger;
        logger.info("Abi AI Framework v{any}", .{root.version});
        logger.info("GPU={s} SIMD={s} Tracy={s}", .{
            if (self.config.enable_gpu) "on" else "off",
            if (self.config.enable_simd) "on" else "off",
            if (self.config.enable_tracy) "on" else "off",
        });

        switch (mode) {
            .default => try self.runDefault(),
            .tui => try self.runTUI(),
            .agent => try self.runAgent(),
            .ml => try self.runML(),
            .bench => try self.runBenchmarks(),
            .web => try self.runWebServer(),
            .cell => try self.runCellInterpreter(),
        }
    }

    fn runDefault(self: *App) !void {
        const logger = core.logging.framework_logger;
        logger.info("Running default mode", .{});

        const values = [_]usize{ 1, 2, 3, 4, 5, 10, 20, 30 };
        const request = Request{ .text = "Sample processing request", .values = &values };

        logger.debug("About to process request...", .{});
        const response = try Abi.process(self.allocator, request);
        defer response.deinit(self.allocator);

        logger.info("Processing completed successfully!", .{});
        logger.info("Result: {s}", .{response.message});
        if (response.metadata) |meta| {
            logger.debug("Metadata: {s}", .{meta});
        }
        logger.info("Default mode completed successfully!", .{});
    }

    fn runTUI(_: *App) !void {
        const tui = @import("tui.zig");
        try tui.run();
    }

    fn runAgent(self: *App) !void {
        const agent = @import("agent.zig");
        var ai_agent = try agent.Agent.init(self.allocator, .{});
        defer ai_agent.deinit();
        try ai_agent.start();
    }

    fn runML(_: *App) !void {
        const logger = core.logging.framework_logger;
        logger.warn("ML mode not yet implemented", .{});
    }

    fn runBenchmarks(_: *App) !void {
        const logger = core.logging.framework_logger;
        logger.warn("Benchmarks not yet implemented", .{});
    }

    fn runWebServer(self: *App) !void {
        const web = @import("web_server.zig");
        var server = try web.WebServer.init(self.allocator, .{});
        defer server.deinit();
        try server.start();
    }

    fn runCellInterpreter(_: *App) !void {
        const logger = core.logging.framework_logger;
        logger.warn("Cell interpreter not yet implemented", .{});
    }
};

fn parseArgs(allocator: std.mem.Allocator) !struct { mode: Mode, config: Config } {
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var mode = Mode.default;
    var config = Config{};

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--mode")) {
            i += 1;
            if (i >= args.len) return Error.InvalidArguments;
            mode = modeFromString(args[i]) orelse return Error.ModeNotFound;
        } else if (std.mem.eql(u8, arg, "--no-gpu")) {
            config.enable_gpu = false;
        } else if (std.mem.eql(u8, arg, "--no-simd")) {
            config.enable_simd = false;
        } else if (std.mem.eql(u8, arg, "--tracy")) {
            config.enable_tracy = true;
        } else if (std.mem.eql(u8, arg, "--threads")) {
            i += 1;
            if (i >= args.len) return Error.InvalidArguments;
            config.worker_threads = try std.fmt.parseInt(u32, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--help")) {
            printHelp();
            std.process.exit(0);
        }
    }

    return .{ .mode = mode, .config = config };
}

fn printHelp() void {
    std.debug.print(
        \\Abi AI Framework
        \\Usage: abi [options]
        \\  --mode <mode>      Mode: default|tui|agent|ml|bench|web|cell
        \\  --no-gpu           Disable GPU acceleration
        \\  --no-simd          Disable SIMD acceleration
        \\  --tracy            Enable Tracy profiling
        \\  --threads <n>      Worker thread count
        \\  --help             Show help
        \\
    , .{});
}

pub fn main() !void {
    const parsed = parseArgs(std.heap.page_allocator) catch {
        printHelp();
        return;
    };

    var app = try App.init(parsed.config);
    defer app.deinit();

    try app.run(parsed.mode);
}

test "request validation" {
    const req = Request{ .text = "test", .values = &[_]usize{1} };
    try req.validate();
}

test "aviva computation" {
    const values = [_]usize{ 1, 2, 3, 4, 5 };
    const sum = try Aviva.computeSum(&values);
    try std.testing.expectEqual(@as(usize, 15), sum);
}

test "Abi integration" {
    const allocator = std.testing.allocator;

    const text = try allocator.dupe(u8, "Test integration");
    defer allocator.free(text);
    const values = try allocator.dupe(usize, &[_]usize{ 10, 20, 30 });
    defer allocator.free(values);

    const req = Request{ .text = text, .values = values };
    const res = try Abi.process(allocator, req);
    defer res.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 60), res.result);
    try std.testing.expect(std.mem.indexOf(u8, res.message, "completed") != null);
    try std.testing.expect(res.metadata != null);
}

test "Mode parsing" {
    try std.testing.expectEqual(Mode.tui, Mode.fromString("tui").?);
    try std.testing.expectEqual(Mode.agent, Mode.fromString("agent").?);
    try std.testing.expectEqual(@as(?Mode, null), Mode.fromString("nonexistent"));
}
