const std = @import("std");

const abi = @import("abi");
const Framework = abi.framework.runtime.Framework;
const FrameworkOptions = abi.framework.config.FrameworkOptions;
const Feature = abi.framework.config.Feature;
const Agent = abi.ai.agent.Agent;
const AgentConfig = abi.ai.agent.AgentConfig;
const db_helpers = abi.database.db_helpers.helpers;

pub const ExitCode = enum(u8) {
    success = 0,
    usage = 1,
    config = 2,
    runtime = 3,
    io = 4,
    backend_missing = 5,
};

pub const Channels = struct {
    out: std.io.AnyWriter,
    err: std.io.AnyWriter,
};

pub fn printJson(out: std.io.AnyWriter, comptime fmt: []const u8, args: anytype) !void {
    try out.print(fmt, args);
    try out.print("\n", .{});
}

pub const Logger = struct {
    pub const Level = enum(u8) {
        @"error" = 1,
        warn = 2,
        info = 3,
        debug = 4,
        trace = 5,
    };

    level: Level,
    writer: std.io.AnyWriter,

    fn allows(self: Logger, target: Level) bool {
        return @intFromEnum(target) <= @intFromEnum(self.level);
    }

    pub fn log(self: Logger, level: Level, comptime fmt: []const u8, args: anytype) !void {
        if (!self.allows(level)) return;
        try self.writer.print(fmt, args);
    }

    pub fn info(self: Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.info, fmt, args);
    }

    pub fn warn(self: Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.warn, fmt, args);
    }

    pub fn err(self: Logger, comptime fmt: []const u8, args: anytype) !void {
        try self.log(.@"error", fmt, args);
    }
};

const SessionDatabase = struct {
    allocator: std.mem.Allocator,
    dim: ?usize = null,
    next_id: u64 = 1,
    entries: std.ArrayList(VectorEntry),

    const VectorEntry = struct {
        id: u64,
        values: []f32,
        metadata: ?[]u8,
    };

    pub const SearchResult = struct {
        id: u64,
        distance: f32,
    };

    pub const Error = error{
        Empty,
        InvalidVector,
        DimensionMismatch,
        InvalidK,
        OutOfMemory,
    };

    pub fn init(allocator: std.mem.Allocator) SessionDatabase {
        return .{
            .allocator = allocator,
            .entries = std.ArrayList(VectorEntry).init(allocator),
        };
    }

    pub fn deinit(self: *SessionDatabase) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.values);
            if (entry.metadata) |meta| {
                self.allocator.free(meta);
            }
        }
        self.entries.deinit();
    }

    pub fn insert(self: *SessionDatabase, vector: []const f32, metadata: ?[]const u8) Error!u64 {
        if (vector.len == 0) return Error.InvalidVector;
        if (self.dim) |dim| {
            if (vector.len != dim) return Error.DimensionMismatch;
        } else {
            self.dim = vector.len;
        }

        const stored = try self.allocator.dupe(f32, vector);
        errdefer self.allocator.free(stored);

        var stored_meta: ?[]u8 = null;
        if (metadata) |meta| {
            const duplicated = try self.allocator.dupe(u8, meta);
            errdefer self.allocator.free(duplicated);
            stored_meta = duplicated;
        }

        const id = self.next_id;
        self.next_id += 1;
        try self.entries.append(.{ .id = id, .values = stored, .metadata = stored_meta });
        return id;
    }

    pub fn count(self: *const SessionDatabase) usize {
        return self.entries.items.len;
    }

    pub fn search(self: *SessionDatabase, query: []const f32, k: usize) Error![]SearchResult {
        if (self.dim == null or self.entries.items.len == 0) return Error.Empty;
        if (query.len == 0) return Error.InvalidVector;
        if (query.len != self.dim.?) return Error.DimensionMismatch;
        if (k == 0) return Error.InvalidK;

        var results = try std.ArrayList(SearchResult).initCapacity(self.allocator, self.entries.items.len);
        defer results.deinit();

        for (self.entries.items) |entry| {
            var sum: f32 = 0.0;
            var idx: usize = 0;
            while (idx < query.len) : (idx += 1) {
                const diff = query[idx] - entry.values[idx];
                sum += diff * diff;
            }
            const dist = std.math.sqrt(sum);
            try results.append(.{ .id = entry.id, .distance = dist });
        }

        std.sort.heap(SearchResult, results.items, {}, struct {
            fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
                if (a.distance == b.distance) return a.id < b.id;
                return a.distance < b.distance;
            }
        }.lessThan);

        const total = @min(results.items.len, k);
        const owned = try self.allocator.alloc(SearchResult, total);
        @memcpy(owned, results.items[0..total]);
        return owned;
    }
};

pub const Cli = struct {
    allocator: std.mem.Allocator,
    channels: Channels,
    logger: Logger,
    framework: Framework,
    database: SessionDatabase,
    json_mode: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        channels: Channels,
        json_mode: bool,
        log_level: Logger.Level,
    ) !Cli {
        const framework = try Framework.init(allocator, FrameworkOptions{});
        return .{
            .allocator = allocator,
            .channels = channels,
            .logger = Logger{ .level = log_level, .writer = channels.err },
            .framework = framework,
            .database = SessionDatabase.init(allocator),
            .json_mode = json_mode,
        };
    }

    pub fn deinit(self: *Cli) void {
        self.database.deinit();
        self.framework.deinit();
    }

    pub fn dispatch(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.printHelp();
            return .usage;
        }

        const command = args[0];
        const tail = args[1..];

        if (std.mem.eql(u8, command, "help")) {
            try self.printHelp();
            return .success;
        } else if (std.mem.eql(u8, command, "features")) {
            return try self.handleFeatures(tail);
        } else if (std.mem.eql(u8, command, "agent")) {
            return try self.handleAgent(tail);
        } else if (std.mem.eql(u8, command, "db")) {
            return try self.handleDatabase(tail);
        } else if (std.mem.eql(u8, command, "gpu")) {
            return try self.handleGpu(tail);
        }

        try self.logger.err("Unknown command: {s}\n", .{command});
        return .usage;
    }

    fn handleFeatures(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0 or isHelp(args)) {
            try self.printFeaturesHelp();
            return .success;
        }

        const sub = args[0];
        const tail = args[1..];

        if (std.mem.eql(u8, sub, "list")) {
            if (tail.len != 0) {
                try self.logger.err("features list does not accept extra arguments\n", .{});
                return .usage;
            }
            try self.emitFeatureList();
            return .success;
        }

        if (std.mem.eql(u8, sub, "enable")) {
            return try self.toggleFeatures(tail, true);
        }

        if (std.mem.eql(u8, sub, "disable")) {
            return try self.toggleFeatures(tail, false);
        }

        try self.logger.err("Unknown features subcommand: {s}\n", .{sub});
        return .usage;
    }

    fn emitFeatureList(self: *Cli) !void {
        if (self.json_mode) {
            var buffer = std.ArrayList(u8).init(self.allocator);
            defer buffer.deinit();

            try buffer.appendSlice("{\"features\":{");
            var first = true;
            inline for (std.meta.fields(Feature)) |field| {
                const feature = @as(Feature, @enumFromInt(field.value));
                const enabled = self.framework.isFeatureEnabled(feature);
                if (!first) try buffer.appendSlice(",");
                first = false;
                try buffer.writer().print("\"{s}\":{s}", .{ field.name, if (enabled) "true" else "false" });
            }
            try buffer.appendSlice("}}");
            try printJson(self.channels.out, "{s}", .{buffer.items});
        } else {
            try self.logger.info("Enabled features ({d}):\n", .{self.framework.featureCount()});
            inline for (std.meta.fields(Feature)) |field| {
                const feature = @as(Feature, @enumFromInt(field.value));
                const enabled = self.framework.isFeatureEnabled(feature);
                try self.logger.info(
                    "  - {s}: {s}\n",
                    .{ field.name, if (enabled) "enabled" else "disabled" },
                );
            }
        }
    }

    fn toggleFeatures(self: *Cli, args: [][]const u8, enabled: bool) !ExitCode {
        if (args.len == 0) {
            try self.logger.err("Specify at least one feature to toggle\n", .{});
            return .usage;
        }

        var changed = std.ArrayList([]const u8).init(self.allocator);
        defer changed.deinit();

        for (args) |token| {
            const feature = parseFeature(token) orelse {
                try self.logger.err("Unknown feature: {s}\n", .{token});
                return .usage;
            };
            const modified = if (enabled)
                self.framework.enableFeature(feature)
            else
                self.framework.disableFeature(feature);
            if (modified) try changed.append(token);
        }

        if (self.json_mode) {
            var list_buffer = std.ArrayList(u8).init(self.allocator);
            defer list_buffer.deinit();
            try list_buffer.appendSlice("[");
            for (changed.items, 0..) |name, idx| {
                if (idx != 0) try list_buffer.appendSlice(",");
                try list_buffer.writer().print("\"{s}\"", .{name});
            }
            try list_buffer.appendSlice("]");
            try printJson(
                self.channels.out,
                "{\"status\":\"ok\",\"action\":\"{s}\",\"updated\":{s}}",
                .{ if (enabled) "enable" else "disable", list_buffer.items },
            );
        } else {
            const label = if (enabled) "enabled" else "disabled";
            if (changed.items.len == 0) {
                try self.logger.warn("No features were {s}\n", .{label});
            } else {
                const action_word = if (enabled) "Enabled" else "Disabled";
                try self.logger.info("{s} {d} feature(s)\n", .{ action_word, changed.items.len });
            }
        }

        return .success;
    }

    fn handleAgent(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0 or isHelp(args)) {
            try self.printAgentHelp();
            return .success;
        }

        if (!std.mem.eql(u8, args[0], "run")) {
            try self.logger.err("Unknown agent subcommand: {s}\n", .{args[0]});
            return .usage;
        }

        var name: []const u8 = "EchoAgent";
        var message: ?[]const u8 = null;

        var idx: usize = 1;
        while (idx < args.len) : (idx += 1) {
            const token = args[idx];
            if (std.mem.eql(u8, token, "--name")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--name requires a value\n", .{});
                    return .usage;
                }
                name = args[idx];
            } else if (std.mem.eql(u8, token, "--message")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--message requires a value\n", .{});
                    return .usage;
                }
                message = args[idx];
            } else {
                try self.logger.err("Unknown agent flag: {s}\n", .{token});
                return .usage;
            }
        }

        var agent = try Agent.init(self.allocator, AgentConfig{ .name = name });
        defer agent.deinit();

        const input = try self.readAgentInput(message);
        defer self.allocator.free(@constCast(input));

        const reply = try agent.process(input, self.allocator);
        defer self.allocator.free(@constCast(reply));

        if (self.json_mode) {
            try printJson(self.channels.out, "{\"status\":\"ok\",\"reply\":\"{s}\"}", .{reply});
        } else {
            try self.logger.info("Agent {s} replied: {s}\n", .{ name, reply });
        }

        return .success;
    }

    fn readAgentInput(self: *Cli, explicit: ?[]const u8) ![]const u8 {
        if (explicit) |value| {
            return try self.allocator.dupe(u8, value);
        }

        var reader = std.io.getStdIn().reader();
        var buffer = std.ArrayList(u8).init(self.allocator);
        errdefer buffer.deinit();

        var temp: [256]u8 = undefined;
        while (true) {
            const read_bytes = try reader.read(&temp);
            if (read_bytes == 0) break;
            try buffer.appendSlice(temp[0..read_bytes]);
        }

        return try buffer.toOwnedSlice();
    }

    fn handleDatabase(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0 or isHelp(args)) {
            try self.printDatabaseHelp();
            return .success;
        }

        const sub = args[0];
        const tail = args[1..];

        if (std.mem.eql(u8, sub, "insert")) {
            return try self.databaseInsert(tail);
        }
        if (std.mem.eql(u8, sub, "search")) {
            return try self.databaseSearch(tail);
        }

        try self.logger.err("Unknown db subcommand: {s}\n", .{sub});
        return .usage;
    }

    const VectorSource = union(enum) {
        @"inline": []const u8,
        file: []const u8,
    };

    fn databaseInsert(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.logger.err("db insert requires arguments\n", .{});
            return .usage;
        }

        var source: ?VectorSource = null;
        var metadata: ?[]const u8 = null;

        var idx: usize = 0;
        while (idx < args.len) : (idx += 1) {
            const token = args[idx];
            if (std.mem.eql(u8, token, "--vec")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--vec requires a value\n", .{});
                    return .usage;
                }
                source = .{ .@"inline" = args[idx] };
            } else if (std.mem.eql(u8, token, "--vec-file")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--vec-file requires a path\n", .{});
                    return .usage;
                }
                source = .{ .file = args[idx] };
            } else if (std.mem.eql(u8, token, "--meta")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--meta requires a value\n", .{});
                    return .usage;
                }
                metadata = args[idx];
            } else {
                try self.logger.err("Unknown db flag: {s}\n", .{token});
                return .usage;
            }
        }

        if (source == null) {
            try self.logger.err("Provide a vector using --vec or --vec-file\n", .{});
            return .usage;
        }

        const vector = try self.loadVector(source.?);
        defer self.allocator.free(vector);

        const id = self.database.insert(vector, metadata) catch |err| {
            return switch (err) {
                SessionDatabase.Error.DimensionMismatch => blk: {
                    try self.logger.err("Vector dimension mismatch\n", .{});
                    break :blk ExitCode.config;
                },
                SessionDatabase.Error.InvalidVector => blk: {
                    try self.logger.err("Invalid vector payload\n", .{});
                    break :blk ExitCode.usage;
                },
                SessionDatabase.Error.OutOfMemory => blk: {
                    try self.logger.err("Out of memory handling vector\n", .{});
                    break :blk ExitCode.runtime;
                },
                else => blk: {
                    try self.logger.err("Database error\n", .{});
                    break :blk ExitCode.runtime;
                },
            };
        };

        if (self.json_mode) {
            try printJson(
                self.channels.out,
                "{\"status\":\"ok\",\"id\":{d},\"count\":{d}}",
                .{ id, self.database.count() },
            );
        } else {
            try self.logger.info("Inserted vector {d} (dim={d})\n", .{ id, vector.len });
        }

        return .success;
    }

    fn databaseSearch(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0) {
            try self.logger.err("db search requires arguments\n", .{});
            return .usage;
        }

        var source: ?VectorSource = null;
        var k: usize = 5;

        var idx: usize = 0;
        while (idx < args.len) : (idx += 1) {
            const token = args[idx];
            if (std.mem.eql(u8, token, "--vec")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--vec requires a value\n", .{});
                    return .usage;
                }
                source = .{ .@"inline" = args[idx] };
            } else if (std.mem.eql(u8, token, "--vec-file")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--vec-file requires a path\n", .{});
                    return .usage;
                }
                source = .{ .file = args[idx] };
            } else if (std.mem.eql(u8, token, "-k") or std.mem.eql(u8, token, "--k")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--k requires an integer\n", .{});
                    return .usage;
                }
                k = std.fmt.parseUnsigned(usize, args[idx], 10) catch {
                    try self.logger.err("Invalid value for k\n", .{});
                    return .usage;
                };
            } else {
                try self.logger.err("Unknown db flag: {s}\n", .{token});
                return .usage;
            }
        }

        if (source == null) {
            try self.logger.err("Provide a vector using --vec or --vec-file\n", .{});
            return .usage;
        }

        const vector = try self.loadVector(source.?);
        defer self.allocator.free(vector);

        const results = self.database.search(vector, k) catch |err| {
            return switch (err) {
                SessionDatabase.Error.Empty => blk: {
                    try self.logger.err("Database has no vectors\n", .{});
                    break :blk ExitCode.runtime;
                },
                SessionDatabase.Error.DimensionMismatch => blk: {
                    try self.logger.err("Vector dimension mismatch\n", .{});
                    break :blk ExitCode.config;
                },
                SessionDatabase.Error.InvalidVector, SessionDatabase.Error.InvalidK => blk: {
                    try self.logger.err("Invalid search parameters\n", .{});
                    break :blk ExitCode.usage;
                },
                SessionDatabase.Error.OutOfMemory => blk: {
                    try self.logger.err("Out of memory processing search\n", .{});
                    break :blk ExitCode.runtime;
                },
            };
        };
        defer self.allocator.free(results);

        if (self.json_mode) {
            var buffer = std.ArrayList(u8).init(self.allocator);
            defer buffer.deinit();
            try buffer.appendSlice("{\"results\":[");
            for (results, 0..) |res, idx_other| {
                if (idx_other != 0) try buffer.appendSlice(",");
                try buffer.writer().print("{\"id\":{d},\"distance\":{d:.6}}", .{ res.id, res.distance });
            }
            try buffer.appendSlice("]}");
            try printJson(self.channels.out, "{s}", .{buffer.items});
        } else {
            if (results.len == 0) {
                try self.logger.warn("No neighbors found\n", .{});
            } else {
                try self.logger.info("Top {d} neighbors:\n", .{results.len});
                for (results) |res| {
                    try self.logger.info("  - id={d} distance={d:.4}\n", .{ res.id, res.distance });
                }
            }
        }

        return .success;
    }

    fn loadVector(self: *Cli, source: VectorSource) ![]f32 {
        return switch (source) {
            .@"inline" => |text| blk: {
                const trimmed = std.mem.trim(u8, text, "[] \t\r\n");
                break :blk try db_helpers.parseVector(self.allocator, trimmed);
            },
            .file => |path| blk: {
                const file = try std.fs.cwd().openFile(path, .{});
                defer file.close();
                const data = try file.readToEndAlloc(self.allocator, std.math.maxInt(usize));
                defer self.allocator.free(data);
                const trimmed = std.mem.trim(u8, data, "[] \t\r\n");
                break :blk try db_helpers.parseVector(self.allocator, trimmed);
            },
        };
    }

    fn handleGpu(self: *Cli, args: [][]const u8) !ExitCode {
        if (args.len == 0 or isHelp(args)) {
            try self.printGpuHelp();
            return .success;
        }

        if (!std.mem.eql(u8, args[0], "bench")) {
            try self.logger.err("Unknown gpu subcommand: {s}\n", .{args[0]});
            return .usage;
        }

        var size = MatSize{ .m = 32, .n = 32, .p = 32 };
        var repeats: usize = 1;

        var idx: usize = 1;
        while (idx < args.len) : (idx += 1) {
            const token = args[idx];
            if (std.mem.eql(u8, token, "--size")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--size requires MxN or MxNxP\n", .{});
                    return .usage;
                }
                size = parseMatSize(args[idx]) catch {
                    try self.logger.err("Invalid --size value\n", .{});
                    return .usage;
                };
            } else if (std.mem.eql(u8, token, "--repeats")) {
                idx += 1;
                if (idx >= args.len) {
                    try self.logger.err("--repeats requires an integer\n", .{});
                    return .usage;
                }
                repeats = std.fmt.parseUnsigned(usize, args[idx], 10) catch {
                    try self.logger.err("Invalid repeats count\n", .{});
                    return .usage;
                };
            } else {
                try self.logger.err("Unknown gpu flag: {s}\n", .{token});
                return .usage;
            }
        }

        const stats = try runCpuBench(self.allocator, size, repeats);
        defer self.allocator.free(stats.output);

        if (self.json_mode) {
            try printJson(
                self.channels.out,
                "{\"cpu_ms\":{d:.3},\"gpu\":\"unavailable\",\"size\":{\"m\":{d},\"n\":{d},\"p\":{d}}}",
                .{ stats.cpu_ms, size.m, size.n, size.p },
            );
        } else {
            try self.logger.info("CPU fallback completed in {d:.3} ms (repeats={d})\n", .{ stats.cpu_ms, repeats });
            try self.logger.warn("GPU backend unavailable; used CPU fallback\n", .{});
        }

        return .success;
    }

    fn printHelp(self: *Cli) !void {
        const message = "Usage: abi <command> [options]\n\nCommands:\n  features   list|enable|disable\n  agent      run\n  db         insert|search\n  gpu        bench\n";
        try self.logger.info("{s}", .{message});
    }

    fn printFeaturesHelp(self: *Cli) !void {
        const text = "features <list|enable|disable> [feature...]\nFeatures: ai, database, gpu, web, monitoring, connectors, simd\n";
        try self.logger.info("{s}", .{text});
    }

    fn printAgentHelp(self: *Cli) !void {
        const text = "agent run [--name <str>] [--message <str>]\nReads stdin when --message is omitted.\n";
        try self.logger.info("{s}", .{text});
    }

    fn printDatabaseHelp(self: *Cli) !void {
        const text =
            "db insert --vec <comma-values>|--vec-file <path> [--meta <json>]\n" ++
            "db search --vec <comma-values>|--vec-file <path> [-k <n>]\n";
        try self.logger.info("{s}", .{text});
    }

    fn printGpuHelp(self: *Cli) !void {
        const text = "gpu bench [--size MxN(xP)] [--repeats <n>]\n";
        try self.logger.info("{s}", .{text});
    }

    const MatSize = struct {
        m: usize,
        n: usize,
        p: usize,
    };

    const CpuBenchResult = struct {
        cpu_ms: f64,
        output: []f32,
    };

    fn parseMatSize(text: []const u8) !MatSize {
        var parts = std.mem.splitScalar(u8, text, 'x');
        var values: [3]usize = .{ 0, 0, 0 };
        var count: usize = 0;
        while (parts.next()) |piece| {
            if (count >= values.len) return error.InvalidSize;
            values[count] = std.fmt.parseUnsigned(usize, piece, 10) catch return error.InvalidSize;
            count += 1;
        }
        if (count < 2) return error.InvalidSize;
        if (count == 2) values[2] = values[1];
        return .{ .m = values[0], .n = values[1], .p = values[2] };
    }

    fn runCpuBench(allocator: std.mem.Allocator, size: MatSize, repeats: usize) !CpuBenchResult {
        const total = size.m * size.p;
        const output = try allocator.alloc(f32, total);
        errdefer allocator.free(output);

        const a = try allocator.alloc(f32, size.m * size.n);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, size.n * size.p);
        defer allocator.free(b);

        for (a, 0..) |*item, idx| item.* = @floatFromInt((idx % 7) + 1);
        for (b, 0..) |*item, idx| item.* = @floatFromInt((idx % 5) + 1);

        const start = std.time.microTimestamp();
        var repeat_idx: usize = 0;
        while (repeat_idx < repeats) : (repeat_idx += 1) {
            matmul(output, a, b, size.m, size.n, size.p);
        }
        const elapsed = std.time.microTimestamp() - start;
        const ms = @as(f64, @floatFromInt(elapsed)) / 1000.0;

        return .{ .cpu_ms = ms, .output = output };
    }
};

fn matmul(out: []f32, a: []const f32, b: []const f32, m: usize, n: usize, p: usize) void {
    var i: usize = 0;
    while (i < m) : (i += 1) {
        var k: usize = 0;
        while (k < p) : (k += 1) {
            var sum: f32 = 0.0;
            var j: usize = 0;
            while (j < n) : (j += 1) {
                sum += a[i * n + j] * b[j * p + k];
            }
            out[i * p + k] = sum;
        }
    }
}
fn parseFeature(name: []const u8) ?Feature {
    inline for (std.meta.fields(Feature)) |field| {
        if (std.ascii.eqlIgnoreCase(name, field.name)) {
            return @as(Feature, @enumFromInt(field.value));
        }
    }
    return null;
}

fn isHelp(args: [][]const u8) bool {
    if (args.len == 0) return false;
    return std.mem.eql(u8, args[0], "--help") or std.mem.eql(u8, args[0], "-h");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const raw_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, raw_args);

    if (raw_args.len == 0) return;

    var idx: usize = 1;
    var json_mode = false;
    var help_flag = false;
    var log_level: Logger.Level = .info;

    while (idx < raw_args.len) {
        const arg = raw_args[idx];
        if (std.mem.eql(u8, arg, "--json")) {
            json_mode = true;
            idx += 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            help_flag = true;
            idx += 1;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--log-level=")) {
            const value = arg["--log-level=".len..];
            log_level = std.meta.stringToEnum(Logger.Level, value) orelse {
                try std.io.getStdErr().writer().print("Invalid log level: {s}\n", .{value});
                std.process.exit(@intFromEnum(ExitCode.usage));
            };
            idx += 1;
            continue;
        }
        break;
    }

    if (json_mode and log_level == .info) {
        log_level = .@"error";
    }

    var cli = try Cli.init(
        allocator,
        .{
            .out = std.io.getStdOut().writer().any(),
            .err = std.io.getStdErr().writer().any(),
        },
        json_mode,
        log_level,
    );
    defer cli.deinit();

    if (help_flag or idx >= raw_args.len) {
        try cli.printHelp();
        return;
    }

    const exit_code = cli.dispatch(raw_args[idx..]) catch |err| {
        try cli.logger.err("fatal error: {s}\n", .{@errorName(err)});
        std.process.exit(@intFromEnum(ExitCode.runtime));
        unreachable;
    };

    if (exit_code != .success) {
        std.process.exit(@intFromEnum(exit_code));
    }
}

test "session database handles missing metadata without invalid free" {
    var db = SessionDatabase.init(std.testing.allocator);
    defer db.deinit();

    const vector = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectEqual(@as(u64, 1), try db.insert(vector[0..], null));
    try std.testing.expectEqual(@as(usize, 1), db.count());
}

test "session database frees metadata when append fails" {
    var backing: [96]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&backing);

    {
        var db = SessionDatabase.init(fba.allocator());
        defer db.deinit();

        const vector = [_]f32{ 1.0, 2.0, 3.0 };
        const metadata = "meta";

        try std.testing.expectError(SessionDatabase.Error.OutOfMemory, db.insert(vector[0..], metadata));
        try std.testing.expectEqual(@as(usize, 0), db.count());
        try std.testing.expectEqual(@as(usize, 0), fba.end_index);
    }

    try std.testing.expectEqual(@as(usize, 0), fba.end_index);
}

const TestChannels = struct {
    out_buf: std.ArrayList(u8),
    err_buf: std.ArrayList(u8),

    fn init(allocator: std.mem.Allocator) TestChannels {
        return .{
            .out_buf = std.ArrayList(u8).init(allocator),
            .err_buf = std.ArrayList(u8).init(allocator),
        };
    }

    fn deinit(self: *TestChannels) void {
        self.out_buf.deinit();
        self.err_buf.deinit();
    }

    fn channels(self: *TestChannels) Channels {
        return .{
            .out = self.out_buf.writer().any(),
            .err = self.err_buf.writer().any(),
        };
    }
};

test "features list uses stderr in human mode" {
    var tc = TestChannels.init(std.testing.allocator);
    defer tc.deinit();

    var cli = try Cli.init(std.testing.allocator, tc.channels(), false, .info);
    defer cli.deinit();

    try std.testing.expectEqual(ExitCode.success, try cli.dispatch(&.{ "features", "list" }));
    try std.testing.expectEqual(@as(usize, 0), tc.out_buf.items.len);
    try std.testing.expect(tc.err_buf.items.len > 0);
}

test "features list emits json in json mode" {
    var tc = TestChannels.init(std.testing.allocator);
    defer tc.deinit();

    var cli = try Cli.init(std.testing.allocator, tc.channels(), true, .@"error");
    defer cli.deinit();

    try std.testing.expectEqual(ExitCode.success, try cli.dispatch(&.{ "features", "list" }));
    try std.testing.expect(tc.out_buf.items.len > 0);
    try std.testing.expectEqual(@as(usize, 0), tc.err_buf.items.len);

    const expected = "{\"features\":{\"ai\":true";
    try std.testing.expect(std.mem.startsWith(u8, tc.out_buf.items, expected));
}
