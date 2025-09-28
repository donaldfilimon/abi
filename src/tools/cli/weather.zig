const std = @import("std");
const abi = @import("abi");
const services = @import("services");
const wdbx = @import("wdbx");
const common = @import("common.zig");

pub const command = common.Command{
    .name = "weather",
    .summary = "Ingest and query weather embeddings",
    .usage = "abi weather <ingest|query> [flags]",
    .details =
    "  ingest  Fetch weather data and store embedding\n" ++
    "  query   Search nearest weather entries\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    if (args.len < 3) {
        std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
        return;
    }

    const sub = args[2];
    if (std.mem.eql(u8, sub, "ingest")) {
        var db_path: ?[]const u8 = null;
        var api_key: ?[]const u8 = null;
        var city: ?[]const u8 = null;
        var units: []const u8 = "metric";
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--db") and i + 1 < args.len) {
                i += 1;
                db_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--apikey") and i + 1 < args.len) {
                i += 1;
                api_key = args[i];
            } else if (std.mem.eql(u8, args[i], "--city") and i + 1 < args.len) {
                i += 1;
                city = args[i];
            } else if (std.mem.eql(u8, args[i], "--units") and i + 1 < args.len) {
                i += 1;
                units = args[i];
            }
        }
        if (db_path == null or api_key == null or city == null) {
            std.debug.print("weather ingest requires --db, --apikey and --city\n", .{});
            return;
        }

        const base_cfg = services.WeatherConfig{ .api_key = api_key.?, .units = units };
        const cfg = services.WeatherConfig.fromEnv(allocator, base_cfg);
        var svc = try services.WeatherService.init(allocator, cfg);
        defer svc.deinit();
        var wd = try svc.getCurrentWeather(city.?);
        defer wd.deinit(allocator);

        const embed = try weatherToEmbedding(allocator, wd);
        defer allocator.free(embed);

        var db = try wdbx.Db.open(db_path.?, true);
        defer db.close();
        if (db.getDimension() == 0) try db.init(@intCast(embed.len));
        const id = try db.addEmbedding(embed);
        std.debug.print("Ingested weather for {s}, id={d}, dim={d}\n", .{ wd.city, id, embed.len });
        return;
    }

    if (std.mem.eql(u8, sub, "query")) {
        var db_path: ?[]const u8 = null;
        var city: ?[]const u8 = null;
        var k: usize = 5;
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--db") and i + 1 < args.len) {
                i += 1;
                db_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--city") and i + 1 < args.len) {
                i += 1;
                city = args[i];
            } else if (std.mem.eql(u8, args[i], "--k") and i + 1 < args.len) {
                i += 1;
                k = try std.fmt.parseInt(usize, args[i], 10);
            }
        }
        if (db_path == null or city == null) {
            std.debug.print("weather query requires --db and --city\n", .{});
            return;
        }
        var db = try wdbx.Db.open(db_path.?, false);
        defer db.close();

        const q = try simpleCityEmbedding(allocator, city.?, db.getDimension());
        defer allocator.free(q);
        const results = try db.search(q, k, allocator);
        defer allocator.free(results);
        std.debug.print("Found {d} matches for {s}\n", .{ results.len, city.? });
        for (results, 0..) |r, idx| {
            std.debug.print("  {d}: id={d} score={d:.6}\n", .{ idx, r.index, r.score });
        }
        return;
    }

    std.debug.print("Unknown weather subcommand: {s}\n", .{sub});
}

fn weatherToEmbedding(allocator: std.mem.Allocator, w: services.WeatherData) ![]f32 {
    const v = try allocator.alloc(f32, 16);
    @memset(v, 0);
    v[0] = w.temperature;
    v[1] = w.feels_like;
    v[2] = @floatFromInt(w.humidity);
    v[3] = @floatFromInt(w.pressure);
    v[4] = w.wind_speed;
    v[5] = @floatFromInt(w.wind_direction);
    v[6] = @floatFromInt(w.visibility);
    v[7] = @floatFromInt(w.timestamp % 100000);
    v[8] = @floatFromInt(@as(u32, @intCast(std.hash_map.hashString(w.description))) & 0xFFFF);
    v[9] = @floatFromInt(@as(u32, @intCast(std.hash_map.hashString(w.icon))) & 0xFFFF);
    v[10] = @floatFromInt(@as(u32, @intCast(std.hash_map.hashString(w.city))) & 0xFFFF);
    v[11] = @floatFromInt(@as(u32, @intCast(std.hash_map.hashString(w.country))) & 0xFFFF);
    abi.VectorOps.normalize(v, v);
    return v;
}

fn simpleCityEmbedding(allocator: std.mem.Allocator, city: []const u8, dim_u16: u16) ![]f32 {
    var dim: usize = @intCast(dim_u16);
    if (dim == 0) dim = 16;
    const v = try allocator.alloc(f32, dim);
    @memset(v, 0);
    const h = std.hash_map.hashString(city);
    for (v, 0..) |*out, i| {
        out.* = @floatFromInt(((h >> @intCast(i % 24)) & 0xFF));
    }
    abi.VectorOps.normalize(v, v);
    return v;
}
