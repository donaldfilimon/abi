//! Dynamic 3D neural network visualization for terminal UI.

const std = @import("std");
const abi = @import("abi");
const shared_time = abi.shared.time;

const Config = struct {
    frames: u32 = 240,
    fps: u32 = 30,
    width: usize = 100,
    height: usize = 34,
    spin_speed: f32 = 0.045,
    pulse_speed: f32 = 0.13,
};

const Node = struct {
    x: f32,
    y: f32,
    phase: f32,
};

const Projected = struct {
    x: i32,
    y: i32,
    depth: f32,
    activity: f32,
};

pub fn run(allocator: std.mem.Allocator, _: std.Io, args: []const [:0]const u8) !void {
    try runVisualizer(allocator, args);
}

pub fn runVisualizer(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    if (containsHelp(args)) {
        printHelp();
        return;
    }

    var cfg = Config{};
    var layer_sizes_buf: [16]u16 = undefined;
    layer_sizes_buf[0] = 8;
    layer_sizes_buf[1] = 16;
    layer_sizes_buf[2] = 12;
    layer_sizes_buf[3] = 6;
    var layer_count: usize = 4;

    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = std.mem.sliceTo(args[i], 0);

        if (std.mem.eql(u8, arg, "--frames")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            cfg.frames = try std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10);
            continue;
        }
        if (std.mem.eql(u8, arg, "--fps")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            cfg.fps = try std.fmt.parseInt(u32, std.mem.sliceTo(args[i], 0), 10);
            if (cfg.fps == 0) cfg.fps = 1;
            continue;
        }
        if (std.mem.eql(u8, arg, "--width")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            cfg.width = @max(@as(usize, 40), try std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10));
            continue;
        }
        if (std.mem.eql(u8, arg, "--height")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            cfg.height = @max(@as(usize, 14), try std.fmt.parseInt(usize, std.mem.sliceTo(args[i], 0), 10));
            continue;
        }
        if (std.mem.eql(u8, arg, "--layers")) {
            i += 1;
            if (i >= args.len) return error.InvalidArgument;
            layer_count = try parseLayerSpec(std.mem.sliceTo(args[i], 0), &layer_sizes_buf);
            continue;
        }
    }

    const layers = layer_sizes_buf[0..layer_count];
    const canvas_h = cfg.height - 2;

    var layer_offsets = try allocator.alloc(usize, layers.len + 1);
    defer allocator.free(layer_offsets);
    layer_offsets[0] = 0;
    for (layers, 0..) |layer_size, idx| {
        layer_offsets[idx + 1] = layer_offsets[idx] + layer_size;
    }
    const total_nodes = layer_offsets[layers.len];

    const nodes = try allocator.alloc(Node, total_nodes);
    defer allocator.free(nodes);
    buildNodes(nodes, layers, layer_offsets);

    var projected = try allocator.alloc(?Projected, total_nodes);
    defer allocator.free(projected);

    var canvas = try allocator.alloc(u8, cfg.width * canvas_h);
    defer allocator.free(canvas);

    std.debug.print("\x1b[?25l\x1b[2J", .{});
    defer std.debug.print("\x1b[?25h\x1b[0m\n", .{});

    const sleep_ms = @max(@as(u32, 1), @divFloor(@as(u32, 1000), cfg.fps));
    var frame: u32 = 0;
    while (cfg.frames == 0 or frame < cfg.frames) : (frame += 1) {
        @memset(canvas, ' ');

        const t = @as(f32, @floatFromInt(frame));
        for (nodes, 0..) |node, idx| {
            projected[idx] = projectNode(node, t, cfg, cfg.width, canvas_h);
        }

        drawEdges(canvas, cfg.width, canvas_h, projected, layers, layer_offsets);
        drawNodes(canvas, cfg.width, canvas_h, projected);

        std.debug.print("\x1b[H", .{});
        std.debug.print("ABI Neural 3D | frame {d}", .{frame + 1});
        if (cfg.frames > 0) std.debug.print("/{d}", .{cfg.frames});
        std.debug.print(" | fps {d} | layers ", .{cfg.fps});
        for (layers, 0..) |layer_size, idx| {
            if (idx > 0) std.debug.print(",", .{});
            std.debug.print("{d}", .{layer_size});
        }
        std.debug.print("\n", .{});
        std.debug.print("Ctrl-C to stop. Tunables: --layers 8,16,12,6 --frames 240 --fps 30\n", .{});

        var row: usize = 0;
        while (row < canvas_h) : (row += 1) {
            const start = row * cfg.width;
            std.debug.print("{s}\n", .{canvas[start .. start + cfg.width]});
        }

        shared_time.sleepMs(sleep_ms);
    }
}

fn containsHelp(args: []const [:0]const u8) bool {
    for (args) |arg| {
        const value = std.mem.sliceTo(arg, 0);
        if (std.mem.eql(u8, value, "--help") or std.mem.eql(u8, value, "-h") or std.mem.eql(u8, value, "help")) {
            return true;
        }
    }
    return false;
}

fn parseLayerSpec(spec: []const u8, out: []u16) !usize {
    var count: usize = 0;
    var iter = std.mem.splitScalar(u8, spec, ',');
    while (iter.next()) |raw| {
        const value = std.mem.trim(u8, raw, " \t");
        if (value.len == 0) continue;
        if (count >= out.len) return error.TooManyLayers;

        const size = try std.fmt.parseInt(u16, value, 10);
        if (size == 0) return error.InvalidLayerSize;
        out[count] = size;
        count += 1;
    }
    if (count < 2) return error.InvalidLayerSpec;
    return count;
}

fn buildNodes(nodes: []Node, layers: []const u16, offsets: []const usize) void {
    const layer_denom = if (layers.len > 1)
        @as(f32, @floatFromInt(layers.len - 1))
    else
        1.0;

    for (layers, 0..) |layer_size, layer_idx| {
        const x = if (layers.len == 1)
            0.0
        else
            (@as(f32, @floatFromInt(layer_idx)) / layer_denom) * 2.8 - 1.4;
        const y_denom = if (layer_size > 1)
            @as(f32, @floatFromInt(layer_size - 1))
        else
            1.0;

        var node_idx: usize = 0;
        while (node_idx < layer_size) : (node_idx += 1) {
            const y = if (layer_size == 1)
                0.0
            else
                (@as(f32, @floatFromInt(node_idx)) / y_denom) * 2.2 - 1.1;
            const phase_deg = (layer_idx * 37 + node_idx * 17) % 360;
            nodes[offsets[layer_idx] + node_idx] = .{
                .x = x,
                .y = y,
                .phase = @as(f32, @floatFromInt(phase_deg)) * (std.math.pi / 180.0),
            };
        }
    }
}

fn projectNode(node: Node, t: f32, cfg: Config, width: usize, height: usize) ?Projected {
    const pulse = std.math.sin(t * cfg.pulse_speed + node.phase);
    const z = 0.45 * pulse;
    const y = node.y + 0.12 * std.math.cos(t * (cfg.pulse_speed * 0.65) + node.phase * 1.7);

    const yaw = t * cfg.spin_speed;
    const pitch = 0.55 + t * (cfg.spin_speed * 0.47);
    const cy = std.math.cos(yaw);
    const sy = std.math.sin(yaw);
    const cx = std.math.cos(pitch);
    const sx = std.math.sin(pitch);

    const rx = node.x * cy - z * sy;
    const rz1 = node.x * sy + z * cy;
    const ry = y * cx - rz1 * sx;
    const rz = y * sx + rz1 * cx + 4.3;

    if (rz <= 0.25) return null;

    const perspective = 1.85 / rz;
    const cx_screen = @as(f32, @floatFromInt(width - 1)) * 0.5;
    const cy_screen = @as(f32, @floatFromInt(height - 1)) * 0.5;

    const sx_screen = cx_screen + rx * perspective * @as(f32, @floatFromInt(width)) * 0.42;
    const sy_screen = cy_screen - ry * perspective * @as(f32, @floatFromInt(height)) * 0.72;

    const ix = @as(i32, @intFromFloat(sx_screen));
    const iy = @as(i32, @intFromFloat(sy_screen));
    if (ix < 0 or iy < 0) return null;
    if (ix >= @as(i32, @intCast(width)) or iy >= @as(i32, @intCast(height))) return null;

    return .{
        .x = ix,
        .y = iy,
        .depth = rz,
        .activity = pulse,
    };
}

fn drawEdges(
    canvas: []u8,
    width: usize,
    height: usize,
    projected: []const ?Projected,
    layers: []const u16,
    offsets: []const usize,
) void {
    if (layers.len < 2) return;

    var layer_idx: usize = 0;
    while (layer_idx + 1 < layers.len) : (layer_idx += 1) {
        const current_count = layers[layer_idx];
        const next_count = layers[layer_idx + 1];
        const current_offset = offsets[layer_idx];
        const next_offset = offsets[layer_idx + 1];

        var src_idx: usize = 0;
        while (src_idx < current_count) : (src_idx += 1) {
            const source = projected[current_offset + src_idx] orelse continue;
            const mapped = if (current_count > 1)
                (src_idx * next_count) / current_count
            else
                0;

            const fan = @min(@as(usize, 3), next_count);
            var fan_idx: usize = 0;
            while (fan_idx < fan) : (fan_idx += 1) {
                const step = if (fan <= 1) 0 else fan_idx * @max(@as(usize, 1), next_count / fan);
                const dst_idx = @min(next_count - 1, mapped + step);
                const target = projected[next_offset + dst_idx] orelse continue;
                const edge_depth = (source.depth + target.depth) * 0.5;
                drawLine(
                    canvas,
                    width,
                    height,
                    source.x,
                    source.y,
                    target.x,
                    target.y,
                    edgeChar(edge_depth),
                );
            }
        }
    }
}

fn drawNodes(canvas: []u8, width: usize, height: usize, projected: []const ?Projected) void {
    for (projected) |p| {
        const point = p orelse continue;
        setPixel(canvas, width, height, point.x, point.y, nodeChar(point.activity));
    }
}

fn setPixel(canvas: []u8, width: usize, height: usize, x: i32, y: i32, ch: u8) void {
    if (x < 0 or y < 0) return;
    if (x >= @as(i32, @intCast(width)) or y >= @as(i32, @intCast(height))) return;
    const ux: usize = @intCast(x);
    const uy: usize = @intCast(y);
    canvas[uy * width + ux] = ch;
}

fn drawLine(
    canvas: []u8,
    width: usize,
    height: usize,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    ch: u8,
) void {
    var ax = x0;
    var ay = y0;
    const dx: i32 = @intCast(@abs(x1 - x0));
    const sx: i32 = if (x0 < x1) 1 else -1;
    const dy: i32 = -@as(i32, @intCast(@abs(y1 - y0)));
    const sy: i32 = if (y0 < y1) 1 else -1;
    var err = dx + dy;

    while (true) {
        setPixel(canvas, width, height, ax, ay, ch);
        if (ax == x1 and ay == y1) break;

        const e2 = err * 2;
        if (e2 >= dy) {
            if (ax == x1) break;
            err += dy;
            ax += sx;
        }
        if (e2 <= dx) {
            if (ay == y1) break;
            err += dx;
            ay += sy;
        }
    }
}

fn nodeChar(activity: f32) u8 {
    const abs_activity = @abs(activity);
    if (abs_activity > 0.85) return '@';
    if (abs_activity > 0.65) return 'O';
    if (abs_activity > 0.35) return 'o';
    return '*';
}

fn edgeChar(depth: f32) u8 {
    if (depth < 3.7) return '*';
    if (depth < 4.4) return ':';
    return '.';
}

fn printHelp() void {
    std.debug.print(
        \\Usage: abi ui neural [options]
        \\
        \\Render a dynamic 3D neural-network view in the terminal.
        \\
        \\Options:
        \\  --frames <n>         Number of frames (default: 240, 0 = run until Ctrl-C)
        \\  --fps <n>            Target frame rate (default: 30)
        \\  --width <n>          Render width (default: 100, min: 40)
        \\  --height <n>         Render height (default: 34, min: 14)
        \\  --layers <csv>       Layer sizes (default: 8,16,12,6)
        \\  -h, --help           Show this help
        \\
        \\Examples:
        \\  abi ui neural
        \\  abi ui neural --layers 12,24,24,12,4 --frames 0
        \\  abi ui neural --fps 60 --width 140 --height 44
        \\
    , .{});
}
