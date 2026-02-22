//! Brain Animation View — Universe/Cosmos 3D neural network visualization.
//!
//! Renders a rotating neural network as a cosmic starfield with density-scaled
//! Unicode characters. Node activity drives brightness, size, and color through
//! 256-color ANSI. Depth-based rendering creates parallax — near nodes appear
//! bright and large, far nodes dim and small.
//!
//! Topology: Input(8) → Hidden1(16) → Hidden2(24) → Output(16) = 64 nodes

const std = @import("std");
const terminal_mod = @import("terminal.zig");
const themes = @import("themes.zig");

const Terminal = terminal_mod.Terminal;
const Theme = themes.Theme;

// ===============================================================================
// Configuration
// ===============================================================================

pub const Config = struct {
    spin_speed: f32 = 0.04,
    pulse_speed: f32 = 0.12,
};

/// Fixed brain topology: 4 layers with 64 total nodes.
/// Input(8) → Hidden1(16) → Hidden2(24) → Output(16)
const LAYER_SIZES = [_]u16{ 8, 16, 24, 16 };
const TOTAL_NODES = 64;
const LAYER_OFFSETS = [_]usize{ 0, 8, 24, 48, 64 };

// ===============================================================================
// Cosmos Character Sets
// ===============================================================================

/// Density-scaled node characters (8 levels, low → high activity)
const COSMOS_CHARS = [_][]const u8{ "\xC2\xB7", "\xE2\x88\x99", "\xE2\x88\x98", "\xE2\x97\x8B", "\xE2\x97\x89", "\xE2\x97\x8F", "\xE2\x9C\xA6", "\xE2\x98\x85" };
// ·  ∙  ∘  ○  ◉  ●  ✦  ★

/// Edge trail characters (sparse → dense)
const EDGE_CHARS = [_][]const u8{ "\xC2\xB7", "\xE2\x88\x99", "\xE2\x80\xA2" };
// ·  ∙  •

/// Background star characters for starfield
const STAR_CHARS = [_][]const u8{ ".", "\xC2\xB7", "\xE2\x88\x99" };
// .  ·  ∙

/// 256-color ANSI codes by layer (dim → bright spectrum)
const LayerColors = struct {
    /// Green spectrum for input layer (inserts)
    const input_dim = "\x1b[38;5;22m";
    const input_mid = "\x1b[38;5;34m";
    const input_bright = "\x1b[38;5;46m";

    /// Blue spectrum for hidden layers (search)
    const hidden_dim = "\x1b[38;5;17m";
    const hidden_mid = "\x1b[38;5;27m";
    const hidden_bright = "\x1b[38;5;39m";

    /// Gold spectrum for output layer (learning)
    const output_dim = "\x1b[38;5;94m";
    const output_mid = "\x1b[38;5;178m";
    const output_bright = "\x1b[38;5;226m";

    /// Purple for connections
    const edge_dim = "\x1b[38;5;53m";
    const edge_mid = "\x1b[38;5;98m";

    /// Background stars
    const star_dim = "\x1b[38;5;236m";
    const star_mid = "\x1b[38;5;240m";
};

// ===============================================================================
// Types
// ===============================================================================

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

/// Pre-computed background star position
const BackgroundStar = struct {
    x: u16,
    y: u16,
    char_idx: u8,
    twinkle_phase: u8,
};

// ===============================================================================
// Brain Animation
// ===============================================================================

pub const BrainAnimation = struct {
    nodes: [TOTAL_NODES]Node,
    projected: [TOTAL_NODES]?Projected,
    node_activity: [TOTAL_NODES]f32,
    frame: u64,
    cfg: Config,
    // Background starfield (fixed positions, pre-seeded)
    stars: [32]BackgroundStar,
    star_count: u8,
    last_width: u16,
    last_height: u16,

    pub fn init() BrainAnimation {
        var self: BrainAnimation = .{
            .nodes = undefined,
            .projected = [_]?Projected{null} ** TOTAL_NODES,
            .node_activity = [_]f32{0.0} ** TOTAL_NODES,
            .frame = 0,
            .cfg = .{},
            .stars = undefined,
            .star_count = 0,
            .last_width = 0,
            .last_height = 0,
        };
        buildNodes(&self.nodes);
        return self;
    }

    /// Map external data metrics into node activity levels.
    pub fn updateFromData(self: *BrainAnimation, activity: *const [TOTAL_NODES]f32) void {
        self.node_activity = activity.*;
        self.frame +%= 1;
    }

    /// Adjust animation dynamics based on training state.
    pub fn updateTrainingDynamics(self: *BrainAnimation, loss: f32, _: f32, lr: f32) void {
        self.cfg.spin_speed = 0.02 + @min(0.08, loss * 0.1);
        self.cfg.pulse_speed = 0.06 + @min(0.2, lr * 1000.0);
    }

    /// Advance one animation frame.
    pub fn tick(self: *BrainAnimation) void {
        self.frame +%= 1;
    }

    /// Render the cosmos brain animation into the terminal area.
    pub fn render(
        self: *BrainAnimation,
        term: *Terminal,
        theme: *const Theme,
        start_row: u16,
        start_col: u16,
        width: u16,
        height: u16,
    ) !void {
        if (width < 20 or height < 8) return;

        const t = @as(f32, @floatFromInt(self.frame));
        const w: usize = @intCast(width);
        const h: usize = @intCast(height);

        // Rebuild starfield if terminal size changed
        if (width != self.last_width or height != self.last_height) {
            self.seedStarfield(width, height);
            self.last_width = width;
            self.last_height = height;
        }

        // Project all nodes
        for (&self.nodes, 0..) |node, idx| {
            self.projected[idx] = projectNode(node, t, self.cfg, w, h);
        }

        // Layer 1: Background starfield
        try self.renderStarfield(term, start_row, start_col);

        // Layer 2: Edges (connection trails)
        try self.renderEdges(term, theme, start_row, start_col);

        // Layer 3: Nodes (cosmos density)
        try self.renderNodes(term, theme, start_row, start_col);
    }

    // ===================================================================
    // Starfield
    // ===================================================================

    fn seedStarfield(self: *BrainAnimation, width: u16, height: u16) void {
        // Deterministic pseudo-random starfield from frame-independent seed
        var seed: u32 = 0xDEAD_BEEF;
        var count: u8 = 0;
        const max_stars: u8 = 32;
        const target = @min(max_stars, @as(u8, @intCast(@min(@as(usize, 32), (@as(usize, width) * @as(usize, height)) / 80))));

        while (count < target) {
            seed = seed *% 1103515245 +% 12345;
            const x = @as(u16, @intCast((seed >> 16) % @as(u32, width)));
            seed = seed *% 1103515245 +% 12345;
            const y = @as(u16, @intCast((seed >> 16) % @as(u32, height)));
            seed = seed *% 1103515245 +% 12345;
            const char_idx = @as(u8, @intCast((seed >> 16) % STAR_CHARS.len));
            seed = seed *% 1103515245 +% 12345;
            const phase = @as(u8, @intCast((seed >> 16) % 60));

            self.stars[count] = .{
                .x = x,
                .y = y,
                .char_idx = char_idx,
                .twinkle_phase = phase,
            };
            count += 1;
        }
        self.star_count = count;
    }

    fn renderStarfield(self: *const BrainAnimation, term: *Terminal, start_row: u16, start_col: u16) !void {
        const frame_mod = @as(u8, @intCast(self.frame % 60));

        for (self.stars[0..self.star_count]) |star| {
            // Twinkle: only visible ~40% of the time based on phase
            const visible = ((frame_mod +% star.twinkle_phase) % 60) < 24;
            if (!visible) continue;

            try setCursorAt(term, start_row, start_col, star.y, star.x);
            // Alternate dim colors
            const color = if (((frame_mod +% star.twinkle_phase) % 60) < 12)
                LayerColors.star_dim
            else
                LayerColors.star_mid;
            try term.write(color);
            try term.write(STAR_CHARS[star.char_idx]);
            try term.write("\x1b[0m");
        }
    }

    // ===================================================================
    // Edges (connection trails)
    // ===================================================================

    fn renderEdges(
        self: *const BrainAnimation,
        term: *Terminal,
        _: *const Theme,
        start_row: u16,
        start_col: u16,
    ) !void {
        var layer_idx: usize = 0;
        while (layer_idx + 1 < LAYER_SIZES.len) : (layer_idx += 1) {
            const src_count = LAYER_SIZES[layer_idx];
            const dst_count = LAYER_SIZES[layer_idx + 1];
            const src_off = LAYER_OFFSETS[layer_idx];
            const dst_off = LAYER_OFFSETS[layer_idx + 1];

            var src_idx: usize = 0;
            while (src_idx < src_count) : (src_idx += 1) {
                const src = self.projected[src_off + src_idx] orelse continue;
                const src_activity = self.node_activity[src_off + src_idx];

                const fan = @min(@as(usize, 3), dst_count);
                const mapped = if (src_count > 1) (src_idx * dst_count) / src_count else 0;

                for (0..fan) |fi| {
                    const step = if (fan <= 1) 0 else fi * @max(@as(usize, 1), dst_count / fan);
                    const di = @min(dst_count - 1, mapped + step);
                    const dst = self.projected[dst_off + di] orelse continue;
                    const dst_activity = self.node_activity[dst_off + di];

                    const edge_val = (src_activity + dst_activity) * 0.5;

                    // 3-point trail instead of single midpoint
                    const points = [_][2]i32{
                        .{ src.x + @divTrunc(dst.x - src.x, 4), src.y + @divTrunc(dst.y - src.y, 4) },
                        .{ @divTrunc(src.x + dst.x, 2), @divTrunc(src.y + dst.y, 2) },
                        .{ dst.x - @divTrunc(dst.x - src.x, 4), dst.y - @divTrunc(dst.y - src.y, 4) },
                    };

                    const color = if (edge_val > 0.5) LayerColors.edge_mid else LayerColors.edge_dim;

                    for (points, 0..) |pt, pi| {
                        if (pt[0] >= 0 and pt[1] >= 0) {
                            const char_idx = if (edge_val > 0.6) @min(pi, EDGE_CHARS.len - 1) else 0;
                            try setCursorAt(term, start_row, start_col, @intCast(pt[1]), @intCast(pt[0]));
                            try term.write(color);
                            try term.write(EDGE_CHARS[char_idx]);
                            try term.write("\x1b[0m");
                        }
                    }
                }
            }
        }
    }

    // ===================================================================
    // Nodes (cosmos density rendering)
    // ===================================================================

    fn renderNodes(
        self: *const BrainAnimation,
        term: *Terminal,
        _: *const Theme,
        start_row: u16,
        start_col: u16,
    ) !void {
        for (0..TOTAL_NODES) |idx| {
            const p = self.projected[idx] orelse continue;
            if (p.x < 0 or p.y < 0) continue;

            const activity = self.node_activity[idx];
            const layer = nodeLayer(idx);

            // Depth-based size: near (low depth) = bigger char, far = smaller
            const depth_factor = 1.0 - @min(1.0, @max(0.0, (p.depth - 3.0) / 3.0));
            const effective = @min(1.0, activity * (0.5 + 0.5 * depth_factor));

            const color = cosmosNodeColor(layer, effective, depth_factor);
            const char = cosmosNodeChar(effective);

            try setCursorAt(term, start_row, start_col, @intCast(p.y), @intCast(p.x));
            try term.write(color);
            try term.write(char);
            try term.write("\x1b[0m");
        }
    }
};

// ===============================================================================
// Projection Math
// ===============================================================================

fn buildNodes(nodes: *[TOTAL_NODES]Node) void {
    const layer_denom: f32 = @as(f32, @floatFromInt(LAYER_SIZES.len - 1));

    for (&LAYER_SIZES, 0..) |layer_size, layer_idx| {
        const x = (@as(f32, @floatFromInt(layer_idx)) / layer_denom) * 2.8 - 1.4;
        const y_denom = if (layer_size > 1) @as(f32, @floatFromInt(layer_size - 1)) else 1.0;

        for (0..layer_size) |node_idx| {
            const y = if (layer_size == 1)
                0.0
            else
                (@as(f32, @floatFromInt(node_idx)) / y_denom) * 2.2 - 1.1;
            const phase_deg = (layer_idx * 37 + node_idx * 17) % 360;
            nodes[LAYER_OFFSETS[layer_idx] + node_idx] = .{
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

// ===============================================================================
// Cosmos Helpers
// ===============================================================================

fn nodeLayer(idx: usize) usize {
    if (idx < 8) return 0;
    if (idx < 24) return 1;
    if (idx < 48) return 2;
    return 3;
}

/// 256-color by layer + activity + depth
fn cosmosNodeColor(layer: usize, activity: f32, depth_factor: f32) []const u8 {
    // High activity + near = bright, low activity or far = dim
    const brightness = activity * (0.5 + 0.5 * depth_factor);

    if (brightness > 0.65) {
        return switch (layer) {
            0 => LayerColors.input_bright,
            1, 2 => LayerColors.hidden_bright,
            3 => LayerColors.output_bright,
            else => LayerColors.hidden_bright,
        };
    }
    if (brightness > 0.3) {
        return switch (layer) {
            0 => LayerColors.input_mid,
            1, 2 => LayerColors.hidden_mid,
            3 => LayerColors.output_mid,
            else => LayerColors.hidden_mid,
        };
    }
    return switch (layer) {
        0 => LayerColors.input_dim,
        1, 2 => LayerColors.hidden_dim,
        3 => LayerColors.output_dim,
        else => LayerColors.hidden_dim,
    };
}

/// Density-scaled cosmos character (8 levels)
fn cosmosNodeChar(activity: f32) []const u8 {
    const idx: usize = @intFromFloat(@min(7.0, @max(0.0, activity * 8.0)));
    return COSMOS_CHARS[idx];
}

fn setCursorAt(term: *Terminal, base_row: u16, base_col: u16, row: u16, col: u16) !void {
    var buf: [16]u8 = undefined;
    const seq = std.fmt.bufPrint(&buf, "\x1b[{d};{d}H", .{
        @as(u32, base_row) + row + 1,
        @as(u32, base_col) + col + 1,
    }) catch return;
    try term.write(seq);
}
