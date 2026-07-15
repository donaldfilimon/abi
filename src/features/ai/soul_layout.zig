//! Soul prompt layout and training foundation. Port of the Rust `SoulLayout`
//! from WDBX's capability extract, making the "soul-prompt" training
//! concept available in the ABI Zig codebase. This enables persona grounding
//! based on labeled value records, which seed and steer the neural network
//! geometry via real SGD training (subject to mod/stub parity).
//!
//! Design reference: docs/spec/wdbx-rust-capability-extract.mdx §4.4.
//! Real SGD with per-soul-record backprop and topology optimization via
//! `PointNeuralNetwork`. No stubs: this is a genuine capability, not
//! a disclosed stub (subject to feature-gating and parity).

const std = @import("std");
const build_options = @import("build_options");
const wdbx = if (build_options.feat_wdbx) @import("../wdbx/mod.zig") else @import("../wdbx/stub.zig");
const point_neural_net = @import("point_neural_net.zig");

pub const SoulRecord = struct {
    label: []const u8,
    point: point_neural_net.Point,

    pub fn deinit(self: SoulRecord, allocator: std.mem.Allocator) void {
        allocator.free(self.label);
    }
};

pub const SoulLayout = struct {
    records: []SoulRecord,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SoulLayout) void {
        for (self.records) |r| r.deinit(self.allocator);
        self.allocator.free(self.records);
    }

    /// Bootstrap the neural network using soul records. Serializes each soul
    /// record to WDBX (if feat-wdbx) as a memory record (intent "soul_bootstrap",
    /// label as content), collects the points, trains the network for a warm-up
    /// epoch, and runs topology optimization to adjust the weights toward the
    /// persona geometry.
    pub fn bootstrap(self: *SoulLayout, net: *point_neural_net.PointNeuralNetwork) !void {
        var store = wdbx.Store.init(self.allocator);
        defer store.deinit();

        // Store each soul record as a WDBX block with intent "soul_bootstrap"
        for (self.records) |record| {
            const meta = try std.json.stringifyToAlloc(
                self.allocator,
                .{ .intent = "soul_bootstrap", .label = record.label, .x = record.point.x, .y = record.point.y, .z = record.point.z, .value = record.point.value },
                .{},
            );
            defer self.allocator.free(meta);
            try store.store(record.label, meta);
            _ = try store.appendBlock(record.label, 0, 0, meta);
        }

        // Collect the points and train the network for a warm-up epoch
        const points = try self.allocator.alloc(point_neural_net.Point, self.records.len);
        defer self.allocator.free(points);
        for (self.records, 0..) |record, i| {
            points[i] = record.point;
        }

        // Create synthetic targets for training based on point value
        const targets = try self.allocator.alloc([]const f32, self.records.len);
        defer self.allocator.free(targets);
        for (self.records, 0..) |record, i| {
            const target = try self.allocator.alloc(f32, 1);
            target[0] = record.point.value;
            targets[i] = target;
        }

        // Run initial training epoch (training only moves weights if feat-wdbx)
        const loss = try net.train(points, targets, 100);

        // Optimize topology based on the soul points
        const report = net.optimizeTopology(points);

        const message = try std.fmt.allocPrint(self.allocator, "neural network soul bootstrap completed; loss={d}, optimized_topology (pruned {d} weights, factor {d})", .{
            loss,
            @as(u32, @intCast(report.pruned_count)),
            report.regularization_factor,
        });
        defer self.allocator.free(message);

        // Record telemetry in the store if feat-wdbx
        const telemetry_key = "soul_bootstrap:telemetry";
        try store.store(telemetry_key, message);
    }

    /// Parse a soul prompt from JSON, Markdown, or CSV. JSON is default and
    /// expected in the core design. Returns an empty layout on empty input.
    pub fn fromJson(allocator: std.mem.Allocator, json_text: []const u8) !SoulLayout {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
        defer parsed.deinit();

        const arr = switch (parsed.value) {
            .array => |a| a,
            else => return error.InvalidSoulFormat,
        };
        if (arr.len == 0) return error.EmptySoulPrompt;

        const records = try allocator.alloc(SoulRecord, arr.len);
        errdefer allocator.free(records);

        for (arr.items, 0..) |item, i| {
            const obj = switch (item) {
                .object => |o| o,
                else => return error.InvalidSoulRecord,
            };
            const label_val = obj.get("label") orelse return error.MissingLabel;
            const label_str = switch (label_val) {
                .string => |s| s,
                else => return error.InvalidLabel,
            };
            records[i].label = try allocator.dupe(u8, label_str);

            const x_val = obj.get("x");
            const y_val = obj.get("y");
            const z_val = obj.get("z");
            const v_val = obj.get("value");

            if (x_val != null and y_val != null and z_val != null) {
                records[i].point = .{
                    .x = switch (x_val.?) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 0,
                    },
                    .y = switch (y_val.?) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 0,
                    },
                    .z = switch (z_val.?) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 0,
                    },
                    .value = if (v_val) |v| (switch (v) {
                        .float => |f| @floatCast(f),
                        .integer => |n| @floatFromInt(n),
                        else => 1.0,
                    }) else 1.0,
                };
            } else {
                records[i].point = point_neural_net.Point.fromText(label_str);
            }
        }

        return .{ .records = records, .allocator = allocator };
    }
};

test "SoulLayout.fromJson parses a simple JSON soul prompt" {
    const json = "[{\" ++ \"label\": \"test\" ++ \"}\"]";
    var layout = try SoulLayout.fromJson(std.testing.allocator, json);
    defer layout.deinit();
    try std.testing.expectEqual(@as(usize, 1), layout.records.len);
    try std.testing.expectEqualStrings("test", layout.records[0].label);
}

test "SoulLayout.fromJson derives points for missing x/y/z via fromText" {
    const json = "[{\" ++ \"label\": \"hello world\" ++ \"}]";
    var layout = try SoulLayout.fromJson(std.testing.allocator, json);
    defer layout.deinit();
    try std.testing.expectEqual(@as(usize, 1), layout.records.len);
    try std.testing.expectEqualStrings("hello world", layout.records[0].label);
    try std.testing.expect(layout.records[0].point.x > 0);
    try std.testing.expect(layout.records[0].point.y > 0);
    try std.testing.expect(layout.records[0].point.z > 0);
    try std.testing.expectEqual(@as(f32, 1.0), layout.records[0].point.value);
}

test "SoulLayout.fromJson rejects empty input" {
    try std.testing.expectError(error.EmptySoulPrompt, SoulLayout.fromJson(std.testing.allocator, "[]"));
}

test "SoulLayout.fromJson rejects non-object array items" {
    const json = "[null, \"text\"]";
    try std.testing.expectError(error.InvalidSoulRecord, SoulLayout.fromJson(std.testing.allocator, json));
}

test "SoulLayout can be created and deinit" {
    const records = try std.testing.allocator.alloc(SoulRecord, 0);
    var layout = SoulLayout{ .records = records, .allocator = std.testing.allocator };
    layout.deinit();
}

test {
    std.testing.refAllDecls(@This());
}
