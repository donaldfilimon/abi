const std = @import("std");

const DataRow = struct {
    x1: f64,
    x2: f64,
    y: f64,
};

fn readDataset(allocator: std.mem.Allocator, path: []const u8) ![]DataRow {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var reader = file.reader();
    var rows = std.ArrayList(DataRow).init(allocator);
    var buf: [256]u8 = undefined;
    while (true) {
        const line = (try reader.readUntilDelimiterOrEof(&buf, '\n')) orelse break;
        var it = std.mem.splitScalar(u8, line, ',');
        const p1 = it.next() orelse continue;
        const p2 = it.next() orelse continue;
        const p3 = it.next() orelse continue;
        const x1 = try std.fmt.parseFloat(f64, std.mem.trim(u8, p1, " \t\r\n"));
        const x2 = try std.fmt.parseFloat(f64, std.mem.trim(u8, p2, " \t\r\n"));
        const y = try std.fmt.parseFloat(f64, std.mem.trim(u8, p3, " \t\r\n"));
        try rows.append(.{ .x1 = x1, .x2 = x2, .y = y });
    }
    return rows.toOwnedSlice();
}

fn logistic(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

fn train(data: []const DataRow, iterations: u32, lr: f64) struct { w: [2]f64, b: f64 } {
    var w: [2]f64 = .{ 0.0, 0.0 };
    var b: f64 = 0.0;
    for (data) |d| {
        _ = d; // ensure data used later
    }
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        var grad_w0: f64 = 0.0;
        var grad_w1: f64 = 0.0;
        var grad_b: f64 = 0.0;
        for (data) |d| {
            const z = w[0] * d.x1 + w[1] * d.x2 + b;
            const yhat = logistic(z);
            const err = yhat - d.y;
            grad_w0 += err * d.x1;
            grad_w1 += err * d.x2;
            grad_b += err;
        }
        const n = @as(f64, @floatFromInt(data.len));
        w[0] -= lr * grad_w0 / n;
        w[1] -= lr * grad_w1 / n;
        b -= lr * grad_b / n;
    }
    return .{ .w = .{ w[0], w[1] }, .b = b };
}

fn saveModel(path: []const u8, w: [2]f64, b: f64) !void {
    var file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writer().print("{d} {d} {d}\n", .{ w[0], w[1], b });
}

fn loadModel(path: []const u8) !struct { w: [2]f64, b: f64 } {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var buf: [128]u8 = undefined;
    const line = (try file.reader().readUntilDelimiterOrEof(&buf, '\n')) orelse "";
    var it = std.mem.splitScalar(u8, line, ' ');
    const w0 = try std.fmt.parseFloat(f64, it.next() orelse return error.InvalidData);
    const w1 = try std.fmt.parseFloat(f64, it.next() orelse return error.InvalidData);
    const b = try std.fmt.parseFloat(f64, it.next() orelse return error.InvalidData);
    return .{ .w = .{ w0, w1 }, .b = b };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var args = std.process.args();
    _ = args.next();
    const cmd = args.next() orelse {
        std.log.err("usage: local_ml.zig (train <data.csv> <model>)|(predict <model> <x1> <x2>)", .{});
        return;
    };

    if (std.mem.eql(u8, cmd, "train")) {
        const data_path = args.next() orelse {
            std.log.err("train requires data path", .{});
            return;
        };
        const model_path = args.next() orelse {
            std.log.err("train requires model path", .{});
            return;
        };
        const data = try readDataset(alloc, data_path);
        defer alloc.free(data);
        const model = train(data, 1000, 0.1);
        const w = model.w;
        const b = model.b;
        try saveModel(model_path, w, b);
    } else if (std.mem.eql(u8, cmd, "predict")) {
        const model_path = args.next() orelse {
            std.log.err("predict requires model path", .{});
            return;
        };
        const sx1 = args.next() orelse {
            std.log.err("predict requires x1", .{});
            return;
        };
        const sx2 = args.next() orelse {
            std.log.err("predict requires x2", .{});
            return;
        };
        const x1 = try std.fmt.parseFloat(f64, sx1);
        const x2 = try std.fmt.parseFloat(f64, sx2);
        const model = try loadModel(model_path);
        const w = model.w;
        const b = model.b;
        const z = w[0] * x1 + w[1] * x2 + b;
        const prob = logistic(z);
        std.log.info("probability: {d}", .{prob});
    } else {
        std.log.err("unknown command", .{});
    }
}
