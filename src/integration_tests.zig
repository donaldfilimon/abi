const std = @import("std");
const test_helpers = @import("testing/test_helpers.zig");
const wdbx = @import("features/wdbx/mod.zig");
const constitution = @import("features/ai/constitution.zig");
const gpu_mod = @import("features/gpu/mod.zig");
const router = @import("features/ai/router.zig");

const AgentProfile = @import("features/ai/mod.zig").AgentProfile;
const analyzeSentiment = router.analyzeSentiment;
const selectProfile = router.selectBestProfile;

test "wdbx index insert and search" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    const v1 = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    _ = try store.putVector(&.{ 0.95, 0.05, 0.0, 0.0 });
    _ = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    _ = try store.putVector(&.{ 0.1, 0.9, 0.0, 0.0 });

    try std.testing.expectEqual(@as(usize, 4), store.vectorCount());

    const results = try store.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    try std.testing.expect(results.len == 2);
    try std.testing.expect(results[0].score >= results[1].score);
    try std.testing.expect(results[0].score > 0.9);
    try std.testing.expectEqual(v1, results[0].id);
}

test "wdbx block chain integrity" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    const h1 = try store.appendBlock("abbey", 1, 2, "block metadata 1");
    const h2 = try store.appendBlock("aviva", 3, 4, "block metadata 2");
    _ = try store.appendBlock("abi", 5, 6, "block metadata 3");

    try std.testing.expectEqual(@as(usize, 3), store.blockCount());

    var it = store.chain.iterator();
    defer store.chain.releaseIterator();

    const first = it.next().?.data;
    const second = it.next().?.data;
    const third = it.next().?.data;

    try std.testing.expectEqualStrings("abbey", first.profile);
    try std.testing.expectEqualStrings("aviva", second.profile);
    try std.testing.expectEqualStrings("abi", third.profile);

    const zero_id = std.mem.zeroes([32]u8);
    try std.testing.expect(std.mem.eql(u8, &first.prev_id, &zero_id));

    try std.testing.expect(std.mem.eql(u8, &second.prev_id, &h1));

    try std.testing.expect(std.mem.eql(u8, &third.prev_id, &h2));
}

test "wdbx vector dimension validation" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    _ = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });

    try std.testing.expectError(
        error.DimensionMismatch,
        store.putVector(&.{ 1.0, 0.0 }),
    );

    try std.testing.expectError(
        error.InvalidVector,
        store.putVector(&.{}),
    );
}

test "ai profile routing" {
    const abbey_input = "analyze the logical structure of this system";
    const abbey_weights = analyzeSentiment(abbey_input);
    try std.testing.expect(abbey_weights.w_abbey > abbey_weights.w_aviva);
    try std.testing.expect(abbey_weights.w_abbey > abbey_weights.w_abi);
    try std.testing.expectEqual(AgentProfile.abbey, selectProfile(abbey_weights));

    const aviva_input = "imagine creative possibilities and explore new ideas";
    const aviva_weights = analyzeSentiment(aviva_input);
    try std.testing.expect(aviva_weights.w_aviva > aviva_weights.w_abbey);
    try std.testing.expect(aviva_weights.w_aviva > aviva_weights.w_abi);
    try std.testing.expectEqual(AgentProfile.aviva, selectProfile(aviva_weights));

    const abi_input = "execute deploy run the build quickly";
    const abi_weights = analyzeSentiment(abi_input);
    try std.testing.expect(abi_weights.w_abi > abi_weights.w_abbey);
    try std.testing.expect(abi_weights.w_abi > abi_weights.w_aviva);
    try std.testing.expectEqual(AgentProfile.abi, selectProfile(abi_weights));
}

test "constitution validation" {
    const empty_result = constitution.Constitution.validate("");
    try std.testing.expect(!empty_result.passed);
    try std.testing.expect(empty_result.violations.isSet(@intFromEnum(constitution.Principle.truthfulness)));

    const clean_result = constitution.Constitution.validate("this is a safe and helpful response for everyone");
    try std.testing.expect(clean_result.passed);

    const harm_result = constitution.Constitution.validate("this could cause harm to users");
    try std.testing.expect(!harm_result.passed);
    try std.testing.expect(harm_result.violations.isSet(@intFromEnum(constitution.Principle.safety)));

    const privacy_result = constitution.Constitution.validate("your password is personal data");
    try std.testing.expect(!privacy_result.passed);
    try std.testing.expect(privacy_result.violations.isSet(@intFromEnum(constitution.Principle.privacy)));

    const all_principles = [_]constitution.Principle{
        .truthfulness,
        .safety,
        .helpfulness,
        .fairness,
        .privacy,
        .transparency,
    };
    const eval_result = constitution.Constitution.evaluateResponse("here is how to do it safely", &all_principles);
    try std.testing.expect(eval_result.scores[@intFromEnum(constitution.Principle.helpfulness)] > 0.7);

    const eval_empty = constitution.Constitution.evaluateResponse("", &all_principles);
    try std.testing.expect(!eval_empty.passed);
    for (all_principles) |p| {
        try std.testing.expect(eval_empty.violations.isSet(@intFromEnum(p)));
    }
}

test "connector lifecycle" {
    var connector = test_helpers.MockConnector.init(std.testing.allocator);
    defer connector.deinit();

    try std.testing.expect(!connector.initialized);
    try std.testing.expectEqual(@as(usize, 0), connector.call_count);

    try connector.initialize();
    try std.testing.expect(connector.initialized);

    try connector.addResponse("mock response alpha");
    try connector.addResponse("mock response beta");
    try connector.addResponse("mock response gamma");

    const r1 = try connector.send("query 1");
    try std.testing.expectEqualStrings("mock response alpha", r1);
    try std.testing.expectEqual(@as(usize, 1), connector.call_count);

    const r2 = try connector.send("query 2");
    try std.testing.expectEqualStrings("mock response beta", r2);

    const r3 = try connector.send("query 3");
    try std.testing.expectEqualStrings("mock response gamma", r3);

    try std.testing.expectError(error.NoMoreResponses, connector.send("query 4"));

    connector.close();
    try std.testing.expect(!connector.initialized);
}

test "mcp json-rpc handling" {
    const allocator = std.testing.allocator;

    const JsonRpcRequest = struct {
        jsonrpc: []const u8,
        method: []const u8,
        id: ?std.json.Value = null,
        params: ?std.json.Value = null,
    };

    const initialize_req =
        \\{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2024-11-05"}}
    ;
    const parsed_init = try std.json.parseFromSlice(JsonRpcRequest, allocator, initialize_req, .{ .ignore_unknown_fields = true });
    defer parsed_init.deinit();
    try std.testing.expectEqualStrings("2.0", parsed_init.value.jsonrpc);
    try std.testing.expectEqualStrings("initialize", parsed_init.value.method);

    const ping_req =
        \\{"jsonrpc":"2.0","method":"ping","id":2}
    ;
    const parsed_ping = try std.json.parseFromSlice(JsonRpcRequest, allocator, ping_req, .{ .ignore_unknown_fields = true });
    defer parsed_ping.deinit();
    try std.testing.expectEqualStrings("ping", parsed_ping.value.method);

    const shutdown_req =
        \\{"jsonrpc":"2.0","method":"shutdown","id":3}
    ;
    const parsed_shutdown = try std.json.parseFromSlice(JsonRpcRequest, allocator, shutdown_req, .{ .ignore_unknown_fields = true });
    defer parsed_shutdown.deinit();
    try std.testing.expectEqualStrings("shutdown", parsed_shutdown.value.method);

    const init_result = try std.json.parseFromSlice(std.json.Value, allocator,
        \\{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"abi-mcp","version":"0.1.0"}}
    , .{});
    defer init_result.deinit();
    try std.testing.expect(init_result.value.object.get("protocolVersion") != null);

    const ping_result = try std.json.parseFromSlice(std.json.Value, allocator, "{}", .{});
    defer ping_result.deinit();
    try std.testing.expect(ping_result.value.object.keys().len == 0);

    const shutdown_result = try std.json.parseFromSlice(std.json.Value, allocator, "null", .{});
    defer shutdown_result.deinit();
    try std.testing.expect(shutdown_result.value == .null);

    const invalid_req =
        \\{"jsonrpc":"1.0","method":"invalid","id":99}
    ;
    const parsed_invalid = try std.json.parseFromSlice(JsonRpcRequest, allocator, invalid_req, .{ .ignore_unknown_fields = true });
    defer parsed_invalid.deinit();
    try std.testing.expect(!std.mem.eql(u8, parsed_invalid.value.jsonrpc, "2.0"));
}

test "gpu vector operations" {
    const ops = gpu_mod.vectorOps();

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const dot_result = try ops.dot(&a, &b);
    try std.testing.expect(dot_result > 0);

    const sim = try ops.cosineSimilarity(&a, &a);
    try test_helpers.assertAlmostEqual(@as(f64, @floatCast(sim)), 1.0, 0.001);
}

test {
    std.testing.refAllDecls(@This());
}
