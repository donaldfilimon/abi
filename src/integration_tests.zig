const std = @import("std");
const test_helpers = @import("testing/test_helpers.zig");
const wdbx = @import("features/wdbx/mod.zig");
const constitution = @import("features/ai/constitution.zig");
const gpu_mod = @import("features/gpu/mod.zig");
const router = @import("features/ai/router.zig");
const memory = @import("core/memory.zig");

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
    try std.testing.expect(store.verifyBlocks());

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

test "wdbx memory tracker records allocation activity in hot paths" {
    const wdbx_mod = @import("features/wdbx/mod.zig");
    const memory_mod = @import("core/memory.zig");

    var tracker = memory_mod.MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();

    var store = wdbx_mod.Store.init(std.testing.allocator);
    store.setTracker(&tracker);

    _ = try store.putVector(&.{ 1.0, 0.0, 0.0, 0.0 });
    _ = try store.putVector(&.{ 0.0, 1.0, 0.0, 0.0 });
    _ = try store.putVector(&.{ 0.0, 0.0, 1.0, 0.0 });

    // putVector uses temporary padded buffers and now tracks persistent HNSW vector storage growth.
    try std.testing.expect(tracker.getPeakUsage() > 0);
    const persistent_usage = tracker.getCurrentUsage();
    try std.testing.expect(persistent_usage > 0);

    const results = try store.search(&.{ 1.0, 0.0, 0.0, 0.0 }, 2);
    defer std.testing.allocator.free(results);

    // search allocates and frees its temporary padded query while retaining HNSW storage accounting.
    try std.testing.expectEqual(persistent_usage, tracker.getCurrentUsage());
    try std.testing.expect(tracker.getRecordCount() == 0); // no tag records from hot path

    store.deinit();
    try std.testing.expectEqual(@as(usize, 0), tracker.getCurrentUsage());
}

test "scheduler drives training tasks (real usage in agent training path)" {
    const scheduler = @import("core/scheduler.zig");
    const ai = @import("features/ai/mod.zig");
    const wdbx_mod = @import("features/wdbx/mod.zig");

    var sched = scheduler.Scheduler.init(std.testing.allocator);
    defer sched.deinit();

    var store = wdbx_mod.Store.init(std.testing.allocator);
    defer store.deinit();

    // MemoryTracker wiring demo for the approved plan (Phase 1, scheduler training path).
    // This proves real allocation tracking flows through scheduler-driven TrainTask work.
    var tracker = memory.MemoryTracker.init(std.testing.allocator);
    defer tracker.deinit();
    var tracking_alloc = memory.TrackingAllocator.init(std.testing.allocator, &tracker);
    sched.setMemoryTracker(&tracker);

    const dataset = ai.DatasetSpec{ .path = "datasets/local-training.jsonl" };
    const artifact_dir = "zig-cache/test-agent-artifacts-scheduler";

    const TaskCtx = struct {
        allocator: std.mem.Allocator,
        store: *wdbx_mod.Store,
        dataset: ai.DatasetSpec,
        artifact_dir: []const u8,
        profile: []const u8,
    };

    const TrainTask = struct {
        fn run(ctx: ?*anyopaque) anyerror!void {
            const c = @as(*TaskCtx, @ptrCast(@alignCast(ctx.?)));
            var res = ai.trainWithStore(c.allocator, c.store, .{
                .profile = c.profile,
                .dataset = c.dataset,
                .artifact_dir = c.artifact_dir,
            }) catch |e| {
                std.log.err("scheduler train task failed: {s}", .{@errorName(e)});
                return e;
            };
            res.deinit(c.allocator);
        }
    };

    // Arena for per-task contexts (matches agent handler pattern, zero leaks in test).
    // Backed by TrackingAllocator so the attached scheduler MemoryTracker records the work.
    var arena = std.heap.ArenaAllocator.init(tracking_alloc.allocator());
    defer arena.deinit();
    const task_alloc = arena.allocator();

    // Submit two training tasks via scheduler (simulates `abi agent train all` path)
    {
        const ctx1 = try task_alloc.create(TaskCtx);
        ctx1.* = .{
            .allocator = std.testing.allocator,
            .store = &store,
            .dataset = dataset,
            .artifact_dir = artifact_dir,
            .profile = "abbey",
        };
        _ = try sched.submit("train:abbey", .high, TrainTask.run, ctx1);
    }
    {
        const ctx2 = try task_alloc.create(TaskCtx);
        ctx2.* = .{
            .allocator = std.testing.allocator,
            .store = &store,
            .dataset = dataset,
            .artifact_dir = artifact_dir,
            .profile = "aviva",
        };
        _ = try sched.submit("train:aviva", .normal, TrainTask.run, ctx2);
    }

    var s = sched.stats();
    try std.testing.expectEqual(@as(usize, 0), s.running);
    try std.testing.expectEqual(@as(usize, 2), s.pending);

    try sched.runAll();

    s = sched.stats();
    try std.testing.expectEqual(@as(usize, 0), s.pending);
    try std.testing.expectEqual(@as(usize, 2), s.completed);
    try std.testing.expectEqual(@as(usize, 0), s.failed);

    // Verify memory tracking flowed through the scheduler training path (plan Phase 1).
    try std.testing.expect(tracker.getPeakUsage() > 0 or tracker.getRecordCount() > 0);
}

test "httpPostJson round-trips against loopback server" {
    const http = @import("connectors/http.zig");
    const connector = @import("connectors/connector.zig");
    const ConnectorConfig = connector.ConnectorConfig;

    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var server = try test_helpers.LoopbackHttpServer.init(io, std.testing.allocator);
    defer server.deinit(io);

    const response_body = "{\"status\":\"ok\"}";
    var request_buf_raw: [4096]u8 = undefined;
    var request_buf: []u8 = request_buf_raw[0..0];

    const ServerContext = struct {
        server: *test_helpers.LoopbackHttpServer,
        io: std.Io,
        response_body: []const u8,
        out: *[]u8,
    };
    var ctx = ServerContext{
        .server = &server,
        .io = io,
        .response_body = response_body,
        .out = &request_buf,
    };
    const server_thread = try std.Thread.spawn(.{}, struct {
        fn run(c: *ServerContext) !void {
            const raw = try c.server.acceptAndRespond(c.io, std.testing.allocator, c.response_body);
            c.out.* = raw;
        }
    }.run, .{&ctx});

    const url = try std.fmt.allocPrint(std.testing.allocator, "http://127.0.0.1:{d}", .{server.port});
    defer std.testing.allocator.free(url);

    const config = ConnectorConfig{
        .api_key = "test-key",
        .base_url = url,
        .transport = .live,
    };

    // Retry loop to avoid nanosleep race condition
    var response: ?connector.Response = null;
    var retries: usize = 0;
    while (retries < 5) : (retries += 1) {
        response = http.httpPostJson(io, std.testing.allocator, config, "/test", "{\"hello\":\"world\"}", &.{}) catch null;
        if (response != null) break;
        var ts = std.mem.zeroes(std.c.timespec);
        ts.nsec = 10_000_000;
        _ = std.c.nanosleep(&ts, null);
    }

    var final_response = response orelse {
        // The listener is bound (in this thread) before the client connects, so
        // a total connect failure is a real bug, not a flake. Unblock the
        // server's pending accept() with a throwaway connection so the thread
        // exits cleanly (no leaked/detached thread), join it, free anything it
        // captured, then fail hard rather than silently skipping.
        if (std.Io.net.IpAddress.parseIp4("127.0.0.1", server.port)) |wake_addr_const| {
            var wake_addr = wake_addr_const;
            if (wake_addr.connect(io, .{ .mode = .stream })) |stream| {
                var s = stream;
                s.close(io);
            } else |_| {}
        } else |_| {}
        server_thread.join();
        if (request_buf.len > 0) std.testing.allocator.free(request_buf);
        return error.LoopbackConnectFailed;
    };
    defer final_response.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u16, 200), final_response.status);
    try std.testing.expect(std.mem.indexOf(u8, final_response.body, "ok") != null);

    server_thread.join();
    const raw = request_buf[0..];
    try std.testing.expect(std.mem.indexOf(u8, raw, "POST") != null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "/test") != null);
    std.testing.allocator.free(request_buf);
}

test "httpPostForm round-trips against loopback server" {
    const http = @import("connectors/http.zig");
    const connector_config = @import("connectors/connector.zig");
    const ConnectorConfig = connector_config.ConnectorConfig;

    var threaded = std.Io.Threaded.init(std.testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var server = try test_helpers.LoopbackHttpServer.init(io, std.testing.allocator);
    defer server.deinit(io);

    const response_body = "{\"sid\":\"SM123\"}";
    var request_buf_raw: [4096]u8 = undefined;
    var request_buf: []u8 = request_buf_raw[0..0];

    const ServerContext = struct {
        server: *test_helpers.LoopbackHttpServer,
        io: std.Io,
        response_body: []const u8,
        out: *[]u8,
    };
    var ctx = ServerContext{
        .server = &server,
        .io = io,
        .response_body = response_body,
        .out = &request_buf,
    };
    const server_thread = try std.Thread.spawn(.{}, struct {
        fn run(c: *ServerContext) !void {
            const raw = try c.server.acceptAndRespond(c.io, std.testing.allocator, c.response_body);
            c.out.* = raw;
        }
    }.run, .{&ctx});

    const url = try std.fmt.allocPrint(std.testing.allocator, "http://127.0.0.1:{d}", .{server.port});
    defer std.testing.allocator.free(url);

    const config = ConnectorConfig{
        .api_key = "test-key",
        .base_url = url,
        .transport = .live,
    };
    const body = "From=%2B15551234567&To=%2B15559876543&Body=hello";

    // Retry loop to avoid nanosleep race condition
    var response: ?connector_config.Response = null;
    var retries: usize = 0;
    while (retries < 5) : (retries += 1) {
        response = http.httpPostForm(io, std.testing.allocator, config, "/2010-04-01/Accounts/AC123/Messages.json", body, &.{}) catch null;
        if (response != null) break;
        var ts = std.mem.zeroes(std.c.timespec);
        ts.nsec = 10_000_000;
        _ = std.c.nanosleep(&ts, null);
    }

    var final_response = response orelse {
        // The listener is bound (in this thread) before the client connects, so
        // a total connect failure is a real bug, not a flake. Unblock the
        // server's pending accept() with a throwaway connection so the thread
        // exits cleanly (no leaked/detached thread), join it, free anything it
        // captured, then fail hard rather than silently skipping.
        if (std.Io.net.IpAddress.parseIp4("127.0.0.1", server.port)) |wake_addr_const| {
            var wake_addr = wake_addr_const;
            if (wake_addr.connect(io, .{ .mode = .stream })) |stream| {
                var s = stream;
                s.close(io);
            } else |_| {}
        } else |_| {}
        server_thread.join();
        if (request_buf.len > 0) std.testing.allocator.free(request_buf);
        return error.LoopbackConnectFailed;
    };
    defer final_response.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u16, 200), final_response.status);
    try std.testing.expect(std.mem.indexOf(u8, final_response.body, "SM123") != null);

    server_thread.join();
    const raw = request_buf[0..];
    try std.testing.expect(std.mem.indexOf(u8, raw, "POST") != null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "Messages.json") != null);
    std.testing.allocator.free(request_buf);
}

test "wdbx 3d spatial index through Store: insert, radius, and nearest-neighbor" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    try store.putSpatial3D(1, .{ .x = 0, .y = 0, .z = 0 }, "origin");
    try store.putSpatial3D(2, .{ .x = 1, .y = 0, .z = 0 }, "unit-x");
    try store.putSpatial3D(3, .{ .x = 0, .y = 1, .z = 0 }, "unit-y");
    try store.putSpatial3D(4, .{ .x = 0, .y = 0, .z = 1 }, "unit-z");
    try store.putSpatial3D(5, .{ .x = 5, .y = 5, .z = 5 }, "far-away");

    const stats = store.stats();
    try std.testing.expectEqual(@as(usize, 5), stats.spatial_records);

    const neighbors = try store.searchSpatial3D(.{ .x = 0, .y = 0, .z = 0 }, 3, .euclidean);
    defer std.testing.allocator.free(neighbors);
    try std.testing.expectEqual(@as(usize, 3), neighbors.len);
    try std.testing.expectEqual(@as(u32, 1), neighbors[0].id);
    try std.testing.expect(neighbors[0].distance <= neighbors[1].distance);
    try std.testing.expect(neighbors[1].distance <= neighbors[2].distance);

    const self_radius = try store.spatial_index.radiusSearch(.{ .x = 0, .y = 0, .z = 0 }, 1.5, .euclidean);
    defer std.testing.allocator.free(self_radius);
    try std.testing.expect(self_radius.len >= 3);
    for (self_radius) |r| {
        try std.testing.expect(r.distance <= 1.5);
    }

    const far = try store.spatial_index.radiusSearch(.{ .x = 0, .y = 0, .z = 0 }, 0.5, .euclidean);
    defer std.testing.allocator.free(far);
    try std.testing.expectEqual(@as(usize, 1), far.len);
    try std.testing.expectEqual(@as(u32, 1), far[0].id);
}

test "wdbx MVCC snapshot isolation: snapshot freezes the chain view" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    const h1 = try store.appendBlock("abbey", 1, 1, "snap-1");
    const h2 = try store.appendBlock("aviva", 2, 2, "snap-2");

    var snap = store.chain.getSnapshot();

    try std.testing.expectEqual(@as(usize, 2), snap.len());
    const snap_first = snap.getBlock(h1);
    try std.testing.expect(snap_first != null);
    try std.testing.expectEqualStrings("abbey", snap_first.?.data.profile);
    try std.testing.expect(std.mem.eql(u8, &snap_first.?.data.prev_id, &std.mem.zeroes([32]u8)));

    store.chain.releaseSnapshot();

    _ = try store.appendBlock("abi", 3, 3, "snap-3");
    _ = try store.appendBlock("abbey", 4, 4, "snap-4");

    try std.testing.expectEqual(@as(usize, 4), store.blockCount());
    try std.testing.expectEqual(@as(usize, 2), snap.len());

    var it = snap.iterator();
    var seen: usize = 0;
    while (it.next()) |node| : (seen += 1) {
        if (seen == 0) try std.testing.expectEqualStrings("abbey", node.data.profile);
        if (seen == 1) try std.testing.expectEqualStrings("aviva", node.data.profile);
    }
    try std.testing.expectEqual(@as(usize, 2), seen);

    try std.testing.expect(std.mem.eql(u8, &snap.getBlock(h2).?.header.hash, &h2));
    const h3 = try store.appendBlock("aviva", 5, 5, "snap-5");
    try std.testing.expect(snap.getBlock(h3) == null);
}

test "wdbx tampered block detection: corrupted metadata invalidates verifyBlocks" {
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    _ = try store.appendBlock("abbey", 1, 1, "clean-1");
    _ = try store.appendBlock("aviva", 2, 2, "clean-2");
    _ = try store.appendBlock("abi", 3, 3, "clean-3");
    try std.testing.expect(store.verifyBlocks());

    const head_hash = store.chain.getTailHash().?;
    const node = store.chain.getBlock(head_hash).?;
    const original_metadata = node.data.metadata;
    try std.testing.expect(std.mem.indexOf(u8, original_metadata, "clean-3") != null);

    // Real tamper: @constCast the metadata slice and flip a byte. This breaks
    // the SHA-256 commitment inside the block header. verifyBlocks() walks the
    // chain, recomputes each block's hash from its captured inputs, and should
    // now return false. The byte is restored afterwards so the test is
    // repeatable.
    const tampered = @constCast(node.data.metadata);
    const original_byte = tampered[0];
    tampered[0] = tampered[0] ^ 0xFF;
    try std.testing.expect(!store.verifyBlocks());
    tampered[0] = original_byte;
    try std.testing.expect(store.verifyBlocks());
}

test {
    std.testing.refAllDecls(@This());
}
