const std = @import("std");
const abi = @import("abi");
const usage = @import("cli_usage");
const handlers = @import("mcp_handlers");

fn expectString(value: std.json.Value, expected: []const u8) !void {
    const actual = switch (value) {
        .string => |s| s,
        else => return error.ExpectedString,
    };
    try std.testing.expectEqualStrings(expected, actual);
}

fn expectObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |obj| obj,
        else => error.ExpectedObject,
    };
}

fn expectArray(value: std.json.Value) !std.json.Array {
    return switch (value) {
        .array => |arr| arr,
        else => error.ExpectedArray,
    };
}

fn expectRequiredFields(required_value: std.json.Value, expected: []const []const u8) !void {
    const required = try expectArray(required_value);
    try std.testing.expectEqual(expected.len, required.items.len);
    for (expected, required.items) |name, item| {
        try expectString(item, name);
    }
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
}

fn expectInteger(value: std.json.Value, expected: u32) !void {
    const actual = switch (value) {
        .integer => |n| blk: {
            if (n < 0) return error.NegativeInteger;
            break :blk @as(u32, @intCast(n));
        },
        else => return error.ExpectedInteger,
    };
    try std.testing.expectEqual(expected, actual);
}

fn expectCompletionMetadataJson(allocator: std.mem.Allocator, metadata: []const u8, model: []const u8, query_id: u32, response_id: u32) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, metadata, .{});
    defer parsed.deinit();

    const obj = try expectObject(parsed.value);
    try expectString(obj.get("kind") orelse return error.MissingKind, "completion");
    try expectString(obj.get("model") orelse return error.MissingModel, model);
    const profile = switch (obj.get("profile") orelse return error.MissingProfile) {
        .string => |s| s,
        else => return error.ExpectedString,
    };
    try std.testing.expect(profile.len > 0);
    _ = obj.get("audit_passed") orelse return error.MissingAuditPassed;
    _ = obj.get("input_bytes") orelse return error.MissingInputBytes;
    _ = obj.get("output_bytes") orelse return error.MissingOutputBytes;
    try expectInteger(obj.get("query_vector_id") orelse return error.MissingQueryVector, query_id);
    try expectInteger(obj.get("response_vector_id") orelse return error.MissingResponseVector, response_id);
}

test "root public namespaces are frozen" {
    inline for (.{
        "interfaces",
        "foundation",
        "features",
        "registry",
        "config",
        "connectors",
        "memory",
        "scheduler",
        "plugins",
    }) |decl_name| {
        try std.testing.expect(@hasDecl(abi, decl_name));
    }
}

test "feature module surfaces expose safe defaults" {
    const features = abi.features;
    inline for (.{
        "ai",
        "accelerator",
        "gpu",
        "mlir",
        "os_control",
        "shaders",
        "tui",
        "wdbx",
        "mobile",
    }) |decl_name| {
        try std.testing.expect(@hasDecl(features, decl_name));
    }

    const gpu_caps = features.gpu.backendCapabilitiesList();
    try std.testing.expectEqual(@as(usize, 7), gpu_caps.len);
    try std.testing.expect(features.gpu.detectBackend().message.len > 0);
    const ops = features.gpu.vectorOps();
    try std.testing.expectEqual(@as(f32, 32), try ops.dot(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));

    const accelerator_selection = features.accelerator.selectBackend(.training);
    try std.testing.expect(accelerator_selection.message.len > 0);

    try std.testing.expect(features.shaders.compilerStatus().message.len > 0);
    try std.testing.expect(features.mlir.toolchainStatus().message.len > 0);
    try std.testing.expect(features.mobile.detectPlatform().message.len > 0);

    const dashboard = try features.tui.renderDashboard(std.testing.allocator, .{ .title = "ABI" });
    defer std.testing.allocator.free(dashboard);
    try std.testing.expect(dashboard.len > 0);

    const command_decision = features.os_control.validateCommand(.{ .argv = &.{"ls"} }, .{ .workspace_root = "/tmp/work" });
    try std.testing.expect(command_decision.message.len > 0);

    var store = features.wdbx.Store.init(std.testing.allocator);
    defer store.deinit();
    const stats = store.stats();
    try std.testing.expect(stats.acceleration.message.len > 0);

    var completion = try features.ai.complete(std.testing.allocator, .{ .input = "contract surface", .model = "abi-contract" });
    defer completion.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings("abi-contract", completion.model);
    try std.testing.expect(completion.output.len > 0);
}

test "nested feature public surfaces are frozen across feature flags" {
    const ai = abi.features.ai;
    try std.testing.expect(@hasDecl(ai.streaming.openai, "OpenAIRequest"));
    try std.testing.expect(@hasDecl(ai.streaming.openai, "OpenAIStreamChunk"));
    try std.testing.expect(@hasDecl(ai.streaming.openai, "parseRequest"));
    try std.testing.expect(@hasDecl(ai.streaming.openai, "handleOpenAIChatCompletions"));
    try std.testing.expect(@hasDecl(ai.constitution, "Principle"));
    try std.testing.expect(@hasDecl(ai.constitution, "AuditResult"));
    try std.testing.expect(@hasDecl(ai.constitution, "Constitution"));
    try std.testing.expect(@hasDecl(ai.profile, "ProfileWeights"));
    try std.testing.expect(@hasDecl(ai.profile.ProfileWeights, "normalize"));
    try std.testing.expect(@hasDecl(ai.profile, "SentimentKeyword"));
    try std.testing.expect(@hasDecl(ai.profile, "SENTIMENT_KEYWORDS"));

    const wdbx = abi.features.wdbx;
    try std.testing.expect(@hasDecl(wdbx.spatial_3d, "SpatialIndex3D"));
    try std.testing.expect(@hasDecl(wdbx.spatial_3d, "euclideanDistance"));
    try std.testing.expect(@hasDecl(wdbx.spatial_3d, "manhattanDistance"));
    try std.testing.expect(@hasDecl(wdbx.spatial_3d, "cosineDistance"));
    try std.testing.expect(@hasDecl(wdbx.spatial_3d, "calculateDistance"));
    try std.testing.expect(@hasDecl(wdbx.index, "Candidate"));
    try std.testing.expect(@hasDecl(wdbx.index.HnswNode, "initEdges"));
    try std.testing.expect(@hasDecl(wdbx.storage.BlockChain, "releaseIterator"));
}

test "AI completion WDBX persistence is opt-in and append-only" {
    const allocator = std.testing.allocator;
    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();

    const initial = store.stats();
    var skipped = try abi.features.ai.completeWithStore(allocator, &store, .{
        .input = "contract skip persistence",
        .model = "abi-contract",
        .store_result = false,
    });
    defer skipped.deinit(allocator);

    try std.testing.expect(skipped.output.len > 0);
    try std.testing.expect(skipped.query_vector_id == null);
    try std.testing.expect(skipped.response_vector_id == null);
    try std.testing.expect(skipped.block_id == null);

    const after_skip = store.stats();
    try std.testing.expectEqual(initial.kv_entries, after_skip.kv_entries);
    try std.testing.expectEqual(initial.vectors, after_skip.vectors);
    try std.testing.expectEqual(initial.blocks, after_skip.blocks);

    var first = try abi.features.ai.completeWithStore(allocator, &store, .{
        .input = "contract persist first completion",
        .model = "abi-contract",
        .store_result = true,
    });
    defer first.deinit(allocator);

    if (first.query_vector_id == null) {
        try std.testing.expect(first.response_vector_id == null);
        try std.testing.expect(first.block_id == null);
        try std.testing.expectEqual(@as(usize, 0), store.count());
        try std.testing.expectEqual(@as(usize, 0), store.vectorCount());
        try std.testing.expectEqual(@as(usize, 0), store.blockCount());
        try std.testing.expect(store.verifyBlocks());
        return;
    }

    const first_qid = first.query_vector_id.?;
    const first_rid = first.response_vector_id orelse return error.MissingResponseVector;
    const first_block_id = first.block_id orelse return error.MissingCompletionBlock;

    try std.testing.expectEqual(@as(usize, 1), store.count());
    try std.testing.expectEqual(@as(usize, 2), store.vectorCount());
    try std.testing.expectEqual(@as(usize, 1), store.blockCount());
    try std.testing.expect(store.verifyBlocks());

    const first_key = try std.fmt.allocPrint(allocator, "completion:{d}", .{first_qid});
    defer allocator.free(first_key);
    const first_metadata = store.get(first_key) orelse return error.MissingCompletionMetadata;
    try expectCompletionMetadataJson(allocator, first_metadata, "abi-contract", first_qid, first_rid);

    const first_block = store.lastBlock() orelse return error.MissingCompletionBlock;
    const zero_id = std.mem.zeroes([32]u8);
    try std.testing.expect(std.mem.eql(u8, &first_block.id, &first_block_id));
    try std.testing.expect(std.mem.eql(u8, &first_block.prev_id, &zero_id));
    try std.testing.expectEqual(first_qid, first_block.query_id);
    try std.testing.expectEqual(first_rid, first_block.response_id);

    var second = try abi.features.ai.completeWithStore(allocator, &store, .{
        .input = "contract persist second completion",
        .model = "abi-contract-2",
        .store_result = true,
    });
    defer second.deinit(allocator);

    const second_qid = second.query_vector_id orelse return error.MissingQueryVector;
    const second_rid = second.response_vector_id orelse return error.MissingResponseVector;
    const second_block_id = second.block_id orelse return error.MissingCompletionBlock;

    try std.testing.expect(second_qid != first_qid);
    try std.testing.expectEqual(@as(usize, 2), store.count());
    try std.testing.expectEqual(@as(usize, 4), store.vectorCount());
    try std.testing.expectEqual(@as(usize, 2), store.blockCount());
    try std.testing.expect(store.verifyBlocks());

    const second_key = try std.fmt.allocPrint(allocator, "completion:{d}", .{second_qid});
    defer allocator.free(second_key);
    const second_metadata = store.get(second_key) orelse return error.MissingCompletionMetadata;
    try expectCompletionMetadataJson(allocator, second_metadata, "abi-contract-2", second_qid, second_rid);

    const second_block = store.lastBlock() orelse return error.MissingCompletionBlock;
    try std.testing.expect(std.mem.eql(u8, &second_block.id, &second_block_id));
    try std.testing.expect(std.mem.eql(u8, &second_block.prev_id, &first_block_id));
    try std.testing.expectEqual(second_qid, second_block.query_id);
    try std.testing.expectEqual(second_rid, second_block.response_id);

    const final_stats = store.stats();
    try std.testing.expectEqual(@as(usize, 2), final_stats.kv_entries);
    try std.testing.expectEqual(@as(usize, 4), final_stats.vectors);
    try std.testing.expectEqual(@as(usize, 2), final_stats.blocks);
    try std.testing.expectEqual(@as(?usize, 4), final_stats.vector_dimensions);
    try std.testing.expectEqual(second_rid + 1, final_stats.next_vector_id);
}

test "AI completion invalid or disabled input does not mutate WDBX" {
    const allocator = std.testing.allocator;
    var store = abi.features.wdbx.Store.init(allocator);
    defer store.deinit();

    const initial = store.stats();
    if (abi.features.ai.completeWithStore(allocator, &store, .{
        .input = "",
        .model = "abi-contract",
        .store_result = true,
    })) |result| {
        var completion = result;
        defer completion.deinit(allocator);
        try std.testing.expect(completion.query_vector_id == null);
        try std.testing.expect(completion.response_vector_id == null);
        try std.testing.expect(completion.block_id == null);
    } else |err| {
        try std.testing.expect(err == error.InvalidCompletionInput);
    }

    const after = store.stats();
    try std.testing.expectEqual(initial.kv_entries, after.kv_entries);
    try std.testing.expectEqual(initial.vectors, after.vectors);
    try std.testing.expectEqual(initial.blocks, after.blocks);
    try std.testing.expect(store.verifyBlocks());
}

test "CLI command surface is frozen" {
    const expected = [_][]const u8{
        "help",   "complete", "train",  "agent", "backends",
        "plugin", "auth",     "twilio", "tui",   "dashboard",
    };
    try std.testing.expectEqual(expected.len, usage.commands.len);
    for (expected, usage.commands) |name, cmd| {
        try std.testing.expectEqualStrings(name, cmd.name);
        try std.testing.expect(std.mem.startsWith(u8, cmd.usage, "abi "));
        try std.testing.expect(cmd.summary.len > 0);
    }
}

test "MCP tools/list includes contract tools" {
    const json = try handlers.handleToolsListJson(std.testing.allocator);
    defer std.testing.allocator.free(json);

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();

    const root = try expectObject(parsed.value);
    const tools = try expectArray(root.get("tools") orelse return error.MissingTools);
    const expected = [_]struct { name: []const u8, required: []const []const u8 }{
        .{ .name = "ai_run", .required = &.{"input"} },
        .{ .name = "ai_complete", .required = &.{"input"} },
        .{ .name = "ai_train", .required = &.{ "profile", "dataset" } },
        .{ .name = "wdbx_query", .required = &.{"query"} },
    };
    try std.testing.expectEqual(expected.len, tools.items.len);

    for (expected, tools.items) |contract, item| {
        const tool = try expectObject(item);
        try expectString(tool.get("name") orelse return error.MissingToolName, contract.name);

        const description = switch (tool.get("description") orelse return error.MissingDescription) {
            .string => |s| s,
            else => return error.ExpectedString,
        };
        try std.testing.expect(description.len > 0);

        const input_schema = try expectObject(tool.get("inputSchema") orelse return error.MissingInputSchema);
        try expectString(input_schema.get("type") orelse return error.MissingInputSchemaType, "object");
        const properties = try expectObject(input_schema.get("properties") orelse return error.MissingProperties);
        if (std.mem.eql(u8, contract.name, "ai_run")) {
            try std.testing.expect(properties.get("profile") == null);
        } else if (std.mem.eql(u8, contract.name, "ai_train")) {
            _ = properties.get("format") orelse return error.MissingDatasetFormatProperty;
            _ = properties.get("artifact_dir") orelse return error.MissingArtifactDirProperty;
        }
        try expectRequiredFields(input_schema.get("required") orelse return error.MissingRequiredFields, contract.required);
    }
}

test "MCP initialize advertises abi-mcp" {
    const json = try handlers.handleInitializeJson(std.testing.allocator, null);
    defer std.testing.allocator.free(json);

    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();

    const root = try expectObject(parsed.value);
    try expectString(root.get("protocolVersion") orelse return error.MissingProtocolVersion, "2024-11-05");
    _ = try expectObject(root.get("capabilities") orelse return error.MissingCapabilities);
    const server_info = try expectObject(root.get("serverInfo") orelse return error.MissingServerInfo);
    try expectString(server_info.get("name") orelse return error.MissingServerName, "abi-mcp");
    const version = switch (server_info.get("version") orelse return error.MissingServerVersion) {
        .string => |s| s,
        else => return error.ExpectedString,
    };
    try std.testing.expect(version.len > 0);
}

test "WDBX store contract preserves search ordering and block metadata" {
    const wdbx = abi.features.wdbx;
    var store = wdbx.Store.init(std.testing.allocator);
    defer store.deinit();

    const query_id = store.putVector(&.{ 1, 0, 0, 0 }) catch |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
        const manifest = try store.exportManifest(std.testing.allocator);
        defer std.testing.allocator.free(manifest);
        try std.testing.expect(std.mem.indexOf(u8, manifest, "\"disabled\":true") != null);
        return;
    };
    const near_id = try store.putVector(&.{ 0.95, 0.05, 0, 0 });
    _ = try store.putVector(&.{ 0, 1, 0, 0 });

    const results = try store.search(&.{ 1, 0, 0, 0 }, 3);
    defer std.testing.allocator.free(results);
    try std.testing.expect(results.len > 0);
    try std.testing.expect(results.len <= 3);
    try std.testing.expectEqual(query_id, results[0].id);
    var idx: usize = 1;
    while (idx < results.len) : (idx += 1) {
        try std.testing.expect(results[idx - 1].score >= results[idx].score);
    }

    const metadata = "model=contract;profile=abbey;kind=completion";
    const block_id = try store.appendBlock("abbey", query_id, near_id, metadata);
    const block = store.lastBlock() orelse return error.MissingWdbxBlock;
    try std.testing.expectEqualSlices(u8, &block_id, &block.id);
    try std.testing.expectEqualStrings("abbey", block.profile);
    try std.testing.expectEqual(query_id, block.query_id);
    try std.testing.expectEqual(near_id, block.response_id);
    try std.testing.expectEqualStrings(metadata, block.metadata);

    try store.store("snapshot:latest", metadata);
    const manifest = try store.exportManifest(std.testing.allocator);
    defer std.testing.allocator.free(manifest);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"kv_entries\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"vectors\":3") != null);
    try std.testing.expect(std.mem.indexOf(u8, manifest, "\"blocks\":1") != null);
}

test "WDBX block snapshots expose appended metadata when enabled" {
    const wdbx = abi.features.wdbx;
    var chain = wdbx.storage.BlockChain.init(std.testing.allocator);
    defer chain.deinit();

    const metadata = "source=contract;roundtrip=true";
    const block_id = chain.append("aviva", 7, 8, metadata) catch |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
        try std.testing.expectEqual(@as(usize, 0), chain.len());
        return;
    };

    const snapshot = chain.getSnapshot();
    defer chain.releaseSnapshot();

    try std.testing.expectEqual(@as(usize, 1), snapshot.len());
    const block = snapshot.getBlock(block_id) orelse return error.MissingSnapshotBlock;
    try std.testing.expectEqualStrings("aviva", block.data.profile);
    try std.testing.expectEqual(@as(u32, 7), block.data.query_id);
    try std.testing.expectEqual(@as(u32, 8), block.data.response_id);
    try std.testing.expectEqualStrings(metadata, block.data.metadata);
}
