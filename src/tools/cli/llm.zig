const std = @import("std");
const connectors = @import("connectors");
const plugins = @import("plugins");
const wdbx = @import("wdbx");
const common = @import("common.zig");
const ml = @import("ml_support.zig");

pub const command = common.Command{
    .name = "llm",
    .summary = "Work with language model embeddings and training",
    .usage = "abi llm <embed|query|train> [flags]",
    .details = "  embed   Generate embeddings for text\n" ++
        "  query   Search embeddings database\n" ++
        "  train   Train local ML models\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    if (args.len < 3) {
        std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
        return;
    }

    const sub = args[2];
    var api_key_owned: ?[]u8 = null;
    defer if (api_key_owned) |buf| {
        @memset(buf, 0);
        allocator.free(buf);
    };

    if (std.mem.eql(u8, sub, "embed")) {
        var db_path: ?[]const u8 = null;
        var provider: []const u8 = "ollama";
        var host: []const u8 = "http://localhost:11434";
        var model: []const u8 = "nomic-embed-text";
        var api_key: []const u8 = "";
        var text: ?[]const u8 = null;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--db") and i + 1 < args.len) {
                i += 1;
                db_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--provider") and i + 1 < args.len) {
                i += 1;
                provider = args[i];
            } else if (std.mem.eql(u8, args[i], "--host") and i + 1 < args.len) {
                i += 1;
                host = args[i];
            } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
                i += 1;
                model = args[i];
            } else if (std.mem.eql(u8, args[i], "--api-key") and i + 1 < args.len) {
                i += 1;
                api_key = args[i];
            } else if (std.mem.eql(u8, args[i], "--text") and i + 1 < args.len) {
                i += 1;
                text = args[i];
            }
        }

        if (db_path == null or text == null) {
            std.debug.print("llm embed requires --db and --text\n", .{});
            return;
        }

        if (std.mem.eql(u8, provider, "openai") and api_key.len == 0) {
            if (std.process.getEnvVarOwned(allocator, "OPENAI_API_KEY")) |buf| {
                api_key_owned = buf;
                api_key = buf;
            } else |_| {}
        }

        const cfg: connectors.ProviderConfig = if (std.mem.eql(u8, provider, "openai"))
            .{ .openai = .{ .base_url = "https://api.openai.com/v1", .api_key = api_key, .model = model } }
        else
            .{ .ollama = .{ .host = host, .model = model } };

        var registry = try plugins.createRegistry(allocator);
        defer registry.deinit();
        try registry.registerBuiltinInterface(connectors.plugin.getInterface());
        const iface = connectors.plugin.getInterface();
        var plugin = try plugins.interface.createPlugin(allocator, iface);
        defer plugins.interface.destroyPlugin(allocator, plugin);
        var plugin_cfg = plugins.types.PluginConfig.init(allocator);
        defer plugin_cfg.deinit();
        try plugin_cfg.setParameter("provider", provider);
        if (std.mem.eql(u8, provider, "openai")) {
            try plugin_cfg.setParameter("base_url", "https://api.openai.com/v1");
            try plugin_cfg.setParameter("api_key", api_key);
            try plugin_cfg.setParameter("model", model);
        } else {
            try plugin_cfg.setParameter("host", host);
            try plugin_cfg.setParameter("model", model);
        }
        try plugin.initialize(&plugin_cfg);
        try plugin.start();
        if (plugin.getApi("embedding")) |p| {
            const emb_api = @as(*const connectors.plugin.EmbeddingApi, @ptrCast(@alignCast(p)));
            var out_ptr: [*]f32 = undefined;
            var out_len: usize = 0;
            const rc = emb_api.embed_text(plugin.context.?, text.?.ptr, text.?.len, &out_ptr, &out_len);
            if (rc == 0) {
                const emb = out_ptr[0..out_len];
                defer emb_api.free_vector(plugin.context.?, out_ptr, out_len);
                var db = try wdbx.Db.open(db_path.?, true);
                defer db.close();
                if (db.getDimension() == 0) try db.init(@intCast(emb.len));
                const id = try db.addEmbedding(emb);
                std.debug.print("Embedded text added, id={d}, dim={d}\n", .{ id, emb.len });
                return;
            }
        }

        const emb = try connectors.embedText(allocator, cfg, text.?);
        defer allocator.free(emb);
        var db = try wdbx.Db.open(db_path.?, true);
        defer db.close();
        if (db.getDimension() == 0) try db.init(@intCast(emb.len));
        const id = try db.addEmbedding(emb);
        std.debug.print("Embedded text added, id={d}, dim={d}\n", .{ id, emb.len });
        return;
    }

    if (std.mem.eql(u8, sub, "query")) {
        var db_path: ?[]const u8 = null;
        var text: ?[]const u8 = null;
        var k: usize = 5;
        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--db") and i + 1 < args.len) {
                i += 1;
                db_path = args[i];
            } else if (std.mem.eql(u8, args[i], "--text") and i + 1 < args.len) {
                i += 1;
                text = args[i];
            } else if (std.mem.eql(u8, args[i], "--k") and i + 1 < args.len) {
                i += 1;
                k = try std.fmt.parseInt(usize, args[i], 10);
            }
        }
        if (db_path == null or text == null) {
            std.debug.print("llm query requires --db and --text\n", .{});
            return;
        }
        var db = try wdbx.Db.open(db_path.?, false);
        defer db.close();

        const q = try simpleHashEmbedding(allocator, text.?, db.getDimension());
        defer allocator.free(q);
        const results = try db.search(q, k, allocator);
        defer allocator.free(results);
        std.debug.print("Found {d} results for query\n", .{results.len});
        for (results, 0..) |r, idx| {
            std.debug.print("  {d}: id={d} score={d:.6}\n", .{ idx, r.index, r.score });
        }
        return;
    }

    if (std.mem.eql(u8, sub, "train")) {
        var data_path: ?[]const u8 = null;
        var output_path: ?[]const u8 = null;
        var model_type: ?[]const u8 = null;
        var epochs: ?usize = null;
        var learning_rate: ?f32 = null;
        var batch_size: ?usize = null;
        var use_gpu = false;
        var threads: ?usize = null;

        var i: usize = 3;
        while (i < args.len) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--data") and i + 1 < args.len) {
                data_path = args[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--output") and i + 1 < args.len) {
                output_path = args[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
                model_type = args[i + 1];
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--epochs") and i + 1 < args.len) {
                epochs = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--lr") and i + 1 < args.len) {
                learning_rate = try std.fmt.parseFloat(f32, args[i + 1]);
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--batch-size") and i + 1 < args.len) {
                batch_size = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            } else if (std.mem.eql(u8, args[i], "--gpu")) {
                use_gpu = true;
            } else if (std.mem.eql(u8, args[i], "--threads") and i + 1 < args.len) {
                threads = try std.fmt.parseInt(usize, args[i + 1], 10);
                i += 1;
            }
        }

        if (data_path == null) {
            std.debug.print("Usage: abi llm train --data <path> [--output <path>] [--model <type>] [--epochs N] [--lr RATE] [--batch-size N] [--gpu] [--threads N]\n", .{});
            return;
        }

        const final_output = output_path orelse "model.bin";
        const final_model_type = model_type orelse "neural";
        const final_epochs = epochs orelse 100;
        const final_lr = learning_rate orelse 0.001;
        const final_batch_size = batch_size orelse 32;
        const final_threads = threads orelse 1;

        std.debug.print(
            "Training {s} model on {s}...\n",
            .{ final_model_type, data_path.? },
        );
        std.debug.print(
            "Epochs: {}, Learning Rate: {}, Batch Size: {}, GPU: {}, Threads: {}\n",
            .{ final_epochs, final_lr, final_batch_size, use_gpu, final_threads },
        );

        var training_data = try ml.loadTrainingData(allocator, data_path.?);
        defer training_data.deinit();

        if (std.mem.eql(u8, final_model_type, "neural")) {
            try ml.trainNeuralNetwork(allocator, training_data, final_output, final_epochs, final_lr, final_batch_size, use_gpu);
        } else if (std.mem.eql(u8, final_model_type, "linear")) {
            try ml.trainLinearModel(allocator, training_data, final_output, final_epochs, final_lr);
        } else {
            std.debug.print("Unknown model type: {s}\n", .{final_model_type});
            return;
        }

        std.debug.print("Training completed. Model saved to: {s}\n", .{final_output});
        return;
    }

    std.debug.print("Unknown llm subcommand: {s}\n", .{sub});
}

fn simpleHashEmbedding(allocator: std.mem.Allocator, text: []const u8, dim_u16: u16) ![]f32 {
    var dim: usize = @intCast(dim_u16);
    if (dim == 0) dim = 16;
    const v = try allocator.alloc(f32, dim);
    @memset(v, 0);
    const h = std.hash_map.hashString(text);
    for (v, 0..) |*out, i| {
        out.* = @floatFromInt(((h >> @intCast(i % 24)) & 0xFF));
    }
    return v;
}
