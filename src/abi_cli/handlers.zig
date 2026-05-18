const std = @import("std");
const abi = @import("../root.zig");
const usage_mod = @import("usage.zig");

pub fn handleBackends() !u8 {
    const gpu_status = abi.features.gpu.detectBackend();
    const native_gpu = abi.features.gpu.nativeKernelStatus();
    const training = abi.features.accelerator.selectBackend(.training);
    const shader_status = abi.features.shaders.compilerStatus();
    const shader = try abi.features.shaders.compile(std.heap.page_allocator, .{
        .name = "status",
        .source = "fn main() void {}",
    });
    defer shader.deinit(std.heap.page_allocator);
    const mlir_status = abi.features.mlir.toolchainStatus();
    const lowered = try abi.features.mlir.lower(std.heap.page_allocator, .{
        .name = "status",
        .operations = &.{"matmul"},
    });
    defer lowered.deinit(std.heap.page_allocator);

    std.debug.print("GPU: {s} available={any} accelerated={any}\n", .{
        abi.features.gpu.backendName(gpu_status.backend),
        gpu_status.available,
        gpu_status.accelerated,
    });
    std.debug.print("Native GPU kernels: linked={any} ({s})\n", .{ native_gpu.linked, native_gpu.message });
    std.debug.print("Accelerator: {s} ({s})\n", .{ abi.features.accelerator.backendName(training.backend), training.message });
    std.debug.print("Shader: {s} backend={s} compiler_available={any} ({s})\n", .{ abi.features.shaders.languageName(shader.language), shader.backend, shader_status.available, shader_status.message });
    std.debug.print("MLIR: {s} backend={s} toolchain_available={any} ({s})\n", .{ abi.features.mlir.dialectName(lowered.dialect), lowered.target_backend, mlir_status.available, mlir_status.message });
    return 0;
}

pub fn handleTrain(allocator: std.mem.Allocator, input: []const u8) !u8 {
    const response = try abi.features.ai.run(allocator, input);
    defer allocator.free(response);
    std.debug.print("{s}\n", .{response});
    return 0;
}

pub fn handleTwilio(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 4 or !std.mem.eql(u8, args[2], "simulate")) return usage_mod.usageError("usage: abi twilio simulate <input>");

    const input = args[3];
    const agent_reply = try abi.features.ai.run(allocator, input);
    defer allocator.free(agent_reply);

    var client = abi.connectors.twilio.Client.init(allocator, abi.connectors.twilio.TwilioConfig.local());
    defer client.deinit();

    var response = try client.handleConversationRelayEvent(allocator, .{
        .kind = .user_transcript,
        .conversation_id = "local-conversation",
        .customer_id = "local-customer",
        .transcript = input,
    }, agent_reply);
    defer response.deinit(allocator);

    std.debug.print("Twilio ConversationRelay simulation\n", .{});
    std.debug.print("response: {s}\n", .{response.text});
    if (response.escalation) |payload| {
        std.debug.print("escalation: true\n", .{});
        std.debug.print("reason: {s}\n", .{payload.reason_code});
        std.debug.print("routing_hints: {s}\n", .{payload.routing_hints});
        std.debug.print("summary: {s}\n", .{payload.summary});
    } else {
        std.debug.print("escalation: false\n", .{});
    }
    return 0;
}

pub fn handleAgent(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi agent <plan|train|tui|os> ...");

    const sub_cmd = args[2];
    if (std.mem.eql(u8, sub_cmd, "plan")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi agent plan <input>");
        const result = try abi.features.ai.runAgent(allocator, .{ .name = "cli-agent", .instructions = "Plan only; do not execute.", .dry_run = true }, args[3]);
        defer result.deinit(allocator);
        std.debug.print("{s}\n", .{result.output});
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "train")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi agent train <abbey|aviva|abi|all>");
        var store = abi.features.wdbx.Store.init(allocator);
        defer store.deinit();

        const dataset = abi.features.ai.DatasetSpec{ .path = "datasets/local-training.jsonl" };
        const artifact_dir = "zig-cache/agent-artifacts";
        const result = if (std.mem.eql(u8, args[3], "all"))
            try abi.features.ai.trainKnownProfiles(allocator, &store, dataset, artifact_dir)
        else
            try abi.features.ai.trainWithStore(allocator, &store, .{
                .profile = args[3],
                .dataset = dataset,
                .artifact_dir = artifact_dir,
            });
        defer result.deinit(allocator);

        std.debug.print("{s}: {s} ({d} wdbx record(s), backend={s})\n", .{ result.profile, result.message, store.count(), result.acceleration_backend });
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "tui")) {
        if (args.len != 3) return usage_mod.usageError("usage: abi agent tui");
        return renderTui(allocator);
    } else if (std.mem.eql(u8, sub_cmd, "os") and args.len >= 5) {
        return handleAgentOs(io, allocator, args);
    } else {
        return usage_mod.usageError("usage: abi agent <plan|train|tui|os dry-run|os execute> ...");
    }
}

pub fn handleAgentOs(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    _ = allocator;
    const os_cmd = args[3];
    const start: usize = if (std.mem.eql(u8, os_cmd, "execute") and args.len >= 6 and std.mem.eql(u8, args[4], "--confirm")) 5 else 4;
    if (start >= args.len) return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");

    const request = abi.features.os_control.CommandRequest{
        .intent = .read_only,
        .argv = args[start..],
    };

    if (std.mem.eql(u8, os_cmd, "dry-run")) {
        const rendered = try abi.features.os_control.renderDryRun(std.heap.page_allocator, request);
        defer std.heap.page_allocator.free(rendered);
        std.debug.print("{s}\n", .{rendered});
        return 0;
    } else if (std.mem.eql(u8, os_cmd, "execute")) {
        if (start != 5) return usage_mod.usageError("usage: abi agent os execute --confirm <cmd> [args...]");
        const workspace_root = if (std.c.getenv("PWD")) |pwd| std.mem.span(pwd) else "/";
        const policy = abi.features.os_control.Policy{
            .workspace_root = workspace_root,
            .dry_run_only = false,
            .allow_execution = true,
            .allowed_commands = &.{ "true", "pwd", "ls", "whoami", "date" },
            .require_confirmation = true,
        };
        const execute_request = abi.features.os_control.CommandRequest{
            .intent = .read_only,
            .argv = args[start..],
            .confirm_execution = true,
        };
        const result = try abi.features.os_control.executeConfirmed(std.heap.page_allocator, io, execute_request, policy);
        std.debug.print("{s}\n", .{result.message});
        return result.exit_code orelse 0;
    } else {
        return usage_mod.usageError("usage: abi agent os <dry-run|execute --confirm> <cmd> [args...]");
    }
}

pub fn handleAuth(io_mod: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len < 3) return usage_mod.usageError("usage: abi auth <signin|logout|status> [args...]");

    const sub_cmd = args[2];
    if (std.mem.eql(u8, sub_cmd, "status")) {
        var creds = try abi.foundation.credentials.loadCredentials(allocator);
        defer creds.deinit(allocator);

        std.debug.print("Authentication Status:\n", .{});
        std.debug.print("  OpenAI:    {s}\n", .{if (creds.openai_api_key != null) "configured" else "not configured"});
        std.debug.print("  Anthropic: {s}\n", .{if (creds.anthropic_api_key != null) "configured" else "not configured"});
        std.debug.print("  Discord:   {s}\n", .{if (creds.discord_token != null) "configured" else "not configured"});
        std.debug.print("  Grok:      {s}\n", .{if (creds.grok_api_key != null) "configured" else "not configured"});
        std.debug.print("  Twilio:    {s}\n", .{if (creds.twilio_account_sid != null and creds.twilio_auth_token != null) "configured" else "not configured"});
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "logout")) {
        const path = try abi.foundation.credentials.getCredentialsPath(allocator);
        defer allocator.free(path);
        if (abi.foundation.io.fileExists(path)) {
            var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{});
            defer threaded.deinit();
            try std.Io.Dir.deleteFileAbsolute(threaded.io(), path);
            std.debug.print("Logged out. Credentials cleared.\n", .{});
        } else {
            std.debug.print("No credentials found.\n", .{});
        }
        return 0;
    } else if (std.mem.eql(u8, sub_cmd, "signin")) {
        if (args.len != 4) return usage_mod.usageError("usage: abi auth signin <openai|anthropic|discord|grok|twilio>");
        const service = args[3];

        var creds = try abi.foundation.credentials.loadCredentials(allocator);
        defer creds.deinit(allocator);

        var buf: [1024]u8 = undefined;
        var stdin_reader = std.Io.File.stdin().reader(io_mod, &buf);

        if (std.mem.eql(u8, service, "openai")) {
            const key = try readSecretLine(&stdin_reader, "Enter API key/token for openai: ");
            try abi.foundation.credentials.replaceOwnedString(allocator, &creds.openai_api_key, key);
        } else if (std.mem.eql(u8, service, "anthropic")) {
            const key = try readSecretLine(&stdin_reader, "Enter API key/token for anthropic: ");
            try abi.foundation.credentials.replaceOwnedString(allocator, &creds.anthropic_api_key, key);
        } else if (std.mem.eql(u8, service, "discord")) {
            const key = try readSecretLine(&stdin_reader, "Enter API key/token for discord: ");
            try abi.foundation.credentials.replaceOwnedString(allocator, &creds.discord_token, key);
        } else if (std.mem.eql(u8, service, "grok")) {
            const key = try readSecretLine(&stdin_reader, "Enter API key/token for grok: ");
            try abi.foundation.credentials.replaceOwnedString(allocator, &creds.grok_api_key, key);
        } else if (std.mem.eql(u8, service, "twilio")) {
            const sid = try readSecretLine(&stdin_reader, "Enter Twilio Account SID: ");
            const token = try readSecretLine(&stdin_reader, "Enter Twilio Auth Token: ");
            try abi.foundation.credentials.replaceOwnedString(allocator, &creds.twilio_account_sid, sid);
            try abi.foundation.credentials.replaceOwnedString(allocator, &creds.twilio_auth_token, token);
        } else {
            return usage_mod.usageError("unknown service; use openai, anthropic, discord, grok, or twilio");
        }

        try abi.foundation.credentials.saveCredentials(allocator, creds);
        std.debug.print("Credentials saved for {s}.\n", .{service});
        return 0;
    } else {
        return usage_mod.usageError("usage: abi auth <signin|logout|status>");
    }
}

fn readSecretLine(stdin_reader: anytype, prompt: []const u8) ![]const u8 {
    std.debug.print("{s}", .{prompt});
    const line = (try stdin_reader.interface.takeDelimiter('\n')) orelse return error.EndOfStream;
    return abi.foundation.utils.trimWhitespace(line);
}

pub fn handlePlugin(allocator: std.mem.Allocator, args: []const []const u8) !u8 {
    if (args.len != 3 or !std.mem.eql(u8, args[2], "list")) return usage_mod.usageError("usage: abi plugin list");

    var registry = abi.registry.Registry.init(allocator);
    defer registry.deinit();
    try registry.loadPlugins();

    std.debug.print("Installed Plugins:\n", .{});
    var it = registry.modules.iterator();
    while (it.next()) |entry| {
        std.debug.print("  - {s}: {s}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    }
    return 0;
}

pub fn renderTui(allocator: std.mem.Allocator) !u8 {
    _ = allocator;
    try abi.features.tui.initScreen();
    defer abi.features.tui.deinitScreen();
    try abi.features.tui.render(.{ .width = 80, .height = 24 });
    return 0;
}
