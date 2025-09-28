const std = @import("std");
const abi = @import("abi");
const common = @import("common.zig");
const persona_manifest = abi.ai.persona_manifest;

pub const command = common.Command{
    .name = "chat",
    .summary = "Interact with the ABI conversational agent",
    .usage = "abi chat [--persona <type>] [--backend <provider>] [--model <name>] [--persona-manifest <path>] [--profile <name>] [--interactive] [message]",
    .details = "  --persona          Select persona (creative, analytical, helpful)\n" ++
        "  --persona-manifest Load persona manifest (JSON/TOML)\n" ++
        "  --profile          Select environment profile from manifest\n" ++
        "  --backend          Choose backend provider (openai, ollama)\n" ++
        "  --model            Model identifier\n" ++
        "  --interactive      Start interactive chat session\n",
    .run = run,
};

pub fn run(ctx: *common.Context, args: [][:0]u8) !void {
    const allocator = ctx.allocator;
    const CliError = error{ ManifestRequired, PersonaNotFound, EnvironmentNotFound, EmptyManifest };

    var persona: ?[]const u8 = null;
    var backend: ?[]const u8 = null;
    var model: ?[]const u8 = null;
    var interactive: bool = false;
    var message: ?[]const u8 = null;
    var manifest_path: ?[]const u8 = null;
    var profile_name: ?[]const u8 = null;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--persona") and i + 1 < args.len) {
            persona = std.mem.span(args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--backend") and i + 1 < args.len) {
            backend = std.mem.span(args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--model") and i + 1 < args.len) {
            model = std.mem.span(args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--persona-manifest") and i + 1 < args.len) {
            manifest_path = std.mem.span(args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--profile") and i + 1 < args.len) {
            profile_name = std.mem.span(args[i + 1]);
            i += 1;
        } else if (std.mem.eql(u8, args[i], "--interactive")) {
            interactive = true;
        } else if (i == 2 and !std.mem.startsWith(u8, args[i], "--")) {
            message = std.mem.span(args[i]);
        }
    }

    if (manifest_path == null and profile_name != null) {
        std.debug.print("--profile requires --persona-manifest\n", .{});
        return CliError.ManifestRequired;
    }

    var manifest: ?persona_manifest.PersonaManifest = null;
    defer if (manifest) |*m| m.deinit();

    if (manifest_path) |path| {
        var loaded = try persona_manifest.loadFromFile(allocator, path);
        try loaded.validate();
        manifest = loaded;
    }

    var selected_persona_name: []const u8 = persona orelse "";
    var selected_persona_profile: ?*const persona_manifest.PersonaProfile = null;
    var selected_environment_profile: ?*const persona_manifest.EnvironmentProfile = null;

    if (manifest) |*m| {
        if (m.personas.len == 0) {
            std.debug.print("persona manifest '{s}' contains no personas\n", .{m.source_path});
            return CliError.EmptyManifest;
        }

        if (profile_name) |name| {
            selected_environment_profile = m.findEnvironment(name) orelse {
                std.debug.print("environment profile '{s}' not found in manifest\n", .{name});
                return CliError.EnvironmentNotFound;
            };
        } else {
            selected_environment_profile = m.defaultEnvironment();
        }

        if (selected_persona_name.len == 0) {
            if (selected_environment_profile) |profile| {
                if (profile.default_persona) |name| {
                    selected_persona_name = name;
                }
            }
        }

        if (selected_persona_name.len == 0) {
            selected_persona_name = m.defaultPersona().?.name;
        }

        selected_persona_profile = m.findPersona(selected_persona_name) orelse {
            std.debug.print("persona '{s}' not found in manifest\n", .{selected_persona_name});
            return CliError.PersonaNotFound;
        };
    }

    if (selected_persona_name.len == 0) {
        selected_persona_name = "creative";
    }

    const final_persona = selected_persona_name;
    const final_backend = backend orelse "openai";
    const final_model = model orelse "gpt-3.5-turbo";

    var agent_config = abi.ai.enhanced_agent.AgentConfig{
        .name = "ABI Assistant",
        .enable_logging = true,
        .max_concurrent_requests = 5,
    };

    if (selected_persona_profile) |profile| {
        agent_config.temperature = profile.temperature;
        agent_config.top_p = profile.top_p;
    }

    var agent = try abi.ai.enhanced_agent.EnhancedAgent.init(allocator, agent_config);
    defer agent.deinit();

    if (message) |msg| {
        const response = try agent.processInput(msg);
        defer allocator.free(response);
        std.debug.print("{s}\n", .{response});
        return;
    }

    if (interactive) {
        std.debug.print("Chat mode (type 'quit' to exit, 'help' for commands)\n", .{});
        std.debug.print("Persona: {s}, Backend: {any}, Model: {any}\n", .{ final_persona, final_backend, final_model });
        if (selected_persona_profile) |profile| {
            std.debug.print("  • Temperature: {d:.2}, Top-p: {d:.2}\n", .{ profile.temperature, profile.top_p });
            std.debug.print("  • Tools: ", .{});
            if (profile.tools.len == 0) {
                std.debug.print("(none)\n", .{});
            } else {
                for (profile.tools, 0..) |tool, idx| {
                    if (idx > 0) std.debug.print(", ", .{});
                    std.debug.print("{s}", .{tool});
                }
                std.debug.print("\n", .{});
            }
            std.debug.print("  • Rate limit: {d} rpm\n", .{ profile.rate_limits.requests_per_minute });
        }
        if (selected_environment_profile) |profile| {
            std.debug.print("Environment profile: {s}\n", .{profile.name});
            std.debug.print("  • Streaming: {s}\n", .{if (profile.streaming) "enabled" else "disabled"});
            std.debug.print("  • Function calling: {s}\n", .{if (profile.function_calling) "enabled" else "disabled"});
            std.debug.print("  • Log sink: {s}\n", .{profile.log_sink});
        }
        std.debug.print("Interactive Chat Mode - Type 'quit' to exit, 'help' for commands\n", .{});
        std.debug.print("Note: Full interactive mode requires additional I/O implementation\n", .{});
        std.debug.print("For now, this is a demonstration of the chat framework.\n", .{});
        return;
    }

    std.debug.print("Usage: {s}\n{s}", .{ command.usage, command.details orelse "" });
}
