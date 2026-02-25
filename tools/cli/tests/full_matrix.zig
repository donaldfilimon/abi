const std = @import("std");
const cli_catalog = @import("catalog.zig");

const CommandSubcommands = struct {
    command: []const u8,
    subcommands: []const []const u8,
};

const command_names = blk: {
    var names: [cli_catalog.commands.len + 2][]const u8 = undefined;
    var idx: usize = 0;
    for (cli_catalog.commands) |command| {
        names[idx] = command.name;
        idx += 1;
    }
    names[idx] = "version";
    idx += 1;
    names[idx] = "help";
    break :blk names;
};

const command_subcommands = blk: {
    var count: usize = 0;
    for (cli_catalog.commands) |command| {
        if (command.subcommands.len > 0) count += 1;
    }

    var items: [count]CommandSubcommands = undefined;
    var idx: usize = 0;
    for (cli_catalog.commands) |command| {
        if (command.subcommands.len == 0) continue;
        items[idx] = .{
            .command = command.name,
            .subcommands = command.subcommands,
        };
        idx += 1;
    }
    break :blk items;
};

pub const EntryKind = enum {
    oneshot,
    serve_probe,
    pty_session,
    long_running_probe,
};

pub const CwdMode = enum {
    temp_workspace,
    repo_copy,
};

pub const EnvProfile = enum {
    isolated,
};

pub const ExitPolicy = enum {
    zero_only,
    allow_signal_after_probe,
};

pub const Entry = struct {
    id: []const u8,
    args: []const []const u8,
    kind: EntryKind,
    timeout_ms: u32,
    requires: []const []const u8,
    cwd_mode: CwdMode,
    env_profile: EnvProfile,
    exit_policy: ExitPolicy,
};

const Matrix = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayListUnmanaged(Entry) = .empty,

    fn init(allocator: std.mem.Allocator) Matrix {
        return .{ .allocator = allocator };
    }

    fn deinit(self: *Matrix) void {
        for (self.entries.items) |entry| {
            self.allocator.free(entry.id);
            self.allocator.free(entry.args);
            self.allocator.free(entry.requires);
        }
        self.entries.deinit(self.allocator);
    }

    fn add(
        self: *Matrix,
        id: []const u8,
        args: []const []const u8,
        kind: EntryKind,
        timeout_ms: u32,
        requires: []const []const u8,
        cwd_mode: CwdMode,
        env_profile: EnvProfile,
        exit_policy: ExitPolicy,
    ) !void {
        const id_copy = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(id_copy);

        const args_copy = try self.allocator.alloc([]const u8, args.len);
        errdefer self.allocator.free(args_copy);
        @memcpy(args_copy, args);

        const req_copy = try self.allocator.alloc([]const u8, requires.len);
        errdefer self.allocator.free(req_copy);
        @memcpy(req_copy, requires);

        try self.entries.append(self.allocator, .{
            .id = id_copy,
            .args = args_copy,
            .kind = kind,
            .timeout_ms = timeout_ms,
            .requires = req_copy,
            .cwd_mode = cwd_mode,
            .env_profile = env_profile,
            .exit_policy = exit_policy,
        });
    }
};

const EntryBuild = struct {
    kind: EntryKind = .oneshot,
    timeout_ms: u32 = 20_000,
    requires: std.ArrayListUnmanaged([]const u8) = .empty,
    cwd_mode: CwdMode = .temp_workspace,
    env_profile: EnvProfile = .isolated,
    exit_policy: ExitPolicy = .zero_only,

    fn deinit(self: *EntryBuild, allocator: std.mem.Allocator) void {
        self.requires.deinit(allocator);
    }
};

fn isAny(value: []const u8, options: []const []const u8) bool {
    for (options) |option| {
        if (std.mem.eql(u8, value, option)) return true;
    }
    return false;
}

fn addTopLevelEntries(matrix: *Matrix) !void {
    for (command_names) |command| {
        const id = try std.fmt.allocPrint(matrix.allocator, "top.{s}", .{command});
        defer matrix.allocator.free(id);

        var kind: EntryKind = .oneshot;
        var timeout_ms: u32 = 20_000;
        var cwd_mode: CwdMode = .temp_workspace;
        var exit_policy: ExitPolicy = .zero_only;

        if (isAny(command, &.{ "ui", "agent" })) {
            kind = .pty_session;
            timeout_ms = 45_000;
            exit_policy = .allow_signal_after_probe;
        }

        if (std.mem.eql(u8, command, "gendocs")) {
            cwd_mode = .repo_copy;
            timeout_ms = 60_000;
        }

        if (std.mem.eql(u8, command, "embed")) {
            const args = [_][]const u8{ command, "--help" };
            try matrix.add(id, &args, kind, timeout_ms, &.{}, cwd_mode, .isolated, exit_policy);
        } else {
            const args = [_][]const u8{command};
            try matrix.add(id, &args, kind, timeout_ms, &.{}, cwd_mode, .isolated, exit_policy);
        }
    }
}

fn addArg(list: *std.ArrayListUnmanaged([]const u8), allocator: std.mem.Allocator, arg: []const u8) !void {
    try list.append(allocator, arg);
}

fn addReq(build: *EntryBuild, allocator: std.mem.Allocator, req: []const u8) !void {
    try build.requires.append(allocator, req);
}

fn addFirstLevelEntry(matrix: *Matrix, command: []const u8, subcommand: []const u8) !void {
    var args = std.ArrayListUnmanaged([]const u8).empty;
    defer args.deinit(matrix.allocator);

    var build = EntryBuild{};
    defer build.deinit(matrix.allocator);

    var use_help_fallback = true;

    try addArg(&args, matrix.allocator, command);
    try addArg(&args, matrix.allocator, subcommand);

    if (std.mem.eql(u8, command, "db")) {
        if (isAny(subcommand, &.{ "stats", "optimize" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "serve")) {
            use_help_fallback = false;
            build.kind = .serve_probe;
            build.timeout_ms = 45_000;
            build.exit_policy = .allow_signal_after_probe;
            try addArg(&args, matrix.allocator, "--addr");
            try addArg(&args, matrix.allocator, "127.0.0.1:18081");
        }
    } else if (std.mem.eql(u8, command, "gpu")) {
        use_help_fallback = false;
    } else if (std.mem.eql(u8, command, "network")) {
        if (isAny(subcommand, &.{ "status", "list", "nodes" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "register")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "node-full");
            try addArg(&args, matrix.allocator, "127.0.0.1:9001");
        } else if (std.mem.eql(u8, subcommand, "unregister")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "node-full");
        } else if (std.mem.eql(u8, subcommand, "touch")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "node-full");
        } else if (std.mem.eql(u8, subcommand, "set-status")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "node-full");
            try addArg(&args, matrix.allocator, "healthy");
        }
    } else if (std.mem.eql(u8, command, "config")) {
        if (isAny(subcommand, &.{ "show", "env", "path" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "init")) {
            use_help_fallback = false;
            build.cwd_mode = .repo_copy;
            try addArg(&args, matrix.allocator, "-o");
            try addArg(&args, matrix.allocator, "abi-cli-full-config.json");
        }
    } else if (std.mem.eql(u8, command, "discord")) {
        try addReq(&build, matrix.allocator, "env:DISCORD_BOT_TOKEN");
        try addReq(&build, matrix.allocator, "net:discord");

        if (isAny(subcommand, &.{ "status", "info", "guilds" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "send")) {
            use_help_fallback = false;
            try addReq(&build, matrix.allocator, "env:ABI_TEST_DISCORD_CHANNEL_ID");
            try addArg(&args, matrix.allocator, "${ABI_TEST_DISCORD_CHANNEL_ID}");
            try addArg(&args, matrix.allocator, "abi-cli-full send test");
        } else if (std.mem.eql(u8, subcommand, "channel")) {
            use_help_fallback = false;
            try addReq(&build, matrix.allocator, "env:ABI_TEST_DISCORD_CHANNEL_ID");
            try addArg(&args, matrix.allocator, "${ABI_TEST_DISCORD_CHANNEL_ID}");
        } else if (std.mem.eql(u8, subcommand, "commands")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "list");
        } else if (std.mem.eql(u8, subcommand, "webhook")) {
            use_help_fallback = false;
            try addReq(&build, matrix.allocator, "env:ABI_TEST_DISCORD_WEBHOOK_URL");
            try addArg(&args, matrix.allocator, "${ABI_TEST_DISCORD_WEBHOOK_URL}");
            try addArg(&args, matrix.allocator, "abi-cli-full webhook test");
        }
    } else if (std.mem.eql(u8, command, "llm")) {
        if (std.mem.eql(u8, subcommand, "providers") or std.mem.eql(u8, subcommand, "plugins")) {
            use_help_fallback = false;
            if (std.mem.eql(u8, subcommand, "plugins")) {
                try addArg(&args, matrix.allocator, "list");
            }
        } else if (std.mem.eql(u8, subcommand, "run")) {
            use_help_fallback = false;
            try addReq(&build, matrix.allocator, "env:ABI_TEST_ENABLE_LLM_RUN");
            try addReq(&build, matrix.allocator, "env:OLLAMA_HOST");
            try addReq(&build, matrix.allocator, "env:ABI_TEST_OLLAMA_MODEL");
            try addReq(&build, matrix.allocator, "net:ollama");
            try addArg(&args, matrix.allocator, "--model");
            try addArg(&args, matrix.allocator, "${ABI_TEST_OLLAMA_MODEL}");
            try addArg(&args, matrix.allocator, "--prompt");
            try addArg(&args, matrix.allocator, "full matrix run");
            try addArg(&args, matrix.allocator, "--backend");
            try addArg(&args, matrix.allocator, "ollama");
            try addArg(&args, matrix.allocator, "--strict-backend");
            try addArg(&args, matrix.allocator, "--max-tokens");
            try addArg(&args, matrix.allocator, "8");
        } else if (std.mem.eql(u8, subcommand, "session")) {
            use_help_fallback = false;
            build.kind = .pty_session;
            build.timeout_ms = 60_000;
            build.exit_policy = .allow_signal_after_probe;
            try addReq(&build, matrix.allocator, "env:OLLAMA_HOST");
            try addReq(&build, matrix.allocator, "env:ABI_TEST_OLLAMA_MODEL");
            try addReq(&build, matrix.allocator, "net:ollama");
            try addArg(&args, matrix.allocator, "--model");
            try addArg(&args, matrix.allocator, "${ABI_TEST_OLLAMA_MODEL}");
            try addArg(&args, matrix.allocator, "--backend");
            try addArg(&args, matrix.allocator, "ollama");
            try addArg(&args, matrix.allocator, "--strict-backend");
        } else if (std.mem.eql(u8, subcommand, "serve")) {
            use_help_fallback = false;
            build.kind = .serve_probe;
            build.timeout_ms = 45_000;
            build.exit_policy = .allow_signal_after_probe;
            try addReq(&build, matrix.allocator, "env:ABI_TEST_GGUF_MODEL_PATH");
            try addArg(&args, matrix.allocator, "-m");
            try addArg(&args, matrix.allocator, "${ABI_TEST_GGUF_MODEL_PATH}");
            try addArg(&args, matrix.allocator, "-a");
            try addArg(&args, matrix.allocator, "127.0.0.1:18080");
        }
    } else if (std.mem.eql(u8, command, "model")) {
        if (isAny(subcommand, &.{ "list", "path", "help" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "info")) {
            use_help_fallback = false;
            try addReq(&build, matrix.allocator, "env:ABI_TEST_GGUF_MODEL_PATH");
            try addArg(&args, matrix.allocator, "${ABI_TEST_GGUF_MODEL_PATH}");
        } else if (std.mem.eql(u8, subcommand, "download")) {
            use_help_fallback = false;
            try addReq(&build, matrix.allocator, "env:ABI_TEST_ENABLE_MODEL_DOWNLOAD");
            try addReq(&build, matrix.allocator, "env:ABI_TEST_MODEL_SPEC");
            try addReq(&build, matrix.allocator, "net:model-host");
            try addArg(&args, matrix.allocator, "${ABI_TEST_MODEL_SPEC}");
        } else if (std.mem.eql(u8, subcommand, "search")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "llama");
        }
    } else if (std.mem.eql(u8, command, "train")) {
        if (std.mem.eql(u8, subcommand, "info") or std.mem.eql(u8, subcommand, "auto")) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "run")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "--epochs");
            try addArg(&args, matrix.allocator, "1");
            try addArg(&args, matrix.allocator, "--batch-size");
            try addArg(&args, matrix.allocator, "4");
            try addArg(&args, matrix.allocator, "--sample-count");
            try addArg(&args, matrix.allocator, "16");
        } else if (std.mem.eql(u8, subcommand, "self")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "--skip-improve");
        } else if (std.mem.eql(u8, subcommand, "generate-data")) {
            use_help_fallback = false;
            build.cwd_mode = .repo_copy;
            try addArg(&args, matrix.allocator, "--output");
            try addArg(&args, matrix.allocator, "abi-cli-full-train.bin");
            try addArg(&args, matrix.allocator, "--num-samples");
            try addArg(&args, matrix.allocator, "8");
            try addArg(&args, matrix.allocator, "--seq-length");
            try addArg(&args, matrix.allocator, "8");
        } else if (std.mem.eql(u8, subcommand, "monitor")) {
            use_help_fallback = false;
            build.kind = .pty_session;
            build.timeout_ms = 60_000;
            build.exit_policy = .allow_signal_after_probe;
        }
    } else if (std.mem.eql(u8, command, "task")) {
        if (isAny(subcommand, &.{ "list", "ls", "stats" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "add")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "full-matrix-task");
        }
    } else if (std.mem.eql(u8, command, "plugins")) {
        if (isAny(subcommand, &.{ "list", "search", "help" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "info")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "openai-connector");
        } else if (std.mem.eql(u8, subcommand, "enable") or std.mem.eql(u8, subcommand, "disable")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "openai-connector");
        }
    } else if (std.mem.eql(u8, command, "profile")) {
        if (isAny(subcommand, &.{ "show", "list", "help" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "create")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "full-matrix");
        } else if (std.mem.eql(u8, subcommand, "switch")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "default");
        } else if (std.mem.eql(u8, subcommand, "delete")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "full-matrix");
        } else if (std.mem.eql(u8, subcommand, "set")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "temperature");
            try addArg(&args, matrix.allocator, "0.8");
        } else if (std.mem.eql(u8, subcommand, "get")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "temperature");
        } else if (std.mem.eql(u8, subcommand, "api-key")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "list");
        } else if (std.mem.eql(u8, subcommand, "export")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "profile-export.json");
        } else if (std.mem.eql(u8, subcommand, "import")) {
            use_help_fallback = false;
            try addArg(&args, matrix.allocator, "profile-export.json");
        }
    } else if (std.mem.eql(u8, command, "toolchain")) {
        if (isAny(subcommand, &.{ "install", "zig", "zls", "update" })) {
            use_help_fallback = false;
            build.kind = .long_running_probe;
            build.timeout_ms = 180_000;
            build.cwd_mode = .repo_copy;
            build.exit_policy = .allow_signal_after_probe;
            try addReq(&build, matrix.allocator, "tool:git");
            try addReq(&build, matrix.allocator, "tool:cmake");
            try addReq(&build, matrix.allocator, "tool:llvm-config");
        } else {
            use_help_fallback = false;
        }
    } else if (std.mem.eql(u8, command, "mcp")) {
        if (std.mem.eql(u8, subcommand, "serve")) {
            use_help_fallback = false;
            build.kind = .serve_probe;
            build.timeout_ms = 45_000;
            build.exit_policy = .allow_signal_after_probe;
        } else {
            use_help_fallback = false;
        }
    } else if (std.mem.eql(u8, command, "acp")) {
        if (std.mem.eql(u8, subcommand, "serve")) {
            use_help_fallback = false;
            build.kind = .serve_probe;
            build.timeout_ms = 45_000;
            build.exit_policy = .allow_signal_after_probe;
            try addArg(&args, matrix.allocator, "--port");
            try addArg(&args, matrix.allocator, "18082");
        } else {
            use_help_fallback = false;
        }
    } else if (std.mem.eql(u8, command, "ui")) {
        if (isAny(subcommand, &.{ "launch", "gpu", "train" })) {
            use_help_fallback = false;
            build.kind = .pty_session;
            build.timeout_ms = 60_000;
            build.exit_policy = .allow_signal_after_probe;
        } else {
            use_help_fallback = false;
        }
    } else if (std.mem.eql(u8, command, "bench")) {
        use_help_fallback = false;
        if (std.mem.eql(u8, subcommand, "micro")) {
            try addArg(&args, matrix.allocator, "noop");
        } else if (std.mem.eql(u8, subcommand, "compare-training")) {
            // In-memory optimizer comparison (AdamW vs Adam vs SGD)
        }
    } else if (std.mem.eql(u8, command, "completions")) {
        use_help_fallback = false;
    } else if (std.mem.eql(u8, command, "multi-agent")) {
        if (isAny(subcommand, &.{ "info", "list", "status" })) {
            use_help_fallback = false;
        }
    } else if (std.mem.eql(u8, command, "ralph")) {
        if (isAny(subcommand, &.{ "status", "skills" })) {
            use_help_fallback = false;
        } else if (std.mem.eql(u8, subcommand, "init")) {
            use_help_fallback = false;
            build.cwd_mode = .repo_copy;
        }
    }

    if (use_help_fallback) {
        try addArg(&args, matrix.allocator, "--help");
    }

    const id = try std.fmt.allocPrint(matrix.allocator, "sub.{s}.{s}", .{ command, subcommand });
    defer matrix.allocator.free(id);

    try matrix.add(
        id,
        args.items,
        build.kind,
        build.timeout_ms,
        build.requires.items,
        build.cwd_mode,
        build.env_profile,
        build.exit_policy,
    );
}

fn addFirstLevelEntries(matrix: *Matrix) !void {
    for (command_subcommands) |entry| {
        for (entry.subcommands) |subcommand| {
            try addFirstLevelEntry(matrix, entry.command, subcommand);
        }
    }
}

fn addNestedEntries(matrix: *Matrix) !void {
    // Explicit nested trees.
    try matrix.add(
        "nested.discord.commands.list",
        &.{ "discord", "commands", "list" },
        .oneshot,
        20_000,
        &.{ "env:DISCORD_BOT_TOKEN", "net:discord" },
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.discord.commands.create",
        &.{ "discord", "commands", "create", "abi-full-ping", "ABI full ping" },
        .oneshot,
        20_000,
        &.{ "env:DISCORD_BOT_TOKEN", "net:discord" },
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.discord.commands.delete",
        &.{ "discord", "commands", "delete", "${ABI_TEST_DISCORD_COMMAND_ID}" },
        .oneshot,
        20_000,
        &.{ "env:DISCORD_BOT_TOKEN", "env:ABI_TEST_DISCORD_COMMAND_ID", "net:discord" },
        .temp_workspace,
        .isolated,
        .zero_only,
    );

    try matrix.add(
        "nested.profile.api-key.list",
        &.{ "profile", "api-key", "list" },
        .oneshot,
        20_000,
        &.{},
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.profile.api-key.set",
        &.{ "profile", "api-key", "set", "openai", "${OPENAI_API_KEY}" },
        .oneshot,
        20_000,
        &.{"env:OPENAI_API_KEY"},
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.profile.api-key.remove",
        &.{ "profile", "api-key", "remove", "openai" },
        .oneshot,
        20_000,
        &.{},
        .temp_workspace,
        .isolated,
        .zero_only,
    );

    try matrix.add(
        "nested.ralph.skills.list",
        &.{ "ralph", "skills", "list" },
        .oneshot,
        20_000,
        &.{},
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.ralph.skills.add",
        &.{ "ralph", "skills", "add", "full-matrix skill" },
        .oneshot,
        20_000,
        &.{},
        .repo_copy,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.ralph.skills.clear",
        &.{ "ralph", "skills", "clear" },
        .oneshot,
        20_000,
        &.{},
        .repo_copy,
        .isolated,
        .zero_only,
    );

    // Explicit provider-bearing embed vectors (not from completion metadata).
    try matrix.add(
        "nested.embed.openai",
        &.{ "embed", "--provider", "openai", "--text", "full matrix openai" },
        .oneshot,
        30_000,
        &.{ "env:OPENAI_API_KEY", "net:openai" },
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.embed.mistral",
        &.{ "embed", "--provider", "mistral", "--text", "full matrix mistral" },
        .oneshot,
        30_000,
        &.{ "env:MISTRAL_API_KEY", "net:mistral" },
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.embed.cohere",
        &.{ "embed", "--provider", "cohere", "--text", "full matrix cohere" },
        .oneshot,
        30_000,
        &.{ "env:COHERE_API_KEY", "net:cohere" },
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.embed.ollama",
        &.{ "embed", "--provider", "ollama", "--text", "full matrix ollama" },
        .oneshot,
        30_000,
        &.{ "env:OLLAMA_HOST", "env:ABI_TEST_OLLAMA_MODEL", "net:ollama" },
        .temp_workspace,
        .isolated,
        .zero_only,
    );

    try matrix.add(
        "nested.ui.list-themes",
        &.{ "ui", "--list-themes" },
        .oneshot,
        20_000,
        &.{},
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.ui.launch.list-themes",
        &.{ "ui", "launch", "--list-themes" },
        .oneshot,
        20_000,
        &.{},
        .temp_workspace,
        .isolated,
        .zero_only,
    );
    try matrix.add(
        "nested.ui.gpu.list-themes",
        &.{ "ui", "gpu", "--list-themes" },
        .oneshot,
        20_000,
        &.{},
        .temp_workspace,
        .isolated,
        .zero_only,
    );
}

fn addAliasEntries(matrix: *Matrix) !void {
    try matrix.add("alias.info", &.{"info"}, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add("alias.sysinfo", &.{"sysinfo"}, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add("alias.launch.list-themes", &.{ "launch", "--list-themes" }, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add("alias.start.list-themes", &.{ "start", "--list-themes" }, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add("alias.ls", &.{ "ls", "stats" }, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add("alias.run", &.{ "run", "quick" }, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add("alias.chat", &.{ "chat", "run", "--help" }, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add("alias.reasoning", &.{ "reasoning", "providers" }, .oneshot, 20_000, &.{}, .temp_workspace, .isolated, .zero_only);
    try matrix.add(
        "alias.serve",
        &.{ "serve", "serve", "-m", "${ABI_TEST_GGUF_MODEL_PATH}", "-a", "127.0.0.1:18180" },
        .serve_probe,
        45_000,
        &.{ "env:ABI_TEST_GGUF_MODEL_PATH", "file:ABI_TEST_GGUF_MODEL_PATH" },
        .temp_workspace,
        .isolated,
        .allow_signal_after_probe,
    );
}

fn buildMatrix(allocator: std.mem.Allocator) !Matrix {
    var matrix = Matrix.init(allocator);
    errdefer matrix.deinit();

    try addTopLevelEntries(&matrix);
    try addFirstLevelEntries(&matrix);
    try addNestedEntries(&matrix);
    try addAliasEntries(&matrix);

    return matrix;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const allocator = gpa_state.allocator();

    var args_iter = try std.process.Args.Iterator.initAllocator(init.args, allocator);
    defer args_iter.deinit();

    var args = std.ArrayListUnmanaged([]const u8).empty;
    defer args.deinit(allocator);

    _ = args_iter.next(); // executable path
    while (args_iter.next()) |arg| {
        try args.append(allocator, arg);
    }

    var json_out_path: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.items.len) : (i += 1) {
        const arg = args.items[i];
        if (std.mem.eql(u8, arg, "--json-out")) {
            i += 1;
            if (i >= args.items.len) {
                std.debug.print("error: --json-out requires a path\n", .{});
                std.process.exit(1);
            }
            json_out_path = args.items[i];
        }
    }

    var matrix = try buildMatrix(allocator);
    defer matrix.deinit();

    var json_writer: std.Io.Writer.Allocating = .init(allocator);
    defer json_writer.deinit();
    try std.json.Stringify.value(
        matrix.entries.items,
        .{ .whitespace = .indent_2 },
        &json_writer.writer,
    );
    try json_writer.writer.writeByte('\n');
    const json_data = try json_writer.toOwnedSlice();
    defer allocator.free(json_data);

    if (json_out_path) |path| {
        var io_backend = std.Io.Threaded.init(allocator, .{});
        defer io_backend.deinit();
        const io = io_backend.io();

        var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, json_data);
        std.debug.print("Wrote full CLI matrix ({d} entries) to {s}\n", .{ matrix.entries.items.len, path });
        return;
    }

    std.debug.print("{s}", .{json_data});
}

test {
    std.testing.refAllDecls(@This());
}
