const std = @import("std");
const types = @import("../types.zig");
const errors = @import("../errors.zig");
const manifest = @import("manifest.zig");
const abi_v1 = @import("native_abi_v1.zig");

const default_symbol: [:0]const u8 = "abi_llm_plugin_v1";

const LoadedPlugin = struct {
    lib: std.DynLib,
    vtable: *const abi_v1.PluginV1,

    fn deinit(self: *LoadedPlugin) void {
        self.lib.close();
    }
};

pub fn generate(
    allocator: std.mem.Allocator,
    plugin_entry: manifest.PluginEntry,
    cfg: types.GenerateConfig,
) !types.GenerateResult {
    if (plugin_entry.kind != .native) return errors.ProviderError.InvalidPlugin;
    if (!plugin_entry.enabled) return errors.ProviderError.PluginDisabled;

    var loaded = try load(allocator, plugin_entry);
    defer loaded.deinit();

    const model_z = try allocator.dupeZ(u8, cfg.model);
    defer allocator.free(model_z);

    const prompt_z = try allocator.dupeZ(u8, cfg.prompt);
    defer allocator.free(prompt_z);

    var request = abi_v1.GenerateRequest{
        .model = model_z.ptr,
        .prompt = prompt_z.ptr,
        .max_tokens = cfg.max_tokens,
        .temperature = cfg.temperature,
        .top_p = cfg.top_p,
        .top_k = cfg.top_k,
        .repetition_penalty = cfg.repetition_penalty,
    };

    var response = abi_v1.GenerateResponse{};
    const status_code = loaded.vtable.generate(&request, &response);

    defer {
        if (response.release) |release| {
            release(&response);
        }
    }

    if (status_code != @intFromEnum(abi_v1.Status.ok)) {
        return switch (abi_v1.statusFromCode(status_code)) {
            .not_available => errors.ProviderError.NotAvailable,
            .invalid_request => errors.ProviderError.InvalidPlugin,
            .failed => errors.ProviderError.GenerationFailed,
            .ok => errors.ProviderError.GenerationFailed,
        };
    }

    if (response.text_len == 0) return errors.ProviderError.GenerationFailed;

    const model_used = if (response.model_len > 0)
        try allocator.dupe(u8, response.model_ptr[0..response.model_len])
    else
        try allocator.dupe(u8, cfg.model);

    errdefer allocator.free(model_used);

    return .{
        .provider = .plugin_native,
        .model_used = model_used,
        .content = try allocator.dupe(u8, response.text_ptr[0..response.text_len]),
    };
}

fn load(allocator: std.mem.Allocator, plugin_entry: manifest.PluginEntry) !LoadedPlugin {
    const path = plugin_entry.library_path orelse return errors.ProviderError.InvalidPlugin;

    var lib = std.DynLib.open(path) catch return errors.ProviderError.NotAvailable;
    errdefer lib.close();

    var owned_symbol: ?[:0]u8 = null;
    defer if (owned_symbol) |symbol| allocator.free(symbol);

    const symbol: [:0]const u8 = if (plugin_entry.symbol) |value| blk: {
        const symbol_z = try allocator.dupeZ(u8, value);
        owned_symbol = symbol_z;
        break :blk symbol_z;
    } else default_symbol;

    const get_plugin = lib.lookup(abi_v1.GetPluginFn, symbol) orelse return errors.ProviderError.SymbolMissing;

    const plugin = get_plugin();
    if (plugin.abi_version != abi_v1.ABI_VERSION) {
        return errors.ProviderError.AbiVersionMismatch;
    }

    if (plugin.is_available) |is_available| {
        if (!is_available()) return errors.ProviderError.NotAvailable;
    }

    return .{
        .lib = lib,
        .vtable = plugin,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "generate rejects non-native plugin kind" {
    const allocator = std.testing.allocator;
    const entry = manifest.PluginEntry{
        .id = @constCast("http-plugin"),
        .kind = .http,
        .enabled = true,
        .library_path = @constCast("/nonexistent/lib.dylib"),
    };

    const cfg = types.GenerateConfig{
        .model = "test-model",
        .prompt = "hello",
    };

    const result = generate(allocator, entry, cfg);
    try std.testing.expectError(errors.ProviderError.InvalidPlugin, result);
}

test "generate rejects disabled native plugin" {
    const allocator = std.testing.allocator;
    const entry = manifest.PluginEntry{
        .id = @constCast("disabled-native"),
        .kind = .native,
        .enabled = false,
        .library_path = @constCast("/nonexistent/lib.dylib"),
    };

    const cfg = types.GenerateConfig{
        .model = "test-model",
        .prompt = "hello",
    };

    const result = generate(allocator, entry, cfg);
    try std.testing.expectError(errors.ProviderError.PluginDisabled, result);
}

test "generate rejects native plugin without library_path" {
    const allocator = std.testing.allocator;
    const entry = manifest.PluginEntry{
        .id = @constCast("no-path"),
        .kind = .native,
        .enabled = true,
        .library_path = null,
    };

    const cfg = types.GenerateConfig{
        .model = "test-model",
        .prompt = "hello",
    };

    // load() will return InvalidPlugin because library_path is null
    const result = generate(allocator, entry, cfg);
    try std.testing.expectError(errors.ProviderError.InvalidPlugin, result);
}

test "load returns NotAvailable for nonexistent library" {
    const allocator = std.testing.allocator;
    const entry = manifest.PluginEntry{
        .id = @constCast("missing-lib"),
        .kind = .native,
        .enabled = true,
        .library_path = @constCast("/nonexistent/path/to/lib.dylib"),
    };

    const result = load(allocator, entry);
    try std.testing.expectError(errors.ProviderError.NotAvailable, result);
}

test {
    std.testing.refAllDecls(@This());
}
