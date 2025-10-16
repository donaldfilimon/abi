//! Persona manifest loader for runtime-configurable agent personas.
//!
//! Supports JSON and TOML sources describing prompts, tool availability,
//! safety filters, sampling parameters, and runtime toggles.

const std = @import("std");
const profile = @import("../../core/profile.zig");

pub const Allocator = std.mem.Allocator;

pub const PersonaManifestError = error{
    InvalidFormat,
    MissingField,
    InvalidValue,
    DuplicatePersona,
    UnsupportedFormat,
};

pub const ManifestFormat = enum { json, toml };

pub const PersonaArchetype = enum {
    adaptive,
    technical,
    empathetic,
};

pub fn archetypeToString(archetype: PersonaArchetype) []const u8 {
    return switch (archetype) {
        .adaptive => "adaptive",
        .technical => "technical",
        .empathetic => "empathetic",
    };
}

pub fn parseArchetype(value: []const u8) ?PersonaArchetype {
    if (std.ascii.eqlIgnoreCase(value, "adaptive")) return .adaptive;
    if (std.ascii.eqlIgnoreCase(value, "technical")) return .technical;
    if (std.ascii.eqlIgnoreCase(value, "empathetic")) return .empathetic;
    return null;
}

pub fn parseLoggingSink(value: []const u8) ?profile.LoggingSink {
    if (std.ascii.eqlIgnoreCase(value, "stdout")) return .stdout;
    if (std.ascii.eqlIgnoreCase(value, "stderr")) return .stderr;
    if (std.ascii.eqlIgnoreCase(value, "file")) return .file;
    return null;
}

pub const PersonaManifest = struct {
    allocator: Allocator,
    name: []const u8 = "",
    owns_name: bool = false,
    description: []const u8 = "",
    owns_description: bool = false,
    system_prompt: []const u8 = "",
    owns_system_prompt: bool = false,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    rate_limit_per_minute: u32 = 60,
    streaming: bool = true,
    function_calling: bool = true,
    logging_sink: profile.LoggingSink = .stdout,
    archetype: PersonaArchetype = .adaptive,
    tools: std.ArrayListUnmanaged([]const u8) = .{},
    safety_filters: std.ArrayListUnmanaged([]const u8) = .{},

    pub fn init(allocator: Allocator) PersonaManifest {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *PersonaManifest) void {
        if (self.owns_name) self.allocator.free(self.name);
        if (self.owns_description) self.allocator.free(self.description);
        if (self.owns_system_prompt) self.allocator.free(self.system_prompt);
        clearStringList(&self.tools, self.allocator);
        self.tools.deinit(self.allocator);
        clearStringList(&self.safety_filters, self.allocator);
        self.safety_filters.deinit(self.allocator);
    }

    pub fn toolsSlice(self: *const PersonaManifest) []const []const u8 {
        return self.tools.items;
    }

    pub fn safetyFiltersSlice(self: *const PersonaManifest) []const []const u8 {
        return self.safety_filters.items;
    }

    pub fn validate(self: PersonaManifest) PersonaManifestError!void {
        if (self.name.len == 0) return PersonaManifestError.MissingField;
        if (self.system_prompt.len == 0) return PersonaManifestError.MissingField;
        if (self.temperature < 0 or self.temperature > 2.0) return PersonaManifestError.InvalidValue;
        if (self.top_p < 0 or self.top_p > 1) return PersonaManifestError.InvalidValue;
        if (self.rate_limit_per_minute == 0) return PersonaManifestError.InvalidValue;
    }

    pub fn setTools(self: *PersonaManifest, values: []const []const u8) !void {
        try replaceStringList(self.allocator, &self.tools, values);
    }

    pub fn setSafetyFilters(self: *PersonaManifest, values: []const []const u8) !void {
        try replaceStringList(self.allocator, &self.safety_filters, values);
    }

    fn assignString(self: *PersonaManifest, field: *[]const u8, owns: *bool, value: []const u8) !void {
        if (owns.*) self.allocator.free(field.*);
        const copy = try self.allocator.dupe(u8, value);
        field.* = copy;
        owns.* = true;
    }

    fn assignOptionalString(self: *PersonaManifest, field: *[]const u8, owns: *bool, maybe_value: ?[]const u8) !void {
        if (maybe_value) |value| {
            try self.assignString(field, owns, value);
        }
    }
};

pub const PersonaRegistry = struct {
    allocator: Allocator,
    manifests: std.ArrayList(PersonaManifest),
    index: std.StringHashMap(usize),

    pub fn init(allocator: Allocator) PersonaRegistry {
        return .{
            .allocator = allocator,
            .manifests = std.ArrayList(PersonaManifest).init(allocator),
            .index = std.StringHashMap(usize).init(allocator),
        };
    }

    pub fn deinit(self: *PersonaRegistry) void {
        for (self.manifests.items) |*manifest| {
            manifest.deinit();
        }
        self.manifests.deinit();
        var it = self.index.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.index.deinit();
    }

    pub fn addManifest(self: *PersonaRegistry, manifest: PersonaManifest) PersonaManifestError!void {
        const idx = self.manifests.items.len;
        try self.manifests.append(manifest);
        errdefer {
            var removed = self.manifests.pop();
            removed.deinit();
        }

        const gop = try self.index.getOrPut(self.manifests.items[idx].name);
        if (gop.found_existing) {
            var removed = self.manifests.pop();
            removed.deinit();
            return PersonaManifestError.DuplicatePersona;
        }

        const key_copy = try self.allocator.dupe(u8, self.manifests.items[idx].name);
        errdefer self.allocator.free(key_copy);
        gop.key_ptr.* = key_copy;
        gop.value_ptr.* = idx;
    }

    pub fn get(self: *const PersonaRegistry, name: []const u8) ?*const PersonaManifest {
        if (self.index.get(name)) |entry| {
            return &self.manifests.items[entry.*];
        }
        return null;
    }

    pub fn slice(self: *const PersonaRegistry) []const PersonaManifest {
        return self.manifests.items;
    }
};

pub fn detectFormatFromExtension(path: []const u8) ?ManifestFormat {
    const ext = std.fs.path.extension(path);
    if (std.ascii.eqlIgnoreCase(ext, ".json")) return .json;
    if (std.ascii.eqlIgnoreCase(ext, ".toml")) return .toml;
    return null;
}

pub fn loadManifestFromSlice(allocator: Allocator, data: []const u8, format: ManifestFormat) PersonaManifestError!PersonaManifest {
    return switch (format) {
        .json => parseJsonManifest(allocator, data),
        .toml => parseTomlManifest(allocator, data),
    };
}

pub fn loadManifestFromFile(allocator: Allocator, dir: std.fs.Dir, path: []const u8) PersonaManifestError!PersonaManifest {
    const format = detectFormatFromExtension(path) orelse return PersonaManifestError.UnsupportedFormat;
    const max_size: usize = 512 * 1024;
    var file = try dir.openFile(path, .{});
    defer file.close();
    const data = try file.readToEndAlloc(allocator, max_size);
    defer allocator.free(data);
    return loadManifestFromSlice(allocator, data, format);
}

pub fn loadRegistryFromDir(allocator: Allocator, path: []const u8) PersonaManifestError!PersonaRegistry {
    var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();

    var registry = PersonaRegistry.init(allocator);
    errdefer registry.deinit();

    var it = dir.iterate();
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        const format = detectFormatFromExtension(entry.name) orelse continue;
        const data = try dir.readFileAlloc(allocator, entry.name, 512 * 1024);
        defer allocator.free(data);
        var manifest = try loadManifestFromSlice(allocator, data, format);
        registry.addManifest(manifest) catch |err| {
            manifest.deinit();
            return err;
        };
    }

    return registry;
}

fn clearStringList(list: *std.ArrayListUnmanaged([]const u8), allocator: Allocator) void {
    for (list.items) |item| {
        allocator.free(item);
    }
    list.clearRetainingCapacity();
}

fn replaceStringList(allocator: Allocator, list: *std.ArrayListUnmanaged([]const u8), values: []const []const u8) !void {
    clearStringList(list, allocator);
    for (values) |value| {
        const copy = try allocator.dupe(u8, value);
        errdefer allocator.free(copy);
        try list.append(allocator, copy);
    }
}

const ManifestWire = struct {
    name: []const u8,
    description: ?[]const u8 = null,
    system_prompt: []const u8,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    rate_limit_per_minute: ?u32 = null,
    streaming: ?bool = null,
    function_calling: ?bool = null,
    logging_sink: ?[]const u8 = null,
    archetype: ?[]const u8 = null,
    tools: ?[]const []const u8 = null,
    safety_filters: ?[]const []const u8 = null,
};

fn parseJsonManifest(allocator: Allocator, data: []const u8) PersonaManifestError!PersonaManifest {
    var parsed = try std.json.parseFromSlice(ManifestWire, allocator, data, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    var manifest = PersonaManifest.init(allocator);
    errdefer manifest.deinit();

    const wire = parsed.value;
    try manifest.assignString(&manifest.name, &manifest.owns_name, wire.name);
    try manifest.assignString(&manifest.system_prompt, &manifest.owns_system_prompt, wire.system_prompt);
    try manifest.assignOptionalString(&manifest.description, &manifest.owns_description, wire.description);
    if (wire.temperature) |temp| manifest.temperature = temp;
    if (wire.top_p) |tp| manifest.top_p = tp;
    if (wire.rate_limit_per_minute) |rl| manifest.rate_limit_per_minute = rl;
    if (wire.streaming) |flag| manifest.streaming = flag;
    if (wire.function_calling) |flag| manifest.function_calling = flag;
    if (wire.logging_sink) |sink_name| {
        manifest.logging_sink = parseLoggingSink(sink_name) orelse return PersonaManifestError.InvalidValue;
    }
    if (wire.archetype) |archetype_name| {
        manifest.archetype = parseArchetype(archetype_name) orelse return PersonaManifestError.InvalidValue;
    }
    if (wire.tools) |values| try manifest.setTools(values);
    if (wire.safety_filters) |values| try manifest.setSafetyFilters(values);

    try manifest.validate();
    return manifest;
}

fn parseTomlManifest(allocator: Allocator, data: []const u8) PersonaManifestError!PersonaManifest {
    var manifest = PersonaManifest.init(allocator);
    errdefer manifest.deinit();

    var line_iter = std.mem.tokenizeScalar(u8, data, '\n');
    while (line_iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0 or trimmed[0] == '#') continue;
        const parts = std.mem.splitOnce(u8, trimmed, "=") orelse return PersonaManifestError.InvalidFormat;
        const key = std.mem.trim(u8, parts.first, " \t\r");
        const raw_value = std.mem.trim(u8, parts.second, " \t\r");

        if (std.mem.eql(u8, key, "name")) {
            const value = try parseTomlString(raw_value);
            try manifest.assignString(&manifest.name, &manifest.owns_name, value);
        } else if (std.mem.eql(u8, key, "description")) {
            const value = try parseTomlString(raw_value);
            try manifest.assignString(&manifest.description, &manifest.owns_description, value);
        } else if (std.mem.eql(u8, key, "system_prompt")) {
            const value = try parseTomlString(raw_value);
            try manifest.assignString(&manifest.system_prompt, &manifest.owns_system_prompt, value);
        } else if (std.mem.eql(u8, key, "temperature")) {
            manifest.temperature = try parseTomlFloat(raw_value);
        } else if (std.mem.eql(u8, key, "top_p")) {
            manifest.top_p = try parseTomlFloat(raw_value);
        } else if (std.mem.eql(u8, key, "rate_limit_per_minute")) {
            manifest.rate_limit_per_minute = try parseTomlInt(raw_value);
        } else if (std.mem.eql(u8, key, "streaming")) {
            manifest.streaming = try parseTomlBool(raw_value);
        } else if (std.mem.eql(u8, key, "function_calling")) {
            manifest.function_calling = try parseTomlBool(raw_value);
        } else if (std.mem.eql(u8, key, "logging_sink")) {
            const sink_name = try parseTomlString(raw_value);
            manifest.logging_sink = parseLoggingSink(sink_name) orelse return PersonaManifestError.InvalidValue;
        } else if (std.mem.eql(u8, key, "archetype")) {
            const archetype_name = try parseTomlString(raw_value);
            manifest.archetype = parseArchetype(archetype_name) orelse return PersonaManifestError.InvalidValue;
        } else if (std.mem.eql(u8, key, "tools")) {
            try parseTomlArrayIntoList(allocator, raw_value, &manifest.tools);
        } else if (std.mem.eql(u8, key, "safety_filters")) {
            try parseTomlArrayIntoList(allocator, raw_value, &manifest.safety_filters);
        }
    }

    try manifest.validate();
    return manifest;
}

fn parseTomlString(raw: []const u8) PersonaManifestError![]const u8 {
    if (raw.len < 2 or raw[0] != '"' or raw[raw.len - 1] != '"') return PersonaManifestError.InvalidFormat;
    return raw[1 .. raw.len - 1];
}

fn parseTomlArrayIntoList(allocator: Allocator, raw: []const u8, list: *std.ArrayListUnmanaged([]const u8)) !void {
    clearStringList(list, allocator);
    if (raw.len < 2 or raw[0] != '[' or raw[raw.len - 1] != ']') return PersonaManifestError.InvalidFormat;
    const inner = std.mem.trim(u8, raw[1 .. raw.len - 1], " \t\r");
    if (inner.len == 0) return;
    var iter = std.mem.tokenizeScalar(u8, inner, ',');
    while (iter.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t\r");
        const value = try parseTomlString(trimmed);
        const copy = try allocator.dupe(u8, value);
        errdefer allocator.free(copy);
        try list.append(allocator, copy);
    }
}

fn parseTomlBool(raw: []const u8) PersonaManifestError!bool {
    if (std.ascii.eqlIgnoreCase(raw, "true")) return true;
    if (std.ascii.eqlIgnoreCase(raw, "false")) return false;
    return PersonaManifestError.InvalidFormat;
}

fn parseTomlFloat(raw: []const u8) PersonaManifestError!f32 {
    return std.fmt.parseFloat(f32, raw) catch PersonaManifestError.InvalidFormat;
}

fn parseTomlInt(raw: []const u8) PersonaManifestError!u32 {
    return std.fmt.parseInt(u32, raw, 10) catch PersonaManifestError.InvalidFormat;
}

test "load manifest from json" {
    const testing = std.testing;
    const json =
        \\{
        \\  "name": "creative",
        \\  "system_prompt": "You are creative",
        \\  "temperature": 0.8,
        \\  "top_p": 0.85,
        \\  "rate_limit_per_minute": 120,
        \\  "tools": ["search", "code"],
        \\  "safety_filters": ["toxicity"]
        \\}
    ;
    var manifest = try loadManifestFromSlice(testing.allocator, json, .json);
    defer manifest.deinit();
    try testing.expectEqualStrings("creative", manifest.name);
    try testing.expectEqualStrings("You are creative", manifest.system_prompt);
    try testing.expectEqual(@as(f32, 0.8), manifest.temperature);
    try testing.expectEqual(@as(u32, 120), manifest.rate_limit_per_minute);
    try testing.expectEqual(@as(usize, 2), manifest.tools.items.len);
}

test "load manifest from toml" {
    const testing = std.testing;
    const toml =
        \\name = "empathetic"
        \\system_prompt = "Respond with care"
        \\temperature = 0.6
        \\top_p = 0.9
        \\rate_limit_per_minute = 30
        \\tools = ["memory"]
        \\safety_filters = ["toxicity", "bias"]
    ;
    var manifest = try loadManifestFromSlice(testing.allocator, toml, .toml);
    defer manifest.deinit();
    try testing.expectEqualStrings("empathetic", manifest.name);
    try testing.expectEqualStrings("Respond with care", manifest.system_prompt);
    try testing.expectEqual(@as(usize, 2), manifest.safety_filters.items.len);
}
