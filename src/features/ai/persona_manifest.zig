const std = @import("std");

pub const ManifestError = error{
    InvalidFormat,
    MissingField,
    InvalidValue,
    DuplicatePersona,
    DuplicateProfile,
    UnsupportedExtension,
};

pub const RateLimits = struct {
    requests_per_minute: u32 = 60,
    tokens_per_minute: ?u32 = null,
    concurrent_requests: ?u32 = null,

    pub fn validate(self: RateLimits) ManifestError!void {
        if (self.requests_per_minute == 0) return ManifestError.InvalidValue;
        if (self.tokens_per_minute) |tokens| {
            if (tokens == 0) return ManifestError.InvalidValue;
        }
        if (self.concurrent_requests) |value| {
            if (value == 0) return ManifestError.InvalidValue;
        }
    }
};

pub const PersonaProfile = struct {
    name: []const u8,
    system_prompt: []const u8,
    description: []const u8,
    tools: []const []const u8,
    safety_filters: []const []const u8,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    streaming: bool = true,
    function_calling: bool = true,
    rate_limits: RateLimits = .{},
    log_sink: []const u8,

    pub fn validate(self: PersonaProfile) ManifestError!void {
        if (self.name.len == 0) return ManifestError.MissingField;
        if (self.system_prompt.len == 0) return ManifestError.MissingField;
        if (self.temperature < 0.0 or self.temperature > 2.0) return ManifestError.InvalidValue;
        if (self.top_p < 0.0 or self.top_p > 1.0) return ManifestError.InvalidValue;
        try self.rate_limits.validate();
    }
};

pub const EnvironmentProfile = struct {
    name: []const u8,
    streaming: bool = true,
    function_calling: bool = true,
    log_sink: []const u8,
    default_persona: ?[]const u8 = null,

    pub fn validate(self: EnvironmentProfile) ManifestError!void {
        if (self.name.len == 0) return ManifestError.MissingField;
        if (self.log_sink.len == 0) return ManifestError.InvalidValue;
    }
};

pub const PersonaManifest = struct {
    arena: std.heap.ArenaAllocator,
    personas: []const PersonaProfile,
    environments: []const EnvironmentProfile,
    source_path: []const u8,

    pub fn deinit(self: *PersonaManifest) void {
        self.arena.deinit();
    }

    pub fn findPersona(self: *const PersonaManifest, name: []const u8) ?*const PersonaProfile {
        for (self.personas) |*persona| {
            if (std.mem.eql(u8, persona.name, name)) return persona;
        }
        return null;
    }

    pub fn findEnvironment(self: *const PersonaManifest, name: []const u8) ?*const EnvironmentProfile {
        for (self.environments) |*profile| {
            if (std.mem.eql(u8, profile.name, name)) return profile;
        }
        return null;
    }

    pub fn defaultPersona(self: *const PersonaManifest) ?*const PersonaProfile {
        return if (self.personas.len == 0) null else &self.personas[0];
    }

    pub fn defaultEnvironment(self: *const PersonaManifest) ?*const EnvironmentProfile {
        return if (self.environments.len == 0) null else &self.environments[0];
    }

    pub fn validate(self: *const PersonaManifest) ManifestError!void {
        for (self.personas) |persona| {
            try persona.validate();
        }
        for (self.environments) |profile| {
            try profile.validate();
            if (profile.default_persona) |name| {
                if (self.findPersona(name) == null) return ManifestError.InvalidValue;
            }
        }
    }
};

pub const LoadError = ManifestError || std.mem.Allocator.Error || std.fs.File.OpenError || std.fs.File.ReadError || std.json.ParseError;

const max_manifest_size: usize = 8 * 1024 * 1024;

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) LoadError!PersonaManifest {
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const contents = try file.readToEndAlloc(allocator, max_manifest_size);
    defer allocator.free(contents);

    const ext = std.fs.path.extension(path);
    if (std.mem.eql(u8, ext, ".json")) {
        return loadFromJson(allocator, path, contents);
    }
    if (std.mem.eql(u8, ext, ".toml")) {
        return loadFromToml(allocator, path, contents);
    }
    return ManifestError.UnsupportedExtension;
}

fn loadFromJson(allocator: std.mem.Allocator, path: []const u8, bytes: []const u8) LoadError!PersonaManifest {
    var scratch = std.heap.ArenaAllocator.init(allocator);
    defer scratch.deinit();
    const scratch_alloc = scratch.allocator();

    var manifest_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer manifest_arena.deinit();
    const manifest_alloc = manifest_arena.allocator();

    var parsed = try std.json.parseFromSlice(std.json.Value, scratch_alloc, bytes, .{});
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return ManifestError.InvalidFormat;
    const root_obj = root.object;

    const personas_value = root_obj.get("personas") orelse return ManifestError.MissingField;
    const personas_array = personas_value.array orelse return ManifestError.InvalidFormat;

    var seen_personas = std.StringHashMapUnmanaged(void){};
    defer seen_personas.deinit(scratch_alloc);

    var persona_storage = try manifest_alloc.alloc(PersonaProfile, personas_array.items.len);
    var persona_index: usize = 0;

    for (personas_array.items) |entry| {
        if (entry != .object) return ManifestError.InvalidFormat;
        const persona_obj = entry.object;

        const name_value = persona_obj.get("name") orelse return ManifestError.MissingField;
        const name_text = name_value.string orelse return ManifestError.InvalidFormat;
        if (seen_personas.contains(name_text)) return ManifestError.DuplicatePersona;
        try seen_personas.put(scratch_alloc, name_text, {});
        const name_copy = try manifest_alloc.dupe(u8, name_text);

        const prompt_value = persona_obj.get("system_prompt") orelse return ManifestError.MissingField;
        const prompt_text = prompt_value.string orelse return ManifestError.InvalidFormat;
        const prompt_copy = try manifest_alloc.dupe(u8, prompt_text);

        const description_copy = blk: {
            if (persona_obj.get("description")) |value| {
                const text = value.string orelse return ManifestError.InvalidFormat;
                break :blk try manifest_alloc.dupe(u8, text);
            }
            break :blk try manifest_alloc.dupe(u8, "");
        };

        const temperature = blk: {
            if (persona_obj.get("temperature")) |value| {
                break :blk try coerceJsonFloat(value);
            }
            break :blk 0.7;
        };

        const top_p = blk: {
            if (persona_obj.get("top_p")) |value| {
                break :blk try coerceJsonFloat(value);
            }
            break :blk 0.9;
        };

        const streaming = blk: {
            if (persona_obj.get("streaming")) |value| {
                const boolean = value.bool orelse return ManifestError.InvalidFormat;
                break :blk boolean;
            }
            break :blk true;
        };

        const function_calling = blk: {
            if (persona_obj.get("function_calling")) |value| {
                const boolean = value.bool orelse return ManifestError.InvalidFormat;
                break :blk boolean;
            }
            break :blk true;
        };

        const log_sink_copy = blk: {
            if (persona_obj.get("log_sink")) |value| {
                const text = value.string orelse return ManifestError.InvalidFormat;
                break :blk try manifest_alloc.dupe(u8, text);
            }
            break :blk try manifest_alloc.dupe(u8, "stdout");
        };

        const tools = blk: {
            if (persona_obj.get("tools")) |value| {
                const arr = value.array orelse return ManifestError.InvalidFormat;
                var out = try manifest_alloc.alloc([]const u8, arr.items.len);
                for (arr.items, 0..) |tool_value, idx| {
                    const text = tool_value.string orelse return ManifestError.InvalidFormat;
                    out[idx] = try manifest_alloc.dupe(u8, text);
                }
                break :blk out;
            }
            break :blk try allocEmptySlice(manifest_alloc);
        };

        const filters = blk: {
            if (persona_obj.get("safety_filters")) |value| {
                const arr = value.array orelse return ManifestError.InvalidFormat;
                var out = try manifest_alloc.alloc([]const u8, arr.items.len);
                for (arr.items, 0..) |filter_value, idx| {
                    const text = filter_value.string orelse return ManifestError.InvalidFormat;
                    out[idx] = try manifest_alloc.dupe(u8, text);
                }
                break :blk out;
            }
            break :blk try allocEmptySlice(manifest_alloc);
        };

        var rate_limits = RateLimits{};
        if (persona_obj.get("rate_limits")) |value| {
            const obj = value.object orelse return ManifestError.InvalidFormat;
            if (obj.get("requests_per_minute")) |rpm| {
                rate_limits.requests_per_minute = try coerceJsonInt(u32, rpm);
            }
            if (obj.get("tokens_per_minute")) |tpm| {
                rate_limits.tokens_per_minute = try coerceJsonInt(u32, tpm);
            }
            if (obj.get("concurrent_requests")) |concurrent| {
                rate_limits.concurrent_requests = try coerceJsonInt(u32, concurrent);
            }
        }

        var persona = PersonaProfile{
            .name = name_copy,
            .system_prompt = prompt_copy,
            .description = description_copy,
            .tools = tools,
            .safety_filters = filters,
            .temperature = temperature,
            .top_p = top_p,
            .streaming = streaming,
            .function_calling = function_calling,
            .rate_limits = rate_limits,
            .log_sink = log_sink_copy,
        };
        try persona.validate();

        persona_storage[persona_index] = persona;
        persona_index += 1;
    }

    var env_slice: []const EnvironmentProfile = &.{};
    if (root_obj.get("profiles")) |profiles_value| {
        const arr = profiles_value.array orelse return ManifestError.InvalidFormat;
        var seen_profiles = std.StringHashMapUnmanaged(void){};
        defer seen_profiles.deinit(scratch_alloc);

        var env_storage = try manifest_alloc.alloc(EnvironmentProfile, arr.items.len);
        var env_index: usize = 0;
        for (arr.items) |entry| {
            if (entry != .object) return ManifestError.InvalidFormat;
            const profile_obj = entry.object;

            const name_value = profile_obj.get("name") orelse return ManifestError.MissingField;
            const name_text = name_value.string orelse return ManifestError.InvalidFormat;
            if (seen_profiles.contains(name_text)) return ManifestError.DuplicateProfile;
            try seen_profiles.put(scratch_alloc, name_text, {});
            const name_copy = try manifest_alloc.dupe(u8, name_text);

            const streaming = blk: {
                if (profile_obj.get("streaming")) |value| {
                    const boolean = value.bool orelse return ManifestError.InvalidFormat;
                    break :blk boolean;
                }
                break :blk true;
            };

            const function_calling = blk: {
                if (profile_obj.get("function_calling")) |value| {
                    const boolean = value.bool orelse return ManifestError.InvalidFormat;
                    break :blk boolean;
                }
                break :blk true;
            };

            const log_sink_copy = blk: {
                if (profile_obj.get("log_sink")) |value| {
                    const text = value.string orelse return ManifestError.InvalidFormat;
                    break :blk try manifest_alloc.dupe(u8, text);
                }
                break :blk try manifest_alloc.dupe(u8, "stdout");
            };

            const default_persona = blk: {
                if (profile_obj.get("default_persona")) |value| {
                    const text = value.string orelse return ManifestError.InvalidFormat;
                    break :blk try manifest_alloc.dupe(u8, text);
                }
                break :blk null;
            };

            var profile = EnvironmentProfile{
                .name = name_copy,
                .streaming = streaming,
                .function_calling = function_calling,
                .log_sink = log_sink_copy,
                .default_persona = default_persona,
            };
            try profile.validate();

            env_storage[env_index] = profile;
            env_index += 1;
        }
        env_slice = env_storage[0..env_index];
    }

    const path_copy = try manifest_alloc.dupe(u8, path);
    var manifest = PersonaManifest{
        .arena = manifest_arena,
        .personas = persona_storage[0..persona_index],
        .environments = env_slice,
        .source_path = path_copy,
    };
    try manifest.validate();
    return manifest;
}

const PersonaBuilder = struct {
    allocator: std.mem.Allocator,
    name: ?[]const u8 = null,
    system_prompt: ?[]const u8 = null,
    description: ?[]const u8 = null,
    log_sink: ?[]const u8 = null,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    streaming: bool = true,
    function_calling: bool = true,
    tools: std.ArrayList([]const u8),
    safety_filters: std.ArrayList([]const u8),
    rate_limits: RateLimits = .{},

    fn init(allocator: std.mem.Allocator) PersonaBuilder {
        return .{
            .allocator = allocator,
            .tools = std.ArrayList([]const u8).init(allocator),
            .safety_filters = std.ArrayList([]const u8).init(allocator),
        };
    }

    fn reset(self: *PersonaBuilder) void {
        self.* = PersonaBuilder.init(self.allocator);
    }

    fn build(self: *PersonaBuilder) !PersonaProfile {
        const name = self.name orelse return ManifestError.MissingField;
        const prompt = self.system_prompt orelse return ManifestError.MissingField;
        const description = self.description orelse try self.allocator.dupe(u8, "");
        const log_sink = self.log_sink orelse try self.allocator.dupe(u8, "stdout");
        const tools_slice = try self.tools.toOwnedSlice();
        const filter_slice = try self.safety_filters.toOwnedSlice();
        return PersonaProfile{
            .name = name,
            .system_prompt = prompt,
            .description = description,
            .tools = tools_slice,
            .safety_filters = filter_slice,
            .temperature = self.temperature,
            .top_p = self.top_p,
            .streaming = self.streaming,
            .function_calling = self.function_calling,
            .rate_limits = self.rate_limits,
            .log_sink = log_sink,
        };
    }
};

const ProfileBuilder = struct {
    allocator: std.mem.Allocator,
    name: ?[]const u8 = null,
    log_sink: ?[]const u8 = null,
    streaming: bool = true,
    function_calling: bool = true,
    default_persona: ?[]const u8 = null,

    fn init(allocator: std.mem.Allocator) ProfileBuilder {
        return .{ .allocator = allocator };
    }

    fn reset(self: *ProfileBuilder) void {
        self.* = ProfileBuilder.init(self.allocator);
    }

    fn build(self: *ProfileBuilder) !EnvironmentProfile {
        const name = self.name orelse return ManifestError.MissingField;
        const log_sink = self.log_sink orelse try self.allocator.dupe(u8, "stdout");
        return EnvironmentProfile{
            .name = name,
            .streaming = self.streaming,
            .function_calling = self.function_calling,
            .log_sink = log_sink,
            .default_persona = self.default_persona,
        };
    }
};

fn loadFromToml(allocator: std.mem.Allocator, path: []const u8, bytes: []const u8) LoadError!PersonaManifest {
    var manifest_arena = std.heap.ArenaAllocator.init(allocator);
    errdefer manifest_arena.deinit();
    const manifest_alloc = manifest_arena.allocator();

    var personas = std.ArrayList(PersonaProfile).init(manifest_alloc);
    var environments = std.ArrayList(EnvironmentProfile).init(manifest_alloc);

    const State = enum { none, persona, persona_rate_limits, profile };
    var state: State = .none;

    var persona_builder = PersonaBuilder.init(manifest_alloc);
    var profile_builder = ProfileBuilder.init(manifest_alloc);

    var it = std.mem.tokenizeScalar(u8, bytes, '\n');
    while (it.next()) |raw_line| {
        const trimmed = std.mem.trim(u8, raw_line, " \t\r");
        if (trimmed.len == 0) continue;
        if (trimmed[0] == '#') continue;

        if (trimmed[0] == '[') {
            if (std.mem.eql(u8, trimmed, "[[persona]]")) {
                if (state == .persona or state == .persona_rate_limits) {
                    var persona = try persona_builder.build();
                    try persona.validate();
                    try personas.append(persona);
                    persona_builder.reset();
                }
                if (state == .profile) {
                    var profile = try profile_builder.build();
                    try profile.validate();
                    try environments.append(profile);
                    profile_builder.reset();
                }
                state = .persona;
                continue;
            }
            if (std.mem.eql(u8, trimmed, "[persona.rate_limits]")) {
                if (state != .persona and state != .persona_rate_limits) return ManifestError.InvalidFormat;
                state = .persona_rate_limits;
                continue;
            }
            if (std.mem.eql(u8, trimmed, "[[profile]]")) {
                if (state == .persona or state == .persona_rate_limits) {
                    var persona = try persona_builder.build();
                    try persona.validate();
                    try personas.append(persona);
                    persona_builder.reset();
                }
                if (state == .profile) {
                    var profile = try profile_builder.build();
                    try profile.validate();
                    try environments.append(profile);
                    profile_builder.reset();
                }
                state = .profile;
                continue;
            }
            return ManifestError.InvalidFormat;
        }

        const eq_index = std.mem.indexOfScalar(u8, trimmed, '=') orelse return ManifestError.InvalidFormat;
        const key = std.mem.trim(u8, trimmed[0..eq_index], " \t");
        const value = std.mem.trim(u8, trimmed[eq_index + 1 ..], " \t");

        switch (state) {
            .persona => try parsePersonaKey(&persona_builder, key, value),
            .persona_rate_limits => try parsePersonaRateLimitKey(&persona_builder, key, value),
            .profile => try parseProfileKey(&profile_builder, key, value),
            .none => return ManifestError.InvalidFormat,
        }
    }

    if (state == .persona or state == .persona_rate_limits) {
        var persona = try persona_builder.build();
        try persona.validate();
        try personas.append(persona);
    } else if (state == .profile) {
        var profile = try profile_builder.build();
        try profile.validate();
        try environments.append(profile);
    }

    const persona_slice = try personas.toOwnedSlice();
    const env_slice = try environments.toOwnedSlice();
    const path_copy = try manifest_alloc.dupe(u8, path);

    var manifest = PersonaManifest{
        .arena = manifest_arena,
        .personas = persona_slice,
        .environments = env_slice,
        .source_path = path_copy,
    };
    try manifest.validate();
    return manifest;
}

fn parsePersonaKey(builder: *PersonaBuilder, key: []const u8, value: []const u8) ManifestError!void {
    if (std.mem.eql(u8, key, "name")) {
        builder.name = try builder.allocator.dupe(u8, try parseTomlString(value));
    } else if (std.mem.eql(u8, key, "system_prompt")) {
        builder.system_prompt = try builder.allocator.dupe(u8, try parseTomlString(value));
    } else if (std.mem.eql(u8, key, "description")) {
        builder.description = try builder.allocator.dupe(u8, try parseTomlString(value));
    } else if (std.mem.eql(u8, key, "log_sink")) {
        builder.log_sink = try builder.allocator.dupe(u8, try parseTomlString(value));
    } else if (std.mem.eql(u8, key, "temperature")) {
        builder.temperature = try parseTomlFloat(value);
    } else if (std.mem.eql(u8, key, "top_p")) {
        builder.top_p = try parseTomlFloat(value);
    } else if (std.mem.eql(u8, key, "streaming")) {
        builder.streaming = try parseTomlBool(value);
    } else if (std.mem.eql(u8, key, "function_calling")) {
        builder.function_calling = try parseTomlBool(value);
    } else if (std.mem.eql(u8, key, "tools")) {
        try parseTomlStringArray(builder.allocator, value, &builder.tools);
    } else if (std.mem.eql(u8, key, "safety_filters")) {
        try parseTomlStringArray(builder.allocator, value, &builder.safety_filters);
    } else {
        return ManifestError.InvalidFormat;
    }
}

fn parsePersonaRateLimitKey(builder: *PersonaBuilder, key: []const u8, value: []const u8) ManifestError!void {
    if (std.mem.eql(u8, key, "requests_per_minute")) {
        builder.rate_limits.requests_per_minute = try parseTomlInt(u32, value);
    } else if (std.mem.eql(u8, key, "tokens_per_minute")) {
        builder.rate_limits.tokens_per_minute = try parseTomlInt(u32, value);
    } else if (std.mem.eql(u8, key, "concurrent_requests")) {
        builder.rate_limits.concurrent_requests = try parseTomlInt(u32, value);
    } else {
        return ManifestError.InvalidFormat;
    }
}

fn parseProfileKey(builder: *ProfileBuilder, key: []const u8, value: []const u8) ManifestError!void {
    if (std.mem.eql(u8, key, "name")) {
        builder.name = try builder.allocator.dupe(u8, try parseTomlString(value));
    } else if (std.mem.eql(u8, key, "log_sink")) {
        builder.log_sink = try builder.allocator.dupe(u8, try parseTomlString(value));
    } else if (std.mem.eql(u8, key, "streaming")) {
        builder.streaming = try parseTomlBool(value);
    } else if (std.mem.eql(u8, key, "function_calling")) {
        builder.function_calling = try parseTomlBool(value);
    } else if (std.mem.eql(u8, key, "default_persona")) {
        builder.default_persona = try builder.allocator.dupe(u8, try parseTomlString(value));
    } else {
        return ManifestError.InvalidFormat;
    }
}

fn parseTomlString(value: []const u8) ManifestError![]const u8 {
    if (value.len < 2) return ManifestError.InvalidFormat;
    const first = value[0];
    const last = value[value.len - 1];
    if ((first == '"' and last == '"') or (first == '\'' and last == '\'')) {
        return value[1 .. value.len - 1];
    }
    return ManifestError.InvalidFormat;
}

fn parseTomlBool(value: []const u8) ManifestError!bool {
    if (std.mem.eql(u8, value, "true")) return true;
    if (std.mem.eql(u8, value, "false")) return false;
    return ManifestError.InvalidFormat;
}

fn parseTomlFloat(value: []const u8) ManifestError!f32 {
    return try std.fmt.parseFloat(f32, value);
}

fn parseTomlInt(comptime T: type, value: []const u8) ManifestError!T {
    return try std.fmt.parseInt(T, value, 10);
}

fn parseTomlStringArray(allocator: std.mem.Allocator, value: []const u8, list: *std.ArrayList([]const u8)) ManifestError!void {
    if (value.len < 2 or value[0] != '[' or value[value.len - 1] != ']') return ManifestError.InvalidFormat;
    const inner = std.mem.trim(u8, value[1 .. value.len - 1], " \t");
    list.clearRetainingCapacity();
    if (inner.len == 0) return;
    var it = std.mem.splitScalar(u8, inner, ',');
    while (it.next()) |segment| {
        const trimmed = std.mem.trim(u8, segment, " \t");
        const parsed = try parseTomlString(trimmed);
        try list.append(try allocator.dupe(u8, parsed));
    }
}

fn coerceJsonFloat(value: std.json.Value) ManifestError!f32 {
    return switch (value) {
        .float => @floatCast(value.float),
        .integer => @floatFromInt(value.integer),
        .number_string => |text| try std.fmt.parseFloat(f32, text),
        else => ManifestError.InvalidFormat,
    };
}

fn coerceJsonInt(comptime T: type, value: std.json.Value) ManifestError!T {
    return switch (value) {
        .integer => @intCast(value.integer),
        .float => |f| @intCast(@as(i64, @intFromFloat(f))),
        .number_string => |text| try std.fmt.parseInt(T, text, 10),
        else => ManifestError.InvalidFormat,
    };
}

fn allocEmptySlice(allocator: std.mem.Allocator) ![]const []const u8 {
    return try allocator.alloc([]const u8, 0);
}

test "load JSON manifest with personas and profiles" {
    const json = \
        "{\n" ++
        "  \"personas\": [\n" ++
        "    {\n" ++
        "      \"name\": \"creative\",\n" ++
        "      \"system_prompt\": \"You are creative\",\n" ++
        "      \"tools\": [\"browser\"],\n" ++
        "      \"safety_filters\": [\"default\"],\n" ++
        "      \"temperature\": 0.9,\n" ++
        "      \"top_p\": 0.85,\n" ++
        "      \"rate_limits\": { \"requests_per_minute\": 120 },\n" ++
        "      \"log_sink\": \"stdout\"\n" ++
        "    }\n" ++
        "  ],\n" ++
        "  \"profiles\": [\n" ++
        "    {\n" ++
        "      \"name\": \"dev\",\n" ++
        "      \"streaming\": true,\n" ++
        "      \"function_calling\": true,\n" ++
        "      \"log_sink\": \"stdout\",\n" ++
        "      \"default_persona\": \"creative\"\n" ++
        "    }\n" ++
        "  ]\n" ++
        "}\n";

    var manifest = try loadFromJson(std.testing.allocator, "personas.json", json);
    defer manifest.deinit();

    try std.testing.expectEqual(@as(usize, 1), manifest.personas.len);
    try std.testing.expectEqual(@as(usize, 1), manifest.environments.len);
    const persona = manifest.findPersona("creative") orelse unreachable;
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), persona.temperature, 0.001);
    const profile = manifest.findEnvironment("dev") orelse unreachable;
    try std.testing.expect(profile.function_calling);
}

test "load TOML manifest with multiple personas" {
    const toml = \
        "[[persona]]\n" ++
        "name = \"analytical\"\n" ++
        "system_prompt = \"You reason carefully\"\n" ++
        "tools = [\"calculator\"]\n" ++
        "safety_filters = [\"default\"]\n" ++
        "temperature = 0.2\n" ++
        "top_p = 0.7\n" ++
        "\n" ++
        "[persona.rate_limits]\n" ++
        "requests_per_minute = 90\n" ++
        "\n" ++
        "[[profile]]\n" ++
        "name = \"prod\"\n" ++
        "streaming = false\n" ++
        "function_calling = true\n" ++
        "log_sink = \"syslog\"\n" ++
        "default_persona = \"analytical\"\n";

    var manifest = try loadFromToml(std.testing.allocator, "personas.toml", toml);
    defer manifest.deinit();

    try std.testing.expectEqual(@as(usize, 1), manifest.personas.len);
    try std.testing.expectEqual(@as(usize, 1), manifest.environments.len);
    const persona = manifest.findPersona("analytical") orelse unreachable;
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), persona.temperature, 0.001);
    try std.testing.expect(!manifest.environments[0].streaming);
}
