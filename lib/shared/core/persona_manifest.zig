//! Persona manifest loader for ABI runtime profiles.
//!
//! Loads persona definitions from JSON manifests so that persona behaviour can
//! be reproduced across environments. The loader keeps all manifest state in an
//! arena allocator which is released when `deinit` is called.

const std = @import("std");

pub const ManifestError = error{
    InvalidManifest,
    IoFailure,
};

/// Runtime representation of a persona manifest.
pub const PersonaManifest = struct {
    arena: std.heap.ArenaAllocator,
    data: ManifestData,

    pub fn deinit(self: *PersonaManifest) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const ManifestData = struct {
    version: u32 = 1,
    defaults: PersonaDefaults,
    personas: []const PersonaDefinition,
};

pub const PersonaDefaults = struct {
    profile: []const u8 = "dev",
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    tools: []const []const u8 = &[_][]const u8{},
    safety: SafetyConfig = .{},
    rate_limits: RateLimitConfig = .{},
};

pub const PersonaDefinition = struct {
    id: []const u8,
    display_name: []const u8,
    description: []const u8,
    system_prompt: []const u8,
    model: ModelConfig,
    tools: []const []const u8,
    safety: SafetyConfig,
    rate_limits: RateLimitConfig,
};

pub const ModelConfig = struct {
    provider: []const u8,
    name: []const u8,
    temperature: f32,
    top_p: f32,
    streaming: bool,
};

pub const SafetyConfig = struct {
    allow_code_execution: bool = false,
    allow_network: bool = false,
    allow_file_system: bool = false,
};

pub const RateLimitConfig = struct {
    requests_per_minute: u32 = 60,
    tokens_per_minute: u32 = 60000,
};

const ManifestJson = struct {
    version: u32 = 1,
    defaults: PersonaDefaultsJson = .{},
    personas: []PersonaDefinitionJson,
};

const PersonaDefaultsJson = struct {
    profile: []const u8 = "dev",
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    tools: []const []const u8 = &[_][]const u8{},
    safety: SafetyConfig = .{},
    rate_limits: RateLimitConfig = .{},
};

const PersonaDefinitionJson = struct {
    id: []const u8,
    display_name: []const u8,
    description: []const u8,
    system_prompt: []const u8,
    model: ModelConfig,
    tools: []const []const u8 = &[_][]const u8{},
    safety: SafetyConfig = .{},
    rate_limits: RateLimitConfig = .{},
};

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) ManifestError!PersonaManifest {
    var file = std.fs.cwd().openFile(path, .{}) catch |err| {
        std.log.err("failed to open persona manifest {s}: {s}", .{ path, @errorName(err) });
        return ManifestError.IoFailure;
    };
    defer file.close();

    const stat = file.stat() catch |err| {
        std.log.err("failed to stat persona manifest {s}: {s}", .{ path, @errorName(err) });
        return ManifestError.IoFailure;
    };

    if (stat.size > 10 * 1024 * 1024) {
        std.log.err("persona manifest too large: {} bytes", .{stat.size});
        return ManifestError.InvalidManifest;
    }

    const bytes = file.readToEndAlloc(allocator, stat.size) catch |err| {
        std.log.err("failed to read persona manifest {s}: {s}", .{ path, @errorName(err) });
        return ManifestError.IoFailure;
    };
    defer allocator.free(bytes);

    return loadFromBytes(allocator, bytes);
}

pub fn loadFromBytes(allocator: std.mem.Allocator, bytes: []const u8) ManifestError!PersonaManifest {
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();

    const arena_alloc = arena.allocator();

    var parsed = std.json.parseFromSlice(ManifestJson, arena_alloc, bytes, .{ .ignore_unknown_fields = false }) catch |err| {
        std.log.err("failed to parse persona manifest: {s}", .{@errorName(err)});
        arena.deinit();
        return ManifestError.InvalidManifest;
    };
    defer parsed.deinit();

    const manifest_json = parsed.value;
    const data = buildManifestData(arena.allocator(), manifest_json) catch |err| {
        std.log.err("persona manifest validation failed: {s}", .{@errorName(err)});
        arena.deinit();
        return ManifestError.InvalidManifest;
    };

    return PersonaManifest{ .arena = arena, .data = data };
}

fn buildManifestData(allocator: std.mem.Allocator, json: ManifestJson) !ManifestData {
    const defaults_value = try copyDefaults(allocator, json.defaults);

    var personas = try allocator.alloc(PersonaDefinition, json.personas.len);
    for (json.personas, 0..) |persona_json, idx| {
        personas[idx] = try copyPersona(allocator, persona_json);
    }

    return ManifestData{
        .version = json.version,
        .defaults = defaults_value,
        .personas = personas,
    };
}

fn copyDefaults(allocator: std.mem.Allocator, persona_defaults: PersonaDefaultsJson) !PersonaDefaults {
    var tools = try allocator.alloc([]const u8, persona_defaults.tools.len);
    for (persona_defaults.tools, 0..) |tool, idx| {
        tools[idx] = try std.mem.dupe(allocator, u8, tool);
    }

    return PersonaDefaults{
        .profile = try std.mem.dupe(allocator, u8, defaults.profile),
        .temperature = defaults.temperature,
        .top_p = defaults.top_p,
        .tools = tools,
        .safety = defaults.safety,
        .rate_limits = defaults.rate_limits,
    };
}

fn copyPersona(allocator: std.mem.Allocator, persona: PersonaDefinitionJson) !PersonaDefinition {
    var tools = try allocator.alloc([]const u8, persona.tools.len);
    for (persona.tools, 0..) |tool, idx| {
        tools[idx] = try std.mem.dupe(allocator, u8, tool);
    }

    return PersonaDefinition{
        .id = try std.mem.dupe(allocator, u8, persona.id),
        .display_name = try std.mem.dupe(allocator, u8, persona.display_name),
        .description = try std.mem.dupe(allocator, u8, persona.description),
        .system_prompt = try std.mem.dupe(allocator, u8, persona.system_prompt),
        .model = .{
            .provider = try std.mem.dupe(allocator, u8, persona.model.provider),
            .name = try std.mem.dupe(allocator, u8, persona.model.name),
            .temperature = persona.model.temperature,
            .top_p = persona.model.top_p,
            .streaming = persona.model.streaming,
        },
        .tools = tools,
        .safety = persona.safety,
        .rate_limits = persona.rate_limits,
    };
}

pub fn findPersona(manifest: *const PersonaManifest, id: []const u8) ?PersonaDefinition {
    for (manifest.data.personas) |persona| {
        if (std.mem.eql(u8, persona.id, id)) {
            return persona;
        }
    }
    return null;
}

pub fn iter(manifest: *const PersonaManifest) []const PersonaDefinition {
    return manifest.data.personas;
}

pub fn defaults(manifest: *const PersonaManifest) PersonaDefaults {
    return manifest.data.defaults;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

test "manifest loads personas" {
    const allocator = std.testing.allocator;
    const json_bytes = "{\n" ++
        "  \"version\": 1,\n" ++
        "  \"defaults\": {\n" ++
        "    \"profile\": \"testing\",\n" ++
        "    \"temperature\": 0.5,\n" ++
        "    \"top_p\": 0.8,\n" ++
        "    \"tools\": [\"code\"],\n" ++
        "    \"safety\": {\n" ++
        "      \"allow_code_execution\": true,\n" ++
        "      \"allow_network\": false,\n" ++
        "      \"allow_file_system\": false\n" ++
        "    },\n" ++
        "    \"rate_limits\": {\n" ++
        "      \"requests_per_minute\": 30,\n" ++
        "      \"tokens_per_minute\": 1000\n" ++
        "    }\n" ++
        "  },\n" ++
        "  \"personas\": [\n" ++
        "    {\n" ++
        "      \"id\": \"testing\",\n" ++
        "      \"display_name\": \"Test Persona\",\n" ++
        "      \"description\": \"desc\",\n" ++
        "      \"system_prompt\": \"prompt\",\n" ++
        "      \"model\": {\n" ++
        "        \"provider\": \"mock\",\n" ++
        "        \"name\": \"mock-model\",\n" ++
        "        \"temperature\": 0.4,\n" ++
        "        \"top_p\": 0.7,\n" ++
        "        \"streaming\": false\n" ++
        "      },\n" ++
        "      \"tools\": [\"code\", \"math\"],\n" ++
        "      \"safety\": {\n" ++
        "        \"allow_code_execution\": true,\n" ++
        "        \"allow_network\": false,\n" ++
        "        \"allow_file_system\": false\n" ++
        "      },\n" ++
        "      \"rate_limits\": {\n" ++
        "        \"requests_per_minute\": 60,\n" ++
        "        \"tokens_per_minute\": 5000\n" ++
        "      }\n" ++
        "    }\n" ++
        "  ]\n" ++
        "}";

    const manifest = try loadFromBytes(allocator, json_bytes);
    defer manifest.deinit();

    try std.testing.expectEqual(@as(u32, 1), manifest.data.version);
    try std.testing.expectEqualStrings("testing", manifest.data.defaults.profile);
    try std.testing.expectEqual(@as(usize, 1), manifest.data.personas.len);

    const persona = manifest.data.personas[0];
    try std.testing.expectEqualStrings("testing", persona.id);
    try std.testing.expectEqual(@as(usize, 2), persona.tools.len);
}
