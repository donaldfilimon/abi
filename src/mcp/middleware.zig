//! Centralized security validation for MCP `tools/call` arguments.
//!
//! Every tool dispatch passes through `validateArguments` before the tool body
//! runs (see `handlers.handleToolsCallJson`). Validation policy is declared
//! per tool as a slice of `FieldSpec` on each `ToolDescriptor`, so the rules
//! live next to the advertised input schema instead of being scattered through
//! the individual handlers.
//!
//! This module depends only on `std` (no cross-feature module imports) so it
//! can sit on the request hot path without crossing the module boundary.

const std = @import("std");

/// Upper bound applied to every string field unless a descriptor overrides it.
/// The transport already caps a whole request at 64 KB (`protocol.MAX_REQUEST_SIZE`);
/// per-field caps stop a single field from consuming the entire budget.
pub const DEFAULT_MAX_FIELD_LEN: usize = 16 * 1024;

/// How a string field is interpreted for validation purposes.
pub const FieldKind = enum {
    /// Arbitrary text (prompts, queries). Null bytes and over-length rejected.
    free_text,
    /// A short identifier (profile, plugin name). Same checks as free_text today,
    /// kept distinct so identifier-specific rules can tighten later.
    identifier,
    /// A filesystem path. Adds path-traversal rejection on top of free_text.
    file_path,
    /// A value constrained to `choices`.
    enum_choice,
};

/// Declarative validation rule for one argument of a tool.
pub const FieldSpec = struct {
    name: []const u8,
    required: bool,
    kind: FieldKind = .free_text,
    /// Error returned when a required field is missing or has the wrong JSON
    /// type. Mirrors the handler's historical per-field error so the frozen MCP
    /// error contract is preserved.
    missing_error: anyerror,
    /// Error returned when an `enum_choice` value is not one of `choices`.
    invalid_error: anyerror = error.InvalidFieldValue,
    max_len: usize = DEFAULT_MAX_FIELD_LEN,
    choices: []const []const u8 = &.{},
};

/// Validates the `arguments` object of a `tools/call` request against `fields`.
/// Returns a normalized error on the first violation; returns cleanly when the
/// tool declares no fields (status / no-argument tools, which never read args).
pub fn validateArguments(fields: []const FieldSpec, params_obj: std.json.ObjectMap) !void {
    if (fields.len == 0) return;

    const args_val = params_obj.get("arguments") orelse return error.MissingArguments;
    const args = switch (args_val) {
        .object => |obj| obj,
        else => return error.MissingArguments,
    };

    for (fields) |field| {
        const value = args.get(field.name) orelse {
            if (field.required) return field.missing_error;
            continue;
        };
        const s = switch (value) {
            .string => |str| str,
            else => return field.missing_error,
        };
        try validateString(field, s);
    }
}

fn validateString(field: FieldSpec, s: []const u8) !void {
    if (std.mem.indexOfScalar(u8, s, 0) != null) return error.InvalidFieldEncoding;
    if (s.len > field.max_len) return error.FieldTooLong;
    switch (field.kind) {
        .free_text, .identifier => {},
        .file_path => try validatePath(s),
        .enum_choice => {
            for (field.choices) |choice| {
                if (std.mem.eql(u8, s, choice)) return;
            }
            return field.invalid_error;
        },
    }
}

/// Rejects path traversal. Absolute paths are intentionally allowed — a local,
/// single-user CLI server legitimately reads absolute dataset paths — so only
/// `..` components, which could escape an intended directory, are blocked.
fn validatePath(path: []const u8) !void {
    var it = std.mem.splitAny(u8, path, "/\\");
    while (it.next()) |segment| {
        if (std.mem.eql(u8, segment, "..")) return error.PathTraversal;
    }
}

// --- Tests ---

fn argsObject(allocator: std.mem.Allocator, json_text: []const u8) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, json_text, .{});
}

test "validateArguments passes valid free text and enum" {
    const fields = [_]FieldSpec{
        .{ .name = "service", .required = true, .kind = .enum_choice, .missing_error = error.MissingConnectorService, .invalid_error = error.UnknownConnector, .choices = &.{ "openai", "grok" } },
        .{ .name = "input", .required = true, .missing_error = error.MissingInput },
    };
    var parsed = try argsObject(std.testing.allocator,
        \\{"arguments":{"service":"openai","input":"hello"}}
    );
    defer parsed.deinit();
    try validateArguments(&fields, parsed.value.object);
}

test "validateArguments empty fields needs no arguments" {
    var parsed = try argsObject(std.testing.allocator, "{}");
    defer parsed.deinit();
    try validateArguments(&.{}, parsed.value.object);
}

test "validateArguments reports the field's missing error" {
    const fields = [_]FieldSpec{.{ .name = "input", .required = true, .missing_error = error.MissingInput }};
    var parsed = try argsObject(std.testing.allocator,
        \\{"arguments":{}}
    );
    defer parsed.deinit();
    try std.testing.expectError(error.MissingInput, validateArguments(&fields, parsed.value.object));
}

test "validateArguments requires an arguments object when fields exist" {
    const fields = [_]FieldSpec{.{ .name = "input", .required = true, .missing_error = error.MissingInput }};
    var parsed = try argsObject(std.testing.allocator,
        \\{"name":"ai_run"}
    );
    defer parsed.deinit();
    try std.testing.expectError(error.MissingArguments, validateArguments(&fields, parsed.value.object));
}

test "validateArguments rejects unknown enum value" {
    const fields = [_]FieldSpec{.{ .name = "service", .required = true, .kind = .enum_choice, .missing_error = error.MissingConnectorService, .invalid_error = error.UnknownConnector, .choices = &.{ "openai", "grok" } }};
    var parsed = try argsObject(std.testing.allocator,
        \\{"arguments":{"service":"pirate"}}
    );
    defer parsed.deinit();
    try std.testing.expectError(error.UnknownConnector, validateArguments(&fields, parsed.value.object));
}

test "validateArguments rejects null bytes" {
    const fields = [_]FieldSpec{.{ .name = "input", .required = true, .missing_error = error.MissingInput }};
    var parsed = try argsObject(std.testing.allocator,
        \\{"arguments":{"input":"ab\u0000cd"}}
    );
    defer parsed.deinit();
    try std.testing.expectError(error.InvalidFieldEncoding, validateArguments(&fields, parsed.value.object));
}

test "validateArguments rejects over-length field" {
    const fields = [_]FieldSpec{.{ .name = "input", .required = true, .missing_error = error.MissingInput, .max_len = 4 }};
    var parsed = try argsObject(std.testing.allocator,
        \\{"arguments":{"input":"toolong"}}
    );
    defer parsed.deinit();
    try std.testing.expectError(error.FieldTooLong, validateArguments(&fields, parsed.value.object));
}

test "validateArguments rejects path traversal but allows plain and absolute paths" {
    const fields = [_]FieldSpec{.{ .name = "dataset", .required = true, .kind = .file_path, .missing_error = error.MissingDataset }};

    var traversal = try argsObject(std.testing.allocator,
        \\{"arguments":{"dataset":"../../etc/passwd"}}
    );
    defer traversal.deinit();
    try std.testing.expectError(error.PathTraversal, validateArguments(&fields, traversal.value.object));

    var plain = try argsObject(std.testing.allocator,
        \\{"arguments":{"dataset":"data/train.jsonl"}}
    );
    defer plain.deinit();
    try validateArguments(&fields, plain.value.object);

    var absolute = try argsObject(std.testing.allocator,
        \\{"arguments":{"dataset":"/tmp/train.jsonl"}}
    );
    defer absolute.deinit();
    try validateArguments(&fields, absolute.value.object);
}

test "validateArguments skips absent optional fields" {
    const fields = [_]FieldSpec{
        .{ .name = "name", .required = true, .kind = .identifier, .missing_error = error.MissingPluginName },
        .{ .name = "input", .required = false, .missing_error = error.MissingInput },
    };
    var parsed = try argsObject(std.testing.allocator,
        \\{"arguments":{"name":"example-plugin"}}
    );
    defer parsed.deinit();
    try validateArguments(&fields, parsed.value.object);
}

test {
    std.testing.refAllDecls(@This());
}
