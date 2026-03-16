const std = @import("std");
const FileStats = @import("fs.zig").FileStats;
const MatchType = @import("results.zig").MatchType;

pub const AstNode = struct {
    node_type: AstNodeType,
    name: []const u8,
    line_number: usize,
    file_path: []const u8,
    children: std.ArrayListUnmanaged(AstNode),
    metadata: std.StringHashMapUnmanaged([]const u8),
    start_pos: usize,
    end_pos: usize,

    pub fn deinit(self: *AstNode, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        // Note: file_path is borrowed from ParsedFile, don't free here
        for (self.children.items) |*child| {
            child.deinit(allocator);
        }
        self.children.deinit(allocator);
        var iter = self.metadata.valueIterator();
        while (iter.next()) |value| {
            allocator.free(value.*);
        }
        self.metadata.deinit(allocator);
    }
};

pub const AstNodeType = enum {
    function,
    struct_type,
    enum_type,
    union_type,
    interface_type,
    class_type,
    const_decl,
    var_decl,
    type_alias,
    import_decl,
    test_decl,
    comment,
    doc_comment,
    error_decl,
    fn_param,
    fn_return_type,
    block,
    if_stmt,
    while_stmt,
    for_stmt,
    switch_stmt,
    field,
    method,
    property,
    other,
};

pub const ParsedFile = struct {
    allocator: std.mem.Allocator,
    file_path: []const u8,
    file_type: []const u8,
    nodes: std.ArrayListUnmanaged(AstNode),
    imports: std.ArrayListUnmanaged([]const u8),
    exports: std.ArrayListUnmanaged([]const u8),
    functions: std.ArrayListUnmanaged([]const u8),
    types: std.ArrayListUnmanaged([]const u8),
    tests: std.ArrayListUnmanaged([]const u8),
    comments: std.ArrayListUnmanaged(AstNode),
    raw_content: []const u8,

    pub fn deinit(self: *ParsedFile) void {
        const allocator = self.allocator;
        if (self.file_path.len > 0) allocator.free(self.file_path);
        if (self.file_type.len > 0) allocator.free(self.file_type);
        for (self.nodes.items) |*node| {
            node.deinit(allocator);
        }
        self.nodes.deinit(allocator);
        for (self.imports.items) |imp| {
            allocator.free(imp);
        }
        self.imports.deinit(allocator);
        for (self.exports.items) |exp| {
            allocator.free(exp);
        }
        self.exports.deinit(allocator);
        for (self.functions.items) |func| {
            allocator.free(func);
        }
        self.functions.deinit(allocator);
        for (self.types.items) |t| {
            allocator.free(t);
        }
        self.types.deinit(allocator);
        for (self.tests.items) |t| {
            allocator.free(t);
        }
        self.tests.deinit(allocator);
        for (self.comments.items) |*comment| {
            comment.deinit(allocator);
        }
        self.comments.deinit(allocator);
        if (self.raw_content.len > 0) allocator.free(self.raw_content);
    }
};

pub const AstParser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) AstParser {
        return AstParser{ .allocator = allocator };
    }

    pub fn deinit(_: *AstParser) void {}

    pub fn parseFile(self: *AstParser, file_stat: *const FileStats, content: []const u8) !ParsedFile {
        var parsed = ParsedFile{
            .allocator = self.allocator,
            .file_path = try self.allocator.dupe(u8, file_stat.path),
            .file_type = try self.allocator.dupe(u8, self.detectFileType(file_stat.path)),
            .nodes = std.ArrayListUnmanaged(AstNode).empty,
            .imports = std.ArrayListUnmanaged([]const u8).empty,
            .exports = std.ArrayListUnmanaged([]const u8).empty,
            .functions = std.ArrayListUnmanaged([]const u8).empty,
            .types = std.ArrayListUnmanaged([]const u8).empty,
            .tests = std.ArrayListUnmanaged([]const u8).empty,
            .comments = std.ArrayListUnmanaged(AstNode).empty,
            .raw_content = try self.allocator.dupe(u8, content),
        };

        try self.extractAst(content, &parsed);

        return parsed;
    }

    fn detectFileType(_: *AstParser, path: []const u8) []const u8 {
        if (std.mem.endsWith(u8, path, ".zig")) return "zig";
        if (std.mem.endsWith(u8, path, ".rs")) return "rust";
        if (std.mem.endsWith(u8, path, ".ts")) return "typescript";
        if (std.mem.endsWith(u8, path, ".js")) return "javascript";
        if (std.mem.endsWith(u8, path, ".py")) return "python";
        if (std.mem.endsWith(u8, path, ".go")) return "go";
        if (std.mem.endsWith(u8, path, ".c") or std.mem.endsWith(u8, path, ".h")) return "c";
        if (std.mem.endsWith(u8, path, ".cpp") or std.mem.endsWith(u8, path, ".hpp")) return "cpp";
        if (std.mem.endsWith(u8, path, ".md") or std.mem.endsWith(u8, path, ".txt")) return "text";
        return "unknown";
    }

    fn extractAst(self: *AstParser, content: []const u8, parsed: *ParsedFile) !void {
        const file_type = parsed.file_type;

        if (std.mem.eql(u8, file_type, "zig")) {
            try self.parseZigAst(content, parsed);
        } else if (std.mem.eql(u8, file_type, "rust")) {
            try self.parseRustAst(content, parsed);
        } else if (std.mem.eql(u8, file_type, "typescript") or std.mem.eql(u8, file_type, "javascript")) {
            try self.parseTypeScriptAst(content, parsed);
        } else {
            try self.parseGenericAst(content, parsed);
        }
    }

    fn parseZigAst(self: *AstParser, content: []const u8, parsed: *ParsedFile) !void {
        var line_number: usize = 1;
        var line_start: usize = 0;

        for (content, 0..) |c, i| {
            if (c == '\n') {
                const line = content[line_start..i];

                if (std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "pub fn") or
                    std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "fn "))
                {
                    const fn_name = try self.extractZigFunctionName(line);
                    if (fn_name.len > 0) {
                        const fn_node = try self.createNode(parsed, .function, fn_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, fn_node);
                        try parsed.functions.append(self.allocator, try self.allocator.dupe(u8, fn_name));
                    }
                }

                if (std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "pub const") or
                    std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "const "))
                {
                    const const_name = try self.extractZigConstName(line);
                    if (const_name.len > 0) {
                        const node = try self.createNode(parsed, .const_decl, const_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, node);
                    }
                }

                if (std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "pub struct") or
                    std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "struct "))
                {
                    const type_name = try self.extractZigTypeName(line);
                    if (type_name.len > 0) {
                        const node = try self.createNode(parsed, .struct_type, type_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, node);
                        try parsed.types.append(self.allocator, try self.allocator.dupe(u8, type_name));
                    }
                }

                if (std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "pub enum") or
                    std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "enum "))
                {
                    const type_name = try self.extractZigTypeName(line);
                    if (type_name.len > 0) {
                        const node = try self.createNode(parsed, .enum_type, type_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, node);
                        try parsed.types.append(self.allocator, try self.allocator.dupe(u8, type_name));
                    }
                }

                if (std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "pub const") and std.mem.indexOf(u8, line, " = @import") != null) {
                    const imp = try self.extractImportPath(line);
                    if (imp.len > 0) {
                        try parsed.imports.append(self.allocator, try self.allocator.dupe(u8, imp));
                    }
                }

                if (std.mem.startsWith(u8, std.mem.trim(u8, line, " \t"), "test ")) {
                    const test_name = try self.extractTestName(line);
                    if (test_name.len > 0) {
                        try parsed.tests.append(self.allocator, try self.allocator.dupe(u8, test_name));
                    }
                }

                if (std.mem.startsWith(u8, line, "///")) {
                    const doc_node = try self.createNode(parsed, .doc_comment, std.mem.trim(u8, line[3..], " \t"), line_number, line_start, i);
                    try parsed.comments.append(self.allocator, doc_node);
                } else if (std.mem.startsWith(u8, line, "//")) {
                    const comment_node = try self.createNode(parsed, .comment, std.mem.trim(u8, line[2..], " \t"), line_number, line_start, i);
                    try parsed.comments.append(self.allocator, comment_node);
                }

                line_start = i + 1;
                line_number += 1;
            }
        }
    }

    fn parseRustAst(self: *AstParser, content: []const u8, parsed: *ParsedFile) !void {
        var line_number: usize = 1;
        var line_start: usize = 0;

        for (content, 0..) |c, i| {
            if (c == '\n') {
                const line = content[line_start..i];
                const trimmed = std.mem.trim(u8, line, " \t");

                if (std.mem.startsWith(u8, trimmed, "pub fn") or std.mem.startsWith(u8, trimmed, "fn ")) {
                    const fn_name = try self.extractRustFunctionName(trimmed);
                    if (fn_name.len > 0) {
                        const fn_node = try self.createNode(parsed, .function, fn_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, fn_node);
                        try parsed.functions.append(self.allocator, try self.allocator.dupe(u8, fn_name));
                    }
                }

                if (std.mem.startsWith(u8, trimmed, "pub struct") or std.mem.startsWith(u8, trimmed, "struct ")) {
                    const type_name = try self.extractRustTypeName(trimmed);
                    if (type_name.len > 0) {
                        const node = try self.createNode(parsed, .struct_type, type_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, node);
                        try parsed.types.append(self.allocator, try self.allocator.dupe(u8, type_name));
                    }
                }

                if (std.mem.startsWith(u8, trimmed, "pub enum") or std.mem.startsWith(u8, trimmed, "enum ")) {
                    const type_name = try self.extractRustTypeName(trimmed);
                    if (type_name.len > 0) {
                        const node = try self.createNode(parsed, .enum_type, type_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, node);
                        try parsed.types.append(self.allocator, try self.allocator.dupe(u8, type_name));
                    }
                }

                if (std.mem.startsWith(u8, trimmed, "use ")) {
                    const imp = try self.extractRustImport(trimmed);
                    if (imp.len > 0) {
                        try parsed.imports.append(self.allocator, try self.allocator.dupe(u8, imp));
                    }
                }

                if (std.mem.startsWith(u8, trimmed, "#[test]")) {
                    const test_line = content[line_start + 7 .. i];
                    if (std.mem.startsWith(u8, std.mem.trim(u8, test_line, " \t"), "fn ")) {
                        const test_name = try self.extractRustFunctionName(std.mem.trim(u8, test_line, " \t"));
                        if (test_name.len > 0) {
                            try parsed.tests.append(self.allocator, try self.allocator.dupe(u8, test_name));
                        }
                    }
                }

                line_start = i + 1;
                line_number += 1;
            }
        }
    }

    fn parseTypeScriptAst(self: *AstParser, content: []const u8, parsed: *ParsedFile) !void {
        var line_number: usize = 1;
        var line_start: usize = 0;

        for (content, 0..) |c, i| {
            if (c == '\n') {
                const line = content[line_start..i];
                const trimmed = std.mem.trim(u8, line, " \t");

                if (std.mem.startsWith(u8, trimmed, "export function") or
                    std.mem.startsWith(u8, trimmed, "export const") or
                    std.mem.startsWith(u8, trimmed, "function ") or
                    std.mem.startsWith(u8, trimmed, "const "))
                {
                    const fn_name = try self.extractTsFunctionName(trimmed);
                    if (fn_name.len > 0) {
                        const fn_node = try self.createNode(parsed, .function, fn_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, fn_node);
                        try parsed.functions.append(self.allocator, try self.allocator.dupe(u8, fn_name));
                    }
                }

                if (std.mem.startsWith(u8, trimmed, "export class") or
                    std.mem.startsWith(u8, trimmed, "class "))
                {
                    const type_name = try self.extractTsTypeName(trimmed);
                    if (type_name.len > 0) {
                        const node = try self.createNode(parsed, .class_type, type_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, node);
                        try parsed.types.append(self.allocator, try self.allocator.dupe(u8, type_name));
                    }
                }

                if (std.mem.startsWith(u8, trimmed, "export interface") or
                    std.mem.startsWith(u8, trimmed, "interface "))
                {
                    const type_name = try self.extractTsTypeName(trimmed);
                    if (type_name.len > 0) {
                        const node = try self.createNode(parsed, .interface_type, type_name, line_number, line_start, i);
                        try parsed.nodes.append(self.allocator, node);
                        try parsed.types.append(self.allocator, try self.allocator.dupe(u8, type_name));
                    }
                }

                if (std.mem.startsWith(u8, trimmed, "import ")) {
                    const imp = try self.extractTsImport(trimmed);
                    if (imp.len > 0) {
                        try parsed.imports.append(self.allocator, try self.allocator.dupe(u8, imp));
                    }
                }

                line_start = i + 1;
                line_number += 1;
            }
        }
    }

    fn parseGenericAst(self: *AstParser, content: []const u8, parsed: *ParsedFile) !void {
        var line_number: usize = 1;
        var line_start: usize = 0;

        for (content, 0..) |c, i| {
            if (c == '\n') {
                const line = content[line_start..i];
                const trimmed = std.mem.trim(u8, line, " \t");

                if (std.mem.indexOf(u8, trimmed, "function") != null or
                    std.mem.indexOf(u8, trimmed, "def ") != null or
                    std.mem.indexOf(u8, trimmed, "func ") != null)
                {
                    const node = try self.createNode(parsed, .function, trimmed, line_number, line_start, i);
                    try parsed.nodes.append(self.allocator, node);
                }

                if (std.mem.startsWith(u8, trimmed, "//") or std.mem.startsWith(u8, trimmed, "#")) {
                    const comment_node = try self.createNode(parsed, .comment, trimmed[2..], line_number, line_start, i);
                    try parsed.comments.append(self.allocator, comment_node);
                }

                line_start = i + 1;
                line_number += 1;
            }
        }
    }

    fn createNode(self: *AstParser, parsed: *ParsedFile, node_type: AstNodeType, name: []const u8, line_number: usize, start_pos: usize, end_pos: usize) !AstNode {
        return AstNode{
            .node_type = node_type,
            .name = try self.allocator.dupe(u8, name),
            .line_number = line_number,
            .file_path = parsed.file_path,
            .children = std.ArrayListUnmanaged(AstNode).empty,
            .metadata = std.StringHashMapUnmanaged([]const u8){},
            .start_pos = start_pos,
            .end_pos = end_pos,
        };
    }

    fn extractZigFunctionName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        const trimmed = std.mem.trim(u8, line, " \t");
        const idx: usize = if (std.mem.startsWith(u8, trimmed, "pub fn "))
            7
        else if (std.mem.startsWith(u8, trimmed, "fn "))
            3
        else
            return "";

        var end = idx;
        while (end < trimmed.len and (std.ascii.isAlphanumeric(trimmed[end]) or trimmed[end] == '_')) {
            end += 1;
        }

        return std.mem.trim(u8, trimmed[idx..end], " \t");
    }

    fn extractZigConstName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        const trimmed = std.mem.trim(u8, line, " \t");
        const idx: usize = if (std.mem.startsWith(u8, trimmed, "pub const ")) 10 else 6;

        var end = idx;
        while (end < trimmed.len and (std.ascii.isAlphanumeric(trimmed[end]) or trimmed[end] == '_')) {
            end += 1;
        }

        return std.mem.trim(u8, trimmed[idx..end], " \t");
    }

    fn extractZigTypeName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        const trimmed = std.mem.trim(u8, line, " \t");
        const idx: usize = if (std.mem.startsWith(u8, trimmed, "pub struct "))
            11
        else if (std.mem.startsWith(u8, trimmed, "struct "))
            7
        else if (std.mem.startsWith(u8, trimmed, "pub enum "))
            9
        else if (std.mem.startsWith(u8, trimmed, "enum "))
            5
        else
            return "";

        var end = idx;
        while (end < trimmed.len and (std.ascii.isAlphanumeric(trimmed[end]) or trimmed[end] == '_')) {
            end += 1;
        }

        return std.mem.trim(u8, trimmed[idx..end], " \t");
    }

    fn extractImportPath(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        if (std.mem.indexOf(u8, line, "= @import(\"")) |start| {
            const begin = start + 10;
            if (std.mem.indexOf(u8, line[begin..], "\"")) |end| {
                return line[begin .. begin + end];
            }
        }
        return "";
    }

    fn extractTestName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        const trimmed = std.mem.trim(u8, line, " \t");
        if (std.mem.startsWith(u8, trimmed, "test \"")) {
            const begin = 6;
            if (std.mem.indexOf(u8, trimmed[begin..], "\"")) |end| {
                return trimmed[begin .. begin + end];
            }
        }
        const idx: usize = 5;
        var end = idx;
        while (end < trimmed.len and (std.ascii.isAlphanumeric(trimmed[end]) or trimmed[end] == '_')) {
            end += 1;
        }
        return trimmed[idx..end];
    }

    fn extractRustFunctionName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        var idx: usize = if (std.mem.startsWith(u8, line, "pub fn ")) 7 else 3;

        while (idx < line.len and line[idx] == ' ') idx += 1;

        var end = idx;
        while (end < line.len and (std.ascii.isAlphanumeric(line[end]) or line[end] == '_')) {
            end += 1;
        }

        return std.mem.trim(u8, line[idx..end], " \t");
    }

    fn extractRustTypeName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        var idx: usize = if (std.mem.startsWith(u8, line, "pub struct "))
            11
        else if (std.mem.startsWith(u8, line, "struct "))
            7
        else if (std.mem.startsWith(u8, line, "pub enum "))
            9
        else if (std.mem.startsWith(u8, line, "enum "))
            5
        else
            return "";

        while (idx < line.len and line[idx] == ' ') idx += 1;

        var end = idx;
        while (end < line.len and (std.ascii.isAlphanumeric(line[end]) or line[end] == '_')) {
            end += 1;
        }

        return std.mem.trim(u8, line[idx..end], " \t");
    }

    fn extractRustImport(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        if (std.mem.startsWith(u8, line, "use ")) {
            var end = line.len;
            if (std.mem.indexOf(u8, line, ";") != null) {
                end = std.mem.indexOf(u8, line, ";").?;
            }
            var start: usize = 4;
            while (start < end and line[start] == ' ') start += 1;
            return line[start..end];
        }
        return "";
    }

    fn extractTsFunctionName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        var idx: usize = if (std.mem.startsWith(u8, line, "export function "))
            16
        else if (std.mem.startsWith(u8, line, "export const "))
            13
        else if (std.mem.startsWith(u8, line, "function "))
            9
        else if (std.mem.startsWith(u8, line, "const "))
            6
        else
            return "";

        while (idx < line.len and line[idx] == ' ') idx += 1;

        var end = idx;
        while (end < line.len and (std.ascii.isAlphanumeric(line[end]) or line[end] == '_')) {
            end += 1;
        }

        return std.mem.trim(u8, line[idx..end], " \t");
    }

    fn extractTsTypeName(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        var idx: usize = if (std.mem.startsWith(u8, line, "export class "))
            13
        else if (std.mem.startsWith(u8, line, "class "))
            6
        else if (std.mem.startsWith(u8, line, "export interface "))
            17
        else if (std.mem.startsWith(u8, line, "interface "))
            10
        else
            return "";

        while (idx < line.len and line[idx] == ' ') idx += 1;

        var end = idx;
        while (end < line.len and (std.ascii.isAlphanumeric(line[end]) or line[end] == '_')) {
            end += 1;
        }

        return std.mem.trim(u8, line[idx..end], " \t");
    }

    fn extractTsImport(self: *AstParser, line: []const u8) ![]const u8 {
        _ = self;
        if (std.mem.startsWith(u8, line, "import ")) {
            var start: usize = 7;
            while (start < line.len and line[start] == ' ') start += 1;

            if (std.mem.startsWith(u8, line[start..], "{")) {
                if (std.mem.indexOf(u8, line, "} from \"")) |end| {
                    const name_end = std.mem.indexOf(u8, line[end + 7 ..], "\"") orelse line.len - (end + 7);
                    return line[end + 7 .. end + 7 + name_end];
                }
            } else {
                var end = start;
                while (end < line.len and (std.ascii.isAlphanumeric(line[end]) or line[end] == '_' or line[end] == '.' or line[end] == '/')) {
                    end += 1;
                }
                if (std.mem.indexOf(u8, line[start..end], "from \"") != null) {
                    const from_idx = std.mem.indexOf(u8, line, "from \"").? + 6;
                    const name_end = std.mem.indexOf(u8, line[from_idx..], "\"") orelse line.len - from_idx;
                    return line[from_idx .. from_idx + name_end];
                }
                return line[start..end];
            }
        }
        return "";
    }

    pub fn nodeToMatchType(node_type: AstNodeType) MatchType {
        return switch (node_type) {
            .function => .function_definition,
            .struct_type, .enum_type, .union_type, .interface_type, .class_type => .type_definition,
            .const_decl, .var_decl => .variable_declaration,
            .import_decl => .import_statement,
            .test_decl => .test_case,
            .comment, .doc_comment => .comment,
            else => .custom,
        };
    }
};

test {
    std.testing.refAllDecls(@This());
}
