//! Aviva Code Generation Module
//!
//! Handles code generation, formatting, and validation for Aviva.
//! Provides clean, well-structured code output with optional explanations.
//!
//! Features:
//! - Language detection and syntax awareness
//! - Code formatting and structure
//! - Optional comment generation
//! - Code block construction

const std = @import("std");
const classifier = @import("classifier.zig");

/// Generated code block.
pub const CodeBlock = struct {
    /// The programming language.
    language: classifier.Language,
    /// The generated code.
    code: []const u8,
    /// Optional explanation.
    explanation: ?[]const u8 = null,
    /// Filename suggestion if applicable.
    filename: ?[]const u8 = null,
    /// Whether this is a complete file or snippet.
    is_complete_file: bool = false,
    /// Lines of code (excluding blank lines and comments).
    loc: usize = 0,
};

/// Code generation options.
pub const GenerationOptions = struct {
    /// Include inline comments.
    include_comments: bool = true,
    /// Include docstrings/documentation.
    include_docs: bool = true,
    /// Include type annotations (where applicable).
    include_types: bool = true,
    /// Include error handling.
    include_error_handling: bool = true,
    /// Prefer verbose variable names.
    verbose_names: bool = false,
    /// Maximum line length.
    max_line_length: usize = 100,
    /// Indentation style.
    indent_style: IndentStyle = .spaces_4,
};

/// Indentation style options.
pub const IndentStyle = enum {
    tabs,
    spaces_2,
    spaces_4,

    pub fn getString(self: IndentStyle) []const u8 {
        return switch (self) {
            .tabs => "\t",
            .spaces_2 => "  ",
            .spaces_4 => "    ",
        };
    }
};

/// Configuration for the code generator.
pub const GeneratorConfig = struct {
    /// Default generation options.
    default_options: GenerationOptions = .{},
    /// Whether to validate syntax (basic checks).
    validate_syntax: bool = true,
    /// Whether to add language markers to output.
    add_language_markers: bool = true,
};

/// Code generator for Aviva.
pub const CodeGenerator = struct {
    allocator: std.mem.Allocator,
    config: GeneratorConfig,

    const Self = @This();

    /// Initialize the code generator.
    pub fn init(allocator: std.mem.Allocator) Self {
        return initWithConfig(allocator, .{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(allocator: std.mem.Allocator, config: GeneratorConfig) Self {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Format raw code into a proper code block.
    pub fn formatCodeBlock(
        self: *Self,
        code: []const u8,
        language: classifier.Language,
        options: ?GenerationOptions,
    ) !CodeBlock {
        const opts = options orelse self.config.default_options;

        // Apply formatting
        const formatted = try self.applyFormatting(code, language, opts);

        // Count lines of code
        const loc = self.countLoc(formatted);

        return .{
            .language = language,
            .code = formatted,
            .loc = loc,
        };
    }

    /// Apply code formatting rules.
    fn applyFormatting(
        self: *Self,
        code: []const u8,
        language: classifier.Language,
        options: GenerationOptions,
    ) ![]const u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(self.allocator);

        // Process line by line
        var lines = std.mem.splitScalar(u8, code, '\n');
        var first = true;

        while (lines.next()) |line| {
            if (!first) {
                try result.append(self.allocator, '\n');
            }
            first = false;

            // Trim trailing whitespace
            const trimmed = std.mem.trimRight(u8, line, " \t\r");

            // Apply line length limit if needed
            if (options.max_line_length > 0 and trimmed.len > options.max_line_length) {
                // For now, just truncate - in production would wrap intelligently
                try result.appendSlice(self.allocator, trimmed[0..options.max_line_length]);
            } else {
                try result.appendSlice(self.allocator, trimmed);
            }
        }

        // Ensure trailing newline
        if (result.items.len > 0 and result.items[result.items.len - 1] != '\n') {
            try result.append(self.allocator, '\n');
        }

        _ = language; // Would use for language-specific formatting

        return result.toOwnedSlice(self.allocator);
    }

    /// Count lines of code (excluding blanks and comments).
    fn countLoc(self: *const Self, code: []const u8) usize {
        _ = self;
        var count: usize = 0;
        var lines = std.mem.splitScalar(u8, code, '\n');

        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0) continue;
            if (std.mem.startsWith(u8, trimmed, "//")) continue;
            if (std.mem.startsWith(u8, trimmed, "#")) continue;
            if (std.mem.startsWith(u8, trimmed, "--")) continue;
            count += 1;
        }

        return count;
    }

    /// Generate a function template.
    pub fn generateFunctionTemplate(
        self: *Self,
        name: []const u8,
        params: []const []const u8,
        return_type: ?[]const u8,
        language: classifier.Language,
        options: ?GenerationOptions,
    ) !CodeBlock {
        const opts = options orelse self.config.default_options;

        var code: std.ArrayListUnmanaged(u8) = .{};
        errdefer code.deinit(self.allocator);

        const indent = opts.indent_style.getString();

        switch (language) {
            .zig => {
                // Generate Zig function
                if (opts.include_docs) {
                    try code.appendSlice(self.allocator, "/// TODO: Add documentation\n");
                }
                try code.appendSlice(self.allocator, "pub fn ");
                try code.appendSlice(self.allocator, name);
                try code.append(self.allocator, '(');
                for (params, 0..) |param, i| {
                    if (i > 0) try code.appendSlice(self.allocator, ", ");
                    try code.appendSlice(self.allocator, param);
                }
                try code.append(self.allocator, ')');
                if (return_type) |rt| {
                    try code.append(self.allocator, ' ');
                    try code.appendSlice(self.allocator, rt);
                } else {
                    try code.appendSlice(self.allocator, " void");
                }
                try code.appendSlice(self.allocator, " {\n");
                try code.appendSlice(self.allocator, indent);
                try code.appendSlice(self.allocator, "// TODO: Implement\n");
                try code.appendSlice(self.allocator, "}\n");
            },
            .python => {
                if (opts.include_docs) {
                    try code.appendSlice(self.allocator, "def ");
                    try code.appendSlice(self.allocator, name);
                    try code.append(self.allocator, '(');
                    for (params, 0..) |param, i| {
                        if (i > 0) try code.appendSlice(self.allocator, ", ");
                        try code.appendSlice(self.allocator, param);
                    }
                    try code.appendSlice(self.allocator, "):\n");
                    try code.appendSlice(self.allocator, indent);
                    try code.appendSlice(self.allocator, "\"\"\"TODO: Add documentation.\"\"\"\n");
                } else {
                    try code.appendSlice(self.allocator, "def ");
                    try code.appendSlice(self.allocator, name);
                    try code.append(self.allocator, '(');
                    for (params, 0..) |param, i| {
                        if (i > 0) try code.appendSlice(self.allocator, ", ");
                        try code.appendSlice(self.allocator, param);
                    }
                    try code.appendSlice(self.allocator, "):\n");
                }
                try code.appendSlice(self.allocator, indent);
                try code.appendSlice(self.allocator, "pass  # TODO: Implement\n");
            },
            .javascript, .typescript => {
                if (opts.include_docs) {
                    try code.appendSlice(self.allocator, "/**\n * TODO: Add documentation\n */\n");
                }
                if (language == .typescript and opts.include_types) {
                    try code.appendSlice(self.allocator, "function ");
                    try code.appendSlice(self.allocator, name);
                    try code.append(self.allocator, '(');
                    for (params, 0..) |param, i| {
                        if (i > 0) try code.appendSlice(self.allocator, ", ");
                        try code.appendSlice(self.allocator, param);
                        try code.appendSlice(self.allocator, ": any");
                    }
                    try code.appendSlice(self.allocator, "): ");
                    try code.appendSlice(self.allocator, return_type orelse "void");
                } else {
                    try code.appendSlice(self.allocator, "function ");
                    try code.appendSlice(self.allocator, name);
                    try code.append(self.allocator, '(');
                    for (params, 0..) |param, i| {
                        if (i > 0) try code.appendSlice(self.allocator, ", ");
                        try code.appendSlice(self.allocator, param);
                    }
                    try code.append(self.allocator, ')');
                }
                try code.appendSlice(self.allocator, " {\n");
                try code.appendSlice(self.allocator, indent);
                try code.appendSlice(self.allocator, "// TODO: Implement\n");
                try code.appendSlice(self.allocator, "}\n");
            },
            .rust => {
                if (opts.include_docs) {
                    try code.appendSlice(self.allocator, "/// TODO: Add documentation\n");
                }
                try code.appendSlice(self.allocator, "fn ");
                try code.appendSlice(self.allocator, name);
                try code.append(self.allocator, '(');
                for (params, 0..) |param, i| {
                    if (i > 0) try code.appendSlice(self.allocator, ", ");
                    try code.appendSlice(self.allocator, param);
                }
                try code.appendSlice(self.allocator, ") ");
                if (return_type) |rt| {
                    try code.appendSlice(self.allocator, "-> ");
                    try code.appendSlice(self.allocator, rt);
                    try code.append(self.allocator, ' ');
                }
                try code.appendSlice(self.allocator, "{\n");
                try code.appendSlice(self.allocator, indent);
                try code.appendSlice(self.allocator, "// TODO: Implement\n");
                try code.appendSlice(self.allocator, indent);
                try code.appendSlice(self.allocator, "todo!()\n");
                try code.appendSlice(self.allocator, "}\n");
            },
            .go => {
                if (opts.include_docs) {
                    try code.appendSlice(self.allocator, "// ");
                    try code.appendSlice(self.allocator, name);
                    try code.appendSlice(self.allocator, " TODO: Add documentation\n");
                }
                try code.appendSlice(self.allocator, "func ");
                try code.appendSlice(self.allocator, name);
                try code.append(self.allocator, '(');
                for (params, 0..) |param, i| {
                    if (i > 0) try code.appendSlice(self.allocator, ", ");
                    try code.appendSlice(self.allocator, param);
                }
                try code.append(self.allocator, ')');
                if (return_type) |rt| {
                    try code.append(self.allocator, ' ');
                    try code.appendSlice(self.allocator, rt);
                }
                try code.appendSlice(self.allocator, " {\n");
                try code.appendSlice(self.allocator, indent);
                try code.appendSlice(self.allocator, "// TODO: Implement\n");
                try code.appendSlice(self.allocator, "}\n");
            },
            else => {
                // Generic template
                try code.appendSlice(self.allocator, "// Function: ");
                try code.appendSlice(self.allocator, name);
                try code.appendSlice(self.allocator, "\n// TODO: Implement for ");
                try code.appendSlice(self.allocator, @tagName(language));
                try code.append(self.allocator, '\n');
            },
        }

        // Calculate loc before toOwnedSlice invalidates items
        const loc = self.countLoc(code.items);
        return .{
            .language = language,
            .code = try code.toOwnedSlice(self.allocator),
            .loc = loc,
        };
    }

    /// Wrap code with markdown code block markers.
    pub fn wrapInMarkdown(
        self: *Self,
        input_code: []const u8,
        language: classifier.Language,
    ) ![]const u8 {
        var result: std.ArrayListUnmanaged(u8) = .{};
        errdefer result.deinit(self.allocator);

        try result.appendSlice(self.allocator, "```");
        try result.appendSlice(self.allocator, self.getLanguageName(language));
        try result.append(self.allocator, '\n');
        try result.appendSlice(self.allocator, input_code);
        if (input_code.len > 0 and input_code[input_code.len - 1] != '\n') {
            try result.append(self.allocator, '\n');
        }
        try result.appendSlice(self.allocator, "```\n");

        return result.toOwnedSlice(self.allocator);
    }

    /// Get the markdown language identifier.
    fn getLanguageName(self: *const Self, language: classifier.Language) []const u8 {
        _ = self;
        return switch (language) {
            .zig => "zig",
            .rust => "rust",
            .python => "python",
            .javascript => "javascript",
            .typescript => "typescript",
            .go => "go",
            .c => "c",
            .cpp => "cpp",
            .java => "java",
            .csharp => "csharp",
            .ruby => "ruby",
            .sql => "sql",
            .bash => "bash",
            .html => "html",
            .css => "css",
            .unknown => "",
        };
    }

    /// Validate basic code structure (very basic checks).
    pub fn validateStructure(
        self: *const Self,
        code: []const u8,
        language: classifier.Language,
    ) ValidationResult {
        _ = self;
        var result = ValidationResult{ .is_valid = true };

        // Count brackets/braces
        var brace_count: i32 = 0;
        var paren_count: i32 = 0;
        var bracket_count: i32 = 0;

        for (code) |c| {
            switch (c) {
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                '(' => paren_count += 1,
                ')' => paren_count -= 1,
                '[' => bracket_count += 1,
                ']' => bracket_count -= 1,
                else => {},
            }
        }

        if (brace_count != 0) {
            result.is_valid = false;
            result.error_message = "Unbalanced braces";
        } else if (paren_count != 0) {
            result.is_valid = false;
            result.error_message = "Unbalanced parentheses";
        } else if (bracket_count != 0) {
            result.is_valid = false;
            result.error_message = "Unbalanced brackets";
        }

        _ = language; // Would use for language-specific validation

        return result;
    }
};

/// Result of code validation.
pub const ValidationResult = struct {
    is_valid: bool,
    error_message: ?[]const u8 = null,
    line_number: ?usize = null,
};

/// Extract code blocks from mixed content.
pub fn extractCodeBlocks(
    allocator: std.mem.Allocator,
    content: []const u8,
) ![]CodeBlock {
    var blocks: std.ArrayListUnmanaged(CodeBlock) = .{};
    errdefer blocks.deinit(allocator);

    var i: usize = 0;
    while (i < content.len) {
        // Look for markdown code block start
        if (std.mem.indexOf(u8, content[i..], "```")) |start_offset| {
            const block_start = i + start_offset + 3;

            // Find language identifier
            var lang_end = block_start;
            while (lang_end < content.len and content[lang_end] != '\n') : (lang_end += 1) {}

            const lang_str = std.mem.trim(u8, content[block_start..lang_end], " \t\r");
            const language = detectLanguageFromString(lang_str);

            // Find block end
            if (std.mem.indexOf(u8, content[lang_end + 1 ..], "```")) |end_offset| {
                const code_start = lang_end + 1;
                const code_end = lang_end + 1 + end_offset;
                const extracted_code = std.mem.trim(u8, content[code_start..code_end], "\n");

                try blocks.append(allocator, .{
                    .language = language,
                    .code = extracted_code,
                });

                i = code_end + 3;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    return blocks.toOwnedSlice(allocator);
}

/// Detect language from a string identifier.
fn detectLanguageFromString(lang_str: []const u8) classifier.Language {
    if (std.mem.eql(u8, lang_str, "zig")) return .zig;
    if (std.mem.eql(u8, lang_str, "rust")) return .rust;
    if (std.mem.eql(u8, lang_str, "python") or std.mem.eql(u8, lang_str, "py")) return .python;
    if (std.mem.eql(u8, lang_str, "javascript") or std.mem.eql(u8, lang_str, "js")) return .javascript;
    if (std.mem.eql(u8, lang_str, "typescript") or std.mem.eql(u8, lang_str, "ts")) return .typescript;
    if (std.mem.eql(u8, lang_str, "go") or std.mem.eql(u8, lang_str, "golang")) return .go;
    if (std.mem.eql(u8, lang_str, "c")) return .c;
    if (std.mem.eql(u8, lang_str, "cpp") or std.mem.eql(u8, lang_str, "c++")) return .cpp;
    if (std.mem.eql(u8, lang_str, "java")) return .java;
    if (std.mem.eql(u8, lang_str, "csharp") or std.mem.eql(u8, lang_str, "cs")) return .csharp;
    if (std.mem.eql(u8, lang_str, "ruby") or std.mem.eql(u8, lang_str, "rb")) return .ruby;
    if (std.mem.eql(u8, lang_str, "sql")) return .sql;
    if (std.mem.eql(u8, lang_str, "bash") or std.mem.eql(u8, lang_str, "sh")) return .bash;
    if (std.mem.eql(u8, lang_str, "html")) return .html;
    if (std.mem.eql(u8, lang_str, "css")) return .css;
    return .unknown;
}

// Tests

test "code generator initialization" {
    const generator = CodeGenerator.init(std.testing.allocator);
    try std.testing.expect(generator.config.validate_syntax);
}

test "format code block" {
    var generator = CodeGenerator.init(std.testing.allocator);

    const block = try generator.formatCodeBlock(
        "fn main() {\n    println!(\"Hello\");\n}\n",
        .rust,
        null,
    );
    defer std.testing.allocator.free(block.code);

    try std.testing.expectEqual(classifier.Language.rust, block.language);
    try std.testing.expect(block.loc > 0);
}

test "generate zig function template" {
    var generator = CodeGenerator.init(std.testing.allocator);

    const params = [_][]const u8{ "x: i32", "y: i32" };
    const block = try generator.generateFunctionTemplate(
        "add",
        &params,
        "i32",
        .zig,
        null,
    );
    defer std.testing.allocator.free(block.code);

    try std.testing.expect(std.mem.indexOf(u8, block.code, "pub fn add") != null);
    try std.testing.expect(std.mem.indexOf(u8, block.code, "i32") != null);
}

test "generate python function template" {
    var generator = CodeGenerator.init(std.testing.allocator);

    const params = [_][]const u8{ "x", "y" };
    const block = try generator.generateFunctionTemplate(
        "add",
        &params,
        null,
        .python,
        null,
    );
    defer std.testing.allocator.free(block.code);

    try std.testing.expect(std.mem.indexOf(u8, block.code, "def add") != null);
}

test "wrap in markdown" {
    var generator = CodeGenerator.init(std.testing.allocator);

    const wrapped = try generator.wrapInMarkdown("print('hello')", .python);
    defer std.testing.allocator.free(wrapped);

    try std.testing.expect(std.mem.indexOf(u8, wrapped, "```python") != null);
    try std.testing.expect(std.mem.indexOf(u8, wrapped, "print") != null);
}

test "validate balanced braces" {
    const generator = CodeGenerator.init(std.testing.allocator);

    const valid = generator.validateStructure("fn main() { }", .zig);
    try std.testing.expect(valid.is_valid);

    const invalid = generator.validateStructure("fn main() { ", .zig);
    try std.testing.expect(!invalid.is_valid);
}

test "count lines of code" {
    const generator = CodeGenerator.init(std.testing.allocator);

    const code =
        \\// Comment
        \\fn main() {
        \\    println!("Hello");
        \\}
        \\
    ;

    const loc = generator.countLoc(code);
    try std.testing.expectEqual(@as(usize, 3), loc); // Excludes comment
}

test "indent style strings" {
    try std.testing.expectEqual(@as(usize, 1), IndentStyle.tabs.getString().len);
    try std.testing.expectEqual(@as(usize, 2), IndentStyle.spaces_2.getString().len);
    try std.testing.expectEqual(@as(usize, 4), IndentStyle.spaces_4.getString().len);
}
