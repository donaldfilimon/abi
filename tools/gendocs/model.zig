const std = @import("std");

pub const Category = enum {
    core,
    compute,
    ai,
    data,
    infrastructure,
    utilities,

    pub fn name(self: Category) []const u8 {
        return switch (self) {
            .core => "Core Framework",
            .compute => "Compute & Runtime",
            .ai => "AI & Machine Learning",
            .data => "Data & Storage",
            .infrastructure => "Infrastructure",
            .utilities => "Utilities",
        };
    }

    pub fn order(self: Category) u8 {
        return switch (self) {
            .core => 0,
            .compute => 1,
            .ai => 2,
            .data => 3,
            .infrastructure => 4,
            .utilities => 5,
        };
    }
};

pub const SymbolKind = enum {
    function,
    constant,
    type_def,
    variable,

    pub fn badge(self: SymbolKind) []const u8 {
        return switch (self) {
            .function => "fn",
            .constant => "const",
            .type_def => "type",
            .variable => "var",
        };
    }
};

pub const SymbolDoc = struct {
    signature: []const u8,
    doc: []const u8,
    kind: SymbolKind,
    line: usize,
    anchor: []const u8,

    pub fn deinit(self: SymbolDoc, allocator: std.mem.Allocator) void {
        allocator.free(self.signature);
        allocator.free(self.doc);
        allocator.free(self.anchor);
    }
};

pub const ModuleDoc = struct {
    name: []const u8,
    path: []const u8,
    description: []const u8,
    category: Category,
    build_flag: []const u8,
    symbols: []SymbolDoc,

    pub fn deinit(self: ModuleDoc, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.path);
        allocator.free(self.description);
        allocator.free(self.build_flag);
        for (self.symbols) |symbol| symbol.deinit(allocator);
        allocator.free(self.symbols);
    }
};

pub const CliCommand = struct {
    name: []const u8,
    description: []const u8,
    aliases: []const []const u8,
    subcommands: []const []const u8,

    pub fn deinit(self: CliCommand, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
        for (self.aliases) |alias| allocator.free(alias);
        allocator.free(self.aliases);
        for (self.subcommands) |sub| allocator.free(sub);
        allocator.free(self.subcommands);
    }
};

pub const FeatureDoc = struct {
    name: []const u8,
    description: []const u8,
    compile_flag: []const u8,
    parent: []const u8, // "" if no parent
    real_module_path: []const u8,
    stub_module_path: []const u8,

    pub fn deinit(self: FeatureDoc, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.description);
        allocator.free(self.compile_flag);
        allocator.free(self.parent);
        allocator.free(self.real_module_path);
        allocator.free(self.stub_module_path);
    }
};

pub const ReadmeSummary = struct {
    path: []const u8,
    title: []const u8,
    summary: []const u8,

    pub fn deinit(self: ReadmeSummary, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.title);
        allocator.free(self.summary);
    }
};

pub const RoadmapDocEntry = struct {
    id: []const u8,
    title: []const u8,
    summary: []const u8,
    track: []const u8,
    track_order: u8,
    horizon: []const u8,
    horizon_order: u8,
    status: []const u8,
    status_order: u8,
    owner: []const u8,
    validation_gate: []const u8,
    plan_slug: []const u8,
    plan_title: []const u8,

    pub fn deinit(self: RoadmapDocEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.title);
        allocator.free(self.summary);
        allocator.free(self.track);
        allocator.free(self.horizon);
        allocator.free(self.status);
        allocator.free(self.owner);
        allocator.free(self.validation_gate);
        allocator.free(self.plan_slug);
        allocator.free(self.plan_title);
    }
};

pub const PlanDocEntry = struct {
    slug: []const u8,
    title: []const u8,
    status: []const u8,
    status_order: u8,
    owner: []const u8,
    scope: []const u8,
    success_criteria: []const []const u8,
    gate_commands: []const []const u8,
    milestones: []const []const u8,

    pub fn deinit(self: PlanDocEntry, allocator: std.mem.Allocator) void {
        allocator.free(self.slug);
        allocator.free(self.title);
        allocator.free(self.status);
        allocator.free(self.owner);
        allocator.free(self.scope);

        for (self.success_criteria) |item| allocator.free(item);
        allocator.free(self.success_criteria);

        for (self.gate_commands) |item| allocator.free(item);
        allocator.free(self.gate_commands);

        for (self.milestones) |item| allocator.free(item);
        allocator.free(self.milestones);
    }
};

pub const GuideSpec = struct {
    slug: []const u8,
    title: []const u8,
    description: []const u8,
    section: []const u8,
    order: usize,
    permalink: []const u8,
    template_path: []const u8,
    /// Feature names that this guide covers (matched against FeatureDoc.name and .parent).
    feature_tags: []const []const u8 = &.{},
};

pub const BuildMeta = struct {
    zig_version: []const u8,
    test_main_pass: usize,
    test_main_skip: usize,
    test_main_total: usize,
    test_feature_pass: usize,
    test_feature_skip: usize,
    test_feature_total: usize,
};

pub const OutputFile = struct {
    path: []const u8,
    content: []const u8,

    pub fn deinit(self: OutputFile, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        allocator.free(self.content);
    }
};

pub fn deinitModuleSlice(allocator: std.mem.Allocator, modules: []ModuleDoc) void {
    for (modules) |mod| mod.deinit(allocator);
    allocator.free(modules);
}

pub fn deinitCommandSlice(allocator: std.mem.Allocator, commands: []CliCommand) void {
    for (commands) |command| command.deinit(allocator);
    allocator.free(commands);
}

pub fn deinitFeatureSlice(allocator: std.mem.Allocator, features: []FeatureDoc) void {
    for (features) |f| f.deinit(allocator);
    allocator.free(features);
}

pub fn deinitReadmeSlice(allocator: std.mem.Allocator, summaries: []ReadmeSummary) void {
    for (summaries) |summary| summary.deinit(allocator);
    allocator.free(summaries);
}

pub fn deinitOutputSlice(allocator: std.mem.Allocator, outputs: []OutputFile) void {
    for (outputs) |output| output.deinit(allocator);
    allocator.free(outputs);
}

pub fn deinitRoadmapSlice(allocator: std.mem.Allocator, entries: []RoadmapDocEntry) void {
    for (entries) |entry| entry.deinit(allocator);
    allocator.free(entries);
}

pub fn deinitPlanSlice(allocator: std.mem.Allocator, entries: []PlanDocEntry) void {
    for (entries) |entry| entry.deinit(allocator);
    allocator.free(entries);
}

pub fn compareModules(_: void, lhs: ModuleDoc, rhs: ModuleDoc) bool {
    if (lhs.category.order() != rhs.category.order()) {
        return lhs.category.order() < rhs.category.order();
    }
    return std.mem.lessThan(u8, lhs.name, rhs.name);
}

pub fn compareSymbols(_: void, lhs: SymbolDoc, rhs: SymbolDoc) bool {
    if (lhs.line != rhs.line) return lhs.line < rhs.line;
    return std.mem.lessThan(u8, lhs.signature, rhs.signature);
}

pub fn compareCommands(_: void, lhs: CliCommand, rhs: CliCommand) bool {
    return std.mem.lessThan(u8, lhs.name, rhs.name);
}

pub fn compareFeatures(_: void, lhs: FeatureDoc, rhs: FeatureDoc) bool {
    return std.mem.lessThan(u8, lhs.name, rhs.name);
}

pub fn compareReadmes(_: void, lhs: ReadmeSummary, rhs: ReadmeSummary) bool {
    return std.mem.lessThan(u8, lhs.path, rhs.path);
}

pub fn compareRoadmapEntries(_: void, lhs: RoadmapDocEntry, rhs: RoadmapDocEntry) bool {
    if (lhs.horizon_order != rhs.horizon_order) return lhs.horizon_order < rhs.horizon_order;
    if (lhs.status_order != rhs.status_order) return lhs.status_order < rhs.status_order;
    if (lhs.track_order != rhs.track_order) return lhs.track_order < rhs.track_order;
    return std.mem.lessThan(u8, lhs.title, rhs.title);
}

pub fn comparePlanEntries(_: void, lhs: PlanDocEntry, rhs: PlanDocEntry) bool {
    if (lhs.status_order != rhs.status_order) return lhs.status_order < rhs.status_order;
    return std.mem.lessThan(u8, lhs.title, rhs.title);
}

pub fn pushOutput(
    allocator: std.mem.Allocator,
    outputs: *std.ArrayListUnmanaged(OutputFile),
    path: []const u8,
    content: []const u8,
) !void {
    try outputs.append(allocator, .{
        .path = try allocator.dupe(u8, path),
        .content = try allocator.dupe(u8, content),
    });
}

pub fn lineSummary(text: []const u8) []const u8 {
    const first_line_end = std.mem.indexOfScalar(u8, text, '\n') orelse text.len;
    return std.mem.trim(u8, text[0..first_line_end], " \t\r\n");
}

pub fn appendTableHeader(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    headers: []const []const u8,
) !void {
    try appendTableRow(allocator, out, headers);
    var i: usize = 0;
    while (i < headers.len) : (i += 1) {
        const divider = if (headers[i].len >= 3) "---" else "--";
        try appendTableCell(allocator, out, divider);
    }
    try out.appendSlice(allocator, "|\n");
}

pub fn appendTableRow(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    cells: []const []const u8,
) !void {
    for (cells) |cell| {
        try appendTableCell(allocator, out, cell);
    }
    try out.appendSlice(allocator, "|\n");
}

fn appendTableCell(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    text: []const u8,
) !void {
    const escaped = try escapeTableCell(allocator, text);
    defer allocator.free(escaped);
    try out.appendSlice(allocator, "| ");
    try out.appendSlice(allocator, escaped);
    try out.appendSlice(allocator, " ");
}

fn escapeTableCell(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var escaped = std.ArrayListUnmanaged(u8).empty;
    errdefer escaped.deinit(allocator);

    for (text) |c| {
        if (c == '|') {
            try escaped.appendSlice(allocator, "\\|");
        } else if (c == '\n' or c == '\r') {
            try escaped.appendSlice(allocator, " ");
        } else {
            try escaped.append(allocator, c);
        }
    }

    return escaped.toOwnedSlice(allocator);
}
