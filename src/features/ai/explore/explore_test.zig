const std = @import("std");
const explore = @import("mod.zig");

test "explore agent creation" {
    const allocator = std.testing.allocator;

    var agent = try explore.createDefaultAgent(allocator);
    agent.deinit();

    try std.testing.expect(true);
}

test "explore config default" {
    const config = explore.ExploreConfig.defaultForLevel(.quick);
    try std.testing.expect(config.level == .quick);
    try std.testing.expect(config.max_files == 1000);
}

test "explore config thorough" {
    const config = explore.ExploreConfig.defaultForLevel(.thorough);
    try std.testing.expect(config.level == .thorough);
    try std.testing.expect(config.max_files == 10000);
}

test "explore result creation" {
    const allocator = std.testing.allocator;

    var result = explore.ExploreResult.init(allocator, "test query", .medium);
    defer result.deinit();

    try std.testing.expectEqualStrings("test query", result.query);
    try std.testing.expect(result.level == .medium);
    try std.testing.expect(result.matches.items.len == 0);
}

test "search pattern literal" {
    const allocator = std.testing.allocator;

    var compiler = explore.search.PatternCompiler.init(allocator);

    var pattern = try compiler.compile("test", explore.search.PatternType.literal, false);
    defer pattern.deinit(allocator);

    try std.testing.expect(pattern.pattern_type == .literal);
}

test "glob matching" {
    try std.testing.expect(explore.search.matchesGlob("*.zig", "main.zig"));
    try std.testing.expect(!explore.search.matchesGlob("*.zig", "main.c"));
    try std.testing.expect(explore.search.matchesGlob("main*", "main.zig"));
}

test "file stats creation" {
    const allocator = std.testing.allocator;
    _ = allocator;

    const test_path = ".";
    const file_stats = explore.fs.FileStats{
        .path = test_path,
        .size_bytes = 0,
        .mtime = 0,
        .ctime = 0,
        .is_directory = true,
        .is_symlink = false,
        .mode = 0,
    };

    try std.testing.expect(file_stats.is_directory);
}

test "file type detection" {
    try std.testing.expectEqualStrings("source", explore.fs.determineFileType("main.zig"));
    try std.testing.expectEqualStrings("source", explore.fs.determineFileType("main.c"));
    try std.testing.expectEqualStrings("test", explore.fs.determineFileType("main.test.zig"));
    try std.testing.expectEqualStrings("documentation", explore.fs.determineFileType("README.md"));
    try std.testing.expectEqualStrings("config", explore.fs.determineFileType("config.json"));
}

test "query understanding intent classification" {
    const allocator = std.testing.allocator;

    var understander = explore.QueryUnderstanding.init(allocator);
    defer understander.deinit();

    var parsed1 = try understander.parse("find all functions");
    try std.testing.expect(parsed1.intent == .find_functions);
    understander.freeParsedQuery(parsed1);

    var parsed2 = try understander.parse("find types and structs");
    try std.testing.expect(parsed2.intent == .find_types);
    understander.freeParsedQuery(parsed2);

    var parsed3 = try understander.parse("show me tests");
    try std.testing.expect(parsed3.intent == .find_tests);
    understander.freeParsedQuery(parsed3);

    var parsed4 = try understander.parse("list imports");
    try std.testing.expect(parsed4.intent == .find_imports);
    understander.freeParsedQuery(parsed4);

    var parsed5 = try understander.parse("find FIXME comments");
    try std.testing.expect(parsed5.intent == .find_comments);
    understander.freeParsedQuery(parsed5);
}

test "query understanding pattern extraction" {
    const allocator = std.testing.allocator;

    var understander = explore.QueryUnderstanding.init(allocator);
    defer understander.deinit();

    var parsed = try understander.parse("find pub fn handler");
    try std.testing.expect(parsed.patterns.len > 0);
    understander.freeParsedQuery(parsed);
}

test "query understanding file extension extraction" {
    const allocator = std.testing.allocator;

    var understander = explore.QueryUnderstanding.init(allocator);
    defer understander.deinit();

    var parsed = try understander.parse("find functions in zig files");
    try std.testing.expect(parsed.file_extensions.len > 0);

    const has_zig = for (parsed.file_extensions) |ext| {
        if (std.mem.eql(u8, ext, ".zig")) break true;
    } else false;
    try std.testing.expect(has_zig);

    understander.freeParsedQuery(parsed);
}

test "query understanding target path extraction" {
    const allocator = std.testing.allocator;

    var understander = explore.QueryUnderstanding.init(allocator);
    defer understander.deinit();

    var parsed = try understander.parse("find in src/api/ handler");
    try std.testing.expect(parsed.target_paths.len > 0);

    const has_src = for (parsed.target_paths) |path| {
        if (std.mem.indexOf(u8, path, "src/") != null) break true;
    } else false;
    try std.testing.expect(has_src);

    understander.freeParsedQuery(parsed);
}

test "parallel explorer creation" {
    const allocator = std.testing.allocator;

    const config = explore.ExploreConfig.defaultForLevel(.medium);
    var result = explore.ExploreResult.init(allocator, "test", .medium);
    defer result.deinit();

    const patterns = [_]explore.SearchPattern{};

    const explorer = explore.ParallelExplorer.init(allocator, config, &result, &patterns);
    _ = explorer;

    try std.testing.expect(true);
}

test "parallel explorer work item" {
    const file_stat = explore.fs.FileStats{
        .path = "test.zig",
        .size_bytes = 100,
        .mtime = 0,
        .ctime = 0,
        .is_directory = false,
        .is_symlink = false,
        .mode = 0,
    };

    const work_item = explore.WorkItem{
        .file_stat = file_stat,
        .depth = 1,
    };

    try std.testing.expect(work_item.depth == 1);
}

test "query intent confidence scoring" {
    const allocator = std.testing.allocator;

    var understander = explore.QueryUnderstanding.init(allocator);
    defer understander.deinit();

    var parsed = try understander.parse("find all functions in src/");
    try std.testing.expect(parsed.confidence > 0.5);

    understander.freeParsedQuery(parsed);
}

test "ast node type to match type conversion" {
    const func_node: explore.AstNodeType = .function;
    try std.testing.expect(explore.AstParser.nodeToMatchType(func_node) == .function_definition);

    const struct_node: explore.AstNodeType = .struct_type;
    try std.testing.expect(explore.AstParser.nodeToMatchType(struct_node) == .type_definition);

    const import_node: explore.AstNodeType = .import_decl;
    try std.testing.expect(explore.AstParser.nodeToMatchType(import_node) == .import_statement);

    const test_node: explore.AstNodeType = .test_decl;
    try std.testing.expect(explore.AstParser.nodeToMatchType(test_node) == .test_case);

    const comment_node: explore.AstNodeType = .comment;
    try std.testing.expect(explore.AstParser.nodeToMatchType(comment_node) == .comment);
}

test "query understanding empty input" {
    const allocator = std.testing.allocator;

    var understander = explore.QueryUnderstanding.init(allocator);
    defer understander.deinit();

    const parsed = try understander.parse("");
    try std.testing.expect(parsed.intent == .find_any);
    try std.testing.expect(parsed.patterns.len == 0);
}

test "query understanding edge case mixed case" {
    const allocator = std.testing.allocator;

    var understander = explore.QueryUnderstanding.init(allocator);
    defer understander.deinit();

    const parsed = try understander.parse("FIND ALL FUNCTIONS");
    try std.testing.expect(parsed.intent == .find_functions);
}

test "explore agent cancellation" {
    const allocator = std.testing.allocator;

    var agent = try explore.createDefaultAgent(allocator);
    defer agent.deinit();

    agent.cancel();

    try std.testing.expect(agent.isCancelled());
}

test "explore config deep defaults" {
    const config = explore.ExploreConfig.defaultForLevel(.deep);
    try std.testing.expect(config.level == .deep);
    try std.testing.expect(config.max_files == 50000);
    try std.testing.expect(config.max_depth == 50);
    try std.testing.expect(config.timeout_ms == 300000);
}

test "parallel explorer cancellation" {
    const allocator = std.testing.allocator;

    const config = explore.ExploreConfig.defaultForLevel(.quick);
    var result = explore.ExploreResult.init(allocator, "test", .quick);
    defer result.deinit();

    const patterns = [_]explore.SearchPattern{};

    var explorer = explore.ParallelExplorer.init(allocator, config, &result, &patterns);

    try std.testing.expect(explorer.getProcessedCount() == 0);
}

test "call graph creation" {
    const allocator = std.testing.allocator;

    var graph = explore.CallGraph.init(allocator);
    defer graph.deinit();

    const func1: explore.Function = .{
        .name = "main",
        .file_path = "test.zig",
        .line = 1,
    };

    const func2: explore.Function = .{
        .name = "helper",
        .file_path = "test.zig",
        .line = 10,
    };

    try graph.addFunction(func1);
    try graph.addFunction(func2);
    try graph.addCall(func1, func2);

    try std.testing.expect(graph.all_functions.items.len == 2);
    try std.testing.expect(graph.edges.items.len == 1);
}

test "call graph get callers and callees" {
    const allocator = std.testing.allocator;

    var graph = explore.CallGraph.init(allocator);
    defer graph.deinit();

    const func1: explore.Function = .{
        .name = "caller",
        .file_path = "test.zig",
        .line = 1,
    };

    const func2: explore.Function = .{
        .name = "callee",
        .file_path = "test.zig",
        .line = 10,
    };

    try graph.addFunction(func1);
    try graph.addFunction(func2);
    try graph.addCall(func1, func2);

    const callees = graph.getCallees("caller");
    try std.testing.expect(callees != null);
    try std.testing.expect(callees.?.len == 1);
    try std.testing.expect(std.mem.eql(u8, callees.?[0].name, "callee"));

    const callers = graph.getCallers("callee");
    try std.testing.expect(callers != null);
    try std.testing.expect(callers.?.len == 1);
    try std.testing.expect(std.mem.eql(u8, callers.?[0].name, "caller"));
}

test "call graph has path" {
    const allocator = std.testing.allocator;

    var graph = explore.CallGraph.init(allocator);
    defer graph.deinit();

    const func1: explore.Function = .{
        .name = "a",
        .file_path = "test.zig",
        .line = 1,
    };

    const func2: explore.Function = .{
        .name = "b",
        .file_path = "test.zig",
        .line = 10,
    };

    const func3: explore.Function = .{
        .name = "c",
        .file_path = "test.zig",
        .line = 20,
    };

    try graph.addFunction(func1);
    try graph.addFunction(func2);
    try graph.addFunction(func3);
    try graph.addCall(func1, func2);
    try graph.addCall(func2, func3);

    try std.testing.expect(graph.hasPathTo("a", "c"));
    try std.testing.expect(!graph.hasPathTo("c", "a"));
}

test "dependency graph creation" {
    const allocator = std.testing.allocator;

    var graph = explore.DependencyGraph.init(allocator);
    defer graph.deinit();

    const mod1: explore.Module = .{
        .name = "module1",
        .file_path = "module1.zig",
        .file_type = "zig",
    };

    const mod2: explore.Module = .{
        .name = "module2",
        .file_path = "module2.zig",
        .file_type = "zig",
    };

    try graph.addModule(mod1);
    try graph.addModule(mod2);
    try graph.addDependency(mod1, mod2, .local);

    try std.testing.expect(graph.all_modules.items.len == 2);
    try std.testing.expect(graph.edges.items.len == 1);
}

test "dependency graph get dependencies and dependents" {
    const allocator = std.testing.allocator;

    var graph = explore.DependencyGraph.init(allocator);
    defer graph.deinit();

    const mod1: explore.Module = .{
        .name = "module1",
        .file_path = "module1.zig",
        .file_type = "zig",
    };

    const mod2: explore.Module = .{
        .name = "module2",
        .file_path = "module2.zig",
        .file_type = "zig",
    };

    try graph.addModule(mod1);
    try graph.addModule(mod2);
    try graph.addDependency(mod1, mod2, .local);

    const deps = graph.getDependencies("module1");
    try std.testing.expect(deps != null);
    try std.testing.expect(deps.?.len == 1);
    try std.testing.expect(std.mem.eql(u8, deps.?[0].module.name, "module2"));

    const dependents = graph.getDependents("module2");
    try std.testing.expect(dependents != null);
    try std.testing.expect(dependents.?.len == 1);
    try std.testing.expect(std.mem.eql(u8, dependents.?[0].module.name, "module1"));
}

test "dependency graph import type classification" {
    const allocator = std.testing.allocator;

    var graph = explore.DependencyGraph.init(allocator);
    defer graph.deinit();

    const mod1: explore.Module = .{
        .name = "module1",
        .file_path = "module1.zig",
        .file_type = "zig",
    };

    const std_mod: explore.Module = .{
        .name = "std",
        .file_path = "std",
        .file_type = "zig",
    };

    const ext_mod: explore.Module = .{
        .name = "external",
        .file_path = "external",
        .file_type = "typescript",
    };

    try graph.addModule(mod1);
    try graph.addModule(std_mod);
    try graph.addModule(ext_mod);

    try graph.addDependency(mod1, std_mod, .std);
    try graph.addDependency(mod1, ext_mod, .external);

    const deps = graph.getDependencies("module1");
    try std.testing.expect(deps != null);
    try std.testing.expect(deps.?.len == 2);
}

test "dependency graph topological sort" {
    const allocator = std.testing.allocator;

    var graph = explore.DependencyGraph.init(allocator);
    defer graph.deinit();

    const mod1: explore.Module = .{
        .name = "a",
        .file_path = "a.zig",
        .file_type = "zig",
    };

    const mod2: explore.Module = .{
        .name = "b",
        .file_path = "b.zig",
        .file_type = "zig",
    };

    const mod3: explore.Module = .{
        .name = "c",
        .file_path = "c.zig",
        .file_type = "zig",
    };

    try graph.addModule(mod1);
    try graph.addModule(mod2);
    try graph.addModule(mod3);
    try graph.addDependency(mod1, mod2, .local);
    try graph.addDependency(mod2, mod3, .local);

    var sorted = try graph.topologicalSort();
    defer sorted.deinit(allocator);

    try std.testing.expect(sorted.items.len == 3);
}
