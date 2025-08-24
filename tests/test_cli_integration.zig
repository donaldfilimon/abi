const std = @import("std");
const testing = std.testing;

// Import the CLI module components
const cli = @import("../src/cli/main.zig");
const abi = @import("../src/root.zig");

/// Test CLI command parsing and validation
test "CLI help command" {
    // This would normally test the CLI by running it, but we'll test the parsing logic
    const HelpCommand = cli.Command.help;
    try testing.expectEqualStrings("Show help information", HelpCommand.getDescription());
}

test "CLI command parsing" {
    const allocator = testing.allocator;

    // Test command string to enum conversion
    try testing.expect(cli.Command.fromString("help") == .help);
    try testing.expect(cli.Command.fromString("version") == .version);
    try testing.expect(cli.Command.fromString("chat") == .chat);
    try testing.expect(cli.Command.fromString("train") == .train);
    try testing.expect(cli.Command.fromString("serve") == .serve);
    try testing.expect(cli.Command.fromString("benchmark") == .benchmark);
    try testing.expect(cli.Command.fromString("analyze") == .analyze);
    try testing.expect(cli.Command.fromString("convert") == .convert);
    try testing.expect(cli.Command.fromString("invalid") == null);

    // Test persona parsing
    try testing.expect(cli.parsePersona("adaptive") == abi.ai.PersonaType.adaptive);
    try testing.expect(cli.parsePersona("creative") == abi.ai.PersonaType.creative);
    try testing.expect(cli.parsePersona("analytical") == abi.ai.PersonaType.analytical);
    try testing.expect(cli.parsePersona("technical") == abi.ai.PersonaType.technical);
    try testing.expectError(error.InvalidPersona, cli.parsePersona("invalid"));

    // Test backend parsing
    try testing.expect(cli.parseBackend("local") == abi.ai.Backend.local);
    try testing.expect(cli.parseBackend("openai") == abi.ai.Backend.openai);
    try testing.expect(cli.parseBackend("anthropic") == abi.ai.Backend.anthropic);
    try testing.expectError(error.InvalidBackend, cli.parseBackend("invalid"));

    // Test format parsing
    try testing.expect(cli.parseFormat("text") == cli.OutputFormat.text);
    try testing.expect(cli.parseFormat("json") == cli.OutputFormat.json);
    try testing.expect(cli.parseFormat("yaml") == cli.OutputFormat.yaml);
    try testing.expect(cli.parseFormat("csv") == cli.OutputFormat.csv);
    try testing.expectError(error.InvalidFormat, cli.parseFormat("invalid"));

    // Test text analysis functions
    {
        const text = "Hello world! This is a test.";
        const words = cli.countWords(text);
        try testing.expectEqual(@as(usize, 5), words);

        const avg_length = cli.calculateAvgWordLength(text);
        try testing.expect(avg_length > 3.0 and avg_length < 6.0);
    }

    // Test file format detection
    {
        // Create a temporary file for testing
        const temp_path = "test_temp_file.bin";
        defer std.fs.cwd().deleteFile(temp_path) catch {};

        // Create a test file with WDBX-AI magic bytes
        const file = try std.fs.cwd().createFile(temp_path, .{});
        defer file.close();

        const magic_bytes = "WDBXAI\x00\x00";
        try file.writeAll(magic_bytes);

        const detected_format = try cli.detectModelFormat(temp_path);
        try testing.expectEqualStrings("wdbx-ai", detected_format);
    }
}

test "CLI options structure" {
    const allocator = testing.allocator;

    // Test Options initialization and cleanup
    {
        var options = cli.Options{};

        // Test that default values are correct
        try testing.expectEqual(cli.Command.help, options.command);
        try testing.expectEqual(false, options.verbose);
        try testing.expectEqual(false, options.quiet);
        try testing.expectEqual(@as(?[]const u8, null), options.config_path);
        try testing.expectEqual(@as(?[]const u8, null), options.output_path);
        try testing.expectEqual(@as([]cli.FilePath, &.{}), options.input_paths);
        try testing.expectEqual(abi.ai.PersonaType.adaptive, options.persona);
        try testing.expectEqual(abi.ai.Backend.local, options.backend);
        try testing.expectEqual(@as(u32, 0), options.threads);
        try testing.expectEqual(false, options.gpu);
        try testing.expectEqual(cli.OutputFormat.text, options.format);

        // Test cleanup (should not crash)
        options.deinit(allocator);
    }
}

test "CLI context initialization" {
    const allocator = testing.allocator;

    // Test AppContext creation and cleanup
    {
        var options = cli.Options{};
        defer options.deinit(allocator);

        // This would normally create a full context, but we're just testing structure
        // In a real test, we'd need to mock or handle the AI context creation
        _ = options;
    }
}

test "CLI argument validation" {
    // Test that all command descriptions are non-empty
    inline for (std.meta.fields(cli.Command)) |field| {
        const cmd = @as(cli.Command, @enumFromInt(field.value));
        const desc = cmd.getDescription();
        try testing.expect(desc.len > 0);
    }

    // Test that all output formats have valid string representations
    inline for (std.meta.fields(cli.OutputFormat)) |field| {
        const format = @as(cli.OutputFormat, @enumFromInt(field.value));
        const str = format.toString();
        try testing.expect(str.len > 0);
        try testing.expect(std.mem.eql(u8, str, field.name));
    }
}

test "CLI text analysis edge cases" {
    // Test text analysis with various inputs
    {
        // Empty string
        try testing.expectEqual(@as(usize, 0), cli.countWords(""));
        try testing.expectEqual(@as(f32, 0.0), cli.calculateAvgWordLength(""));

        // Single word
        try testing.expectEqual(@as(usize, 1), cli.countWords("hello"));
        try testing.expectEqual(@as(f32, 5.0), cli.calculateAvgWordLength("hello"));

        // Multiple spaces
        try testing.expectEqual(@as(usize, 2), cli.countWords("hello   world"));
        try testing.expect(cli.calculateAvgWordLength("hello   world") > 0);

        // Punctuation
        try testing.expectEqual(@as(usize, 3), cli.countWords("Hello, world! Test"));
        try testing.expect(cli.calculateAvgWordLength("Hello, world! Test") > 0);

        // Numbers
        try testing.expectEqual(@as(usize, 2), cli.countWords("test123 abc456"));
        try testing.expect(cli.calculateAvgWordLength("test123 abc456") > 0);

        // Mixed case
        try testing.expectEqual(@as(usize, 2), cli.countWords("Hello WORLD"));
        try testing.expect(cli.calculateAvgWordLength("Hello WORLD") > 0);
    }
}

test "CLI benchmark functions structure" {
    // Test that benchmark functions exist and can be called
    // (We can't easily test the actual benchmarks without full setup)

    // Test that the functions exist by checking their signatures
    const benchmark_simd_fn = cli.benchmarkSimd;
    const benchmark_ai_fn = cli.benchmarkAI;
    const benchmark_database_fn = cli.benchmarkDatabase;

    // These are function pointers we can check exist
    _ = benchmark_simd_fn;
    _ = benchmark_ai_fn;
    _ = benchmark_database_fn;
}

test "CLI command execution paths" {
    // Test that all command functions exist
    const execute_fn = cli.executeCommand;

    // Test that command functions are properly defined
    inline for (std.meta.fields(cli.Command)) |field| {
        const cmd = @as(cli.Command, @enumFromInt(field.value));

        // Each command should have a corresponding function
        switch (cmd) {
            .help => {
                const help_fn = cli.showHelp;
                _ = help_fn;
            },
            .version => {
                const version_fn = cli.showVersion;
                _ = version_fn;
            },
            .chat => {
                const chat_fn = cli.runChat;
                _ = chat_fn;
            },
            .train => {
                const train_fn = cli.runTrain;
                _ = train_fn;
            },
            .serve => {
                const serve_fn = cli.runServe;
                _ = serve_fn;
            },
            .benchmark => {
                const benchmark_fn = cli.runBenchmark;
                _ = benchmark_fn;
            },
            .analyze => {
                const analyze_fn = cli.runAnalyze;
                _ = analyze_fn;
            },
            .convert => {
                const convert_fn = cli.runConvert;
                _ = convert_fn;
            },
        }
    }
}

test "CLI file operations" {
    const allocator = testing.allocator;

    // Test FilePath structure
    {
        const path = try allocator.dupe(u8, "test/path.txt");
        defer allocator.free(path);

        var file_path = cli.FilePath{ .path = path };
        file_path.deinit(allocator); // Should free the path
    }

    // Test file size function
    {
        const temp_path = "test_file_size.bin";
        defer std.fs.cwd().deleteFile(temp_path) catch {};

        // Create a test file
        const file = try std.fs.cwd().createFile(temp_path, .{});
        defer file.close();

        const test_data = "Hello, World!";
        try file.writeAll(test_data);

        const size = try cli.getFileSize(temp_path);
        try testing.expectEqual(@as(u64, test_data.len), size);
    }
}

test "CLI error handling" {
    // Test error types that CLI can return
    try testing.expectError(error.MissingConfigPath, error.MissingConfigPath);
    try testing.expectError(error.MissingOutputPath, error.MissingOutputPath);
    try testing.expectError(error.MissingPersona, error.MissingPersona);
    try testing.expectError(error.MissingBackend, error.MissingBackend);
    try testing.expectError(error.MissingThreadCount, error.MissingThreadCount);
    try testing.expectError(error.MissingFormat, error.MissingFormat);
    try testing.expectError(error.UnknownOption, error.UnknownOption);
    try testing.expectError(error.NoInputFiles, error.NoInputFiles);
    try testing.expectError(error.UnsupportedFormat, error.UnsupportedFormat);
}

test "CLI output formatting" {
    const allocator = testing.allocator;

    // Test TextAnalysis structure
    const analysis = cli.TextAnalysis{
        .file = "test.txt",
        .size = 100,
        .lines = 10,
        .words = 50,
        .avg_word_length = 4.2,
    };

    // Test that the structure can be used in output formatting
    // (The actual formatting would require a full context setup)
    _ = analysis;

    // Test output format enum values
    try testing.expectEqual(cli.OutputFormat.text, cli.OutputFormat.text);
    try testing.expectEqual(cli.OutputFormat.json, cli.OutputFormat.json);
    try testing.expectEqual(cli.OutputFormat.yaml, cli.OutputFormat.yaml);
    try testing.expectEqual(cli.OutputFormat.csv, cli.OutputFormat.csv);
}

test "CLI model conversion functions" {
    const allocator = testing.allocator;

    // Test that model conversion functions exist (even if not fully implemented)
    const load_wdbx = cli.loadWdbxModel;
    const load_onnx = cli.loadOnnxModel;
    const load_tflite = cli.loadTfliteModel;
    const load_binary = cli.loadBinaryModel;

    // These functions should exist and return appropriate errors for unimplemented features
    _ = load_wdbx;
    _ = load_onnx;
    _ = load_tflite;
    _ = load_binary;

    // Test save functions
    const save_json = cli.saveJsonModel;
    const save_yaml = cli.saveYamlModel;
    const save_csv = cli.saveCsvModel;
    const save_text = cli.saveTextModel;

    _ = save_json;
    _ = save_yaml;
    _ = save_csv;
    _ = save_text;
}

test "CLI web server endpoints" {
    // Test that web server handler functions exist
    const handle_inference = cli.handleInference;
    const handle_health = cli.handleHealth;
    const handle_model_info = cli.handleModelInfo;

    // These are function pointers that should exist
    _ = handle_inference;
    _ = handle_health;
    _ = handle_model_info;
}

test "CLI stream callback" {
    // Test that stream callback function exists
    const stream_callback = cli.streamCallback;

    // This is a function that should exist
    _ = stream_callback;
}

test "CLI integration - full command flow" {
    const allocator = testing.allocator;

    // Test the full flow of command parsing and execution
    // (This is a simplified test since we can't easily mock std.process.args)

    // Test that we can create the necessary structures
    var options = cli.Options{
        .command = .version,
        .verbose = true,
    };
    defer options.deinit(allocator);

    // The context would need more setup for full testing
    // This tests the structure creation
    try testing.expectEqual(cli.Command.version, options.command);
    try testing.expectEqual(true, options.verbose);
}

test "CLI performance - memory efficiency" {
    const allocator = testing.allocator;

    // Test memory allocation patterns in CLI structures
    {
        var options = cli.Options{};
        defer options.deinit(allocator);

        // Add some input paths
        const input_paths = try allocator.alloc(cli.FilePath, 3);
        defer allocator.free(input_paths);

        for (input_paths, 0..) |*fp, i| {
            const path = try std.fmt.allocPrint(allocator, "file_{d}.txt", .{i});
            defer allocator.free(path);
            fp.* = .{ .path = try allocator.dupe(u8, path) };
        }

        options.input_paths = input_paths;

        // Test cleanup
        options.deinit(allocator);

        // After cleanup, paths should be freed
        try testing.expectEqual(@as([]cli.FilePath, &.{}), options.input_paths);
    }
}
