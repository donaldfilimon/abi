const std = @import("std");

pub fn build(b: *std.Build) void {
	const target = b.standardTargetOptions(.{});
	const optimize = b.standardOptimizeOption(.{});

	// Library/module exposing the abi interface (src/mod.zig)
	const abi_mod = b.addModule("abi", .{
		.root_source_file = .{ .path = "src/mod.zig" },
	});

	// Executable
	const exe = b.addExecutable(.{
		.name = "abi",
		.root_source_file = .{ .path = "src/main.zig" },
		.target = target,
		.optimize = optimize,
	});
	// Provide the abi module to any code that does @import("abi") in the exe
	exe.root_module.addImport("abi", abi_mod);
	b.installArtifact(exe);

	// Unit tests for core code (aggregate via src/mod.zig)
	const core_tests = b.addTest(.{
		.root_source_file = .{ .path = "src/mod.zig" },
		.target = target,
		.optimize = optimize,
	});
	core_tests.root_module.addImport("abi", abi_mod);

	// Standalone test files under tests/
	const test_files = [_][]const u8{
		"tests/test_memory_management.zig",
		"tests/test_performance_optimizations.zig",
		"tests/test_performance_regression.zig",
		"tests/test_simd_vector.zig",
		"tests/test_weather.zig",
		"tests/test_weather_integration.zig",
		"tests/test_web_server.zig",
		"tests/test_ai.zig",
		"tests/test_cli_integration.zig",
		"tests/test_database.zig",
		"tests/test_database_hnsw.zig",
		"tests/test_database_integration.zig",
		"tests/test_discord_plugin.zig",
	};

	// Module for example plugins referenced by tests
	const discord_plugin_mod = b.addModule("discord_plugin", .{
		.root_source_file = .{ .path = "examples/plugins/discord_plugin.zig" },
	});
	// Expose abi to plugin sources too
	discord_plugin_mod.addImport("abi", abi_mod);

	const all_tests_step = b.step("test", "Run all tests");
	const run_core_tests = b.addRunArtifact(core_tests);
	all_tests_step.dependOn(&run_core_tests.step);

	inline for (test_files) |tf| {
		const t = b.addTest(.{
			.root_source_file = .{ .path = tf },
			.target = target,
			.optimize = optimize,
		});
		// Provide modules commonly imported by tests
		t.root_module.addImport("abi", abi_mod);
		t.root_module.addImport("discord_plugin", discord_plugin_mod);

		const run_t = b.addRunArtifact(t);
		all_tests_step.dependOn(&run_t.step);
	}
}

