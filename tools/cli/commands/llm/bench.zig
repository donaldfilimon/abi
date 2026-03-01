//! LLM bench subcommand - Benchmark model performance.

const std = @import("std");
const context_mod = @import("../../framework/context.zig");
const abi = @import("abi");
const utils = @import("../../utils/mod.zig");
const mod = @import("mod.zig");

// ── Types ──────────────────────────────────────────────────────────────

pub const BenchResult = struct {
    m: u32,
    k: u32,
    n: u32,
    time_ms: f64,
    gflops: f64,
    est_prefill_tok_s: f64,
    est_decode_tok_s: f64,
};

pub const RuntimeBenchSample = struct {
    prompt_tokens: u32,
    generated_tokens: u32,
    elapsed_ms: f64,
    prefill_tok_s: f64,
    decode_tok_s: f64,
};

pub const RuntimeStatsSummary = struct {
    mean: f64 = 0.0,
    p50: f64 = 0.0,
    p90: f64 = 0.0,
    p95: f64 = 0.0,
};

pub const RuntimeSampleSummary = struct {
    prompt_tokens_mean: u32 = 0,
    generated_tokens_mean: u32 = 0,
    elapsed_ms: RuntimeStatsSummary = .{},
    prefill_tok_s: RuntimeStatsSummary = .{},
    decode_tok_s: RuntimeStatsSummary = .{},
};

pub const RuntimeBenchResult = struct {
    backend: abi.features.ai.llm.EngineBackend,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *RuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.runs);
        self.* = undefined;
    }
};

pub const OllamaRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *OllamaRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

pub const MlxRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *MlxRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

pub const VllmRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *VllmRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

pub const LmStudioRuntimeBenchResult = struct {
    model_name: []u8,
    runs: []RuntimeBenchSample,
    summary: RuntimeSampleSummary,

    pub fn deinit(self: *LmStudioRuntimeBenchResult, allocator: std.mem.Allocator) void {
        allocator.free(self.model_name);
        allocator.free(self.runs);
        self.* = undefined;
    }
};

const RuntimeRunRecordJson = struct {
    wall_ms: f64,
    prefill_tps: f64,
    decode_tps: f64,
    prompt_tokens: u32,
    generated_tokens: u32,
};

const RuntimeSummaryJson = struct {
    backend: []const u8,
    runs: usize,
    prompt_tokens_mean: u32,
    generated_tokens_mean: u32,
    wall_ms: RuntimeStatsSummary,
    prefill_tps: RuntimeStatsSummary,
    decode_tps: RuntimeStatsSummary,
    per_run: []const RuntimeRunRecordJson,
};

const OllamaRuntimeSummaryJson = struct {
    backend: []const u8,
    model: []const u8,
    runs: usize,
    prompt_tokens_mean: u32,
    generated_tokens_mean: u32,
    wall_ms: RuntimeStatsSummary,
    prefill_tps: RuntimeStatsSummary,
    decode_tps: RuntimeStatsSummary,
    per_run: []const RuntimeRunRecordJson,
};

// ── Main entry point ───────────────────────────────────────────────────

pub fn runBench(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    if (utils.args.containsHelpArgs(args)) {
        mod.printHelp();
        return;
    }

    var model_path: ?[]const u8 = null;
    var prompt_tokens: u32 = 128;
    var gen_tokens: u32 = 64;
    var runtime_runs: u32 = 3;
    var prompt_text: ?[]const u8 = null;
    var compare_ollama: bool = false;
    var ollama_model: ?[]const u8 = null;
    var compare_mlx: bool = false;
    var mlx_model: ?[]const u8 = null;
    var compare_vllm: bool = false;
    var vllm_model: ?[]const u8 = null;
    var compare_lmstudio: bool = false;
    var lmstudio_model: ?[]const u8 = null;
    var json_output: bool = false;
    var wdbx_out: ?[]const u8 = null;

    var i: usize = 0;
    while (i < args.len) {
        const arg = args[i];
        i += 1;

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--prompt-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                prompt_tokens = std.fmt.parseInt(u32, val, 10) catch 128;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--gen-tokens")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                gen_tokens = std.fmt.parseInt(u32, val, 10) catch 64;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--runs")) {
            if (i < args.len) {
                const val = std.mem.sliceTo(args[i], 0);
                runtime_runs = std.fmt.parseInt(u32, val, 10) catch 3;
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--prompt")) {
            if (i < args.len) {
                prompt_text = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-ollama")) {
            compare_ollama = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--ollama-model")) {
            if (i < args.len) {
                ollama_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-mlx")) {
            compare_mlx = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--mlx-model")) {
            if (i < args.len) {
                mlx_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-vllm")) {
            compare_vllm = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--vllm-model")) {
            if (i < args.len) {
                vllm_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-lmstudio")) {
            compare_lmstudio = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--lmstudio-model")) {
            if (i < args.len) {
                lmstudio_model = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--compare-all")) {
            compare_ollama = true;
            compare_mlx = true;
            compare_vllm = true;
            compare_lmstudio = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--json")) {
            json_output = true;
            continue;
        }

        if (std.mem.eql(u8, std.mem.sliceTo(arg, 0), "--wdbx-out")) {
            if (i < args.len) {
                wdbx_out = std.mem.sliceTo(args[i], 0);
                i += 1;
            }
            continue;
        }

        if (model_path == null) {
            model_path = std.mem.sliceTo(arg, 0);
        }
    }

    if (model_path == null and !compare_ollama and !compare_mlx and !compare_vllm and !compare_lmstudio) {
        utils.output.println("Usage: abi llm bench <model> [--prompt-tokens N] [--gen-tokens N] [--runs N] [--compare-ollama] [--compare-mlx] [--compare-vllm] [--compare-lmstudio] [--compare-all] [--json]", .{});
        return;
    }

    runtime_runs = @max(runtime_runs, 1);

    const bench_prompt = if (prompt_text) |p| p else try buildBenchmarkPrompt(allocator, prompt_tokens);
    defer if (prompt_text == null) allocator.free(@constCast(bench_prompt));

    utils.output.printHeader("LLM Benchmark");
    if (model_path) |path| {
        utils.output.printKeyValueFmt("Local model", "{s}", .{path});
    }
    utils.output.printKeyValueFmt("Prompt", "{s}", .{bench_prompt});
    utils.output.printKeyValueFmt("Prompt tokens", "{d}", .{prompt_tokens});
    utils.output.printKeyValueFmt("Generation tokens", "{d}", .{gen_tokens});
    utils.output.printKeyValueFmt("Runtime runs", "{d}", .{runtime_runs});
    if (compare_ollama) {
        utils.output.println("Compare with Ollama: enabled", .{});
    }
    if (ollama_model) |name| {
        utils.output.printKeyValueFmt("Ollama model override", "{s}", .{name});
    }
    if (compare_mlx) {
        utils.output.println("Compare with MLX: enabled", .{});
    }
    if (mlx_model) |name| {
        utils.output.printKeyValueFmt("MLX model override", "{s}", .{name});
    }
    if (compare_vllm) {
        utils.output.println("Compare with vLLM: enabled", .{});
    }
    if (vllm_model) |name| {
        utils.output.printKeyValueFmt("vLLM model override", "{s}", .{name});
    }
    if (compare_lmstudio) {
        utils.output.println("Compare with LM Studio: enabled", .{});
    }
    if (lmstudio_model) |name| {
        utils.output.printKeyValueFmt("LM Studio model override", "{s}", .{name});
    }
    if (json_output) {
        utils.output.println("Output: JSON", .{});
    }
    if (wdbx_out) |path| {
        utils.output.printKeyValueFmt("WDBX output", "{s}", .{path});
    }
    utils.output.println("", .{});

    // Synthetic compute benchmark (backend-agnostic)
    const bench_result = runComputeBenchmark(allocator, prompt_tokens, gen_tokens);

    utils.output.println("Synthetic Compute Benchmark", .{});
    utils.output.printSeparator(27);
    utils.output.println("  Dimensions: {d}x{d} @ {d}x{d}", .{ bench_result.m, bench_result.k, bench_result.k, bench_result.n });
    utils.output.println("  Time: {d:.2} ms", .{bench_result.time_ms});
    utils.output.println("  GFLOPS: {d:.2}", .{bench_result.gflops});
    utils.output.println("Estimated local throughput from matmul:", .{});
    utils.output.println("  Prefill: ~{d:.0} tokens/sec", .{bench_result.est_prefill_tok_s});
    utils.output.println("  Decode:  ~{d:.0} tokens/sec", .{bench_result.est_decode_tok_s});

    var local_runtime: ?RuntimeBenchResult = null;
    if (model_path) |path| {
        local_runtime = runLocalRuntimeBenchmark(allocator, path, bench_prompt, gen_tokens, runtime_runs) catch |err| blk: {
            utils.output.printWarning("Local runtime benchmark unavailable: {t}", .{err});
            break :blk null;
        };
    }
    defer if (local_runtime) |*res| res.deinit(allocator);

    var ollama_runtime: ?OllamaRuntimeBenchResult = null;
    if (compare_ollama) {
        ollama_runtime = runOllamaRuntimeBenchmark(allocator, bench_prompt, gen_tokens, ollama_model, runtime_runs) catch |err| blk: {
            utils.output.printWarning("Ollama benchmark unavailable: {t}", .{err});
            break :blk null;
        };
    }
    defer if (ollama_runtime) |*res| res.deinit(allocator);

    var mlx_runtime: ?MlxRuntimeBenchResult = null;
    if (compare_mlx) {
        mlx_runtime = runMlxRuntimeBenchmark(allocator, bench_prompt, gen_tokens, mlx_model, runtime_runs) catch |err| blk: {
            utils.output.printWarning("MLX benchmark unavailable: {t}", .{err});
            break :blk null;
        };
    }
    defer if (mlx_runtime) |*res| res.deinit(allocator);

    var vllm_runtime: ?VllmRuntimeBenchResult = null;
    if (compare_vllm) {
        vllm_runtime = runVllmRuntimeBenchmark(allocator, bench_prompt, gen_tokens, vllm_model, runtime_runs) catch |err| blk: {
            utils.output.printWarning("vLLM benchmark unavailable: {t}", .{err});
            break :blk null;
        };
    }
    defer if (vllm_runtime) |*res| res.deinit(allocator);

    var lmstudio_runtime: ?LmStudioRuntimeBenchResult = null;
    if (compare_lmstudio) {
        lmstudio_runtime = runLmStudioRuntimeBenchmark(allocator, bench_prompt, gen_tokens, lmstudio_model, runtime_runs) catch |err| blk: {
            utils.output.printWarning("LM Studio benchmark unavailable: {t}", .{err});
            break :blk null;
        };
    }
    defer if (lmstudio_runtime) |*res| res.deinit(allocator);

    if (local_runtime) |local| {
        utils.output.println("", .{});
        utils.output.println("Local Runtime Benchmark", .{});
        utils.output.printSeparator(23);
        utils.output.printKeyValueFmt("Backend", "{s}", .{local.backend.label()});
        utils.output.printKeyValueFmt("Runs", "{d}", .{local.runs.len});
        utils.output.printKeyValueFmt("Prompt tokens (mean)", "{d}", .{local.summary.prompt_tokens_mean});
        utils.output.printKeyValueFmt("Generated tokens (mean)", "{d}", .{local.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", local.summary.elapsed_ms);
        printRuntimeStats("Prefill tok/s", local.summary.prefill_tok_s);
        printRuntimeStats("Decode tok/s", local.summary.decode_tok_s);
    }

    if (ollama_runtime) |res| {
        utils.output.println("", .{});
        utils.output.println("Ollama Runtime Benchmark", .{});
        utils.output.printSeparator(24);
        utils.output.printKeyValue("Backend", "ollama");
        utils.output.printKeyValueFmt("Model", "{s}", .{res.model_name});
        utils.output.printKeyValueFmt("Runs", "{d}", .{res.runs.len});
        utils.output.printKeyValueFmt("Prompt tokens (mean)", "{d}", .{res.summary.prompt_tokens_mean});
        utils.output.printKeyValueFmt("Generated tokens (mean)", "{d}", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Prefill tok/s", res.summary.prefill_tok_s);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    if (mlx_runtime) |res| {
        utils.output.println("", .{});
        utils.output.println("MLX Runtime Benchmark", .{});
        utils.output.printSeparator(21);
        utils.output.printKeyValue("Backend", "mlx");
        utils.output.printKeyValueFmt("Model", "{s}", .{res.model_name});
        utils.output.printKeyValueFmt("Runs", "{d}", .{res.runs.len});
        utils.output.printKeyValueFmt("Prompt tokens (mean)", "{d}", .{res.summary.prompt_tokens_mean});
        utils.output.printKeyValueFmt("Generated tokens (mean)", "{d}", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Prefill tok/s", res.summary.prefill_tok_s);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    if (vllm_runtime) |res| {
        utils.output.println("", .{});
        utils.output.println("vLLM Runtime Benchmark", .{});
        utils.output.printSeparator(22);
        utils.output.printKeyValue("Backend", "vllm");
        utils.output.printKeyValueFmt("Model", "{s}", .{res.model_name});
        utils.output.printKeyValueFmt("Runs", "{d}", .{res.runs.len});
        utils.output.printKeyValueFmt("Prompt tokens (mean)", "{d}", .{res.summary.prompt_tokens_mean});
        utils.output.printKeyValueFmt("Generated tokens (mean)", "{d}", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    if (lmstudio_runtime) |res| {
        utils.output.println("", .{});
        utils.output.println("LM Studio Runtime Benchmark", .{});
        utils.output.printSeparator(27);
        utils.output.printKeyValue("Backend", "lm-studio");
        utils.output.printKeyValueFmt("Model", "{s}", .{res.model_name});
        utils.output.printKeyValueFmt("Runs", "{d}", .{res.runs.len});
        utils.output.printKeyValueFmt("Prompt tokens (mean)", "{d}", .{res.summary.prompt_tokens_mean});
        utils.output.printKeyValueFmt("Generated tokens (mean)", "{d}", .{res.summary.generated_tokens_mean});
        printRuntimeStats("Wall time ms", res.summary.elapsed_ms);
        printRuntimeStats("Decode tok/s", res.summary.decode_tok_s);
    }

    // Comparison summary table
    {
        const BackendEntry = struct {
            name: []const u8,
            status: []const u8,
            decode_mean: f64,
            decode_p50: f64,
            prefill_mean: f64,
            wall_mean: f64,
        };

        var entries: [5]BackendEntry = undefined;
        var entry_count: usize = 0;

        if (local_runtime) |local| {
            entries[entry_count] = .{
                .name = "local-gguf",
                .status = "OK",
                .decode_mean = local.summary.decode_tok_s.mean,
                .decode_p50 = local.summary.decode_tok_s.p50,
                .prefill_mean = local.summary.prefill_tok_s.mean,
                .wall_mean = local.summary.elapsed_ms.mean,
            };
            entry_count += 1;
        } else if (model_path != null) {
            entries[entry_count] = .{ .name = "local-gguf", .status = "FAIL", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            entry_count += 1;
        }
        if (compare_ollama) {
            if (ollama_runtime) |res| {
                entries[entry_count] = .{
                    .name = "ollama",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "ollama", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }
        if (compare_mlx) {
            if (mlx_runtime) |res| {
                entries[entry_count] = .{
                    .name = "mlx",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "mlx", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }
        if (compare_vllm) {
            if (vllm_runtime) |res| {
                entries[entry_count] = .{
                    .name = "vllm",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "vllm", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }
        if (compare_lmstudio) {
            if (lmstudio_runtime) |res| {
                entries[entry_count] = .{
                    .name = "lm-studio",
                    .status = "OK",
                    .decode_mean = res.summary.decode_tok_s.mean,
                    .decode_p50 = res.summary.decode_tok_s.p50,
                    .prefill_mean = res.summary.prefill_tok_s.mean,
                    .wall_mean = res.summary.elapsed_ms.mean,
                };
            } else {
                entries[entry_count] = .{ .name = "lm-studio", .status = "SKIP", .decode_mean = 0, .decode_p50 = 0, .prefill_mean = 0, .wall_mean = 0 };
            }
            entry_count += 1;
        }

        if (entry_count >= 2) {
            utils.output.printHeader("Backend Comparison");
            utils.output.println("  {s:<14} {s:>14} {s:>14} {s:>12} {s:>6}", .{ "Backend", "Decode tok/s", "Prefill tok/s", "Wall ms", "Status" });
            utils.output.println("  {s:<14} {s:>14} {s:>14} {s:>12} {s:>6}", .{ "-" ** 14, "-" ** 14, "-" ** 14, "-" ** 12, "-" ** 6 });
            for (entries[0..entry_count]) |e| {
                if (std.mem.eql(u8, e.status, "OK")) {
                    if (e.prefill_mean > 0) {
                        utils.output.println("  {s:<14} {d:>7.1} ({d:.1}) {d:>14.1} {d:>12.1} {s:>6}", .{ e.name, e.decode_mean, e.decode_p50, e.prefill_mean, e.wall_mean, e.status });
                    } else {
                        utils.output.println("  {s:<14} {d:>7.1} ({d:.1}) {s:>14} {d:>12.1} {s:>6}", .{ e.name, e.decode_mean, e.decode_p50, "N/A", e.wall_mean, e.status });
                    }
                } else {
                    utils.output.println("  {s:<14} {s:>14} {s:>14} {s:>12} {s:>6}", .{ e.name, "-", "-", "-", e.status });
                }
            }

            // Pairwise decode speed ratios
            utils.output.println("", .{});
            utils.output.println("  Decode Speed Ratios (mean):", .{});
            for (entries[0..entry_count], 0..) |a, ai| {
                if (a.decode_mean <= 0) continue;
                for (entries[0..entry_count], 0..) |b, bi| {
                    if (bi <= ai) continue;
                    if (b.decode_mean <= 0) continue;
                    const ratio = b.decode_mean / a.decode_mean;
                    utils.output.println("    {s} / {s}: {d:.2}x", .{ b.name, a.name, ratio });
                }
            }
        }
    }

    // JSON export
    if (json_output) {
        printBenchJson(
            allocator,
            prompt_tokens,
            gen_tokens,
            runtime_runs,
            bench_result,
            local_runtime,
            ollama_runtime,
            mlx_runtime,
            vllm_runtime,
            lmstudio_runtime,
        );
    }

    if (wdbx_out) |path| {
        appendBenchRecordToWdbx(
            allocator,
            path,
            bench_prompt,
            prompt_tokens,
            gen_tokens,
            runtime_runs,
            bench_result,
            local_runtime,
            ollama_runtime,
        ) catch |err| {
            utils.output.printError("WDBX write failed: {t}", .{err});
        };
    }

    if (!json_output) {
        utils.output.println("", .{});
        utils.output.printInfo("Local runtime requires ABI-native support for the GGUF architecture.", .{});
        utils.output.println("When unsupported, use Ollama fallback for execution and compare decode throughput.", .{});
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

pub fn printRuntimeStats(label: []const u8, stats: RuntimeStatsSummary) void {
    utils.output.println("  {s}: mean={d:.2}, p50={d:.2}, p90={d:.2}, p95={d:.2}", .{
        label,
        stats.mean,
        stats.p50,
        stats.p90,
        stats.p95,
    });
}

pub fn summarizeRuntimeSamples(allocator: std.mem.Allocator, runs: []const RuntimeBenchSample) !RuntimeSampleSummary {
    if (runs.len == 0) return .{};

    const elapsed_values = try allocator.alloc(f64, runs.len);
    defer allocator.free(elapsed_values);

    const prefill_values = try allocator.alloc(f64, runs.len);
    defer allocator.free(prefill_values);

    const decode_values = try allocator.alloc(f64, runs.len);
    defer allocator.free(decode_values);

    var prompt_total: u64 = 0;
    var generated_total: u64 = 0;
    for (runs, 0..) |sample, idx| {
        elapsed_values[idx] = sample.elapsed_ms;
        prefill_values[idx] = sample.prefill_tok_s;
        decode_values[idx] = sample.decode_tok_s;
        prompt_total += sample.prompt_tokens;
        generated_total += sample.generated_tokens;
    }

    const run_count_u64: u64 = @intCast(runs.len);

    return .{
        .prompt_tokens_mean = @intCast(prompt_total / run_count_u64),
        .generated_tokens_mean = @intCast(generated_total / run_count_u64),
        .elapsed_ms = try summarizeF64(allocator, elapsed_values),
        .prefill_tok_s = try summarizeF64(allocator, prefill_values),
        .decode_tok_s = try summarizeF64(allocator, decode_values),
    };
}

pub fn summarizeF64(allocator: std.mem.Allocator, values: []const f64) !RuntimeStatsSummary {
    if (values.len == 0) return .{};

    const sorted = try allocator.dupe(f64, values);
    defer allocator.free(sorted);

    std.mem.sort(f64, sorted, {}, std.sort.asc(f64));

    var total: f64 = 0.0;
    for (values) |v| total += v;
    const count = @as(f64, @floatFromInt(values.len));

    return .{
        .mean = total / count,
        .p50 = percentileFromSorted(sorted, 0.50),
        .p90 = percentileFromSorted(sorted, 0.90),
        .p95 = percentileFromSorted(sorted, 0.95),
    };
}

pub fn percentileFromSorted(sorted: []const f64, quantile: f64) f64 {
    if (sorted.len == 0) return 0.0;

    const q = std.math.clamp(quantile, 0.0, 1.0);
    const last_index = sorted.len - 1;
    const position = q * @as(f64, @floatFromInt(last_index));
    const lower_idx: usize = @intFromFloat(@floor(position));
    const upper_idx: usize = @intFromFloat(@ceil(position));

    if (upper_idx >= sorted.len or lower_idx == upper_idx) {
        return sorted[lower_idx];
    }

    const fraction = position - @as(f64, @floatFromInt(lower_idx));
    return sorted[lower_idx] + (sorted[upper_idx] - sorted[lower_idx]) * fraction;
}

pub fn runComputeBenchmark(allocator: std.mem.Allocator, prompt_tokens: u32, gen_tokens: u32) BenchResult {
    // Typical 7B model dimensions: hidden_size=4096, intermediate_size=11008
    const m: u32 = @max(prompt_tokens, 1);
    const k: u32 = 4096; // hidden_size
    const n: u32 = 4096; // hidden_size (for attention projection)

    // Allocate test matrices
    const a = allocator.alloc(f32, @as(usize, m) * k) catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };
    defer allocator.free(a);

    const b = allocator.alloc(f32, @as(usize, k) * n) catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };
    defer allocator.free(b);

    const c = allocator.alloc(f32, @as(usize, m) * n) catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };
    defer allocator.free(c);

    // Initialize with random-ish values
    for (a, 0..) |*v, idx| {
        v.* = @as(f32, @floatFromInt(idx % 100)) / 100.0 - 0.5;
    }
    for (b, 0..) |*v, idx| {
        v.* = @as(f32, @floatFromInt(idx % 100)) / 100.0 - 0.5;
    }

    // Warmup
    abi.features.ai.llm.ops.matrixMultiply(a, b, c, m, k, n);

    // Benchmark
    const iterations: u32 = 5;
    var timer = abi.services.shared.time.Timer.start() catch {
        return BenchResult{
            .m = m,
            .k = k,
            .n = n,
            .time_ms = 0,
            .gflops = 0,
            .est_prefill_tok_s = 0,
            .est_decode_tok_s = 0,
        };
    };

    for (0..iterations) |_| {
        abi.features.ai.llm.ops.matrixMultiply(a, b, c, m, k, n);
    }

    const elapsed_ns = timer.read();
    const time_ms = @as(f64, @floatFromInt(elapsed_ns)) / (1_000_000.0 * @as(f64, @floatFromInt(iterations)));

    // Calculate GFLOPS: 2*M*N*K operations per matmul
    const flops: f64 = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    const gflops = (flops / (time_ms / 1000.0)) / 1_000_000_000.0;

    // Estimate tokens/sec based on typical 7B model compute
    // A 7B model has ~32 transformer layers, each with multiple matmuls
    // Prefill processes all tokens in parallel, decode processes 1 token at a time
    const layers: f64 = 32.0;
    const matmuls_per_layer: f64 = 4.0; // Q,K,V projection + output projection
    const total_flops_per_token = flops * layers * matmuls_per_layer;

    const est_prefill_tok_s = (gflops * 1_000_000_000.0) / total_flops_per_token * @as(f64, @floatFromInt(prompt_tokens));
    const est_decode_tok_s = (gflops * 1_000_000_000.0) / (total_flops_per_token / @as(f64, @floatFromInt(gen_tokens)));

    return BenchResult{
        .m = m,
        .k = k,
        .n = n,
        .time_ms = time_ms,
        .gflops = gflops,
        .est_prefill_tok_s = @min(est_prefill_tok_s, 10000.0),
        .est_decode_tok_s = @min(est_decode_tok_s, 100.0),
    };
}

pub fn runLocalRuntimeBenchmark(
    allocator: std.mem.Allocator,
    model_path: []const u8,
    prompt: []const u8,
    gen_tokens: u32,
    runs: u32,
) !RuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var engine = abi.features.ai.llm.Engine.init(allocator, .{
        .max_new_tokens = gen_tokens,
        .allow_ollama_fallback = false,
    });
    defer engine.deinit();

    try engine.loadModel(model_path);
    if (engine.getBackend() != .local_gguf) return error.UnsupportedArchitecture;

    for (samples) |*sample| {
        var timer = try abi.services.shared.time.Timer.start();
        const output = try engine.generate(allocator, prompt);
        defer allocator.free(output);
        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        const stats = engine.getStats();
        const prompt_count = if (stats.prompt_tokens > 0) stats.prompt_tokens else estimateTokenCount(prompt);
        const generated_count = if (stats.generated_tokens > 0) stats.generated_tokens else estimateTokenCount(output);

        var decode_tok_s = stats.decodeTokensPerSecond();
        if (decode_tok_s <= 0 and elapsed_ms > 0) {
            decode_tok_s = @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0);
        }

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = stats.prefillTokensPerSecond(),
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .backend = engine.getBackend(),
        .runs = samples,
        .summary = summary,
    };
}

pub fn runOllamaRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !OllamaRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.services.connectors.ollama.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.services.shared.time.Timer.start();
        var res = try client.generate(.{
            .model = client.config.model,
            .prompt = prompt,
            .stream = false,
            .options = .{
                .temperature = 0.7,
                .num_predict = gen_tokens,
                .top_p = 0.9,
                .top_k = 40,
            },
        });
        defer res.deinit(allocator);

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
        const prompt_count = res.prompt_eval_count orelse estimateTokenCount(prompt);
        const generated_count = res.eval_count orelse estimateTokenCount(res.response);

        const prefill_tok_s = blk: {
            if (res.prompt_eval_count) |count| {
                if (res.prompt_eval_duration_ns) |dur| {
                    if (dur > 0) {
                        break :blk @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(dur)) / 1_000_000_000.0);
                    }
                }
            }
            break :blk 0.0;
        };

        var decode_tok_s = blk: {
            if (res.eval_count) |count| {
                if (res.eval_duration_ns) |dur| {
                    if (dur > 0) {
                        break :blk @as(f64, @floatFromInt(count)) / (@as(f64, @floatFromInt(dur)) / 1_000_000_000.0);
                    }
                }
            }
            break :blk 0.0;
        };
        if (decode_tok_s <= 0 and elapsed_ms > 0) {
            decode_tok_s = @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0);
        }

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = prefill_tok_s,
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

pub fn runMlxRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !MlxRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.services.connectors.mlx.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.services.shared.time.Timer.start();

        // MLX uses OpenAI-compatible chat completions; use generate() for simplicity
        const response_text = try client.generate(prompt, gen_tokens);
        defer allocator.free(response_text);

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        // MLX server returns usage in the chat response but generate() consumes it.
        // Estimate token counts from text length as fallback.
        const prompt_count = estimateTokenCount(prompt);
        const generated_count = estimateTokenCount(response_text);

        // Derive throughput from wall time since MLX chat API doesn't expose
        // per-phase timing like Ollama's eval_duration_ns fields.
        const decode_tok_s = if (elapsed_ms > 0)
            @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0)
        else
            0.0;

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = 0.0, // Not available from chat completions API
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

pub fn runVllmRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !VllmRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.services.connectors.vllm.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.services.shared.time.Timer.start();

        var res = try client.chatCompletion(.{
            .model = client.config.model,
            .messages = &.{
                .{ .role = "user", .content = prompt },
            },
            .max_tokens = gen_tokens,
            .temperature = 0.7,
        });

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        const prompt_count = res.usage.prompt_tokens;
        const generated_count = res.usage.completion_tokens;

        // Free response strings
        allocator.free(res.id);
        allocator.free(res.model);
        for (res.choices) |choice| {
            allocator.free(choice.message.role);
            allocator.free(choice.message.content);
            allocator.free(choice.finish_reason);
        }
        allocator.free(res.choices);

        const decode_tok_s = if (elapsed_ms > 0)
            @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0)
        else
            0.0;

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = 0.0,
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

pub fn runLmStudioRuntimeBenchmark(
    allocator: std.mem.Allocator,
    prompt: []const u8,
    gen_tokens: u32,
    model_override: ?[]const u8,
    runs: u32,
) !LmStudioRuntimeBenchResult {
    const run_count: usize = @intCast(@max(runs, @as(u32, 1)));
    const samples = try allocator.alloc(RuntimeBenchSample, run_count);
    errdefer allocator.free(samples);

    var client = try abi.services.connectors.lm_studio.createClient(allocator);
    defer client.deinit();

    if (model_override) |model_name| {
        if (model_name.len > 0) {
            if (client.config.model_owned) {
                allocator.free(@constCast(client.config.model));
            }
            client.config.model = try allocator.dupe(u8, model_name);
            client.config.model_owned = true;
        }
    }

    const model_name_copy = try allocator.dupe(u8, client.config.model);
    errdefer allocator.free(model_name_copy);

    for (samples) |*sample| {
        var timer = try abi.services.shared.time.Timer.start();

        var res = try client.chatCompletion(.{
            .model = client.config.model,
            .messages = &.{
                .{ .role = "user", .content = prompt },
            },
            .max_tokens = gen_tokens,
            .temperature = 0.7,
        });

        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;

        const prompt_count = res.usage.prompt_tokens;
        const generated_count = res.usage.completion_tokens;

        // Free response strings
        allocator.free(res.id);
        allocator.free(res.model);
        for (res.choices) |choice| {
            allocator.free(choice.message.role);
            allocator.free(choice.message.content);
            allocator.free(choice.finish_reason);
        }
        allocator.free(res.choices);

        const decode_tok_s = if (elapsed_ms > 0)
            @as(f64, @floatFromInt(generated_count)) / (elapsed_ms / 1000.0)
        else
            0.0;

        sample.* = .{
            .prompt_tokens = prompt_count,
            .generated_tokens = generated_count,
            .elapsed_ms = elapsed_ms,
            .prefill_tok_s = 0.0,
            .decode_tok_s = decode_tok_s,
        };
    }

    const summary = try summarizeRuntimeSamples(allocator, samples);
    return .{
        .model_name = model_name_copy,
        .runs = samples,
        .summary = summary,
    };
}

pub fn buildBenchmarkPrompt(allocator: std.mem.Allocator, prompt_tokens: u32) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);

    const words = @max(prompt_tokens, 16);
    for (0..words) |idx| {
        if (idx != 0) try out.append(allocator, ' ');
        const token = switch (idx % 8) {
            0 => "analyze",
            1 => "latency",
            2 => "throughput",
            3 => "architecture",
            4 => "token",
            5 => "benchmark",
            6 => "inference",
            else => "runtime",
        };
        try out.appendSlice(allocator, token);
    }

    return out.toOwnedSlice(allocator);
}

pub fn estimateTokenCount(text: []const u8) u32 {
    if (text.len == 0) return 0;

    var count: u32 = 0;
    var in_word = false;
    for (text) |c| {
        const is_space = std.ascii.isWhitespace(c);
        if (is_space) {
            in_word = false;
            continue;
        }
        if (!in_word) {
            count += 1;
            in_word = true;
        }
    }
    return count;
}

pub fn printBenchJson(
    _: std.mem.Allocator,
    prompt_tokens: u32,
    gen_tokens: u32,
    runs: u32,
    compute: BenchResult,
    local_runtime: ?RuntimeBenchResult,
    ollama_runtime: ?OllamaRuntimeBenchResult,
    mlx_runtime: ?MlxRuntimeBenchResult,
    vllm_runtime: ?VllmRuntimeBenchResult,
    lmstudio_runtime: ?LmStudioRuntimeBenchResult,
) void {
    utils.output.println("{{", .{});
    utils.output.println("  \"prompt_tokens\": {d},", .{prompt_tokens});
    utils.output.println("  \"gen_tokens\": {d},", .{gen_tokens});
    utils.output.println("  \"runs\": {d},", .{runs});
    utils.output.println("  \"compute\": {{ \"gflops\": {d:.2}, \"est_prefill_tok_s\": {d:.1}, \"est_decode_tok_s\": {d:.1} }},", .{
        compute.gflops,
        compute.est_prefill_tok_s,
        compute.est_decode_tok_s,
    });
    utils.output.println("  \"backends\": {{", .{});

    var printed_any = false;

    if (local_runtime) |local| {
        if (printed_any) utils.output.println(",", .{});
        utils.output.print("    \"local_gguf\": {{ \"status\": \"ok\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"prefill_tok_s\": {{ \"mean\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            local.summary.decode_tok_s.mean,
            local.summary.decode_tok_s.p50,
            local.summary.decode_tok_s.p90,
            local.summary.prefill_tok_s.mean,
            local.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (ollama_runtime) |res| {
        if (printed_any) utils.output.println(",", .{});
        utils.output.print("    \"ollama\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"prefill_tok_s\": {{ \"mean\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.prefill_tok_s.mean,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (mlx_runtime) |res| {
        if (printed_any) utils.output.println(",", .{});
        utils.output.print("    \"mlx\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (vllm_runtime) |res| {
        if (printed_any) utils.output.println(",", .{});
        utils.output.print("    \"vllm\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }
    if (lmstudio_runtime) |res| {
        if (printed_any) utils.output.println(",", .{});
        utils.output.print("    \"lm_studio\": {{ \"status\": \"ok\", \"model\": \"{s}\", \"decode_tok_s\": {{ \"mean\": {d:.2}, \"p50\": {d:.2}, \"p90\": {d:.2} }}, \"wall_ms\": {{ \"mean\": {d:.2} }} }}", .{
            res.model_name,
            res.summary.decode_tok_s.mean,
            res.summary.decode_tok_s.p50,
            res.summary.decode_tok_s.p90,
            res.summary.elapsed_ms.mean,
        });
        printed_any = true;
    }

    utils.output.println("", .{});
    utils.output.println("  }}", .{});
    utils.output.println("}}", .{});
}

pub fn mapRuntimeRunsForJson(allocator: std.mem.Allocator, runs: []const RuntimeBenchSample) ![]RuntimeRunRecordJson {
    const mapped = try allocator.alloc(RuntimeRunRecordJson, runs.len);
    for (runs, 0..) |sample, idx| {
        mapped[idx] = .{
            .wall_ms = sample.elapsed_ms,
            .prefill_tps = sample.prefill_tok_s,
            .decode_tps = sample.decode_tok_s,
            .prompt_tokens = sample.prompt_tokens,
            .generated_tokens = sample.generated_tokens,
        };
    }
    return mapped;
}

pub fn appendBenchRecordToWdbx(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    prompt: []const u8,
    prompt_tokens_target: u32,
    gen_tokens_target: u32,
    runtime_runs: u32,
    compute: BenchResult,
    local_runtime: ?RuntimeBenchResult,
    ollama_runtime: ?OllamaRuntimeBenchResult,
) !void {
    var handle = try abi.features.database.wdbx.createDatabaseWithConfig(allocator, output_path, .{
        .cache_norms = false,
        .initial_capacity = 0,
        .use_vector_pool = false,
        .thread_safe = false,
    });
    defer abi.features.database.wdbx.closeDatabase(&handle);

    abi.features.database.wdbx.restore(&handle, output_path) catch {};

    const now_ms = abi.services.shared.time.unixMs();
    const stats = abi.features.database.wdbx.getStats(&handle);
    const record_id: u64 = stats.count + 1;

    var local_runs_json: []RuntimeRunRecordJson = &.{};
    defer if (local_runs_json.len > 0) allocator.free(local_runs_json);
    if (local_runtime) |local| {
        local_runs_json = try mapRuntimeRunsForJson(allocator, local.runs);
    }

    var ollama_runs_json: []RuntimeRunRecordJson = &.{};
    defer if (ollama_runs_json.len > 0) allocator.free(ollama_runs_json);
    if (ollama_runtime) |ollama| {
        ollama_runs_json = try mapRuntimeRunsForJson(allocator, ollama.runs);
    }

    const local_payload: ?RuntimeSummaryJson = if (local_runtime) |local| .{
        .backend = local.backend.label(),
        .runs = local.runs.len,
        .prompt_tokens_mean = local.summary.prompt_tokens_mean,
        .generated_tokens_mean = local.summary.generated_tokens_mean,
        .wall_ms = local.summary.elapsed_ms,
        .prefill_tps = local.summary.prefill_tok_s,
        .decode_tps = local.summary.decode_tok_s,
        .per_run = local_runs_json,
    } else null;

    const ollama_payload: ?OllamaRuntimeSummaryJson = if (ollama_runtime) |ollama| .{
        .backend = "ollama",
        .model = ollama.model_name,
        .runs = ollama.runs.len,
        .prompt_tokens_mean = ollama.summary.prompt_tokens_mean,
        .generated_tokens_mean = ollama.summary.generated_tokens_mean,
        .wall_ms = ollama.summary.elapsed_ms,
        .prefill_tps = ollama.summary.prefill_tok_s,
        .decode_tps = ollama.summary.decode_tok_s,
        .per_run = ollama_runs_json,
    } else null;

    const comparison_payload: ?struct {
        ollama_over_local_decode_mean: f64,
        ollama_over_local_decode_p50: f64,
    } = blk: {
        if (local_runtime) |local| {
            if (ollama_runtime) |ollama| {
                if (local.summary.decode_tok_s.mean > 0 and local.summary.decode_tok_s.p50 > 0) {
                    break :blk .{
                        .ollama_over_local_decode_mean = ollama.summary.decode_tok_s.mean / local.summary.decode_tok_s.mean,
                        .ollama_over_local_decode_p50 = ollama.summary.decode_tok_s.p50 / local.summary.decode_tok_s.p50,
                    };
                }
            }
        }
        break :blk null;
    };

    const payload = struct {
        schema_version: []const u8,
        ts_ms: i64,
        config: struct {
            prompt: []const u8,
            prompt_tokens_target: u32,
            gen_tokens_target: u32,
            runs: u32,
        },
        compute: struct {
            m: u32,
            k: u32,
            n: u32,
            time_ms: f64,
            gflops: f64,
            est_prefill_tok_s: f64,
            est_decode_tok_s: f64,
        },
        local: ?RuntimeSummaryJson,
        ollama: ?OllamaRuntimeSummaryJson,
        comparison: @TypeOf(comparison_payload),
    }{
        .schema_version = "abi.llm.bench.v2",
        .ts_ms = now_ms,
        .config = .{
            .prompt = prompt,
            .prompt_tokens_target = prompt_tokens_target,
            .gen_tokens_target = gen_tokens_target,
            .runs = runtime_runs,
        },
        .compute = .{
            .m = compute.m,
            .k = compute.k,
            .n = compute.n,
            .time_ms = compute.time_ms,
            .gflops = compute.gflops,
            .est_prefill_tok_s = compute.est_prefill_tok_s,
            .est_decode_tok_s = compute.est_decode_tok_s,
        },
        .local = local_payload,
        .ollama = ollama_payload,
        .comparison = comparison_payload,
    };

    var metadata_writer: std.Io.Writer.Allocating = .init(allocator);
    defer metadata_writer.deinit();
    try std.json.Stringify.value(payload, .{}, &metadata_writer.writer);
    const metadata = try metadata_writer.toOwnedSlice();
    defer allocator.free(metadata);

    try abi.features.database.wdbx.insertVector(&handle, record_id, &[_]f32{}, metadata);
    try abi.features.database.wdbx.backup(&handle, output_path);
    utils.output.printSuccess("WDBX benchmark record appended: {s}", .{output_path});
}

test {
    std.testing.refAllDecls(@This());
}
