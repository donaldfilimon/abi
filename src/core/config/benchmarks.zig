//! Benchmarks configuration.

pub const BenchmarksConfig = struct {
    /// Default number of warmup iterations per benchmark.
    warmup_iterations: u32 = 3,
    /// Default number of sample iterations for timing.
    sample_iterations: u32 = 10,
    /// Whether to export results as JSON.
    export_json: bool = false,
    /// Output path for exported results (when export_json is true).
    output_path: ?[]const u8 = null,

    pub fn defaults() BenchmarksConfig {
        return .{};
    }
};
