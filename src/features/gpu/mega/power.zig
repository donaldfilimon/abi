//! Power and Energy Monitoring for GPU Backends
//!
//! Tracks energy consumption across GPU backends and provides eco-friendly
//! scheduling recommendations based on power efficiency.

const std = @import("std");
const time = @import("../../../services/shared/time.zig");
const sync = @import("../../../services/shared/sync.zig");
const backend_mod = @import("../backend.zig");

/// Power profile for a GPU backend.
pub const BackendPowerProfile = struct {
    backend_type: backend_mod.Backend,
    tdp_watts: f32,
    idle_watts: f32,
    peak_watts: f32,
    efficiency_tflops_per_watt: f32,
    is_low_power: bool,

    pub fn estimatePower(self: BackendPowerProfile, utilization: f32) f32 {
        const util = @min(1.0, @max(0.0, utilization));
        return self.idle_watts + (self.peak_watts - self.idle_watts) * util;
    }

    pub fn calculateEnergy(self: BackendPowerProfile, duration_ms: u64, utilization: f32) f32 {
        const power = self.estimatePower(utilization);
        const hours = @as(f32, @floatFromInt(duration_ms)) / (1000.0 * 3600.0);
        return power * hours;
    }
};

/// Default power profiles for common GPU backends.
pub const default_profiles = struct {
    pub const cuda = BackendPowerProfile{ .backend_type = .cuda, .tdp_watts = 250, .idle_watts = 15, .peak_watts = 350, .efficiency_tflops_per_watt = 0.08, .is_low_power = false };
    pub const metal = BackendPowerProfile{ .backend_type = .metal, .tdp_watts = 60, .idle_watts = 5, .peak_watts = 100, .efficiency_tflops_per_watt = 0.15, .is_low_power = true };
    pub const vulkan = BackendPowerProfile{ .backend_type = .vulkan, .tdp_watts = 200, .idle_watts = 10, .peak_watts = 300, .efficiency_tflops_per_watt = 0.07, .is_low_power = false };
    pub const fpga = BackendPowerProfile{ .backend_type = .fpga, .tdp_watts = 75, .idle_watts = 15, .peak_watts = 100, .efficiency_tflops_per_watt = 0.02, .is_low_power = true };
    pub const stdgpu = BackendPowerProfile{ .backend_type = .stdgpu, .tdp_watts = 65, .idle_watts = 10, .peak_watts = 95, .efficiency_tflops_per_watt = 0.005, .is_low_power = true };
    pub const webgpu = BackendPowerProfile{ .backend_type = .webgpu, .tdp_watts = 50, .idle_watts = 5, .peak_watts = 80, .efficiency_tflops_per_watt = 0.06, .is_low_power = true };

    pub fn getProfile(backend: backend_mod.Backend) BackendPowerProfile {
        return switch (backend) {
            .cuda => cuda,
            .metal => metal,
            .vulkan => vulkan,
            .fpga => fpga,
            .webgpu => webgpu,
            .stdgpu => stdgpu,
            .opengl, .opengles => vulkan,
            .webgl2 => webgpu,
            .tpu => cuda, // TPU: use high-throughput profile until TPU-specific data
            .simulated => stdgpu,
        };
    }
};

/// Energy statistics for a backend.
pub const BackendEnergyStats = struct {
    total_energy_wh: f32 = 0,
    total_runtime_ms: u64 = 0,
    workload_count: u64 = 0,
    avg_power_watts: f32 = 0,
    peak_power_watts: f32 = 0,
    estimated_co2_grams: f32 = 0,
};

/// Configuration for eco-mode scheduling.
pub const EcoModeConfig = struct {
    enabled: bool = false,
    power_budget_watts: f32 = 0,
    min_efficiency: f32 = 0.05,
    carbon_intensity: f32 = 0.5, // kg CO2 per kWh
    max_co2_per_workload_grams: f32 = 0,
};

/// Aggregate energy report across all backends.
pub const EnergyReport = struct {
    total_wh: f32,
    total_runtime_ms: u64,
    total_workloads: u64,
    avg_power_watts: f32,
    estimated_co2_grams: f32,
    most_efficient_backend: ?backend_mod.Backend,
    least_efficient_backend: ?backend_mod.Backend,
    per_backend: [@typeInfo(backend_mod.Backend).@"enum".fields.len]BackendEnergyStats,
};

/// Monitors power consumption and provides eco-mode scoring.
pub const PowerMonitor = struct {
    allocator: std.mem.Allocator,
    profiles: std.AutoHashMapUnmanaged(backend_mod.Backend, BackendPowerProfile),
    per_backend_stats: [@typeInfo(backend_mod.Backend).@"enum".fields.len]BackendEnergyStats,
    eco_config: EcoModeConfig,
    mutex: sync.Mutex,

    const backend_count = @typeInfo(backend_mod.Backend).@"enum".fields.len;

    pub fn init(allocator: std.mem.Allocator) !*PowerMonitor {
        const self = try allocator.create(PowerMonitor);
        self.* = .{
            .allocator = allocator,
            .profiles = .empty,
            .per_backend_stats = [_]BackendEnergyStats{.{}} ** backend_count,
            .eco_config = .{},
            .mutex = .{},
        };
        try self.loadDefaultProfiles();
        return self;
    }

    pub fn deinit(self: *PowerMonitor) void {
        self.profiles.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn loadDefaultProfiles(self: *PowerMonitor) !void {
        const backends = [_]backend_mod.Backend{
            .cuda,   .vulkan,   .metal,  .fpga,      .tpu, .webgpu, .stdgpu,
            .opengl, .opengles, .webgl2, .simulated,
        };
        for (backends) |backend| {
            try self.profiles.put(self.allocator, backend, default_profiles.getProfile(backend));
        }
    }

    /// Record a completed workload's energy consumption.
    pub fn recordWorkload(self: *PowerMonitor, backend: backend_mod.Backend, duration_ms: u64, utilization: f32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const profile = self.profiles.get(backend) orelse default_profiles.getProfile(backend);
        const power = profile.estimatePower(utilization);
        const energy = profile.calculateEnergy(duration_ms, utilization);

        const idx = @intFromEnum(backend);
        if (idx < backend_count) {
            var stats = &self.per_backend_stats[idx];
            stats.total_energy_wh += energy;
            stats.total_runtime_ms += duration_ms;
            stats.workload_count += 1;
            stats.peak_power_watts = @max(stats.peak_power_watts, power);
            stats.estimated_co2_grams += energy * self.eco_config.carbon_intensity;
            if (stats.total_runtime_ms > 0) {
                stats.avg_power_watts = stats.total_energy_wh / (@as(f32, @floatFromInt(stats.total_runtime_ms)) / (1000.0 * 3600.0));
            }
        }
    }

    /// Get aggregate energy report for all backends.
    pub fn getEnergyReport(self: *PowerMonitor) EnergyReport {
        self.mutex.lock();
        defer self.mutex.unlock();

        var report = EnergyReport{
            .total_wh = 0,
            .total_runtime_ms = 0,
            .total_workloads = 0,
            .avg_power_watts = 0,
            .estimated_co2_grams = 0,
            .most_efficient_backend = null,
            .least_efficient_backend = null,
            .per_backend = self.per_backend_stats,
        };

        for (self.per_backend_stats) |stats| {
            report.total_wh += stats.total_energy_wh;
            report.total_runtime_ms += stats.total_runtime_ms;
            report.total_workloads += stats.workload_count;
            report.estimated_co2_grams += stats.estimated_co2_grams;
        }

        if (report.total_runtime_ms > 0) {
            report.avg_power_watts = report.total_wh / (@as(f32, @floatFromInt(report.total_runtime_ms)) / (1000.0 * 3600.0));
        }
        return report;
    }

    /// Get energy stats for a specific backend.
    pub fn getBackendStats(self: *PowerMonitor, backend: backend_mod.Backend) BackendEnergyStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        const idx = @intFromEnum(backend);
        if (idx < backend_count) return self.per_backend_stats[idx];
        return .{};
    }

    /// Enable eco-mode with the given configuration.
    pub fn enableEcoMode(self: *PowerMonitor, config: EcoModeConfig) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.eco_config = config;
        self.eco_config.enabled = true;
    }

    /// Disable eco-mode.
    pub fn disableEcoMode(self: *PowerMonitor) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.eco_config.enabled = false;
    }

    /// Check if eco-mode is enabled.
    pub fn isEcoModeEnabled(self: *PowerMonitor) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.eco_config.enabled;
    }

    /// Get eco-friendliness score for a backend (higher = more efficient).
    /// Used by scheduler for eco-mode decisions.
    pub fn getEcoScore(self: *PowerMonitor, backend: backend_mod.Backend) f32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.eco_config.enabled) return 1.0;
        const profile = self.profiles.get(backend) orelse default_profiles.getProfile(backend);
        var score: f32 = profile.efficiency_tflops_per_watt * 10.0;
        if (profile.is_low_power) score += 2.0;
        return score;
    }

    /// Set a custom power profile for a backend.
    pub fn setProfile(self: *PowerMonitor, backend: backend_mod.Backend, profile: BackendPowerProfile) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.profiles.put(self.allocator, backend, profile);
    }

    /// Get the power profile for a backend.
    pub fn getProfile(self: *PowerMonitor, backend: backend_mod.Backend) BackendPowerProfile {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.profiles.get(backend) orelse default_profiles.getProfile(backend);
    }
};

test "power monitor" {
    const allocator = std.testing.allocator;
    const monitor = try PowerMonitor.init(allocator);
    defer monitor.deinit();

    // Record a workload
    try monitor.recordWorkload(.cuda, 1000, 0.8);
    const stats = monitor.getBackendStats(.cuda);
    try std.testing.expect(stats.workload_count == 1);
    try std.testing.expect(stats.total_runtime_ms == 1000);

    // Test eco-mode
    monitor.enableEcoMode(.{ .carbon_intensity = 0.5 });
    try std.testing.expect(monitor.isEcoModeEnabled());

    const cuda_score = monitor.getEcoScore(.cuda);
    const metal_score = monitor.getEcoScore(.metal);
    try std.testing.expect(metal_score > cuda_score); // Metal is more efficient
}

test "power profile calculations" {
    const profile = default_profiles.cuda;

    // At 50% utilization, power should be between idle and peak
    const power_50 = profile.estimatePower(0.5);
    try std.testing.expect(power_50 > profile.idle_watts);
    try std.testing.expect(power_50 < profile.peak_watts);

    // Energy calculation
    const energy = profile.calculateEnergy(3600000, 1.0); // 1 hour at full load
    try std.testing.expect(energy > 0);
}

test {
    std.testing.refAllDecls(@This());
}
