//! GPU Monitor TUI Widget
//!
//! Provides real-time GPU backend status, memory usage, and scheduler statistics
//! for the TUI interface. Displays device information with sparkline visualizations.

const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const widgets = @import("widgets.zig");
const unicode = @import("unicode.zig");
const render_utils = @import("render_utils.zig");
const layout = @import("layout.zig");

// ===============================================================================
// Types
// ===============================================================================

/// GPU backend types
pub const BackendType = enum {
    none,
    cuda,
    vulkan,
    metal,
    opengl,
    webgpu,
    fpga,
    simulated,

    pub fn name(self: BackendType) []const u8 {
        return switch (self) {
            .none => "None",
            .cuda => "CUDA",
            .vulkan => "Vulkan",
            .metal => "Metal",
            .opengl => "OpenGL",
            .webgpu => "WebGPU",
            .fpga => "FPGA",
            .simulated => "Simulated",
        };
    }
};

/// GPU device status information
pub const GpuDeviceStatus = struct {
    /// Device name/description
    name: []const u8,
    /// Backend type
    backend_type: BackendType,
    /// Total memory in bytes
    total_memory: u64,
    /// Used memory in bytes
    used_memory: u64,
    /// GPU utilization percentage (0-100)
    utilization: u8,
    /// Temperature in Celsius (0 if unavailable)
    temperature: u8,
    /// Whether the device is active/available
    is_active: bool,

    /// Calculate memory usage percentage
    pub fn memoryUsagePercent(self: GpuDeviceStatus) u8 {
        if (self.total_memory == 0) return 0;
        return @intCast(@min(100, (self.used_memory * 100) / self.total_memory));
    }

    /// Format memory size as human-readable string
    pub fn formatMemory(bytes: u64, buf: []u8) []const u8 {
        if (bytes >= 1024 * 1024 * 1024) {
            const gb = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
            return std.fmt.bufPrint(buf, "{d:.1} GB", .{gb}) catch "?";
        } else if (bytes >= 1024 * 1024) {
            const mb = @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
            return std.fmt.bufPrint(buf, "{d:.1} MB", .{mb}) catch "?";
        } else if (bytes >= 1024) {
            const kb = @as(f64, @floatFromInt(bytes)) / 1024.0;
            return std.fmt.bufPrint(buf, "{d:.1} KB", .{kb}) catch "?";
        } else {
            return std.fmt.bufPrint(buf, "{d} B", .{bytes}) catch "?";
        }
    }
};

/// Scheduler statistics
pub const SchedulerStats = struct {
    /// Total number of tasks scheduled
    total_schedules: u64,
    /// Number of successful task completions
    successful_completions: u64,
    /// Number of failed tasks
    failed_tasks: u64,
    /// Current exploration rate (0.0 - 1.0)
    exploration_rate: f32,
    /// Average task latency in microseconds
    avg_latency_us: u64,
    /// Tasks currently in queue
    queue_depth: u32,

    /// Calculate success rate as percentage
    pub fn successRate(self: SchedulerStats) f32 {
        const total = self.successful_completions + self.failed_tasks;
        if (total == 0) return 100.0;
        return @as(f32, @floatFromInt(self.successful_completions)) / @as(f32, @floatFromInt(total)) * 100.0;
    }
};

/// Memory history buffer for sparkline visualization
pub const MemoryHistory = struct {
    const HISTORY_SIZE = 20;

    /// Ring buffer of memory usage percentages (0-100)
    values: [HISTORY_SIZE]u8,
    /// Current write position
    pos: usize,
    /// Number of values written
    count: usize,

    pub fn init() MemoryHistory {
        return .{
            .values = [_]u8{0} ** HISTORY_SIZE,
            .pos = 0,
            .count = 0,
        };
    }

    pub fn push(self: *MemoryHistory, value: u8) void {
        self.values[self.pos] = value;
        self.pos = (self.pos + 1) % HISTORY_SIZE;
        if (self.count < HISTORY_SIZE) {
            self.count += 1;
        }
    }

    /// Get values in chronological order
    pub fn getValues(self: *const MemoryHistory) [HISTORY_SIZE]u8 {
        var result: [HISTORY_SIZE]u8 = [_]u8{0} ** HISTORY_SIZE;
        if (self.count == 0) return result;

        const start = if (self.count < HISTORY_SIZE)
            0
        else
            self.pos;

        for (0..self.count) |i| {
            const idx = (start + i) % HISTORY_SIZE;
            result[i] = self.values[idx];
        }
        return result;
    }
};

// ===============================================================================
// GPU Monitor Widget
// ===============================================================================

pub const GpuMonitor = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,

    /// Currently tracked devices
    devices: std.ArrayListUnmanaged(GpuDeviceStatus),

    /// Memory history per device (indexed by device index)
    memory_histories: std.ArrayListUnmanaged(MemoryHistory),

    /// Current scheduler statistics
    scheduler_stats: SchedulerStats,

    /// Update counter for demo/simulation
    update_counter: u64,

    /// Sparkline characters (from lowest to highest)
    const sparkline_chars = [_][]const u8{ " ", "\u{2581}", "\u{2582}", "\u{2583}", "\u{2584}", "\u{2585}", "\u{2586}", "\u{2587}", "\u{2588}" };

    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) GpuMonitor {
        return .{
            .allocator = allocator,
            .term = term,
            .theme = theme,
            .devices = .empty,
            .memory_histories = .empty,
            .scheduler_stats = .{
                .total_schedules = 0,
                .successful_completions = 0,
                .failed_tasks = 0,
                .exploration_rate = 0.1,
                .avg_latency_us = 0,
                .queue_depth = 0,
            },
            .update_counter = 0,
        };
    }

    pub fn deinit(self: *GpuMonitor) void {
        self.devices.deinit(self.allocator);
        self.memory_histories.deinit(self.allocator);
    }

    /// Add a device to monitor
    pub fn addDevice(self: *GpuMonitor, device: GpuDeviceStatus) !void {
        try self.devices.append(self.allocator, device);
        try self.memory_histories.append(self.allocator, MemoryHistory.init());
    }

    /// Clear all devices
    pub fn clearDevices(self: *GpuMonitor) void {
        self.devices.clearRetainingCapacity();
        self.memory_histories.clearRetainingCapacity();
    }

    /// Update device status at given index
    pub fn updateDevice(self: *GpuMonitor, index: usize, device: GpuDeviceStatus) void {
        if (index < self.devices.items.len) {
            self.devices.items[index] = device;
            // Update memory history
            if (index < self.memory_histories.items.len) {
                self.memory_histories.items[index].push(device.memoryUsagePercent());
            }
        }
    }

    /// Set scheduler statistics
    pub fn setSchedulerStats(self: *GpuMonitor, stats: SchedulerStats) void {
        self.scheduler_stats = stats;
    }

    /// Update with simulated data (for demo purposes)
    pub fn update(self: *GpuMonitor) !void {
        self.update_counter += 1;

        // If no devices, add simulated ones for demo
        if (self.devices.items.len == 0) {
            try self.addDevice(.{
                .name = "NVIDIA RTX 4090 (Simulated)",
                .backend_type = .cuda,
                .total_memory = 24 * 1024 * 1024 * 1024, // 24 GB
                .used_memory = 8 * 1024 * 1024 * 1024, // 8 GB
                .utilization = 45,
                .temperature = 65,
                .is_active = true,
            });
            try self.addDevice(.{
                .name = "Vulkan Device (Simulated)",
                .backend_type = .vulkan,
                .total_memory = 8 * 1024 * 1024 * 1024, // 8 GB
                .used_memory = 2 * 1024 * 1024 * 1024, // 2 GB
                .utilization = 20,
                .temperature = 55,
                .is_active = true,
            });
        }

        // Simulate changing values
        for (self.devices.items, 0..) |*device, i| {
            // Vary utilization and memory slightly
            const variation = @as(i16, @intCast(self.update_counter % 20)) - 10;
            const new_util = @as(i16, device.utilization) + variation;
            device.utilization = @intCast(@max(5, @min(95, new_util)));

            // Vary memory usage
            const mem_base: u64 = device.total_memory / 3;
            const mem_variation = (self.update_counter *% 1024 *% 1024 *% 100) % (device.total_memory / 4);
            device.used_memory = mem_base + mem_variation;

            // Update history
            if (i < self.memory_histories.items.len) {
                self.memory_histories.items[i].push(device.memoryUsagePercent());
            }
        }

        // Simulate scheduler stats changes
        self.scheduler_stats.total_schedules += 1;
        if (self.update_counter % 10 != 0) {
            self.scheduler_stats.successful_completions += 1;
        } else {
            self.scheduler_stats.failed_tasks += 1;
        }
        self.scheduler_stats.exploration_rate = 0.1 + @as(f32, @floatFromInt(self.update_counter % 50)) / 500.0;
        self.scheduler_stats.avg_latency_us = 100 + (self.update_counter % 500);
        self.scheduler_stats.queue_depth = @intCast(self.update_counter % 10);
    }

    /// Render the GPU monitor widget
    pub fn render(self: *GpuMonitor, start_row: u16, start_col: u16, width: u16, height: u16) !void {
        var current_row = start_row;

        // Render header
        try self.renderHeader(current_row, start_col, width);
        current_row += 2;

        // Render each device
        for (self.devices.items, 0..) |device, i| {
            if (current_row >= start_row + height - 4) break;

            const history = if (i < self.memory_histories.items.len)
                &self.memory_histories.items[i]
            else
                null;

            const rows_used = try self.renderDevice(device, history, current_row, start_col, width);
            current_row += rows_used + 1;
        }

        // Render scheduler stats at the bottom
        if (current_row < start_row + height - 2) {
            try self.renderSchedulerStats(current_row, start_col, width);
        }
    }

    fn renderHeader(self: *GpuMonitor, row: u16, col: u16, width: u16) !void {
        if (width < 4) return; // Too narrow to render

        // Position cursor
        try self.setCursorPosition(row, col);

        // Draw top border
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.tl);
        try render_utils.writeRepeat(self.term, widgets.box.h, width -| 2);
        try self.term.write(widgets.box.tr);
        try self.term.write(self.theme.reset);

        // Title line
        try self.setCursorPosition(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" ");
        try self.term.write(self.theme.bold);
        try self.term.write(self.theme.info);
        const title = "GPU Monitor";
        try self.term.write(title);
        try self.term.write(self.theme.reset);

        // Device count
        var buf: [32]u8 = undefined;
        const count_str = std.fmt.bufPrint(&buf, " ({d} devices)", .{self.devices.items.len}) catch "";
        try self.term.write(self.theme.text_dim);
        try self.term.write(count_str);
        try self.term.write(self.theme.reset);

        // Pad and close
        const content_len = unicode.displayWidth(title) + unicode.displayWidth(count_str) + 2;
        if (content_len < width - 2) {
            try render_utils.writeRepeat(self.term, " ", width - 2 - content_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
    }

    fn renderDevice(self: *GpuMonitor, device: GpuDeviceStatus, history: ?*const MemoryHistory, row: u16, col: u16, width: u16) !u16 {
        if (width < 4) return 0; // Too narrow to render

        var current_row = row;

        // Device separator
        try self.setCursorPosition(current_row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.lsep);
        try render_utils.writeRepeat(self.term, widgets.box.h, width -| 2);
        try self.term.write(widgets.box.rsep);
        try self.term.write(self.theme.reset);
        current_row += 1;

        // Device name and backend
        try self.setCursorPosition(current_row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" ");

        // Status indicator
        if (device.is_active) {
            try self.term.write(self.theme.success);
            try self.term.write("\u{25CF} "); // Filled circle
        } else {
            try self.term.write(self.theme.@"error");
            try self.term.write("\u{25CB} "); // Empty circle
        }
        try self.term.write(self.theme.reset);

        // Device name (truncated to fit available width)
        try self.term.write(self.theme.bold);
        const max_name_cols = if (width > 20) width - 20 else 0;
        const truncated_name = unicode.truncateToWidth(device.name, max_name_cols);
        const name_display_width = unicode.displayWidth(truncated_name);
        try self.term.write(truncated_name);
        try self.term.write(self.theme.reset);

        // Backend badge
        const backend_label = device.backend_type.name();
        try self.term.write(" ");
        try self.term.write(self.theme.text_muted);
        try self.term.write("[");
        try self.term.write(backend_label);
        try self.term.write("]");
        try self.term.write(self.theme.reset);

        // Close line
        const line1_len = 4 + name_display_width + 3 + unicode.displayWidth(backend_label);
        const inner1 = @as(usize, width) -| 2;
        if (line1_len < inner1) {
            try render_utils.writeRepeat(self.term, " ", inner1 - line1_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        current_row += 1;

        // Memory line
        try self.setCursorPosition(current_row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write("   Memory: ");

        // Memory bar
        const mem_pct = device.memoryUsagePercent();
        try self.renderProgressBar(mem_pct, 20);

        // Memory values
        var used_buf: [16]u8 = undefined;
        var total_buf: [16]u8 = undefined;
        const used_str = GpuDeviceStatus.formatMemory(device.used_memory, &used_buf);
        const total_str = GpuDeviceStatus.formatMemory(device.total_memory, &total_buf);

        var mem_str_buf: [48]u8 = undefined;
        const mem_str = std.fmt.bufPrint(&mem_str_buf, " {s}/{s} ({d}%)", .{ used_str, total_str, mem_pct }) catch "";
        try self.term.write(self.theme.text_dim);
        try self.term.write(mem_str);
        try self.term.write(self.theme.reset);

        const line2_len = 11 + 20 + unicode.displayWidth(mem_str);
        const inner2 = @as(usize, width) -| 2;
        if (line2_len < inner2) {
            try render_utils.writeRepeat(self.term, " ", inner2 - line2_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        current_row += 1;

        // Utilization and Temperature line
        try self.setCursorPosition(current_row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write("   Util: ");

        // Utilization bar
        try self.renderProgressBar(device.utilization, 10);

        var util_buf: [8]u8 = undefined;
        const util_str = std.fmt.bufPrint(&util_buf, " {d:>3}%", .{device.utilization}) catch "";
        try self.term.write(util_str);

        // Temperature
        try self.term.write("  Temp: ");
        if (device.temperature > 0) {
            // Color based on temperature
            if (device.temperature >= 80) {
                try self.term.write(self.theme.@"error");
            } else if (device.temperature >= 70) {
                try self.term.write(self.theme.warning);
            } else {
                try self.term.write(self.theme.success);
            }
            var temp_buf: [8]u8 = undefined;
            const temp_str = std.fmt.bufPrint(&temp_buf, "{d}C", .{device.temperature}) catch "";
            try self.term.write(temp_str);
            try self.term.write(self.theme.reset);
        } else {
            try self.term.write(self.theme.text_muted);
            try self.term.write("N/A");
            try self.term.write(self.theme.reset);
        }

        const line3_len = 9 + 10 + unicode.displayWidth(util_str) + 8 + 4;
        const inner3 = @as(usize, width) -| 2;
        if (line3_len < inner3) {
            try render_utils.writeRepeat(self.term, " ", inner3 - line3_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        current_row += 1;

        // Sparkline history line (if available)
        if (history) |h| {
            try self.setCursorPosition(current_row, col);
            try self.term.write(self.theme.border);
            try self.term.write(widgets.box.v);
            try self.term.write(self.theme.reset);
            try self.term.write("   History: ");

            try self.renderSparkline(h);

            const line4_len = 12 + MemoryHistory.HISTORY_SIZE;
            if (line4_len < width - 2) {
                try render_utils.writeRepeat(self.term, " ", width - 2 - line4_len);
            }
            try self.term.write(self.theme.border);
            try self.term.write(widgets.box.v);
            try self.term.write(self.theme.reset);
            current_row += 1;
        }

        return current_row - row;
    }

    fn renderSchedulerStats(self: *GpuMonitor, row: u16, col: u16, width: u16) !void {
        if (width < 4) return; // Too narrow to render

        // Separator
        try self.setCursorPosition(row, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.lsep);
        try render_utils.writeRepeat(self.term, widgets.box.h, width -| 2);
        try self.term.write(widgets.box.rsep);
        try self.term.write(self.theme.reset);

        // Stats line 1
        try self.setCursorPosition(row + 1, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);
        try self.term.write(" ");
        try self.term.write(self.theme.bold);
        const sched_label = "Scheduler:";
        try self.term.write(sched_label);
        try self.term.write(self.theme.reset);

        var buf: [64]u8 = undefined;
        const stats_str = std.fmt.bufPrint(&buf, " Tasks: {d} | Success: {d:.1}% | Queue: {d}", .{
            self.scheduler_stats.total_schedules,
            self.scheduler_stats.successRate(),
            self.scheduler_stats.queue_depth,
        }) catch "";
        try self.term.write(self.theme.text_dim);
        try self.term.write(stats_str);
        try self.term.write(self.theme.reset);

        const line1_len = 1 + unicode.displayWidth(sched_label) + unicode.displayWidth(stats_str);
        if (line1_len < width - 2) {
            try render_utils.writeRepeat(self.term, " ", width - 2 - line1_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);

        // Stats line 2
        try self.setCursorPosition(row + 2, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);

        var buf2: [64]u8 = undefined;
        const stats_str2 = std.fmt.bufPrint(&buf2, " Explore: {d:.2} | Latency: {d}us", .{
            self.scheduler_stats.exploration_rate,
            self.scheduler_stats.avg_latency_us,
        }) catch "";
        try self.term.write(self.theme.text_dim);
        try self.term.write(stats_str2);
        try self.term.write(self.theme.reset);

        const line2_len = unicode.displayWidth(stats_str2);
        if (line2_len < width - 2) {
            try render_utils.writeRepeat(self.term, " ", width - 2 - line2_len);
        }
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.v);
        try self.term.write(self.theme.reset);

        // Bottom border
        try self.setCursorPosition(row + 3, col);
        try self.term.write(self.theme.border);
        try self.term.write(widgets.box.bl);
        try render_utils.writeRepeat(self.term, widgets.box.h, width -| 2);
        try self.term.write(widgets.box.br);
        try self.term.write(self.theme.reset);
    }

    fn renderProgressBar(self: *GpuMonitor, percent: u8, width: usize) !void {
        const filled = (@as(usize, percent) * width) / 100;

        // Color based on percentage
        if (percent >= 90) {
            try self.term.write(self.theme.@"error");
        } else if (percent >= 70) {
            try self.term.write(self.theme.warning);
        } else {
            try self.term.write(self.theme.success);
        }

        for (0..width) |i| {
            if (i < filled) {
                try self.term.write("\u{2588}"); // Full block
            } else {
                try self.term.write(self.theme.text_muted);
                try self.term.write("\u{2591}"); // Light shade
            }
        }
        try self.term.write(self.theme.reset);
    }

    fn renderSparkline(self: *GpuMonitor, history: *const MemoryHistory) !void {
        const values = history.getValues();

        try self.term.write(self.theme.info);
        for (values) |val| {
            // Map 0-100 to sparkline character index (0-8)
            const idx: usize = @min(8, val / 12);
            try self.term.write(sparkline_chars[idx]);
        }
        try self.term.write(self.theme.reset);
    }

    fn setCursorPosition(self: *GpuMonitor, row: u16, col: u16) !void {
        // Delegate to terminal.moveTo (0-indexed), converting from 1-indexed.
        try self.term.moveTo(row -| 1, col -| 1);
    }
};

// ===============================================================================
// Tests
// ===============================================================================

test "GpuDeviceStatus memory usage percent" {
    const device = GpuDeviceStatus{
        .name = "Test GPU",
        .backend_type = .cuda,
        .total_memory = 1000,
        .used_memory = 250,
        .utilization = 50,
        .temperature = 60,
        .is_active = true,
    };

    try std.testing.expectEqual(@as(u8, 25), device.memoryUsagePercent());
}

test "GpuDeviceStatus memory usage percent zero total" {
    const device = GpuDeviceStatus{
        .name = "Test GPU",
        .backend_type = .vulkan,
        .total_memory = 0,
        .used_memory = 0,
        .utilization = 0,
        .temperature = 0,
        .is_active = false,
    };

    try std.testing.expectEqual(@as(u8, 0), device.memoryUsagePercent());
}

test "SchedulerStats success rate" {
    const stats = SchedulerStats{
        .total_schedules = 100,
        .successful_completions = 90,
        .failed_tasks = 10,
        .exploration_rate = 0.1,
        .avg_latency_us = 100,
        .queue_depth = 5,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 90.0), stats.successRate(), 0.01);
}

test "SchedulerStats success rate zero tasks" {
    const stats = SchedulerStats{
        .total_schedules = 0,
        .successful_completions = 0,
        .failed_tasks = 0,
        .exploration_rate = 0.0,
        .avg_latency_us = 0,
        .queue_depth = 0,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 100.0), stats.successRate(), 0.01);
}

test "MemoryHistory push and get" {
    var history = MemoryHistory.init();

    history.push(10);
    history.push(20);
    history.push(30);

    const values = history.getValues();
    try std.testing.expectEqual(@as(u8, 10), values[0]);
    try std.testing.expectEqual(@as(u8, 20), values[1]);
    try std.testing.expectEqual(@as(u8, 30), values[2]);
}

test "MemoryHistory ring buffer" {
    var history = MemoryHistory.init();

    // Fill buffer
    for (0..MemoryHistory.HISTORY_SIZE) |i| {
        history.push(@intCast(i));
    }

    // Add more to test wrap-around
    history.push(100);
    history.push(101);

    const values = history.getValues();
    // First values should now be 2, 3, 4... then 100, 101
    try std.testing.expectEqual(@as(u8, 2), values[0]);
    try std.testing.expectEqual(@as(u8, 100), values[MemoryHistory.HISTORY_SIZE - 2]);
    try std.testing.expectEqual(@as(u8, 101), values[MemoryHistory.HISTORY_SIZE - 1]);
}

test "GpuMonitor init and deinit" {
    const allocator = std.testing.allocator;
    var term = terminal.Terminal.init(allocator);
    defer term.deinit();

    const theme = &themes.themes.default;
    var monitor = GpuMonitor.init(allocator, &term, theme);
    defer monitor.deinit();

    try std.testing.expectEqual(@as(usize, 0), monitor.devices.items.len);
}

test "GpuMonitor add device" {
    const allocator = std.testing.allocator;
    var term = terminal.Terminal.init(allocator);
    defer term.deinit();

    const theme = &themes.themes.default;
    var monitor = GpuMonitor.init(allocator, &term, theme);
    defer monitor.deinit();

    try monitor.addDevice(.{
        .name = "Test Device",
        .backend_type = .cuda,
        .total_memory = 8 * 1024 * 1024 * 1024,
        .used_memory = 2 * 1024 * 1024 * 1024,
        .utilization = 50,
        .temperature = 65,
        .is_active = true,
    });

    try std.testing.expectEqual(@as(usize, 1), monitor.devices.items.len);
    try std.testing.expectEqual(@as(usize, 1), monitor.memory_histories.items.len);
}

test "GpuMonitor update creates simulated devices" {
    const allocator = std.testing.allocator;
    var term = terminal.Terminal.init(allocator);
    defer term.deinit();

    const theme = &themes.themes.default;
    var monitor = GpuMonitor.init(allocator, &term, theme);
    defer monitor.deinit();

    try monitor.update();

    // Should have created simulated devices
    try std.testing.expect(monitor.devices.items.len > 0);
}

test "BackendType name" {
    try std.testing.expectEqualStrings("CUDA", BackendType.cuda.name());
    try std.testing.expectEqualStrings("Vulkan", BackendType.vulkan.name());
    try std.testing.expectEqualStrings("Metal", BackendType.metal.name());
    try std.testing.expectEqualStrings("None", BackendType.none.name());
}

test {
    std.testing.refAllDecls(@This());
}
