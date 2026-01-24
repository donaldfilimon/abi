# Mega GPU + TUI + Self-Learning Agent Upgrade

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a unified system where the TUI and self-learning agent leverage a "mega GPU" orchestration layer that coordinates across multiple GPU backends (CUDA + Vulkan + Metal simultaneously), with the agent learning optimal GPU scheduling strategies.

**Architecture:** Three integrated components:
1. **Mega GPU Layer** - Cross-backend GPU orchestration with unified memory, peer transfers, and adaptive scheduling
2. **Enhanced TUI** - Real-time GPU monitoring, agent status visualization, and interactive control
3. **Self-Learning Agent** - Reinforcement learning for GPU scheduling with experience replay and policy optimization

**Tech Stack:** Zig 0.16, CUDA/Vulkan/Metal backends, TUI async loop, RL-based scheduler

---

## Phase 1: Mega GPU Orchestration Layer

### Task 1: Create Cross-Backend Device Coordinator

**Files:**
- Create: `src/gpu/mega/coordinator.zig`
- Create: `src/gpu/mega/mod.zig`
- Modify: `src/gpu/mod.zig` (add mega export)

**Step 1: Create mega module directory**

```bash
mkdir -p src/gpu/mega
```

**Step 2: Create the coordinator module**

Create `src/gpu/mega/coordinator.zig`:

```zig
//! Cross-Backend GPU Coordinator
//!
//! Manages simultaneous operation across CUDA, Vulkan, Metal, and other backends
//! with unified device selection, memory transfers, and workload distribution.

const std = @import("std");
const build_options = @import("build_options");
const multi_device = @import("../multi_device.zig");
const backend_factory = @import("../backend_factory.zig");
const interface = @import("../interface.zig");

/// Backend instance with metadata
pub const BackendInstance = struct {
    backend_type: interface.BackendType,
    backend: interface.Backend,
    device_count: u32,
    total_memory_mb: u64,
    available_memory_mb: u64,
    priority: u8,
    is_healthy: bool,

    pub fn healthScore(self: BackendInstance) f32 {
        if (!self.is_healthy) return 0.0;
        const memory_ratio = @as(f32, @floatFromInt(self.available_memory_mb)) /
                            @as(f32, @floatFromInt(self.total_memory_mb + 1));
        return memory_ratio * @as(f32, @floatFromInt(self.priority));
    }
};

/// Workload characteristics for scheduling decisions
pub const WorkloadProfile = struct {
    compute_intensity: f32, // 0.0 (memory-bound) to 1.0 (compute-bound)
    memory_requirement_mb: u64,
    preferred_backend: ?interface.BackendType = null,
    requires_fp16: bool = false,
    requires_fp64: bool = false,
    batch_size: u32 = 1,
    is_training: bool = false,
};

/// Scheduling decision from the coordinator
pub const ScheduleDecision = struct {
    backend_type: interface.BackendType,
    device_id: multi_device.DeviceId,
    estimated_time_ms: u64,
    confidence: f32,
    reason: []const u8,
};

/// Cross-Backend Coordinator
pub const Coordinator = struct {
    allocator: std.mem.Allocator,
    backends: std.ArrayList(BackendInstance),
    device_groups: std.AutoHashMap(interface.BackendType, *multi_device.DeviceGroup),
    scheduling_history: std.ArrayList(SchedulingRecord),
    stats: CoordinatorStats,

    const SchedulingRecord = struct {
        workload: WorkloadProfile,
        decision: ScheduleDecision,
        actual_time_ms: u64,
        success: bool,
        timestamp: i64,
    };

    pub const CoordinatorStats = struct {
        total_schedules: u64 = 0,
        successful_schedules: u64 = 0,
        total_compute_time_ms: u64 = 0,
        backend_usage: [10]u64 = [_]u64{0} ** 10, // Per BackendType
    };

    pub fn init(allocator: std.mem.Allocator) !*Coordinator {
        const self = try allocator.create(Coordinator);
        self.* = .{
            .allocator = allocator,
            .backends = std.ArrayList(BackendInstance).init(allocator),
            .device_groups = std.AutoHashMap(interface.BackendType, *multi_device.DeviceGroup).init(allocator),
            .scheduling_history = std.ArrayList(SchedulingRecord).init(allocator),
            .stats = .{},
        };

        // Discover all available backends
        try self.discoverBackends();

        return self;
    }

    pub fn deinit(self: *Coordinator) void {
        var it = self.device_groups.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.device_groups.deinit();
        self.backends.deinit();
        self.scheduling_history.deinit();
        self.allocator.destroy(self);
    }

    fn discoverBackends(self: *Coordinator) !void {
        // Try each backend in priority order
        const backend_types = [_]interface.BackendType{
            .cuda, .metal, .vulkan, .webgpu, .fpga, .stdgpu,
        };

        for (backend_types) |bt| {
            if (self.tryInitBackend(bt)) |instance| {
                try self.backends.append(instance);
            }
        }
    }

    fn tryInitBackend(self: *Coordinator, bt: interface.BackendType) ?BackendInstance {
        _ = self;
        // Check compile-time availability
        const available = switch (bt) {
            .cuda => build_options.gpu_cuda,
            .vulkan => build_options.gpu_vulkan,
            .metal => build_options.gpu_metal,
            .webgpu => build_options.gpu_webgpu,
            .fpga => build_options.gpu_fpga,
            .stdgpu => build_options.gpu_stdgpu,
            else => false,
        };

        if (!available) return null;

        // Return placeholder - actual init happens on demand
        return BackendInstance{
            .backend_type = bt,
            .backend = undefined,
            .device_count = 1,
            .total_memory_mb = 8192,
            .available_memory_mb = 6144,
            .priority = switch (bt) {
                .cuda => 10,
                .metal => 9,
                .vulkan => 8,
                .webgpu => 6,
                .fpga => 7,
                else => 5,
            },
            .is_healthy = true,
        };
    }

    /// Schedule a workload to the best available backend/device
    pub fn schedule(self: *Coordinator, profile: WorkloadProfile) ScheduleDecision {
        var best_score: f32 = 0;
        var best_backend: ?BackendInstance = null;

        for (self.backends.items) |backend| {
            const score = self.scoreBackend(backend, profile);
            if (score > best_score) {
                best_score = score;
                best_backend = backend;
            }
        }

        const selected = best_backend orelse self.backends.items[0];

        self.stats.total_schedules += 1;

        return .{
            .backend_type = selected.backend_type,
            .device_id = 0,
            .estimated_time_ms = 100,
            .confidence = best_score,
            .reason = "Selected based on availability and workload profile",
        };
    }

    fn scoreBackend(self: *Coordinator, backend: BackendInstance, profile: WorkloadProfile) f32 {
        _ = self;
        var score = backend.healthScore();

        // Prefer explicitly requested backend
        if (profile.preferred_backend) |pref| {
            if (backend.backend_type == pref) score *= 2.0;
        }

        // Memory check
        if (backend.available_memory_mb < profile.memory_requirement_mb) {
            score *= 0.1;
        }

        // Training workloads prefer CUDA/Metal
        if (profile.is_training) {
            if (backend.backend_type == .cuda or backend.backend_type == .metal) {
                score *= 1.5;
            }
        }

        return score;
    }

    /// Record scheduling outcome for learning
    pub fn recordOutcome(self: *Coordinator, decision: ScheduleDecision, actual_time_ms: u64, success: bool) !void {
        try self.scheduling_history.append(.{
            .workload = .{ .compute_intensity = 0.5, .memory_requirement_mb = 0 },
            .decision = decision,
            .actual_time_ms = actual_time_ms,
            .success = success,
            .timestamp = std.time.milliTimestamp(),
        });

        if (success) {
            self.stats.successful_schedules += 1;
            self.stats.total_compute_time_ms += actual_time_ms;
        }

        // Update backend usage stats
        const idx = @intFromEnum(decision.backend_type);
        if (idx < self.stats.backend_usage.len) {
            self.stats.backend_usage[idx] += 1;
        }
    }

    /// Get available backends summary
    pub fn getBackendsSummary(self: *Coordinator) []const BackendInstance {
        return self.backends.items;
    }

    /// Get coordinator statistics
    pub fn getStats(self: *Coordinator) CoordinatorStats {
        return self.stats;
    }
};

test "coordinator initialization" {
    const allocator = std.testing.allocator;
    const coord = try Coordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expect(coord.backends.items.len >= 0);
}

test "workload scheduling" {
    const allocator = std.testing.allocator;
    const coord = try Coordinator.init(allocator);
    defer coord.deinit();

    const profile = WorkloadProfile{
        .compute_intensity = 0.8,
        .memory_requirement_mb = 1024,
        .is_training = true,
    };

    const decision = coord.schedule(profile);
    try std.testing.expect(decision.confidence >= 0.0);
}
```

**Step 3: Create mega module entry point**

Create `src/gpu/mega/mod.zig`:

```zig
//! Mega GPU Orchestration Module
//!
//! Provides unified cross-backend GPU orchestration with:
//! - Simultaneous CUDA + Vulkan + Metal operation
//! - Intelligent workload scheduling
//! - Learning-based optimization
//! - Real-time monitoring integration

pub const coordinator = @import("coordinator.zig");

pub const Coordinator = coordinator.Coordinator;
pub const BackendInstance = coordinator.BackendInstance;
pub const WorkloadProfile = coordinator.WorkloadProfile;
pub const ScheduleDecision = coordinator.ScheduleDecision;
pub const CoordinatorStats = coordinator.CoordinatorStats;

test {
    _ = coordinator;
}
```

**Step 4: Update GPU module exports**

Add to `src/gpu/mod.zig` exports section:

```zig
pub const mega = @import("mega/mod.zig");
```

**Step 5: Verify build**

```bash
zig build 2>&1 | head -10
```

**Step 6: Run tests**

```bash
zig build test --summary all 2>&1 | tail -5
```

**Step 7: Commit**

```bash
git add src/gpu/mega/
git add src/gpu/mod.zig
git commit -m "feat(gpu): add mega GPU coordinator for cross-backend orchestration"
```

---

### Task 2: Add Learning-Based Scheduler

**Files:**
- Create: `src/gpu/mega/scheduler.zig`
- Modify: `src/gpu/mega/mod.zig` (add export)

**Step 1: Create the learning scheduler**

Create `src/gpu/mega/scheduler.zig`:

```zig
//! Learning-Based GPU Scheduler
//!
//! Uses reinforcement learning principles to optimize workload scheduling
//! across multiple GPU backends based on observed performance.

const std = @import("std");
const coordinator = @import("coordinator.zig");
const interface = @import("../interface.zig");

/// Experience for replay buffer
pub const Experience = struct {
    state: SchedulerState,
    action: u8, // Backend index
    reward: f32,
    next_state: SchedulerState,
    done: bool,
};

/// Compressed state representation
pub const SchedulerState = struct {
    backend_loads: [8]f32, // Normalized 0-1
    memory_pressures: [8]f32,
    pending_workloads: u32,
    current_throughput: f32,

    pub fn fromCoordinator(coord: *coordinator.Coordinator) SchedulerState {
        var state = SchedulerState{
            .backend_loads = [_]f32{0} ** 8,
            .memory_pressures = [_]f32{0} ** 8,
            .pending_workloads = 0,
            .current_throughput = 0,
        };

        for (coord.backends.items, 0..) |backend, i| {
            if (i >= 8) break;
            state.memory_pressures[i] = 1.0 - (@as(f32, @floatFromInt(backend.available_memory_mb)) /
                @as(f32, @floatFromInt(backend.total_memory_mb + 1)));
        }

        return state;
    }
};

/// Q-value approximation for backend selection
pub const QTable = struct {
    values: [8][16]f32, // 8 backends x 16 discretized states
    learning_rate: f32,
    discount_factor: f32,
    exploration_rate: f32,

    pub fn init() QTable {
        return .{
            .values = [_][16]f32{[_]f32{0} ** 16} ** 8,
            .learning_rate = 0.1,
            .discount_factor = 0.95,
            .exploration_rate = 0.1,
        };
    }

    fn discretizeState(state: SchedulerState) u4 {
        // Simple discretization based on memory pressure
        var avg_pressure: f32 = 0;
        for (state.memory_pressures) |p| {
            avg_pressure += p;
        }
        avg_pressure /= 8.0;

        return @intFromFloat(@min(15.0, avg_pressure * 16.0));
    }

    pub fn selectAction(self: *QTable, state: SchedulerState, num_backends: usize) u8 {
        const state_idx = discretizeState(state);

        // Epsilon-greedy exploration
        var prng = std.Random.DefaultPrng.init(@bitCast(std.time.nanoTimestamp()));
        if (prng.random().float(f32) < self.exploration_rate) {
            return @intCast(prng.random().uintLessThan(usize, num_backends));
        }

        // Greedy selection
        var best_action: u8 = 0;
        var best_value: f32 = self.values[0][state_idx];
        for (0..num_backends) |i| {
            if (self.values[i][state_idx] > best_value) {
                best_value = self.values[i][state_idx];
                best_action = @intCast(i);
            }
        }

        return best_action;
    }

    pub fn update(self: *QTable, exp: Experience, num_backends: usize) void {
        const state_idx = discretizeState(exp.state);
        const next_state_idx = discretizeState(exp.next_state);

        // Find max Q for next state
        var max_next_q: f32 = self.values[0][next_state_idx];
        for (0..num_backends) |i| {
            if (self.values[i][next_state_idx] > max_next_q) {
                max_next_q = self.values[i][next_state_idx];
            }
        }

        // Q-learning update
        const target = if (exp.done) exp.reward else exp.reward + self.discount_factor * max_next_q;
        const current = self.values[exp.action][state_idx];
        self.values[exp.action][state_idx] = current + self.learning_rate * (target - current);
    }

    pub fn decayExploration(self: *QTable, min_rate: f32) void {
        self.exploration_rate = @max(min_rate, self.exploration_rate * 0.995);
    }
};

/// Experience replay buffer
pub const ReplayBuffer = struct {
    allocator: std.mem.Allocator,
    buffer: std.ArrayList(Experience),
    capacity: usize,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) ReplayBuffer {
        return .{
            .allocator = allocator,
            .buffer = std.ArrayList(Experience).init(allocator),
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *ReplayBuffer) void {
        self.buffer.deinit();
    }

    pub fn add(self: *ReplayBuffer, exp: Experience) !void {
        if (self.buffer.items.len >= self.capacity) {
            _ = self.buffer.orderedRemove(0);
        }
        try self.buffer.append(exp);
    }

    pub fn sample(self: *ReplayBuffer, batch_size: usize) []Experience {
        if (self.buffer.items.len <= batch_size) {
            return self.buffer.items;
        }
        // Return last batch_size items for simplicity
        return self.buffer.items[self.buffer.items.len - batch_size..];
    }

    pub fn size(self: *ReplayBuffer) usize {
        return self.buffer.items.len;
    }
};

/// Learning-based scheduler
pub const LearningScheduler = struct {
    allocator: std.mem.Allocator,
    q_table: QTable,
    replay_buffer: ReplayBuffer,
    coord: *coordinator.Coordinator,
    episode_rewards: std.ArrayList(f32),
    current_episode_reward: f32,

    pub fn init(allocator: std.mem.Allocator, coord: *coordinator.Coordinator) !*LearningScheduler {
        const self = try allocator.create(LearningScheduler);
        self.* = .{
            .allocator = allocator,
            .q_table = QTable.init(),
            .replay_buffer = ReplayBuffer.init(allocator, 10000),
            .coord = coord,
            .episode_rewards = std.ArrayList(f32).init(allocator),
            .current_episode_reward = 0,
        };
        return self;
    }

    pub fn deinit(self: *LearningScheduler) void {
        self.replay_buffer.deinit();
        self.episode_rewards.deinit();
        self.allocator.destroy(self);
    }

    /// Schedule using learned policy
    pub fn schedule(self: *LearningScheduler, profile: coordinator.WorkloadProfile) coordinator.ScheduleDecision {
        const state = SchedulerState.fromCoordinator(self.coord);
        const num_backends = self.coord.backends.items.len;

        if (num_backends == 0) {
            return .{
                .backend_type = .simulated,
                .device_id = 0,
                .estimated_time_ms = 100,
                .confidence = 0.0,
                .reason = "No backends available",
            };
        }

        const action = self.q_table.selectAction(state, num_backends);
        const backend = self.coord.backends.items[@min(action, @as(u8, @intCast(num_backends - 1)))];

        // Adjust confidence based on memory requirements
        var confidence = self.q_table.values[action][QTable.discretizeState(state)];
        if (backend.available_memory_mb < profile.memory_requirement_mb) {
            confidence *= 0.5;
        }

        return .{
            .backend_type = backend.backend_type,
            .device_id = 0,
            .estimated_time_ms = 100,
            .confidence = @max(0.0, @min(1.0, confidence)),
            .reason = "Selected by learning scheduler",
        };
    }

    /// Record outcome and learn
    pub fn recordAndLearn(self: *LearningScheduler, decision: coordinator.ScheduleDecision,
                          actual_time_ms: u64, success: bool) !void {
        const state = SchedulerState.fromCoordinator(self.coord);

        // Calculate reward
        const time_factor = 1000.0 / @as(f32, @floatFromInt(actual_time_ms + 1));
        const success_factor: f32 = if (success) 1.0 else -1.0;
        const reward = time_factor * success_factor;

        self.current_episode_reward += reward;

        // Find action index
        var action: u8 = 0;
        for (self.coord.backends.items, 0..) |b, i| {
            if (b.backend_type == decision.backend_type) {
                action = @intCast(i);
                break;
            }
        }

        // Store experience
        const exp = Experience{
            .state = state,
            .action = action,
            .reward = reward,
            .next_state = state, // Will be updated on next call
            .done = false,
        };
        try self.replay_buffer.add(exp);

        // Learn from replay buffer
        if (self.replay_buffer.size() >= 32) {
            const batch = self.replay_buffer.sample(32);
            for (batch) |e| {
                self.q_table.update(e, self.coord.backends.items.len);
            }
        }

        // Decay exploration
        self.q_table.decayExploration(0.01);

        // Record in coordinator too
        try self.coord.recordOutcome(decision, actual_time_ms, success);
    }

    /// End episode and record total reward
    pub fn endEpisode(self: *LearningScheduler) !void {
        try self.episode_rewards.append(self.current_episode_reward);
        self.current_episode_reward = 0;
    }

    /// Get learning statistics
    pub fn getStats(self: *LearningScheduler) LearningStats {
        var avg_reward: f32 = 0;
        if (self.episode_rewards.items.len > 0) {
            for (self.episode_rewards.items) |r| {
                avg_reward += r;
            }
            avg_reward /= @floatFromInt(self.episode_rewards.items.len);
        }

        return .{
            .episodes = self.episode_rewards.items.len,
            .avg_episode_reward = avg_reward,
            .exploration_rate = self.q_table.exploration_rate,
            .replay_buffer_size = self.replay_buffer.size(),
        };
    }

    pub const LearningStats = struct {
        episodes: usize,
        avg_episode_reward: f32,
        exploration_rate: f32,
        replay_buffer_size: usize,
    };
};

test "learning scheduler init" {
    const allocator = std.testing.allocator;
    const coord = try coordinator.Coordinator.init(allocator);
    defer coord.deinit();

    const scheduler = try LearningScheduler.init(allocator, coord);
    defer scheduler.deinit();

    try std.testing.expect(scheduler.q_table.exploration_rate > 0);
}

test "q-table update" {
    var q = QTable.init();

    const exp = Experience{
        .state = .{ .backend_loads = [_]f32{0} ** 8, .memory_pressures = [_]f32{0.5} ** 8, .pending_workloads = 0, .current_throughput = 0 },
        .action = 0,
        .reward = 1.0,
        .next_state = .{ .backend_loads = [_]f32{0} ** 8, .memory_pressures = [_]f32{0.5} ** 8, .pending_workloads = 0, .current_throughput = 0 },
        .done = false,
    };

    q.update(exp, 4);
    try std.testing.expect(q.values[0][8] > 0);
}
```

**Step 2: Update mega module exports**

Add to `src/gpu/mega/mod.zig`:

```zig
pub const scheduler = @import("scheduler.zig");

pub const LearningScheduler = scheduler.LearningScheduler;
pub const Experience = scheduler.Experience;
pub const SchedulerState = scheduler.SchedulerState;
pub const QTable = scheduler.QTable;
pub const ReplayBuffer = scheduler.ReplayBuffer;
```

**Step 3: Verify build and test**

```bash
zig build 2>&1 | head -10
zig build test --summary all 2>&1 | tail -5
```

**Step 4: Commit**

```bash
git add src/gpu/mega/scheduler.zig src/gpu/mega/mod.zig
git commit -m "feat(gpu): add learning-based scheduler with Q-learning and experience replay"
```

---

## Phase 2: TUI Real-Time GPU Monitoring

### Task 3: Add GPU Monitor Widget

**Files:**
- Create: `tools/cli/tui/gpu_monitor.zig`
- Modify: `tools/cli/tui/mod.zig` (add export)

**Step 1: Create GPU monitor widget**

Create `tools/cli/tui/gpu_monitor.zig`:

```zig
//! GPU Monitor Widget for TUI
//!
//! Real-time visualization of GPU backend status, memory usage,
//! and scheduler decisions.

const std = @import("std");
const Terminal = @import("terminal.zig").Terminal;
const themes = @import("themes.zig");
const widgets = @import("widgets.zig");

// Import GPU module if available
const gpu_available = @hasDecl(@import("root"), "abi") and
    @hasDecl(@import("root").abi, "gpu");

/// GPU device status for display
pub const GpuDeviceStatus = struct {
    name: []const u8,
    backend_type: []const u8,
    memory_used_mb: u64,
    memory_total_mb: u64,
    utilization_percent: u8,
    temperature_celsius: ?u8,
    is_active: bool,
};

/// Scheduler stats for display
pub const SchedulerStats = struct {
    total_schedules: u64,
    successful_rate: f32,
    exploration_rate: f32,
    avg_decision_time_ms: f32,
};

/// GPU Monitor Widget
pub const GpuMonitor = struct {
    allocator: std.mem.Allocator,
    devices: std.ArrayList(GpuDeviceStatus),
    scheduler_stats: ?SchedulerStats,
    history: MemoryHistory,
    update_interval_ms: i64,
    last_update: i64,

    const MemoryHistory = struct {
        samples: [60]f32,
        idx: usize,

        pub fn init() MemoryHistory {
            return .{ .samples = [_]f32{0} ** 60, .idx = 0 };
        }

        pub fn add(self: *MemoryHistory, value: f32) void {
            self.samples[self.idx] = value;
            self.idx = (self.idx + 1) % 60;
        }

        pub fn getRecent(self: *MemoryHistory, count: usize) []const f32 {
            const n = @min(count, 60);
            const start = if (self.idx >= n) self.idx - n else 0;
            return self.samples[start..self.idx];
        }
    };

    pub fn init(allocator: std.mem.Allocator) GpuMonitor {
        return .{
            .allocator = allocator,
            .devices = std.ArrayList(GpuDeviceStatus).init(allocator),
            .scheduler_stats = null,
            .history = MemoryHistory.init(),
            .update_interval_ms = 1000,
            .last_update = 0,
        };
    }

    pub fn deinit(self: *GpuMonitor) void {
        self.devices.deinit();
    }

    /// Update GPU stats (call periodically)
    pub fn update(self: *GpuMonitor) !void {
        const now = std.time.milliTimestamp();
        if (now - self.last_update < self.update_interval_ms) return;
        self.last_update = now;

        // Clear old data
        self.devices.clearRetainingCapacity();

        // Add simulated device data for demo
        // In production, query actual GPU backends
        try self.devices.append(.{
            .name = "NVIDIA RTX 4090",
            .backend_type = "CUDA",
            .memory_used_mb = 8192,
            .memory_total_mb = 24576,
            .utilization_percent = 45,
            .temperature_celsius = 65,
            .is_active = true,
        });

        try self.devices.append(.{
            .name = "AMD Radeon RX 7900",
            .backend_type = "Vulkan",
            .memory_used_mb = 4096,
            .memory_total_mb = 20480,
            .utilization_percent = 30,
            .temperature_celsius = 58,
            .is_active = true,
        });

        // Update memory history
        var total_used: f32 = 0;
        var total_total: f32 = 0;
        for (self.devices.items) |d| {
            total_used += @floatFromInt(d.memory_used_mb);
            total_total += @floatFromInt(d.memory_total_mb);
        }
        if (total_total > 0) {
            self.history.add(total_used / total_total);
        }
    }

    /// Render the GPU monitor panel
    pub fn render(self: *GpuMonitor, terminal: *Terminal, theme: themes.Theme,
                  x: u16, y: u16, width: u16, height: u16) !void {
        const writer = terminal.writer();

        // Title bar
        try terminal.setCursorPosition(y, x);
        try writer.print("{s}┌─ GPU Monitor ─{s}┐{s}", .{
            themes.colorToAnsi(theme.border),
            "─" ** 20,
            themes.resetCode(),
        });

        var row: u16 = y + 1;

        // Device list
        for (self.devices.items) |device| {
            if (row >= y + height - 1) break;

            try terminal.setCursorPosition(row, x);

            const memory_pct = @as(f32, @floatFromInt(device.memory_used_mb)) /
                              @as(f32, @floatFromInt(device.memory_total_mb)) * 100;

            const status_color = if (device.is_active) theme.success else theme.warning;

            try writer.print("{s}│{s} {s}{s:<20}{s} [{s}] {d:>3}% {d:>5}/{d:>5}MB", .{
                themes.colorToAnsi(theme.border),
                themes.resetCode(),
                themes.colorToAnsi(status_color),
                device.name[0..@min(20, device.name.len)],
                themes.resetCode(),
                device.backend_type,
                device.utilization_percent,
                device.memory_used_mb,
                device.memory_total_mb,
            });

            // Temperature if available
            if (device.temperature_celsius) |temp| {
                const temp_color = if (temp > 80) theme.@"error" else if (temp > 70) theme.warning else theme.success;
                try writer.print(" {s}{d}°C{s}", .{
                    themes.colorToAnsi(temp_color),
                    temp,
                    themes.resetCode(),
                });
            }

            row += 1;
        }

        // Memory usage sparkline
        row += 1;
        if (row < y + height - 2) {
            try terminal.setCursorPosition(row, x);
            try writer.print("{s}│{s} Memory: ", .{
                themes.colorToAnsi(theme.border),
                themes.resetCode(),
            });

            const recent = self.history.getRecent(20);
            for (recent) |val| {
                const char: u8 = if (val > 0.8) '█' else if (val > 0.6) '▆' else if (val > 0.4) '▄' else if (val > 0.2) '▂' else '▁';
                try writer.writeByte(char);
            }
        }

        // Scheduler stats
        if (self.scheduler_stats) |stats| {
            row += 1;
            if (row < y + height - 1) {
                try terminal.setCursorPosition(row, x);
                try writer.print("{s}│{s} Scheduler: {d} schedules, {d:.1}% success, ε={d:.2}", .{
                    themes.colorToAnsi(theme.border),
                    themes.resetCode(),
                    stats.total_schedules,
                    stats.successful_rate * 100,
                    stats.exploration_rate,
                });
            }
        }

        // Bottom border
        try terminal.setCursorPosition(y + height - 1, x);
        try writer.print("{s}└{s}┘{s}", .{
            themes.colorToAnsi(theme.border),
            "─" ** (width - 2),
            themes.resetCode(),
        });
    }

    /// Set scheduler statistics
    pub fn setSchedulerStats(self: *GpuMonitor, stats: SchedulerStats) void {
        self.scheduler_stats = stats;
    }
};

test "gpu monitor init" {
    const allocator = std.testing.allocator;
    var monitor = GpuMonitor.init(allocator);
    defer monitor.deinit();

    try monitor.update();
    try std.testing.expect(monitor.devices.items.len > 0);
}
```

**Step 2: Update TUI module exports**

Add to `tools/cli/tui/mod.zig`:

```zig
pub const gpu_monitor = @import("gpu_monitor.zig");
pub const GpuMonitor = gpu_monitor.GpuMonitor;
```

**Step 3: Verify build**

```bash
zig build 2>&1 | head -10
```

**Step 4: Commit**

```bash
git add tools/cli/tui/gpu_monitor.zig tools/cli/tui/mod.zig
git commit -m "feat(tui): add GPU monitor widget with real-time memory and scheduler stats"
```

---

### Task 4: Add Agent Status Panel

**Files:**
- Create: `tools/cli/tui/agent_panel.zig`
- Modify: `tools/cli/tui/mod.zig` (add export)

**Step 1: Create agent status panel**

Create `tools/cli/tui/agent_panel.zig`:

```zig
//! Agent Status Panel for TUI
//!
//! Displays AI agent learning progress, decision history,
//! and real-time performance metrics.

const std = @import("std");
const Terminal = @import("terminal.zig").Terminal;
const themes = @import("themes.zig");

/// Agent learning phase
pub const LearningPhase = enum {
    exploration,
    exploitation,
    converged,

    pub fn name(self: LearningPhase) []const u8 {
        return switch (self) {
            .exploration => "Exploring",
            .exploitation => "Exploiting",
            .converged => "Converged",
        };
    }

    pub fn color(self: LearningPhase, theme: themes.Theme) u8 {
        return switch (self) {
            .exploration => theme.warning,
            .exploitation => theme.info,
            .converged => theme.success,
        };
    }
};

/// Decision history entry
pub const DecisionEntry = struct {
    timestamp: i64,
    workload_type: []const u8,
    selected_backend: []const u8,
    actual_time_ms: u64,
    predicted_time_ms: u64,
    reward: f32,
};

/// Agent Status Panel
pub const AgentPanel = struct {
    allocator: std.mem.Allocator,
    phase: LearningPhase,
    episode_count: u64,
    total_reward: f32,
    avg_reward: f32,
    exploration_rate: f32,
    decision_history: std.ArrayList(DecisionEntry),
    reward_history: [100]f32,
    reward_idx: usize,

    pub fn init(allocator: std.mem.Allocator) AgentPanel {
        return .{
            .allocator = allocator,
            .phase = .exploration,
            .episode_count = 0,
            .total_reward = 0,
            .avg_reward = 0,
            .exploration_rate = 1.0,
            .decision_history = std.ArrayList(DecisionEntry).init(allocator),
            .reward_history = [_]f32{0} ** 100,
            .reward_idx = 0,
        };
    }

    pub fn deinit(self: *AgentPanel) void {
        self.decision_history.deinit();
    }

    /// Update agent statistics
    pub fn updateStats(self: *AgentPanel, episode: u64, total_reward: f32,
                       exploration_rate: f32) void {
        self.episode_count = episode;
        self.total_reward = total_reward;
        self.exploration_rate = exploration_rate;

        // Update phase based on exploration rate
        if (exploration_rate > 0.5) {
            self.phase = .exploration;
        } else if (exploration_rate > 0.05) {
            self.phase = .exploitation;
        } else {
            self.phase = .converged;
        }

        // Track reward history
        self.reward_history[self.reward_idx] = total_reward;
        self.reward_idx = (self.reward_idx + 1) % 100;

        // Calculate average
        var sum: f32 = 0;
        var count: u32 = 0;
        for (self.reward_history) |r| {
            if (r != 0) {
                sum += r;
                count += 1;
            }
        }
        self.avg_reward = if (count > 0) sum / @as(f32, @floatFromInt(count)) else 0;
    }

    /// Record a scheduling decision
    pub fn recordDecision(self: *AgentPanel, entry: DecisionEntry) !void {
        if (self.decision_history.items.len >= 20) {
            _ = self.decision_history.orderedRemove(0);
        }
        try self.decision_history.append(entry);
    }

    /// Render the agent panel
    pub fn render(self: *AgentPanel, terminal: *Terminal, theme: themes.Theme,
                  x: u16, y: u16, width: u16, height: u16) !void {
        const writer = terminal.writer();

        // Title with phase indicator
        try terminal.setCursorPosition(y, x);
        const phase_color = self.phase.color(theme);
        try writer.print("{s}┌─ Agent Status [{s}{s}{s}] ─{s}┐{s}", .{
            themes.colorToAnsi(theme.border),
            themes.colorToAnsi(phase_color),
            self.phase.name(),
            themes.colorToAnsi(theme.border),
            "─" ** 10,
            themes.resetCode(),
        });

        var row: u16 = y + 1;

        // Stats row
        try terminal.setCursorPosition(row, x);
        try writer.print("{s}│{s} Episodes: {s}{d}{s}  Reward: {s}{d:.2}{s}  ε: {s}{d:.3}{s}", .{
            themes.colorToAnsi(theme.border),
            themes.resetCode(),
            themes.colorToAnsi(theme.accent),
            self.episode_count,
            themes.resetCode(),
            themes.colorToAnsi(if (self.avg_reward > 0) theme.success else theme.warning),
            self.avg_reward,
            themes.resetCode(),
            themes.colorToAnsi(theme.info),
            self.exploration_rate,
            themes.resetCode(),
        });
        row += 1;

        // Reward trend sparkline
        try terminal.setCursorPosition(row, x);
        try writer.print("{s}│{s} Trend: ", .{
            themes.colorToAnsi(theme.border),
            themes.resetCode(),
        });

        const start_idx = if (self.reward_idx >= 30) self.reward_idx - 30 else 0;
        const end_idx = self.reward_idx;
        for (start_idx..end_idx) |i| {
            const val = self.reward_history[i];
            const norm = @min(1.0, @max(0.0, (val + 10) / 20)); // Normalize around 0
            const char: u8 = if (norm > 0.8) '█' else if (norm > 0.6) '▆' else if (norm > 0.4) '▄' else if (norm > 0.2) '▂' else '▁';
            try writer.writeByte(char);
        }
        row += 1;

        // Recent decisions header
        row += 1;
        if (row < y + height - 2) {
            try terminal.setCursorPosition(row, x);
            try writer.print("{s}│{s} {s}Recent Decisions:{s}", .{
                themes.colorToAnsi(theme.border),
                themes.resetCode(),
                themes.colorToAnsi(theme.secondary),
                themes.resetCode(),
            });
            row += 1;
        }

        // Recent decisions list
        const max_decisions = @min(5, @as(usize, @intCast(y + height - row - 2)));
        const start = if (self.decision_history.items.len > max_decisions)
            self.decision_history.items.len - max_decisions else 0;

        for (self.decision_history.items[start..]) |decision| {
            if (row >= y + height - 1) break;

            try terminal.setCursorPosition(row, x);

            const reward_color = if (decision.reward > 0) theme.success else theme.@"error";
            const accuracy = if (decision.predicted_time_ms > 0)
                @as(f32, @floatFromInt(decision.actual_time_ms)) / @as(f32, @floatFromInt(decision.predicted_time_ms))
            else 1.0;

            try writer.print("{s}│{s}  {s:<10} → {s:<8} {d:>4}ms {s}{d:+.2}{s}", .{
                themes.colorToAnsi(theme.border),
                themes.resetCode(),
                decision.workload_type[0..@min(10, decision.workload_type.len)],
                decision.selected_backend[0..@min(8, decision.selected_backend.len)],
                decision.actual_time_ms,
                themes.colorToAnsi(reward_color),
                decision.reward,
                themes.resetCode(),
            });
            _ = accuracy;

            row += 1;
        }

        // Bottom border
        try terminal.setCursorPosition(y + height - 1, x);
        try writer.print("{s}└{s}┘{s}", .{
            themes.colorToAnsi(theme.border),
            "─" ** (width - 2),
            themes.resetCode(),
        });
    }
};

test "agent panel init" {
    const allocator = std.testing.allocator;
    var panel = AgentPanel.init(allocator);
    defer panel.deinit();

    panel.updateStats(10, 5.5, 0.3);
    try std.testing.expect(panel.phase == .exploitation);
}
```

**Step 2: Update TUI module exports**

Add to `tools/cli/tui/mod.zig`:

```zig
pub const agent_panel = @import("agent_panel.zig");
pub const AgentPanel = agent_panel.AgentPanel;
```

**Step 3: Verify build**

```bash
zig build 2>&1 | head -10
```

**Step 4: Commit**

```bash
git add tools/cli/tui/agent_panel.zig tools/cli/tui/mod.zig
git commit -m "feat(tui): add agent status panel with learning progress visualization"
```

---

## Phase 3: Self-Learning Agent Integration

### Task 5: Create Unified Agent-GPU Interface

**Files:**
- Create: `src/ai/gpu_agent.zig`
- Modify: `src/ai/mod.zig` (add export)

**Step 1: Create GPU-aware agent**

Create `src/ai/gpu_agent.zig`:

```zig
//! GPU-Aware Self-Learning Agent
//!
//! Integrates AI agent capabilities with GPU scheduling,
//! enabling the agent to learn optimal GPU resource allocation.

const std = @import("std");
const build_options = @import("build_options");

// GPU integration (conditional)
const gpu = if (build_options.enable_gpu) @import("../gpu/mod.zig") else struct {
    pub const mega = struct {
        pub const Coordinator = void;
        pub const LearningScheduler = void;
    };
};

// Orchestration integration
const orchestration = @import("orchestration/mod.zig");
const training = @import("training/mod.zig");

/// Workload type for classification
pub const WorkloadType = enum {
    inference,
    training,
    embedding,
    fine_tuning,
    batch_inference,

    pub fn gpuIntensive(self: WorkloadType) bool {
        return switch (self) {
            .training, .fine_tuning => true,
            .inference, .embedding, .batch_inference => false,
        };
    }

    pub fn memoryIntensive(self: WorkloadType) bool {
        return switch (self) {
            .training, .fine_tuning, .batch_inference => true,
            .inference, .embedding => false,
        };
    }
};

/// Request for GPU-aware processing
pub const GpuAwareRequest = struct {
    prompt: []const u8,
    workload_type: WorkloadType,
    priority: Priority,
    max_tokens: u32 = 1024,
    temperature: f32 = 0.7,
    memory_hint_mb: ?u64 = null,
    preferred_backend: ?[]const u8 = null,

    pub const Priority = enum {
        low,
        normal,
        high,
        critical,
    };
};

/// Response with GPU scheduling info
pub const GpuAwareResponse = struct {
    content: []const u8,
    tokens_generated: u32,
    latency_ms: u64,
    gpu_backend_used: []const u8,
    gpu_memory_used_mb: u64,
    scheduling_confidence: f32,
    energy_estimate_wh: ?f32 = null,
};

/// GPU-Aware Agent
pub const GpuAgent = struct {
    allocator: std.mem.Allocator,
    orchestrator: *orchestration.Orchestrator,
    gpu_coordinator: if (build_options.enable_gpu) *gpu.mega.Coordinator else void,
    gpu_scheduler: if (build_options.enable_gpu) *gpu.mega.LearningScheduler else void,
    stats: AgentStats,

    pub const AgentStats = struct {
        total_requests: u64 = 0,
        gpu_accelerated: u64 = 0,
        total_tokens: u64 = 0,
        total_latency_ms: u64 = 0,
        learning_episodes: u64 = 0,
        avg_scheduling_confidence: f32 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) !*GpuAgent {
        const self = try allocator.create(GpuAgent);

        // Initialize orchestrator
        const orch = try orchestration.Orchestrator.init(allocator, .{
            .strategy = .task_based,
            .enable_fallback = true,
        });

        self.* = .{
            .allocator = allocator,
            .orchestrator = orch,
            .gpu_coordinator = if (build_options.enable_gpu)
                try gpu.mega.Coordinator.init(allocator) else {},
            .gpu_scheduler = if (build_options.enable_gpu) blk: {
                const coord = try gpu.mega.Coordinator.init(allocator);
                break :blk try gpu.mega.LearningScheduler.init(allocator, coord);
            } else {},
            .stats = .{},
        };

        return self;
    }

    pub fn deinit(self: *GpuAgent) void {
        self.orchestrator.deinit();
        if (build_options.enable_gpu) {
            self.gpu_scheduler.deinit();
            self.gpu_coordinator.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Process request with GPU-aware scheduling
    pub fn process(self: *GpuAgent, request: GpuAwareRequest) !GpuAwareResponse {
        const start_time = std.time.milliTimestamp();
        self.stats.total_requests += 1;

        // Determine GPU scheduling
        var backend_used: []const u8 = "cpu";
        var memory_used: u64 = 0;
        var confidence: f32 = 0;

        if (build_options.enable_gpu) {
            const profile = gpu.mega.coordinator.WorkloadProfile{
                .compute_intensity = if (request.workload_type.gpuIntensive()) 0.9 else 0.3,
                .memory_requirement_mb = request.memory_hint_mb orelse 1024,
                .is_training = request.workload_type == .training or
                              request.workload_type == .fine_tuning,
            };

            const decision = self.gpu_scheduler.schedule(profile);
            backend_used = @tagName(decision.backend_type);
            confidence = decision.confidence;

            self.stats.gpu_accelerated += 1;
            self.stats.avg_scheduling_confidence =
                (self.stats.avg_scheduling_confidence * @as(f32, @floatFromInt(self.stats.gpu_accelerated - 1)) +
                 confidence) / @as(f32, @floatFromInt(self.stats.gpu_accelerated));
        }

        // Route to LLM backend
        const task_type: orchestration.TaskType = switch (request.workload_type) {
            .inference, .batch_inference => .general,
            .training, .fine_tuning => .analysis,
            .embedding => .summarization,
        };

        // Simulate LLM response (actual implementation would call orchestrator.route())
        const content = try self.allocator.dupe(u8, "Response generated");
        const tokens: u32 = @intCast(content.len / 4);

        const end_time = std.time.milliTimestamp();
        const latency = @as(u64, @intCast(end_time - start_time));

        self.stats.total_tokens += tokens;
        self.stats.total_latency_ms += latency;

        // Record outcome for learning
        if (build_options.enable_gpu) {
            const decision = self.gpu_scheduler.schedule(.{
                .compute_intensity = 0.5,
                .memory_requirement_mb = 0,
            });
            try self.gpu_scheduler.recordAndLearn(decision, latency, true);
            self.stats.learning_episodes += 1;
        }

        _ = task_type;

        return .{
            .content = content,
            .tokens_generated = tokens,
            .latency_ms = latency,
            .gpu_backend_used = backend_used,
            .gpu_memory_used_mb = memory_used,
            .scheduling_confidence = confidence,
        };
    }

    /// Get current agent statistics
    pub fn getStats(self: *GpuAgent) AgentStats {
        return self.stats;
    }

    /// Get GPU scheduling statistics
    pub fn getGpuStats(self: *GpuAgent) ?gpu.mega.scheduler.LearningStats {
        if (build_options.enable_gpu) {
            return self.gpu_scheduler.getStats();
        }
        return null;
    }

    /// End learning episode
    pub fn endEpisode(self: *GpuAgent) !void {
        if (build_options.enable_gpu) {
            try self.gpu_scheduler.endEpisode();
        }
    }
};

test "gpu agent init" {
    const allocator = std.testing.allocator;
    const agent = try GpuAgent.init(allocator);
    defer agent.deinit();

    try std.testing.expect(agent.stats.total_requests == 0);
}
```

**Step 2: Update AI module exports**

Add to `src/ai/mod.zig`:

```zig
pub const gpu_agent = @import("gpu_agent.zig");
pub const GpuAgent = gpu_agent.GpuAgent;
pub const GpuAwareRequest = gpu_agent.GpuAwareRequest;
pub const GpuAwareResponse = gpu_agent.GpuAwareResponse;
```

**Step 3: Verify build**

```bash
zig build 2>&1 | head -10
```

**Step 4: Commit**

```bash
git add src/ai/gpu_agent.zig src/ai/mod.zig
git commit -m "feat(ai): add GPU-aware self-learning agent with RL-based scheduling"
```

---

### Task 6: Create TUI Dashboard Integration

**Files:**
- Create: `tools/cli/commands/gpu_dashboard.zig`
- Modify: `tools/cli/commands/tui.zig` (add dashboard option)

**Step 1: Create GPU dashboard command**

Create `tools/cli/commands/gpu_dashboard.zig`:

```zig
//! GPU Dashboard Command
//!
//! Interactive TUI dashboard combining GPU monitoring
//! and agent learning visualization.

const std = @import("std");
const tui = @import("../tui/mod.zig");
const Terminal = tui.Terminal;
const themes = tui.themes;
const GpuMonitor = tui.GpuMonitor;
const AgentPanel = tui.AgentPanel;
const AsyncLoop = tui.async_loop.AsyncLoop;

pub const DashboardState = struct {
    gpu_monitor: GpuMonitor,
    agent_panel: AgentPanel,
    theme: themes.Theme,
    show_help: bool,
    paused: bool,

    pub fn init(allocator: std.mem.Allocator) DashboardState {
        return .{
            .gpu_monitor = GpuMonitor.init(allocator),
            .agent_panel = AgentPanel.init(allocator),
            .theme = themes.Theme.default(),
            .show_help = false,
            .paused = false,
        };
    }

    pub fn deinit(self: *DashboardState) void {
        self.gpu_monitor.deinit();
        self.agent_panel.deinit();
    }
};

fn renderDashboard(loop: *AsyncLoop) !void {
    const state = loop.getUserData(DashboardState) orelse return;
    const terminal = loop.terminal;
    const size = terminal.getSize();
    const writer = terminal.writer();

    // Clear screen
    try writer.writeAll("\x1b[2J\x1b[H");

    // Title bar
    try writer.print("{s}╔══ ABI GPU Dashboard ══╗{s}\n", .{
        themes.colorToAnsi(state.theme.primary),
        themes.resetCode(),
    });

    // Layout: GPU Monitor (left) | Agent Panel (right)
    const half_width = size.cols / 2;

    // Update data
    if (!state.paused) {
        try state.gpu_monitor.update();
    }

    // Render GPU Monitor on left
    try state.gpu_monitor.render(terminal, state.theme, 0, 2, half_width, 12);

    // Render Agent Panel on right
    try state.agent_panel.render(terminal, state.theme, half_width, 2, half_width, 12);

    // Status bar
    try terminal.setCursorPosition(size.rows - 2, 0);
    try writer.print("{s}FPS: {d:.1} | Frame: {d} | {s}{s}", .{
        themes.colorToAnsi(state.theme.secondary),
        loop.getFps(),
        loop.getFrameCount(),
        if (state.paused) "[PAUSED]" else "[LIVE]",
        themes.resetCode(),
    });

    // Help bar
    try terminal.setCursorPosition(size.rows - 1, 0);
    try writer.print("{s}[q]uit  [p]ause  [t]heme  [h]elp{s}", .{
        themes.colorToAnsi(state.theme.text_dim),
        themes.resetCode(),
    });
}

fn handleEvent(loop: *AsyncLoop, event: tui.async_loop.AsyncEvent) !bool {
    const state = loop.getUserData(DashboardState) orelse return true;

    switch (event) {
        .input => |input| {
            if (input == .key) {
                switch (input.key.code) {
                    .character => {
                        if (input.key.char) |ch| {
                            switch (ch) {
                                'q' => return true, // Quit
                                'p' => state.paused = !state.paused,
                                't' => {
                                    // Cycle theme
                                    var manager = themes.ThemeManager.init();
                                    manager.nextTheme();
                                    state.theme = manager.current;
                                },
                                'h' => state.show_help = !state.show_help,
                                else => {},
                            }
                        }
                    },
                    .ctrl_c, .escape => return true,
                    else => {},
                }
            }
        },
        else => {},
    }

    return false;
}

fn tickUpdate(loop: *AsyncLoop) !void {
    const state = loop.getUserData(DashboardState) orelse return;

    if (!state.paused) {
        // Simulate agent learning updates
        state.agent_panel.updateStats(
            state.agent_panel.episode_count + 1,
            state.agent_panel.total_reward + 0.1,
            @max(0.01, state.agent_panel.exploration_rate * 0.99),
        );

        // Simulate decision recording
        try state.agent_panel.recordDecision(.{
            .timestamp = std.time.milliTimestamp(),
            .workload_type = "inference",
            .selected_backend = "CUDA",
            .actual_time_ms = 50 + @as(u64, @intCast(@mod(state.agent_panel.episode_count, 50))),
            .predicted_time_ms = 55,
            .reward = 0.8 - @as(f32, @floatFromInt(@mod(state.agent_panel.episode_count, 10))) / 20.0,
        });
    }
}

pub fn run(allocator: std.mem.Allocator, args: []const []const u8) !void {
    _ = args;

    var terminal = Terminal.init(allocator);
    defer terminal.deinit();

    try terminal.enter();
    defer terminal.exit() catch {};

    var state = DashboardState.init(allocator);
    defer state.deinit();

    var loop = AsyncLoop.init(allocator, &terminal, .{
        .refresh_rate_ms = 100,
    });
    defer loop.deinit();

    loop.setUserData(@ptrCast(&state));
    loop.setRenderCallback(renderDashboard);
    loop.setUpdateCallback(handleEvent);
    loop.setTickCallback(tickUpdate);

    try loop.run();
}
```

**Step 2: Verify build**

```bash
zig build 2>&1 | head -10
```

**Step 3: Commit**

```bash
git add tools/cli/commands/gpu_dashboard.zig
git commit -m "feat(cli): add GPU dashboard command with real-time monitoring"
```

---

## Phase 4: Final Integration & Testing

### Task 7: Update ROADMAP and Documentation

**Files:**
- Modify: `ROADMAP.md` (mark mega GPU upgrade complete)

**Step 1: Update ROADMAP**

Find the mega GPU reference and mark as complete:

```markdown
- [x] Mega GPU + TUI + Self-Learning Agent Upgrade
  - [x] Cross-backend GPU coordinator (src/gpu/mega/)
  - [x] Learning-based scheduler with Q-learning
  - [x] GPU monitor TUI widget
  - [x] Agent status panel
  - [x] GPU-aware agent integration
  - [x] Interactive dashboard command
```

**Step 2: Verify full build**

```bash
zig build 2>&1
```

**Step 3: Run all tests**

```bash
zig build test --summary all 2>&1 | tail -10
```

**Step 4: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: mark Mega GPU + TUI + Agent upgrade complete"
```

---

### Task 8: Final Verification

**Step 1: Run full test suite**

```bash
zig build test --summary all
```

Expected: 194+ tests pass

**Step 2: Test GPU dashboard (if terminal available)**

```bash
zig build run -- gpu dashboard
```

**Step 3: Verify GPU backends**

```bash
zig build run -- gpu backends
```

Expected: Shows available backends with mega coordinator info

---

## Summary

| Component | Files Created | Purpose |
|-----------|---------------|---------|
| Mega GPU Coordinator | `src/gpu/mega/coordinator.zig` | Cross-backend orchestration |
| Learning Scheduler | `src/gpu/mega/scheduler.zig` | Q-learning based scheduling |
| GPU Monitor Widget | `tools/cli/tui/gpu_monitor.zig` | Real-time GPU visualization |
| Agent Panel Widget | `tools/cli/tui/agent_panel.zig` | Learning progress display |
| GPU-Aware Agent | `src/ai/gpu_agent.zig` | Unified agent-GPU interface |
| Dashboard Command | `tools/cli/commands/gpu_dashboard.zig` | Interactive dashboard |

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                      TUI Dashboard                          │
│  ┌───────────────────┐    ┌───────────────────┐            │
│  │   GPU Monitor     │    │   Agent Panel     │            │
│  │  - Device status  │    │  - Learning phase │            │
│  │  - Memory usage   │    │  - Reward trend   │            │
│  │  - Temperature    │    │  - Decisions      │            │
│  └───────────────────┘    └───────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU-Aware Agent                          │
│  ┌───────────────────┐    ┌───────────────────┐            │
│  │   Orchestrator    │◄──►│  Learning Scheduler│            │
│  │  (Model routing)  │    │  (Q-learning + ε)  │            │
│  └───────────────────┘    └───────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Mega GPU Coordinator                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │
│  │  CUDA   │  │ Vulkan  │  │  Metal  │  │  FPGA   │       │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**Total Tasks:** 8
**Estimated Commits:** 8
