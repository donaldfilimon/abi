# Mega GPU + TUI + Self-Learning Agent Upgrade Plan

**Status** – Core stubs for `AsyncTui`, widget placeholders, and agent scaffolding have been added; detailed implementation to follow.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete GPU backend implementations with real hardware support, upgrade TUI with dynamic real-time refresh, and build a self-learning Claude-Code-like agent system in Zig.

**Architecture:**
- GPU: Real Vulkan device enumeration with VK_KHR extensions, CUDA via NVRTC, Metal via Objective-C FFI
- TUI: Event-driven architecture with async refresh, widget system, real-time metrics
- Agent: Tool-calling architecture with code generation, self-improvement loop, system control

**Tech Stack:** Zig 0.16, Vulkan 1.2+, CUDA 12+, Metal 3, ANSI/VT100 terminals, LLM inference

---

## Phase 1: Real Vulkan Backend (Priority: HIGH)

### Task 1.1: Vulkan Real Device Enumeration

**Files:**
- Modify: `src/gpu/backends/vulkan_init.zig`
- Modify: `src/gpu/backends/vulkan.zig`
- Test: `src/gpu/backends/tests/vulkan_test.zig`

**Implementation:**
```zig
// Enhanced device selection with real GPU scoring
pub fn selectBestPhysicalDevice(instance: VkInstance) !VkPhysicalDevice {
    var count: u32 = 0;
    _ = vkEnumeratePhysicalDevices(instance, &count, null);
    if (count == 0) return error.NoDevicesFound;

    const devices = try allocator.alloc(VkPhysicalDevice, count);
    defer allocator.free(devices);
    _ = vkEnumeratePhysicalDevices(instance, &count, devices.ptr);

    var best_device: ?VkPhysicalDevice = null;
    var best_score: u32 = 0;

    for (devices) |device| {
        const score = scorePhysicalDevice(device);
        if (score > best_score) {
            best_score = score;
            best_device = device;
        }
    }

    return best_device orelse error.NoSuitableDevice;
}

fn scorePhysicalDevice(device: VkPhysicalDevice) u32 {
    var props: VkPhysicalDeviceProperties = undefined;
    vkGetPhysicalDeviceProperties(device, &props);

    var score: u32 = 0;

    // Discrete GPUs are preferred
    if (props.deviceType == .discrete_gpu) score += 10000;
    else if (props.deviceType == .integrated_gpu) score += 1000;

    // Add VRAM to score
    var mem_props: VkPhysicalDeviceMemoryProperties = undefined;
    vkGetPhysicalDeviceMemoryProperties(device, &mem_props);

    for (mem_props.memoryHeaps[0..mem_props.memoryHeapCount]) |heap| {
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            score += @intCast(heap.size / (1024 * 1024 * 1024)); // GB of VRAM
        }
    }

    return score;
}
```

### Task 1.2: Vulkan Compute Pipeline Implementation

**Files:**
- Modify: `src/gpu/backends/vulkan_pipelines.zig`
- Create: `src/gpu/backends/vulkan_compute.zig`

**Implementation:**
- Create compute pipeline for shader execution
- Implement descriptor set management
- Add push constants support
- Create command buffer pools

### Task 1.3: Vulkan Memory Management

**Files:**
- Modify: `src/gpu/backends/vulkan_buffers.zig`
- Create: `src/gpu/backends/vulkan_allocator.zig`

**Implementation:**
- VMA-style suballocation
- Device-local + host-visible memory pools
- Staging buffer management
- Memory defragmentation

---

## Phase 2: Complete Other GPU Backends

### Task 2.1: CUDA Backend via NVRTC

**Files:**
- Create: `src/gpu/backends/cuda/mod.zig`
- Create: `src/gpu/backends/cuda/nvrtc.zig`
- Create: `src/gpu/backends/cuda/driver.zig`

**Implementation:**
```zig
// Runtime compilation of CUDA kernels
pub const CudaBackend = struct {
    device: c_int,
    context: CUcontext,

    pub fn compileKernel(source: []const u8, name: []const u8) !CUfunction {
        var program: nvrtcProgram = undefined;
        nvrtcCreateProgram(&program, source.ptr, name.ptr, 0, null, null);

        const opts = [_][*:0]const u8{ "--gpu-architecture=compute_86" };
        const result = nvrtcCompileProgram(program, opts.len, &opts);
        if (result != NVRTC_SUCCESS) {
            // Get compilation log
            var log_size: usize = 0;
            nvrtcGetProgramLogSize(program, &log_size);
            // Handle error...
        }

        // Get PTX
        var ptx_size: usize = 0;
        nvrtcGetPTXSize(program, &ptx_size);
        const ptx = try allocator.alloc(u8, ptx_size);
        nvrtcGetPTX(program, ptx.ptr);

        // Load module
        var module: CUmodule = undefined;
        cuModuleLoadDataEx(&module, ptx.ptr, 0, null, null);

        var func: CUfunction = undefined;
        cuModuleGetFunction(&func, module, name.ptr);

        return func;
    }
};
```

### Task 2.2: Metal Backend via Objective-C FFI

**Files:**
- Create: `src/gpu/backends/metal/mod.zig`
- Create: `src/gpu/backends/metal/objc.zig`
- Create: `src/gpu/backends/metal/shaders.zig`

### Task 2.3: WebGPU Backend via wgpu-native

**Files:**
- Modify: `src/gpu/backends/webgpu.zig`
- Create: `src/gpu/backends/webgpu/wgpu_native.zig`

### Task 2.4: OpenGL Compute Shaders

**Files:**
- Modify: `src/gpu/backends/opengl.zig`
- Create: `src/gpu/backends/opengl/compute.zig`

---

## Phase 3: Dynamic TUI with Real-Time Updates

### Task 3.1: Async Event Loop

**Files:**
- Create: `tools/cli/tui/async_loop.zig`
- Modify: `tools/cli/tui/mod.zig`

**Implementation:**
```zig
pub const AsyncTui = struct {
    terminal: *Terminal,
    event_queue: EventQueue,
    refresh_timer: Timer,
    widgets: WidgetTree,

    pub fn run(self: *AsyncTui) !void {
        while (self.running) {
            // Non-blocking event poll
            if (self.terminal.pollEvent(10)) |event| {
                try self.handleEvent(event);
            }

            // Check refresh timer
            if (self.refresh_timer.expired()) {
                try self.refresh();
                self.refresh_timer.reset();
            }

            // Process async updates
            while (self.event_queue.pop()) |update| {
                try self.applyUpdate(update);
            }
        }
    }

    pub fn scheduleRefresh(self: *AsyncTui, widget_id: u32) void {
        self.event_queue.push(.{ .refresh = widget_id });
    }
};
```

### Task 3.2: Real-Time Metrics Widget

**Files:**
- Create: `tools/cli/tui/widgets/metrics.zig`
- Create: `tools/cli/tui/widgets/gpu_monitor.zig`
- Create: `tools/cli/tui/widgets/system_monitor.zig`

**Implementation:**
```zig
pub const MetricsWidget = struct {
    cpu_usage: f32,
    memory_usage: f32,
    gpu_usage: ?f32,
    gpu_memory: ?f32,
    update_interval_ms: u32 = 1000,

    pub fn render(self: *MetricsWidget, term: *Terminal, rect: Rect) !void {
        // Sparkline for CPU
        try self.renderSparkline(term, rect, self.cpu_history, "CPU");

        // GPU metrics if available
        if (self.gpu_usage) |usage| {
            try self.renderBar(term, rect.offset(0, 1), usage, "GPU");
        }

        // Memory bars
        try self.renderMemoryBars(term, rect.offset(0, 2));
    }

    pub fn update(self: *MetricsWidget) !void {
        self.cpu_usage = try system.getCpuUsage();
        self.memory_usage = try system.getMemoryUsage();

        if (gpu.isAvailable()) {
            self.gpu_usage = try gpu.getUtilization();
            self.gpu_memory = try gpu.getMemoryUsage();
        }
    }
};
```

### Task 3.3: Interactive Dashboard

**Files:**
- Create: `tools/cli/tui/dashboard.zig`
- Modify: `tools/cli/commands/tui.zig`

---

## Phase 4: Self-Learning Agent System (Claude-Code-like)

### Task 4.1: Tool System Architecture

**Files:**
- Create: `src/ai/tools/mod.zig`
- Create: `src/ai/tools/file_tools.zig`
- Create: `src/ai/tools/bash_tools.zig`
- Create: `src/ai/tools/search_tools.zig`

**Implementation:**
```zig
pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    parameters: []const Parameter,
    execute: *const fn (args: ToolArgs) ToolResult,

    pub const Parameter = struct {
        name: []const u8,
        type: ParameterType,
        description: []const u8,
        required: bool,
    };
};

pub const ToolRegistry = struct {
    tools: std.StringHashMap(Tool),

    pub fn register(self: *ToolRegistry, tool: Tool) !void {
        try self.tools.put(tool.name, tool);
    }

    pub fn execute(self: *ToolRegistry, name: []const u8, args: ToolArgs) !ToolResult {
        const tool = self.tools.get(name) orelse return error.UnknownTool;
        return tool.execute(args);
    }
};

// Built-in tools
pub const ReadFileTool = Tool{
    .name = "read_file",
    .description = "Read the contents of a file",
    .parameters = &[_]Tool.Parameter{
        .{ .name = "path", .type = .string, .description = "File path to read", .required = true },
    },
    .execute = readFileImpl,
};

pub const WriteFileTool = Tool{
    .name = "write_file",
    .description = "Write content to a file",
    .parameters = &[_]Tool.Parameter{
        .{ .name = "path", .type = .string, .description = "File path to write", .required = true },
        .{ .name = "content", .type = .string, .description = "Content to write", .required = true },
    },
    .execute = writeFileImpl,
};

pub const BashTool = Tool{
    .name = "bash",
    .description = "Execute a bash command",
    .parameters = &[_]Tool.Parameter{
        .{ .name = "command", .type = .string, .description = "Command to execute", .required = true },
    },
    .execute = bashImpl,
};
```

### Task 4.2: Agent Loop with Tool Calling

**Files:**
- Create: `src/ai/agents/code_agent.zig`
- Create: `src/ai/agents/tool_executor.zig`

**Implementation:**
```zig
pub const CodeAgent = struct {
    llm: *LlmEngine,
    tools: *ToolRegistry,
    context: AgentContext,
    max_iterations: u32 = 50,

    pub fn run(self: *CodeAgent, task: []const u8) !AgentResult {
        var messages = std.ArrayList(Message).init(self.allocator);
        defer messages.deinit();

        // System prompt
        try messages.append(.{
            .role = .system,
            .content = AGENT_SYSTEM_PROMPT,
        });

        // User task
        try messages.append(.{
            .role = .user,
            .content = task,
        });

        var iteration: u32 = 0;
        while (iteration < self.max_iterations) : (iteration += 1) {
            // Get LLM response
            const response = try self.llm.chat(messages.items, .{
                .tools = self.tools.getSchemas(),
                .tool_choice = .auto,
            });

            try messages.append(response);

            // Check for tool calls
            if (response.tool_calls) |calls| {
                for (calls) |call| {
                    const result = try self.tools.execute(call.name, call.arguments);
                    try messages.append(.{
                        .role = .tool,
                        .tool_call_id = call.id,
                        .content = result.output,
                    });
                }
            } else {
                // No tool calls, agent is done
                return .{
                    .success = true,
                    .output = response.content,
                    .iterations = iteration,
                };
            }
        }

        return .{
            .success = false,
            .output = "Max iterations reached",
            .iterations = iteration,
        };
    }
};
```

### Task 4.3: Self-Improvement Loop

**Files:**
- Create: `src/ai/agents/self_improve.zig`
- Create: `src/ai/agents/code_review.zig`

**Implementation:**
```zig
pub const SelfImprovingAgent = struct {
    code_agent: *CodeAgent,
    review_agent: *CodeAgent,
    test_runner: *TestRunner,

    pub fn improveCode(self: *SelfImprovingAgent, file_path: []const u8) !ImprovementResult {
        // 1. Read current code
        const code = try std.fs.cwd().readFileAlloc(self.allocator, file_path, 1024 * 1024);
        defer self.allocator.free(code);

        // 2. Generate improvement suggestions
        const suggestions = try self.review_agent.run(
            std.fmt.allocPrint(self.allocator,
                "Review this code and suggest improvements:\n```\n{s}\n```",
                .{code}
            ),
        );

        // 3. Apply improvements
        const improved = try self.code_agent.run(
            std.fmt.allocPrint(self.allocator,
                "Apply these improvements to the code:\n{s}\n\nOriginal code:\n```\n{s}\n```",
                .{suggestions.output, code}
            ),
        );

        // 4. Run tests to verify
        const test_result = try self.test_runner.runTests(file_path);

        if (test_result.passed) {
            // 5. Commit the improvement
            return .{
                .success = true,
                .diff = improved.output,
                .tests_passed = test_result.passed,
            };
        } else {
            // 6. Rollback and try again
            return .{
                .success = false,
                .error_message = test_result.error_message,
            };
        }
    }
};
```

### Task 4.4: System Control Interface

**Files:**
- Create: `src/ai/system/controller.zig`
- Create: `src/ai/system/permissions.zig`

**Implementation:**
```zig
pub const SystemController = struct {
    permissions: PermissionSet,
    audit_log: *AuditLog,

    pub const Action = enum {
        read_file,
        write_file,
        execute_command,
        network_request,
        gpu_operation,
        database_query,
    };

    pub fn requestPermission(self: *SystemController, action: Action, resource: []const u8) !bool {
        // Check existing permissions
        if (self.permissions.isAllowed(action, resource)) {
            try self.audit_log.log(action, resource, .allowed);
            return true;
        }

        // Request user approval for new permissions
        if (try self.promptUserApproval(action, resource)) {
            try self.permissions.grant(action, resource);
            try self.audit_log.log(action, resource, .granted);
            return true;
        }

        try self.audit_log.log(action, resource, .denied);
        return false;
    }
};
```

---

## Phase 5: Integration and Testing

### Task 5.1: GPU Backend Integration Tests

**Files:**
- Create: `src/gpu/backends/tests/integration_test.zig`

### Task 5.2: TUI Performance Tests

**Files:**
- Create: `tools/cli/tui/tests/performance_test.zig`

### Task 5.3: Agent System Tests

**Files:**
- Create: `src/ai/agents/tests/agent_test.zig`

---

## Summary

| Phase | Components | Priority |
|-------|------------|----------|
| 1 | Real Vulkan Backend | HIGH |
| 2 | CUDA, Metal, WebGPU, OpenGL | HIGH |
| 3 | Dynamic TUI with Real-Time Updates | MEDIUM |
| 4 | Self-Learning Agent System | HIGH |
| 5 | Integration and Testing | MEDIUM |

**Estimated Total:** ~50+ files to create/modify
**Key Dependencies:** Vulkan SDK, CUDA Toolkit (optional), Metal (macOS), wgpu-native
