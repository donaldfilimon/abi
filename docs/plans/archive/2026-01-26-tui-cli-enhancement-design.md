# TUI/CLI Comprehensive Enhancement Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create implementation plan after design approval.

**Goal:** Add comprehensive TUI panels and CLI enhancements for model management, streaming inference, database monitoring, multi-agent workflows, and interactive argument picking.

**Date:** 2026-01-26

---

## 1. Executive Summary

This design adds 6 major components to the ABI Framework's TUI/CLI system:

| Component | Purpose | Priority |
|-----------|---------|----------|
| Model Management Panel | View cached models, download progress, switching | P0 |
| Streaming Inference Dashboard | Real-time TTFT, throughput, connections | P0 |
| Interactive Argument Picker | Prompt for arguments in TUI | P1 |
| Database/Vector Panel | WDBX health, index status, query perf | P1 |
| Multi-Agent Workflow View | Agent coordination, decision history | P2 |
| TUI Launcher Enhancements | Integrate all new panels + missing commands | P0 |

**Estimated effort:** 8-12 tasks across 4-6 implementation sessions.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TUI Enhancement Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Model     │  │  Streaming   │  │   Database   │          │
│  │  Management  │  │  Dashboard   │  │    Panel     │          │
│  │    Panel     │  │              │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Multi-Agent  │  │ Interactive  │  │     TUI      │          │
│  │  Workflow    │  │   Argument   │  │   Launcher   │          │
│  │    View      │  │    Picker    │  │  (Enhanced)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐           │
│  │              Shared Infrastructure               │           │
│  │  • RingBuffer<T>  • Widget Library  • Themes    │           │
│  │  • Event Loop     • Key Handlers    • I/O Backend│          │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.1 Design Principles

1. **Unified panel pattern** - All new panels follow `GpuMonitor` structure
2. **Ring buffers for time-series** - Fixed memory, rolling window for metrics
3. **Consistent theming** - All panels use existing 5-theme system
4. **Non-breaking additions** - Existing commands remain unchanged
5. **Keyboard-first UX** - Single-key shortcuts for all actions

### 2.2 Existing Patterns to Follow

| Pattern | Source File | Usage |
|---------|-------------|-------|
| Panel struct | `gpu_monitor.zig` | Allocator, term, theme fields |
| Ring buffer | `gpu_monitor.zig` | MemoryHistory with pos/count |
| Modal states | `training_panel.zig` | Help, preview, search modes |
| Sparklines | `widgets.zig` | `SparklineChart.render()` |
| Progress bars | `widgets.zig` | `ProgressBar`, `ProgressGauge` |
| Key handling | `tui.zig` | `handleKeyEvent()` switch |

---

## 3. Component Designs

### 3.1 Model Management Panel

**File:** `tools/cli/tui/model_panel.zig`

**Purpose:** Real-time view of cached models, download progress, model switching.

```
╭─ Model Management ──────────────────────────────────────────────╮
│ Cached Models (3)                                    [t] Theme │
├─────────────────────────────────────────────────────────────────┤
│ ● llama-7b-q4         4.2 GB   Ready    ★ Active              │
│ ○ mistral-7b-instruct 4.1 GB   Ready                          │
│ ○ phi-3-mini          2.3 GB   Ready                          │
├─────────────────────────────────────────────────────────────────┤
│ Downloads (1 active)                                            │
│ gemma-2b-it  ████████████░░░░░░░░  58%  1.2 GB/2.1 GB  45 MB/s│
│              ETA: 2m 34s  ▁▂▃▅▆▇▆▅▄▃▂▁▂▃▄▅▆▇                   │
├─────────────────────────────────────────────────────────────────┤
│ [d] Download  [r] Remove  [s] Set Active  [i] Info  [q] Quit  │
╰─────────────────────────────────────────────────────────────────╯
```

**Struct Definition:**

```zig
pub const ModelManagementPanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,
    
    // Data
    cached_models: std.ArrayListUnmanaged(ModelEntry),
    active_downloads: std.ArrayListUnmanaged(DownloadState),
    transfer_rate_history: RingBuffer(f32, 60),
    
    // State
    selected_model: usize,
    scroll_offset: usize,
    active_model_id: ?[]const u8,
    show_details: bool,
    
    // Polling
    last_refresh: i64,
    refresh_interval_ms: u64 = 1000,
    
    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme) ModelManagementPanel;
    pub fn deinit(self: *ModelManagementPanel) void;
    pub fn update(self: *ModelManagementPanel) !void;
    pub fn render(self: *ModelManagementPanel, row: usize, col: usize, width: usize, height: usize) !void;
    pub fn handleKey(self: *ModelManagementPanel, key: events.Key) ?Action;
    pub fn runInteractive(self: *ModelManagementPanel) !void;
};
```

**Data Sources:**
- `src/ai/models/manager.zig` - `Manager.listCachedModels()`
- `src/ai/models/downloader.zig` - `Downloader.getProgress()`
- `src/ai/models/cache.zig` - Cache directory operations

**Key Bindings:**
| Key | Action |
|-----|--------|
| `j`/`↓` | Move selection down |
| `k`/`↑` | Move selection up |
| `d` | Start download dialog |
| `r` | Remove selected model |
| `s` | Set as active model |
| `i` | Show model info |
| `t` | Toggle theme |
| `q` | Quit panel |

---

### 3.2 Streaming Inference Dashboard

**File:** `tools/cli/tui/streaming_dashboard.zig`

**Purpose:** Monitor real-time streaming inference metrics.

```
╭─ Streaming Inference Dashboard ─────────────────────────────────╮
│ Server: http://127.0.0.1:8080  Uptime: 2h 34m  Backend: ollama │
├─────────────────────────────────────────────────────────────────┤
│ ┌─ Time to First Token ─┐  ┌─ Token Throughput ──┐             │
│ │ Current:     127ms    │  │ Current:    45 tok/s│             │
│ │ P50:         98ms     │  │ Peak:       67 tok/s│             │
│ │ P99:         342ms    │  │ Total:     1.2M tok │             │
│ │ ▂▃▄▅▆▇▆▅▄▃▂▁▂▃▄▅▆▇▆▅ │  │ ▅▆▇▆▅▄▃▂▁▂▃▄▅▆▇▆▅▄▃│             │
│ └───────────────────────┘  └────────────────────┘             │
├─────────────────────────────────────────────────────────────────┤
│ Active Connections: 3/100  │  Requests/min: ████████░░ 78     │
│ Queue Depth:        2      │  Errors/hr:    ░░░░░░░░░░  0     │
├─────────────────────────────────────────────────────────────────┤
│ Recent Requests                                                 │
│ 14:32:05  POST /v1/chat  200  127ms  "How do I..."  45 tokens │
│ 14:32:03  POST /v1/chat  200   98ms  "What is..."   32 tokens │
╰─────────────────────────────────────────────────────────────────╯
```

**Struct Definition:**

```zig
pub const StreamingDashboard = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,
    
    // Time-series metrics (ring buffers)
    ttft_history: RingBuffer(u32, 120),      // Time to first token (ms)
    throughput_history: RingBuffer(f32, 120), // Tokens/sec
    connection_history: RingBuffer(u16, 120), // Active connections
    
    // Percentile tracking
    ttft_percentiles: PercentileTracker,
    
    // Current state
    server_status: ServerStatus,
    active_connections: u32,
    max_connections: u32,
    queue_depth: u32,
    total_tokens: u64,
    total_requests: u64,
    error_count: u32,
    uptime_ms: i64,
    
    // Request log (circular)
    recent_requests: RingBuffer(RequestLogEntry, 100),
    request_scroll: usize,
    
    // Polling
    server_endpoint: []const u8,
    last_poll: i64,
    poll_interval_ms: u64 = 500,
    
    pub fn init(...) StreamingDashboard;
    pub fn deinit(self: *StreamingDashboard) void;
    pub fn pollMetrics(self: *StreamingDashboard) !void;
    pub fn render(self: *StreamingDashboard, row: usize, col: usize, width: usize, height: usize) !void;
    pub fn handleKey(self: *StreamingDashboard, key: events.Key) ?Action;
};

const ServerStatus = enum { online, offline, degraded, unknown };

const RequestLogEntry = struct {
    timestamp: i64,
    method: []const u8,
    path: []const u8,
    status_code: u16,
    latency_ms: u32,
    prompt_preview: [64]u8,
    token_count: u32,
};
```

**Data Sources:**
- `src/ai/streaming/server.zig` - Server status endpoint
- `src/ai/streaming/metrics.zig` - Metrics aggregation
- HTTP polling to `/health` and `/metrics` endpoints

**Key Bindings:**
| Key | Action |
|-----|--------|
| `r` | Force refresh |
| `l` | Toggle request log view |
| `c` | Clear statistics |
| `+`/`-` | Adjust poll interval |
| `t` | Toggle theme |
| `q` | Quit panel |

---

### 3.3 Database/Vector Search Panel

**File:** `tools/cli/tui/database_panel.zig`

**Purpose:** Monitor WDBX vector database health and performance.

```
╭─ Database Panel ─ WDBX Vector Store ────────────────────────────╮
│ Status: ● Online   Vectors: 1,234,567   Dimensions: 384        │
├─────────────────────────────────────────────────────────────────┤
│ Indexes                                                         │
│ ┌──────────────────┬────────┬─────────┬──────────────────────┐ │
│ │ Name             │ Type   │ Vectors │ Health               │ │
│ ├──────────────────┼────────┼─────────┼──────────────────────┤ │
│ │ embeddings_main  │ HNSW   │ 892K    │ ████████████████ 98% │ │
│ │ documents_idx    │ IVF-PQ │ 342K    │ ██████████████░░ 87% │ │
│ │ code_search      │ HNSW   │ 12K     │ ████████████████ 100%│ │
│ └──────────────────┴────────┴─────────┴──────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Query Performance              Memory Usage                     │
│ Avg Latency: 12ms             Pool: ██████████░░░░ 2.1/4.0 GB  │
│ P99 Latency: 45ms             Cache: █████░░░░░░░░░ 512 MB     │
│ QPS: ▂▃▄▅▆▇▆▅▄▃▂▁▂▃          Disk: ████████░░░░░░ 8.2 GB     │
├─────────────────────────────────────────────────────────────────┤
│ [o] Optimize  [b] Backup  [r] Rebuild  [s] Stats  [q] Quit    │
╰─────────────────────────────────────────────────────────────────╯
```

**Struct Definition:**

```zig
pub const DatabasePanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,
    
    // Database state
    db_status: DbStatus,
    total_vectors: u64,
    dimensions: u32,
    
    // Index list
    indexes: std.ArrayListUnmanaged(IndexInfo),
    selected_index: usize,
    
    // Performance metrics
    query_latency_history: RingBuffer(u32, 60),
    qps_history: RingBuffer(f32, 60),
    latency_percentiles: PercentileTracker,
    
    // Memory usage
    memory_pool_used: u64,
    memory_pool_total: u64,
    cache_used: u64,
    disk_used: u64,
    
    pub fn init(...) DatabasePanel;
    pub fn deinit(self: *DatabasePanel) void;
    pub fn refresh(self: *DatabasePanel) !void;
    pub fn render(self: *DatabasePanel, row: usize, col: usize, width: usize, height: usize) !void;
    pub fn handleKey(self: *DatabasePanel, key: events.Key) ?Action;
};

const IndexInfo = struct {
    name: []const u8,
    index_type: IndexType,
    vector_count: u64,
    health_percent: u8,
    last_optimized: i64,
};
```

**Data Sources:**
- `src/database/mod.zig` - Database instance
- `src/database/stats.zig` - Query statistics
- `src/database/health.zig` - Index health checks

---

### 3.4 Multi-Agent Workflow View

**File:** `tools/cli/tui/agent_workflow_panel.zig`

**Purpose:** Monitor multi-agent coordination and decision history.

```
╭─ Multi-Agent Workflow View ─────────────────────────────────────╮
│ Workflow: code_review_pipeline   Status: ● Running  Agents: 4  │
├─────────────────────────────────────────────────────────────────┤
│ Agent Pipeline                                                  │
│                                                                 │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐               │
│   │ Analyst │ ───► │Reviewer │ ───► │ Writer  │               │
│   │   ●     │      │   ◐     │      │   ○     │               │
│   └─────────┘      └─────────┘      └─────────┘               │
│        │                                  │                     │
│        └──────────┐          ┌───────────┘                     │
│                   ▼          ▼                                  │
│              ┌─────────────────┐                               │
│              │  Coordinator    │                               │
│              │       ●         │                               │
│              └─────────────────┘                               │
├─────────────────────────────────────────────────────────────────┤
│ Decision Log                                                    │
│ 14:32:15 Coordinator → Analyst: "Analyze PR #123"              │
│ 14:32:18 Analyst → Reviewer: "Found 3 issues" [conf: 0.9]      │
│ 14:32:22 Reviewer: Processing... ████████░░░░ 67%              │
├─────────────────────────────────────────────────────────────────┤
│ [f] Follow  [p] Pause  [d] Details  [l] Full Log  [q] Quit    │
╰─────────────────────────────────────────────────────────────────╯
```

**Struct Definition:**

```zig
pub const AgentWorkflowPanel = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,
    
    // Workflow state
    workflow_name: []const u8,
    workflow_status: WorkflowStatus,
    
    // Agents
    agents: std.ArrayListUnmanaged(AgentInfo),
    agent_connections: std.ArrayListUnmanaged(Connection),
    selected_agent: ?usize,
    
    // Decision log
    decision_log: RingBuffer(DecisionEntry, 500),
    log_scroll: usize,
    follow_mode: bool,
    
    // Progress tracking
    current_step: usize,
    total_steps: usize,
    
    pub fn init(...) AgentWorkflowPanel;
    pub fn deinit(self: *AgentWorkflowPanel) void;
    pub fn update(self: *AgentWorkflowPanel) !void;
    pub fn render(self: *AgentWorkflowPanel, row: usize, col: usize, width: usize, height: usize) !void;
    pub fn handleKey(self: *AgentWorkflowPanel, key: events.Key) ?Action;
};

const AgentInfo = struct {
    id: []const u8,
    name: []const u8,
    persona: []const u8,
    status: AgentStatus,
    progress: ?u8,  // 0-100 if processing
    position: struct { row: u8, col: u8 },
};

const DecisionEntry = struct {
    timestamp: i64,
    from_agent: ?[]const u8,
    to_agent: ?[]const u8,
    message: []const u8,
    confidence: ?f32,
    entry_type: enum { message, decision, progress, error },
};
```

**Data Sources:**
- `src/ai/multi_agent/coordinator.zig` - Workflow state
- `src/ai/multi_agent/messaging.zig` - Message bus
- `src/ai/agents/mod.zig` - Agent registry

---

### 3.5 Interactive Argument Picker

**File:** `tools/cli/utils/picker.zig`

**Purpose:** Prompt for missing arguments when launching commands from TUI.

```
╭─ Command: llm generate ─────────────────────────────────────────╮
│                                                                 │
│ Select Model:                                                   │
│   ● llama-7b-q4 (active)                                       │
│   ○ mistral-7b-instruct                                        │
│   ○ phi-3-mini                                                  │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Prompt: [Enter your prompt...]                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ Write a function that calculates fibonacci numbers          │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Options:                                                        │
│   Max tokens: [256    ]  Temperature: [0.7 ]  Top-p: [0.9 ]   │
│   ☑ Stream output   ☐ Show timing                              │
├─────────────────────────────────────────────────────────────────┤
│                              [Enter] Run   [Esc] Cancel        │
╰─────────────────────────────────────────────────────────────────╯
```

**Core Types:**

```zig
pub const ArgumentPicker = struct {
    allocator: std.mem.Allocator,
    term: *terminal.Terminal,
    theme: *const themes.Theme,
    
    fields: []const Field,
    values: std.StringHashMapUnmanaged(Value),
    current_field: usize,
    
    pub fn init(allocator: std.mem.Allocator, term: *terminal.Terminal, theme: *const themes.Theme, fields: []const Field) ArgumentPicker;
    pub fn deinit(self: *ArgumentPicker) void;
    pub fn run(self: *ArgumentPicker) !?std.StringHashMap([]const u8);
    pub fn render(self: *ArgumentPicker) !void;
    pub fn handleKey(self: *ArgumentPicker, key: events.Key) ?PickerAction;
};

pub const Field = struct {
    name: []const u8,
    label: []const u8,
    description: []const u8,
    field_type: FieldType,
    required: bool,
    default: ?Value,
    options: ?[]const Option,  // For select/multiselect
    validation: ?*const fn([]const u8) ?[]const u8,  // Returns error message or null
};

pub const FieldType = enum {
    text,           // Single-line text input
    textarea,       // Multi-line text input
    number,         // Integer input
    float,          // Decimal input
    select,         // Single selection from options
    multiselect,    // Multiple selection from options
    checkbox,       // Boolean toggle
    file,           // File path with completion
    directory,      // Directory path with completion
};

pub const Value = union(enum) {
    text: []const u8,
    number: i64,
    float: f64,
    boolean: bool,
    list: []const []const u8,
};
```

**Command Schema Definition:**

```zig
// Example: llm generate command schema
pub const llm_generate_schema = [_]Field{
    .{
        .name = "model",
        .label = "Model",
        .description = "The model to use for generation",
        .field_type = .select,
        .required = true,
        .default = null,
        .options = null,  // Populated dynamically from model list
    },
    .{
        .name = "prompt",
        .label = "Prompt",
        .description = "The text prompt for generation",
        .field_type = .textarea,
        .required = true,
        .default = null,
        .options = null,
    },
    .{
        .name = "max_tokens",
        .label = "Max Tokens",
        .description = "Maximum tokens to generate",
        .field_type = .number,
        .required = false,
        .default = .{ .number = 256 },
        .options = null,
    },
    // ... more fields
};
```

---

### 3.6 TUI Launcher Enhancements

**File:** `tools/cli/commands/tui.zig` (modifications)

**Changes:**

1. **Add `model` command to menu:**

```zig
.{
    .label = "model",
    .description = "Model management (download, cache, switch)",
    .action = .{ .command = .model },
    .category = .ai_ml,
    .shortcut = 'm',
    .usage = "model <subcommand>",
    .examples = &[_][]const u8{
        "model list",
        "model download llama-7b",
        "model info mistral",
    },
    .related = &[_][]const u8{ "llm", "agent", "embed" },
},
```

2. **Add new panel launch items:**

```zig
.{
    .label = "model-panel",
    .description = "Interactive model management panel",
    .action = .{ .panel = .model_management },
    .category = .ai_ml,
    .shortcut = 'M',
},
.{
    .label = "streaming-dashboard",
    .description = "Real-time streaming inference metrics",
    .action = .{ .panel = .streaming_dashboard },
    .category = .ai_ml,
    .shortcut = 'S',
},
.{
    .label = "db-panel",
    .description = "Database health and performance monitor",
    .action = .{ .panel = .database },
    .category = .data,
    .shortcut = 'D',
},
.{
    .label = "agent-view",
    .description = "Multi-agent workflow monitor",
    .action = .{ .panel = .agent_workflow },
    .category = .system,
    .shortcut = 'A',
},
```

3. **Update Action enum:**

```zig
const Action = union(enum) {
    command: Command,
    panel: Panel,
    builtin: Builtin,
};

const Panel = enum {
    model_management,
    streaming_dashboard,
    database,
    agent_workflow,
    gpu_monitor,      // Existing
    training,         // Existing
};
```

4. **Implement panel launching:**

```zig
fn launchPanel(state: *TuiState, panel: Panel) !void {
    switch (panel) {
        .model_management => {
            var model_panel = ModelManagementPanel.init(state.allocator, state.terminal, state.theme_manager.currentTheme());
            defer model_panel.deinit();
            try model_panel.runInteractive();
        },
        .streaming_dashboard => {
            var dashboard = StreamingDashboard.init(state.allocator, state.terminal, state.theme_manager.currentTheme());
            defer dashboard.deinit();
            try dashboard.runInteractive();
        },
        // ... other panels
    }
}
```

5. **Interactive argument prompting:**

When a command is selected and requires arguments:

```zig
fn executeCommand(state: *TuiState, command: Command) !void {
    const schema = getCommandSchema(command);
    if (schema) |s| {
        // Check if any required args are missing
        var picker = ArgumentPicker.init(state.allocator, state.terminal, state.theme_manager.currentTheme(), s);
        defer picker.deinit();
        
        const args = try picker.run() orelse return;  // User cancelled
        defer args.deinit();
        
        // Build command args from picker values
        var cmd_args = try buildCommandArgs(state.allocator, args);
        defer cmd_args.deinit();
        
        // Execute command
        try runCommandWithArgs(command, cmd_args.items);
    }
}
```

---

## 4. Shared Infrastructure

### 4.1 Ring Buffer Generic

**File:** `tools/cli/tui/ring_buffer.zig`

```zig
pub fn RingBuffer(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();
        
        data: [capacity]T,
        head: usize,
        count: usize,
        
        pub fn init() Self {
            return .{
                .data = undefined,
                .head = 0,
                .count = 0,
            };
        }
        
        pub fn push(self: *Self, value: T) void {
            self.data[self.head] = value;
            self.head = (self.head + 1) % capacity;
            if (self.count < capacity) self.count += 1;
        }
        
        pub fn toSlice(self: *const Self, buf: []T) []T {
            // Copy in order from oldest to newest
            var i: usize = 0;
            var pos = if (self.count == capacity) self.head else 0;
            while (i < self.count) : (i += 1) {
                buf[i] = self.data[pos];
                pos = (pos + 1) % capacity;
            }
            return buf[0..self.count];
        }
        
        pub fn latest(self: *const Self) ?T {
            if (self.count == 0) return null;
            const idx = if (self.head == 0) capacity - 1 else self.head - 1;
            return self.data[idx];
        }
        
        pub fn average(self: *const Self) f64 {
            if (self.count == 0) return 0;
            var sum: f64 = 0;
            for (self.data[0..self.count]) |v| sum += @as(f64, @floatFromInt(v));
            return sum / @as(f64, @floatFromInt(self.count));
        }
    };
}
```

### 4.2 Percentile Tracker

**File:** `tools/cli/tui/percentile_tracker.zig`

```zig
pub const PercentileTracker = struct {
    samples: std.ArrayListUnmanaged(u32),
    sorted: bool,
    max_samples: usize,
    
    pub fn init(allocator: std.mem.Allocator, max_samples: usize) PercentileTracker;
    pub fn deinit(self: *PercentileTracker) void;
    pub fn add(self: *PercentileTracker, value: u32) void;
    pub fn getPercentile(self: *PercentileTracker, p: u8) u32;  // p = 50 for P50, etc.
    pub fn reset(self: *PercentileTracker) void;
};
```

---

## 5. File Structure

New files to create:

```
tools/cli/tui/
├── model_panel.zig           # Model Management Panel
├── streaming_dashboard.zig   # Streaming Inference Dashboard
├── database_panel.zig        # Database/Vector Panel
├── agent_workflow_panel.zig  # Multi-Agent Workflow View
├── ring_buffer.zig           # Generic ring buffer
└── percentile_tracker.zig    # Percentile statistics

tools/cli/utils/
└── picker.zig                # Interactive Argument Picker

tools/cli/commands/
└── tui.zig                   # (modifications) Add panels and model command
```

Modifications to existing files:

```
tools/cli/commands/tui.zig    # Add new panels, model command
tools/cli/commands/mod.zig    # Ensure model.zig is exported
tools/cli/tui/mod.zig         # Export new panels
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

Each panel should have tests in its file:

```zig
test "ModelManagementPanel renders correctly" {
    // Mock terminal, theme
    // Verify render output
}

test "RingBuffer maintains FIFO order" {
    var buf = RingBuffer(u32, 5).init();
    buf.push(1); buf.push(2); buf.push(3);
    // Verify order
}

test "ArgumentPicker validates required fields" {
    // Test validation logic
}
```

### 6.2 Integration Tests

```zig
test "TUI launcher can launch model panel" {
    // End-to-end test of panel launch
}

test "Streaming dashboard polls metrics" {
    // Test with mock server
}
```

---

## 7. Dependencies

### 7.1 Internal Dependencies

| Component | Depends On |
|-----------|------------|
| Model Panel | `src/ai/models/manager.zig`, `downloader.zig` |
| Streaming Dashboard | `src/ai/streaming/server.zig`, `metrics.zig` |
| Database Panel | `src/database/mod.zig`, `stats.zig`, `health.zig` |
| Agent Workflow | `src/ai/multi_agent/coordinator.zig`, `messaging.zig` |
| Argument Picker | `tools/cli/tui/widgets.zig`, `events.zig` |

### 7.2 No External Dependencies

All components use existing infrastructure:
- Terminal abstraction from `tools/cli/tui/terminal.zig`
- Widgets from `tools/cli/tui/widgets.zig`
- Themes from `tools/cli/tui/themes.zig`
- Events from `tools/cli/tui/events.zig`

---

## 8. Implementation Order

Based on dependencies and value:

| Phase | Components | Rationale |
|-------|------------|-----------|
| 1 | Ring Buffer, Percentile Tracker | Shared infrastructure |
| 2 | TUI Launcher: Add model command | Quick win, unblocks model panel |
| 3 | Model Management Panel | High user value |
| 4 | Interactive Argument Picker | Enables better UX for all commands |
| 5 | Streaming Dashboard | Real-time monitoring |
| 6 | Database Panel | Monitoring value |
| 7 | Multi-Agent Workflow View | Advanced feature |

---

## 9. Open Questions

1. **Polling vs Push**: Should panels poll for data or receive push updates?
   - **Decision**: Start with polling (simpler), add push later if needed.

2. **State Persistence**: Should panel state (scroll position, selections) persist across sessions?
   - **Decision**: No for MVP, consider for future.

3. **Remote Monitoring**: Should dashboards support monitoring remote instances?
   - **Decision**: Yes for streaming dashboard (HTTP endpoint), local only for others initially.

---

## 10. Success Criteria

1. ✅ All 4 new panels render correctly with 5 themes
2. ✅ Model command integrated into TUI launcher
3. ✅ Interactive argument picker works for llm/agent commands
4. ✅ Real-time metrics update at 1 Hz or faster
5. ✅ All new code has unit tests
6. ✅ No regression in existing TUI functionality
7. ✅ Keyboard navigation consistent across all panels

---

## Appendix A: Key Bindings Reference

| Key | Global | Model Panel | Streaming | Database | Agent View |
|-----|--------|-------------|-----------|----------|------------|
| `q` | Quit | Quit | Quit | Quit | Quit |
| `t` | Theme | Theme | Theme | Theme | Theme |
| `?` | Help | Help | Help | Help | Help |
| `j`/`↓` | - | Down | - | Down | Down |
| `k`/`↑` | - | Up | - | Up | Up |
| `Enter` | - | Select | - | Select | Select |
| `d` | - | Download | - | - | Details |
| `r` | - | Remove | Refresh | Rebuild | - |
| `s` | - | Set Active | - | Stats | - |
| `f` | - | - | - | - | Follow |
| `l` | - | - | Log | - | Full Log |

---

## Appendix B: Color Coding

All panels use consistent color coding from theme:

| State | Color | Usage |
|-------|-------|-------|
| Success/Online | `theme.success` (green) | Active model, healthy index |
| Warning/Degraded | `theme.warning` (yellow) | High latency, low health |
| Error/Offline | `theme.error` (red) | Failed download, offline |
| In Progress | `theme.info` (blue) | Downloading, processing |
| Inactive | `theme.text_dim` (gray) | Inactive models, idle agents |
