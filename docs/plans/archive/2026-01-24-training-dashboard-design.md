# Training Dashboard Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A TUI dashboard that shows real-time training progress and historical run comparison, accessible via `train monitor` command or from the TUI launcher.

**Architecture:** Widget-based panel design using existing TUI infrastructure (terminal.zig, themes.zig, widgets.zig). Live mode polls a JSONL metrics file written by TrainingLogger; history mode reads past run logs.

**Tech Stack:** Zig 0.16, existing TUI framework, JSONL for metrics streaming

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Training Monitor: run-2026-01-24-abc123        [Live] ⏱ 00:45:32  │
├────────────────────────────────────┬────────────────────────────────┤
│  Loss                              │  Learning Rate                 │
│  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁    │  ▁▁▁▂▃▄▅▆▇████████▇▆▅▄▃▂▁▁   │
│  train: 0.0234  val: 0.0312        │  current: 0.0001  warmup: 5%   │
│  epoch: 7/10    step: 2340/3500    │  schedule: cosine              │
├────────────────────────────────────┼────────────────────────────────┤
│  Resources                         │  Checkpoints                   │
│  GPU:  ████████████░░░░ 78%        │  ✓ epoch-5.ckpt   12.4 MB      │
│  VRAM: █████████░░░░░░░ 62%        │  ✓ epoch-6.ckpt   12.4 MB      │
│  RAM:  ████░░░░░░░░░░░░ 24%        │  ★ best.ckpt      12.4 MB      │
│  Disk: 2.1 GB used                 │  ⏳ epoch-7.ckpt  (saving...)  │
├─────────────────────────────────────────────────────────────────────┤
│  [r] Refresh  [h] History  [q] Quit  [←/→] Switch runs  [?] Help   │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

**Live Mode:**
```
TrainingLoop → TrainingLogger → metrics.jsonl file → TrainingPanel (polling)
```

**Metric event format (JSONL):**
```json
{ "type": "scalar", "tag": "loss/train", "value": 0.0234, "step": 2340, "ts": 1706123456 }
{ "type": "scalar", "tag": "loss/val", "value": 0.0312, "step": 2340, "ts": 1706123457 }
{ "type": "checkpoint", "path": "epoch-7.ckpt", "size": 13002400, "step": 2340 }
```

**History Mode:**
Scan log directory for past runs, parse metrics files, allow browsing/comparison.

## File Structure

```
tools/cli/
├── commands/
│   └── train.zig              # Add "monitor" subcommand
└── tui/
    ├── training_panel.zig     # NEW: Main panel coordinator
    ├── training_metrics.zig   # NEW: Metrics reader/parser
    └── widgets.zig            # Extend with new chart widgets

src/ai/training/
└── logging.zig                # Extend: add JSONL metrics stream
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| No active training | Show "No active training. Press [h] for history." |
| Metrics file missing | Graceful degradation, show available data only |
| Training crashes | Detect stale timestamps, show "Training stopped" |
| Disk full (checkpoint) | Show warning icon, continue monitoring |
| Invalid run ID | List available runs, suggest closest match |

## YAGNI Decisions

**Included (MVP):**
- Single run view (live or historical)
- Loss, LR, GPU, checkpoints display
- Basic keyboard navigation
- JSONL metrics format

**Deferred (not in v1):**
- Multi-run comparison charts
- Export to PNG/CSV
- Remote training monitoring
- Custom metric configuration
- Alerts/notifications

---

## Implementation Tasks

### Task 1: Add JSONL MetricsStream to logging.zig

**Files:**
- Modify: `src/ai/training/logging.zig`
- Test: `src/ai/training/logging.zig` (inline tests)

**Step 1: Write the failing test**

```zig
test "MetricsStream writes JSONL events" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stream = try MetricsStream.init(allocator, "/tmp/test-metrics.jsonl");
    defer stream.deinit();

    try stream.logScalar("loss/train", 0.5, 100);

    // Read back and verify
    const content = try std.fs.cwd().readFileAlloc(allocator, "/tmp/test-metrics.jsonl", 4096);
    defer allocator.free(content);

    try std.testing.expect(std.mem.indexOf(u8, content, "loss/train") != null);
    try std.testing.expect(std.mem.indexOf(u8, content, "0.5") != null);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test src/ai/training/logging.zig --test-filter "MetricsStream"`
Expected: FAIL with "MetricsStream not defined"

**Step 3: Write minimal implementation**

```zig
pub const MetricsStream = struct {
    allocator: std.mem.Allocator,
    io_backend: std.Io.Threaded,
    file: std.Io.File,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !MetricsStream {
        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = .empty });
        errdefer io_backend.deinit();
        const io = io_backend.io();

        const file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = false });
        return .{ .allocator = allocator, .io_backend = io_backend, .file = file };
    }

    pub fn deinit(self: *MetricsStream) void {
        const io = self.io_backend.io();
        self.file.close(io);
        self.io_backend.deinit();
    }

    pub fn logScalar(self: *MetricsStream, tag: []const u8, value: f32, step: u64) !void {
        const io = self.io_backend.io();
        const ts = std.time.timestamp();
        var buf: [256]u8 = undefined;
        const line = try std.fmt.bufPrint(&buf,
            "{{\"type\":\"scalar\",\"tag\":\"{s}\",\"value\":{d},\"step\":{d},\"ts\":{d}}}\n",
            .{ tag, value, step, ts });
        try self.file.writer(io).writeAll(line);
    }
};
```

**Step 4: Run test to verify it passes**

Run: `zig test src/ai/training/logging.zig --test-filter "MetricsStream"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/ai/training/logging.zig
git commit -m "feat(training): add MetricsStream for JSONL metrics logging"
```

---

### Task 2: Add SparklineChart widget

**Files:**
- Modify: `tools/cli/tui/widgets.zig`
- Test: `tools/cli/tui/widgets.zig` (inline tests)

**Step 1: Write the failing test**

```zig
test "SparklineChart renders values" {
    const values = [_]u8{ 10, 30, 50, 70, 90, 70, 50, 30, 10 };
    var buf: [64]u8 = undefined;
    const result = SparklineChart.render(&values, &buf);

    // Should contain sparkline characters
    try std.testing.expect(result.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result, "▁") != null or
                          std.mem.indexOf(u8, result, "█") != null);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test tools/cli/tui/widgets.zig --test-filter "SparklineChart"`
Expected: FAIL

**Step 3: Write minimal implementation**

```zig
pub const SparklineChart = struct {
    pub const BARS = "▁▂▃▄▅▆▇█";

    /// Render values (0-100) as sparkline into buffer
    pub fn render(values: []const u8, buf: []u8) []const u8 {
        var pos: usize = 0;
        for (values) |v| {
            const idx = @min(7, v / 13); // Map 0-100 to 0-7
            const bar = BARS[idx * 3 ..][0..3]; // UTF-8 chars are 3 bytes
            if (pos + 3 > buf.len) break;
            @memcpy(buf[pos..][0..3], bar);
            pos += 3;
        }
        return buf[0..pos];
    }
};
```

**Step 4: Run test to verify it passes**

Run: `zig test tools/cli/tui/widgets.zig --test-filter "SparklineChart"`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/cli/tui/widgets.zig
git commit -m "feat(tui): add SparklineChart widget for metric visualization"
```

---

### Task 3: Add ProgressGauge widget

**Files:**
- Modify: `tools/cli/tui/widgets.zig`

**Step 1: Write the failing test**

```zig
test "ProgressGauge renders percentage" {
    var buf: [32]u8 = undefined;
    const result = ProgressGauge.render(75, 16, &buf);

    try std.testing.expect(std.mem.indexOf(u8, result, "█") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "░") != null);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test tools/cli/tui/widgets.zig --test-filter "ProgressGauge"`
Expected: FAIL

**Step 3: Write minimal implementation**

```zig
pub const ProgressGauge = struct {
    /// Render percentage (0-100) as progress bar
    pub fn render(percent: u8, width: usize, buf: []u8) []const u8 {
        const filled = (percent * width) / 100;
        var pos: usize = 0;

        var i: usize = 0;
        while (i < width and pos < buf.len) : (i += 1) {
            buf[pos] = if (i < filled) 0xE2 else 0xE2; // █ or ░
            buf[pos + 1] = if (i < filled) 0x96 else 0x96;
            buf[pos + 2] = if (i < filled) 0x88 else 0x91;
            pos += 3;
        }
        return buf[0..pos];
    }
};
```

**Step 4: Run test to verify it passes**

Run: `zig test tools/cli/tui/widgets.zig --test-filter "ProgressGauge"`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/cli/tui/widgets.zig
git commit -m "feat(tui): add ProgressGauge widget for resource bars"
```

---

### Task 4: Create training_metrics.zig parser

**Files:**
- Create: `tools/cli/tui/training_metrics.zig`

**Step 1: Write the failing test**

```zig
test "MetricsParser parses JSONL line" {
    const line = "{\"type\":\"scalar\",\"tag\":\"loss/train\",\"value\":0.5,\"step\":100,\"ts\":1706123456}";
    const event = try MetricsParser.parseLine(line);

    try std.testing.expectEqualStrings("loss/train", event.tag);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), event.value, 0.001);
    try std.testing.expectEqual(@as(u64, 100), event.step);
}
```

**Step 2: Run test to verify it fails**

Run: `zig test tools/cli/tui/training_metrics.zig`
Expected: FAIL (file doesn't exist)

**Step 3: Write minimal implementation**

```zig
//! Training metrics parser for JSONL format.
const std = @import("std");

pub const MetricEvent = struct {
    event_type: EventType,
    tag: []const u8,
    value: f32,
    step: u64,
    timestamp: i64,
};

pub const EventType = enum { scalar, checkpoint };

pub const MetricsParser = struct {
    pub fn parseLine(line: []const u8) !MetricEvent {
        // Simple JSON parsing for known format
        var event = MetricEvent{
            .event_type = .scalar,
            .tag = "",
            .value = 0,
            .step = 0,
            .timestamp = 0,
        };

        // Extract tag
        if (std.mem.indexOf(u8, line, "\"tag\":\"")) |start| {
            const tag_start = start + 7;
            if (std.mem.indexOfPos(u8, line, tag_start, "\"")) |end| {
                event.tag = line[tag_start..end];
            }
        }

        // Extract value
        if (std.mem.indexOf(u8, line, "\"value\":")) |start| {
            const val_start = start + 8;
            var end = val_start;
            while (end < line.len and (line[end] == '.' or line[end] == '-' or
                   (line[end] >= '0' and line[end] <= '9'))) : (end += 1) {}
            event.value = std.fmt.parseFloat(f32, line[val_start..end]) catch 0;
        }

        // Extract step
        if (std.mem.indexOf(u8, line, "\"step\":")) |start| {
            const step_start = start + 7;
            var end = step_start;
            while (end < line.len and line[end] >= '0' and line[end] <= '9') : (end += 1) {}
            event.step = std.fmt.parseInt(u64, line[step_start..end], 10) catch 0;
        }

        return event;
    }
};
```

**Step 4: Run test to verify it passes**

Run: `zig test tools/cli/tui/training_metrics.zig`
Expected: PASS

**Step 5: Commit**

```bash
git add tools/cli/tui/training_metrics.zig
git commit -m "feat(tui): add training_metrics.zig JSONL parser"
```

---

### Task 5: Create training_panel.zig scaffold

**Files:**
- Create: `tools/cli/tui/training_panel.zig`
- Modify: `tools/cli/tui/mod.zig`

**Step 1: Create panel scaffold with init/deinit**

```zig
//! Training Progress Panel for TUI
//!
//! Displays training metrics: loss curves, learning rate,
//! GPU/memory usage, and checkpoint status.

const std = @import("std");
const terminal = @import("terminal.zig");
const themes = @import("themes.zig");
const widgets = @import("widgets.zig");
const metrics = @import("training_metrics.zig");

pub const TrainingPanel = struct {
    allocator: std.mem.Allocator,
    theme: *const themes.Theme,
    mode: Mode,
    run_id: ?[]const u8,

    // Metric buffers
    loss_history: [30]u8,
    lr_history: [30]u8,

    pub const Mode = enum { live, history };

    pub fn init(allocator: std.mem.Allocator, theme: *const themes.Theme) TrainingPanel {
        return .{
            .allocator = allocator,
            .theme = theme,
            .mode = .live,
            .run_id = null,
            .loss_history = [_]u8{50} ** 30,
            .lr_history = [_]u8{50} ** 30,
        };
    }

    pub fn deinit(self: *TrainingPanel) void {
        _ = self;
    }

    pub fn render(self: *TrainingPanel, writer: anytype) !void {
        try self.renderHeader(writer);
        try self.renderLossPanel(writer);
        try self.renderFooter(writer);
    }

    fn renderHeader(self: *TrainingPanel, writer: anytype) !void {
        const mode_str = if (self.mode == .live) "[Live]" else "[History]";
        try writer.print("{s}Training Monitor{s} {s}\n", .{
            self.theme.header, mode_str, self.theme.reset
        });
    }

    fn renderLossPanel(self: *TrainingPanel, writer: anytype) !void {
        var buf: [128]u8 = undefined;
        const sparkline = widgets.SparklineChart.render(&self.loss_history, &buf);
        try writer.print("Loss: {s}\n", .{sparkline});
    }

    fn renderFooter(self: *TrainingPanel, writer: anytype) !void {
        try writer.print("[r] Refresh  [h] History  [q] Quit\n", .{});
    }
};
```

**Step 2: Add to tui/mod.zig**

Add import and re-export.

**Step 3: Run build to verify compilation**

Run: `zig build`
Expected: SUCCESS

**Step 4: Commit**

```bash
git add tools/cli/tui/training_panel.zig tools/cli/tui/mod.zig
git commit -m "feat(tui): add training_panel.zig scaffold"
```

---

### Task 6: Add "monitor" subcommand to train.zig

**Files:**
- Modify: `tools/cli/commands/train.zig`

**Step 1: Add monitor to subcommands array**

**Step 2: Add runMonitor function**

```zig
fn runMonitor(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
    _ = args;
    const tui = @import("../tui/mod.zig");

    var panel = tui.TrainingPanel.init(allocator, &tui.themes.default_theme);
    defer panel.deinit();

    const stdout = std.io.getStdOut().writer();
    try panel.render(stdout);
}
```

**Step 3: Add routing in run function**

**Step 4: Test command**

Run: `zig build run -- train monitor`
Expected: Shows basic panel output

**Step 5: Commit**

```bash
git add tools/cli/commands/train.zig
git commit -m "feat(cli): add 'train monitor' subcommand"
```

---

### Task 7: Wire up live metrics polling

**Files:**
- Modify: `tools/cli/tui/training_panel.zig`

**Step 1: Add pollMetrics function**

**Step 2: Integrate with TrainingLogger**

**Step 3: Test with mock metrics file**

**Step 4: Commit**

```bash
git add tools/cli/tui/training_panel.zig
git commit -m "feat(tui): add live metrics polling to training panel"
```

---

### Task 8: Add full UI layout with quadrants

**Files:**
- Modify: `tools/cli/tui/training_panel.zig`

**Step 1: Implement renderResourcesPanel**

**Step 2: Implement renderCheckpointsPanel**

**Step 3: Implement renderLRPanel**

**Step 4: Test full layout**

Run: `zig build run -- train monitor`
Expected: Four-quadrant layout displays

**Step 5: Commit**

```bash
git add tools/cli/tui/training_panel.zig
git commit -m "feat(tui): complete training panel UI layout"
```

---

### Task 9: Add keyboard navigation

**Files:**
- Modify: `tools/cli/tui/training_panel.zig`

**Step 1: Add handleInput function**

**Step 2: Implement mode switching (h for history)**

**Step 3: Implement refresh (r)**

**Step 4: Test keyboard controls**

**Step 5: Commit**

```bash
git add tools/cli/tui/training_panel.zig
git commit -m "feat(tui): add keyboard navigation to training panel"
```

---

### Task 10: Integration with TUI launcher

**Files:**
- Modify: `tools/cli/commands/tui.zig`
- Modify: `tools/cli/tui/mod.zig`

**Step 1: Register training panel in TUI menu**

**Step 2: Test from TUI launcher**

Run: `zig build run -- tui`
Expected: Training option visible in menu

**Step 3: Commit**

```bash
git add tools/cli/commands/tui.zig tools/cli/tui/mod.zig
git commit -m "feat(tui): integrate training panel with TUI launcher"
```

---

## Verification

After all tasks:

```bash
zig fmt .
zig build test --summary all
zig build run -- train monitor
zig build run -- tui
```
