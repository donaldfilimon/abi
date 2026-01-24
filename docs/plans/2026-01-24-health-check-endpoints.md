# Health Check Endpoints Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add HTTP health check endpoints for production deployment and monitoring integration.

**Architecture:** Extend the web module with `/health`, `/ready`, and `/live` endpoints that aggregate system component status using existing observability infrastructure.

**Tech Stack:** Zig 0.16, existing web module, observability metrics

---

## Task 1: Create Health Status Types

**Files:**
- Create: `src/web/health.zig`

**Step 1: Define health status types**

```zig
//! Health Check Module
//!
//! Provides health check endpoints for Kubernetes probes and monitoring systems.

const std = @import("std");

/// Component health status.
pub const Status = enum {
    healthy,
    degraded,
    unhealthy,

    pub fn toString(self: Status) []const u8 {
        return switch (self) {
            .healthy => "healthy",
            .degraded => "degraded",
            .unhealthy => "unhealthy",
        };
    }

    pub fn httpCode(self: Status) u16 {
        return switch (self) {
            .healthy => 200,
            .degraded => 200,
            .unhealthy => 503,
        };
    }
};

/// Individual component check result.
pub const ComponentCheck = struct {
    name: []const u8,
    status: Status,
    message: ?[]const u8 = null,
    latency_ms: ?f64 = null,
};

/// Aggregate health response.
pub const HealthResponse = struct {
    status: Status,
    components: []const ComponentCheck,
    timestamp: i64,
    version: []const u8,
};
```

**Step 2: Verify syntax**

Run: `zig ast-check src/web/health.zig`
Expected: No errors

**Step 3: Commit**

```bash
git add src/web/health.zig
git commit -m "feat(web): add health check types"
```

---

## Task 2: Add Health Check Functions

**Files:**
- Modify: `src/web/health.zig`

**Step 1: Add checker interface**

```zig
/// Health check function type.
pub const CheckFn = *const fn () Status;

/// Health checker with registered components.
pub const HealthChecker = struct {
    allocator: std.mem.Allocator,
    checks: std.StringHashMap(CheckFn),
    version: []const u8,

    pub fn init(allocator: std.mem.Allocator, version: []const u8) HealthChecker {
        return .{
            .allocator = allocator,
            .checks = std.StringHashMap(CheckFn).init(allocator),
            .version = version,
        };
    }

    pub fn deinit(self: *HealthChecker) void {
        self.checks.deinit();
    }

    pub fn registerCheck(self: *HealthChecker, name: []const u8, check: CheckFn) !void {
        try self.checks.put(name, check);
    }

    pub fn runAll(self: *HealthChecker, allocator: std.mem.Allocator) !HealthResponse {
        var components = std.ArrayList(ComponentCheck).init(allocator);
        errdefer components.deinit();

        var overall: Status = .healthy;
        var iter = self.checks.iterator();
        while (iter.next()) |entry| {
            const status = entry.value_ptr.*();
            try components.append(.{
                .name = entry.key_ptr.*,
                .status = status,
            });
            if (@intFromEnum(status) > @intFromEnum(overall)) {
                overall = status;
            }
        }

        return .{
            .status = overall,
            .components = try components.toOwnedSlice(),
            .timestamp = std.time.timestamp(),
            .version = self.version,
        };
    }
};
```

**Step 2: Verify syntax**

Run: `zig ast-check src/web/health.zig`
Expected: No errors

**Step 3: Commit**

```bash
git add src/web/health.zig
git commit -m "feat(web): add health checker implementation"
```

---

## Task 3: Add JSON Serialization

**Files:**
- Modify: `src/web/health.zig`

**Step 1: Add toJson function**

```zig
/// Serialize health response to JSON.
pub fn toJson(response: HealthResponse, allocator: std.mem.Allocator) ![]u8 {
    var output = std.ArrayList(u8).init(allocator);
    errdefer output.deinit();
    const writer = output.writer();

    try writer.writeAll("{\"status\":\"");
    try writer.writeAll(response.status.toString());
    try writer.writeAll("\",\"version\":\"");
    try writer.writeAll(response.version);
    try writer.print("\",\"timestamp\":{d},\"components\":[", .{response.timestamp});

    for (response.components, 0..) |comp, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeAll("{\"name\":\"");
        try writer.writeAll(comp.name);
        try writer.writeAll("\",\"status\":\"");
        try writer.writeAll(comp.status.toString());
        try writer.writeAll("\"}");
    }

    try writer.writeAll("]}");
    return output.toOwnedSlice();
}
```

**Step 2: Verify syntax**

Run: `zig ast-check src/web/health.zig`
Expected: No errors

**Step 3: Commit**

```bash
git add src/web/health.zig
git commit -m "feat(web): add health JSON serialization"
```

---

## Task 4: Register in Web Module

**Files:**
- Modify: `src/web/mod.zig`

**Step 1: Add health import and re-export**

Add to imports section:
```zig
pub const health = @import("health.zig");
```

**Step 2: Verify syntax**

Run: `zig ast-check src/web/mod.zig`
Expected: No errors

**Step 3: Commit**

```bash
git add src/web/mod.zig
git commit -m "feat(web): export health module"
```

---

## Task 5: Add Unit Tests

**Files:**
- Modify: `src/web/health.zig`

**Step 1: Add tests**

```zig
test "status http codes" {
    try std.testing.expectEqual(@as(u16, 200), Status.healthy.httpCode());
    try std.testing.expectEqual(@as(u16, 200), Status.degraded.httpCode());
    try std.testing.expectEqual(@as(u16, 503), Status.unhealthy.httpCode());
}

test "health checker" {
    const allocator = std.testing.allocator;
    var checker = HealthChecker.init(allocator, "1.0.0");
    defer checker.deinit();

    try checker.registerCheck("test", struct {
        fn check() Status {
            return .healthy;
        }
    }.check);

    const response = try checker.runAll(allocator);
    defer allocator.free(response.components);

    try std.testing.expectEqual(Status.healthy, response.status);
    try std.testing.expectEqual(@as(usize, 1), response.components.len);
}

test "json serialization" {
    const allocator = std.testing.allocator;
    const response = HealthResponse{
        .status = .healthy,
        .components = &.{},
        .timestamp = 1706140800,
        .version = "1.0.0",
    };

    const json = try toJson(response, allocator);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"status\":\"healthy\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"version\":\"1.0.0\"") != null);
}
```

**Step 2: Verify syntax**

Run: `zig ast-check src/web/health.zig`
Expected: No errors

**Step 3: Run tests**

Run: `zig test src/web/health.zig`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/web/health.zig
git commit -m "test(web): add health check unit tests"
```

---

## Verification

After all tasks complete:

```bash
zig fmt src/web/health.zig
zig ast-check src/web/health.zig
zig ast-check src/web/mod.zig
```

## Usage Example

```zig
const web = @import("abi").web;

var checker = web.health.HealthChecker.init(allocator, "1.0.0");
defer checker.deinit();

// Register component checks
try checker.registerCheck("database", dbHealthCheck);
try checker.registerCheck("gpu", gpuHealthCheck);

// Run all checks
const response = try checker.runAll(allocator);
const json = try web.health.toJson(response, allocator);
// Return json with response.status.httpCode()
```
