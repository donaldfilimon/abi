---
title: "Mobile"
description: "Mobile platform support with lifecycle, sensors, and notifications"
section: "Infrastructure"
order: 7
---

# Mobile

The Mobile module provides mobile platform lifecycle management, sensor access,
and push notification support for iOS and Android targets.

- **Build flag:** `-Denable-mobile=false` (default: **disabled**)
- **Namespace:** `abi.mobile`
- **Source:** `src/features/mobile/`

**Note:** Mobile is the only ABI module that defaults to disabled. You must
explicitly enable it with `-Denable-mobile=true` to compile the real
implementation.

## Overview

The mobile module abstracts platform-specific mobile APIs behind a unified Zig
interface. It manages application lifecycle states, reads sensor data, and sends
push notifications through a single API that works across iOS and Android.

Key capabilities:

- **Lifecycle management** -- Track application state: active, background, suspended, terminated
- **Sensor access** -- Read 3-axis sensor data (accelerometer, gyroscope, etc.) by name
- **Push notifications** -- Send notifications with title and body
- **Platform detection** -- Auto-detect or explicitly target iOS or Android
- **Framework integration** -- Context struct for lifecycle management

### Lifecycle States

| State | Description |
|-------|-------------|
| `active` | Application is in the foreground and receiving events |
| `background` | Application is running but not visible |
| `suspended` | Application is in memory but not executing |
| `terminated` | Application has been terminated |

## Quick Start

```zig
const abi = @import("abi");

// Initialize the mobile module
try abi.mobile.init(allocator, .{
    .platform = .auto,
    .enable_sensors = true,
    .enable_notifications = true,
});
defer abi.mobile.deinit();

// Check lifecycle state
const state = abi.mobile.getLifecycleState();
std.debug.print("App state: {t}\n", .{state});

// Read sensor data
const accel = try abi.mobile.readSensor("accelerometer");
std.debug.print("Accel: [{d:.2}, {d:.2}, {d:.2}]\n", .{
    accel.values[0], accel.values[1], accel.values[2],
});

// Send a notification
try abi.mobile.sendNotification("Hello", "Message from ABI");
```

### Using the Context API

```zig
var ctx = try abi.mobile.Context.init(allocator, .{
    .platform = .ios,
    .enable_sensors = true,
    .enable_notifications = true,
});
defer ctx.deinit();

std.debug.print("Platform: {t}\n", .{ctx.config.platform});
std.debug.print("State: {t}\n", .{ctx.state});
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Context` | Framework integration context with lifecycle state |
| `MobileConfig` | Platform selection, sensor and notification toggles |
| `MobilePlatform` | Platform enum: `auto`, `ios`, `android` |
| `LifecycleState` | Application state: `active`, `background`, `suspended`, `terminated` |
| `SensorData` | Sensor reading: `timestamp_ms` and 3-axis `values` array |
| `MobileError` | Error set: FeatureDisabled, PlatformNotSupported, SensorUnavailable, NotificationFailed, OutOfMemory |

### Key Functions

| Function | Description |
|----------|-------------|
| `init(allocator, config) !void` | Initialize the mobile module |
| `deinit() void` | Tear down the mobile module |
| `isEnabled() bool` | Returns `true` if mobile is compiled in |
| `isInitialized() bool` | Returns `true` if the module is initialized |
| `getLifecycleState() LifecycleState` | Get current application lifecycle state |
| `readSensor(name) !SensorData` | Read named sensor (e.g., `"accelerometer"`) |
| `sendNotification(title, body) !void` | Send a push notification |

## Configuration

Mobile is configured through the `MobileConfig` struct:

```zig
const config = abi.mobile.MobileConfig{
    .platform = .auto,              // .auto, .ios, or .android
    .enable_sensors = false,        // enable sensor access
    .enable_notifications = false,  // enable push notifications
};
```

| Field | Default | Description |
|-------|---------|-------------|
| `platform` | `.auto` | Target platform (auto-detect, iOS, or Android) |
| `enable_sensors` | `false` | Enable sensor data access |
| `enable_notifications` | `false` | Enable push notification support |

## CLI Commands

The mobile module does not have a dedicated CLI command. Use the mobile API
programmatically or through the Framework builder.

## Examples

See `examples/mobile.zig` for a complete working example that demonstrates
both the enabled and disabled (stub) paths, including lifecycle state queries,
sensor reads, and error handling:

```bash
# Mobile is disabled by default; enable it explicitly
zig build -Denable-mobile=true run-mobile
```

The example gracefully handles the disabled case, showing stub error responses:

```bash
# Run without enabling mobile to see stub behavior
zig build run-mobile
```

Output in stub mode:

```
Mobile feature is disabled (default).
Enable with: zig build -Denable-mobile=true

--- Available Types (stub mode) ---
MobileConfig: platform, orientation, sensor settings
LifecycleState: active, background, suspended, terminated
SensorData: timestamp + 3-axis values

Lifecycle state: terminated
init: FeatureDisabled (expected -- feature disabled)
readSensor: FeatureDisabled (expected -- feature disabled)
```

## Disabling at Build Time

Mobile is **disabled by default**. To enable it:

```bash
# Enable mobile support
zig build -Denable-mobile=true
```

To explicitly disable (the default):

```bash
zig build -Denable-mobile=false
```

When disabled, all public functions return `error.FeatureDisabled` and
`isEnabled()` returns `false`. The stub's `getLifecycleState()` returns
`.terminated` (rather than `.active` in the real implementation) as a signal
that the module is not available. The stub `Context.init()` also returns
`error.FeatureDisabled` instead of creating a context.

## Related

- [Cloud](cloud.html) -- Serverless cloud function adapters
- [Web](web.html) -- HTTP client utilities
- [Network](network.html) -- Distributed compute and node management

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for new Zig syntax improvements and validation guidance.
