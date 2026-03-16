---
title: errors API
purpose: Generated API reference for errors
last_updated: 2026-03-16
target_zig_version: 0.16.0-dev.2905+5d71e3051
---

# errors

> Composable Error Hierarchy

Defines the framework's error taxonomy as composable error sets.
Feature modules can import and extend these base categories.
`FrameworkError` composes lifecycle, feature, config, and allocator errors.

**Source:** [`src/core/errors.zig`](../../src/core/errors.zig)

**Availability:** Always enabled

---

## API

### <a id="pub-const-lifecycleerror"></a>`pub const LifecycleError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L8)

Lifecycle errors for framework state transitions.

### <a id="pub-const-gpuframeworkerror"></a>`pub const GpuFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L19)

GPU feature errors visible at the framework level.

### <a id="pub-const-aiframeworkerror"></a>`pub const AiFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L28)

AI feature errors visible at the framework level.

### <a id="pub-const-databaseframeworkerror"></a>`pub const DatabaseFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L40)

Database feature errors visible at the framework level.

### <a id="pub-const-networkframeworkerror"></a>`pub const NetworkFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L49)

Network feature errors visible at the framework level.

### <a id="pub-const-observabilityframeworkerror"></a>`pub const ObservabilityFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L57)

Observability feature errors visible at the framework level.

### <a id="pub-const-webframeworkerror"></a>`pub const WebFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L65)

Web feature errors visible at the framework level.

### <a id="pub-const-cloudframeworkerror"></a>`pub const CloudFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L72)

Cloud feature errors visible at the framework level.

### <a id="pub-const-analyticsframeworkerror"></a>`pub const AnalyticsFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L84)

Analytics feature errors visible at the framework level.

### <a id="pub-const-authframeworkerror"></a>`pub const AuthFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L91)

Auth feature errors visible at the framework level.

### <a id="pub-const-messagingframeworkerror"></a>`pub const MessagingFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L99)

Messaging feature errors visible at the framework level.

### <a id="pub-const-cacheframeworkerror"></a>`pub const CacheFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L106)

Cache feature errors visible at the framework level.

### <a id="pub-const-storageframeworkerror"></a>`pub const StorageFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L113)

Storage feature errors visible at the framework level.

### <a id="pub-const-searchframeworkerror"></a>`pub const SearchFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L121)

Search feature errors visible at the framework level.

### <a id="pub-const-allfeatureerrors"></a>`pub const AllFeatureErrors`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L129)

All feature errors combined.

### <a id="pub-const-frameworkerror"></a>`pub const FrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L145)

The complete Framework error set.
Composes lifecycle errors, all feature errors, and infrastructure errors.



---

*Generated automatically by `zig build gendocs`*


## Workflow Contract
- Canonical repo workflow: [AGENTS.md](../../AGENTS.md)
- Active execution tracker: [tasks/todo.md](../../tasks/todo.md)
- Correction log: [tasks/lessons.md](../../tasks/lessons.md)

## Zig Validation
Use `zig build full-check` on supported hosts. On Darwin 25+ / 26+, use `zig fmt --check ...` plus `./tools/scripts/run_build.sh <step>`. For docs generation, use `zig build gendocs` or `./tools/scripts/run_build.sh gendocs` on Darwin.
