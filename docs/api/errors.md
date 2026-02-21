# errors

> Composable error hierarchy for framework operations.

**Source:** [`src/core/errors.zig`](../../src/core/errors.zig)

**Availability:** Always enabled

---

Composable Error Hierarchy

Defines the framework's error taxonomy as composable error sets.
Feature modules can import and extend these base categories.
`FrameworkError` composes lifecycle, feature, config, and allocator errors.

---

## API

### `pub const LifecycleError`

<sup>**const**</sup>

Lifecycle errors for framework state transitions.

### `pub const GpuFrameworkError`

<sup>**const**</sup>

GPU feature errors visible at the framework level.

### `pub const AiFrameworkError`

<sup>**const**</sup>

AI feature errors visible at the framework level.

### `pub const DatabaseFrameworkError`

<sup>**const**</sup>

Database feature errors visible at the framework level.

### `pub const NetworkFrameworkError`

<sup>**const**</sup>

Network feature errors visible at the framework level.

### `pub const ObservabilityFrameworkError`

<sup>**const**</sup>

Observability feature errors visible at the framework level.

### `pub const WebFrameworkError`

<sup>**const**</sup>

Web feature errors visible at the framework level.

### `pub const CloudFrameworkError`

<sup>**const**</sup>

Cloud feature errors visible at the framework level.

### `pub const AnalyticsFrameworkError`

<sup>**const**</sup>

Analytics feature errors visible at the framework level.

### `pub const AuthFrameworkError`

<sup>**const**</sup>

Auth feature errors visible at the framework level.

### `pub const MessagingFrameworkError`

<sup>**const**</sup>

Messaging feature errors visible at the framework level.

### `pub const CacheFrameworkError`

<sup>**const**</sup>

Cache feature errors visible at the framework level.

### `pub const StorageFrameworkError`

<sup>**const**</sup>

Storage feature errors visible at the framework level.

### `pub const SearchFrameworkError`

<sup>**const**</sup>

Search feature errors visible at the framework level.

### `pub const AllFeatureErrors`

<sup>**const**</sup>

All feature errors combined.

### `pub const FrameworkError`

<sup>**const**</sup>

The complete Framework error set.
Composes lifecycle errors, all feature errors, and infrastructure errors.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use [$zig](/Users/donaldfilimon/.codex/skills/zig/SKILL.md) for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
