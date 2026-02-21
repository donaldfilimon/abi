# errors

> Composable error hierarchy for framework operations.

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

<sup>**const**</sup> | [source](../../src/core/errors.zig#L39)

Database feature errors visible at the framework level.

### <a id="pub-const-networkframeworkerror"></a>`pub const NetworkFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L48)

Network feature errors visible at the framework level.

### <a id="pub-const-observabilityframeworkerror"></a>`pub const ObservabilityFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L56)

Observability feature errors visible at the framework level.

### <a id="pub-const-webframeworkerror"></a>`pub const WebFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L64)

Web feature errors visible at the framework level.

### <a id="pub-const-cloudframeworkerror"></a>`pub const CloudFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L71)

Cloud feature errors visible at the framework level.

### <a id="pub-const-analyticsframeworkerror"></a>`pub const AnalyticsFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L83)

Analytics feature errors visible at the framework level.

### <a id="pub-const-authframeworkerror"></a>`pub const AuthFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L90)

Auth feature errors visible at the framework level.

### <a id="pub-const-messagingframeworkerror"></a>`pub const MessagingFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L98)

Messaging feature errors visible at the framework level.

### <a id="pub-const-cacheframeworkerror"></a>`pub const CacheFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L105)

Cache feature errors visible at the framework level.

### <a id="pub-const-storageframeworkerror"></a>`pub const StorageFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L112)

Storage feature errors visible at the framework level.

### <a id="pub-const-searchframeworkerror"></a>`pub const SearchFrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L120)

Search feature errors visible at the framework level.

### <a id="pub-const-allfeatureerrors"></a>`pub const AllFeatureErrors`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L128)

All feature errors combined.

### <a id="pub-const-frameworkerror"></a>`pub const FrameworkError`

<sup>**const**</sup> | [source](../../src/core/errors.zig#L144)

The complete Framework error set.
Composes lifecycle errors, all feature errors, and infrastructure errors.

---

*Generated automatically by `zig build gendocs`*

## Zig Skill
Use the `$zig` Codex skill for ABI Zig 0.16-dev syntax updates, modular build graph guidance, and targeted validation workflows.
