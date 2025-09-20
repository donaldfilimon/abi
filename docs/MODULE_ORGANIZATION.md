# Abi Framework Module Organization

Updated map of the reorganised Abi source tree. The new layout pivots around
feature-oriented directories and shared runtime layers that are orchestrated via
`src/mod.zig`.

## 📁 Module Architecture

```
src/
├── mod.zig               # Top-level entrypoint that wires framework + features
├── main.zig              # Legacy CLI entry
├── root.zig              # Compatibility exports
├── features/             # Feature families exported via src/features/mod.zig
│   ├── ai/               # Agents, model registry, training loops
│   ├── database/         # Vector store, sharding, HTTP adapters
│   ├── gpu/              # GPU compute backends, memory and demos
│   ├── web/              # HTTP/TCP servers, clients and bindings
│   ├── monitoring/       # Telemetry, profiling and regression tooling
│   └── connectors/       # Third-party API integrations and plugin bridges
├── framework/            # Runtime orchestrator, feature registry, lifecycle
│   ├── config.zig
│   ├── feature_manager.zig
│   ├── runtime.zig
│   └── state.zig
├── shared/               # Cross-cutting utilities reused everywhere
│   ├── core/             # Error handling, lifecycle helpers, registry
│   ├── utils/            # HTTP/JSON/math helpers
│   ├── platform/         # OS abstractions and platform introspection
│   ├── logging/          # Structured logging backends
│   └── simd.zig          # Re-exported SIMD helpers
└── simd.zig              # Legacy SIMD entry point (re-exported in shared)```

## 🔧 Module Details

### Feature Modules (`features/`)
- **Purpose**: House capability-specific logic grouped by feature families.
- **Components**:
  - `ai/`: agents, transformers, reinforcement learning, data structures.
  - `database/`: WDBX vector database, sharding, HTTP façade.
  - `gpu/`: compute kernels, backend detection, memory pools, demos.
  - `web/`: HTTP client/server stacks, C bindings, weather demo.
  - `monitoring/`: metrics, tracing, regression analysis, Prometheus exports.
  - `connectors/`: OpenAI/Ollama bridges and plugin-facing adapters.
- **Dependencies**: Heavily reuse `shared/*` utilities and are orchestrated via
  `framework/`.

### Framework Runtime (`framework/`)
- **Purpose**: Central coordination layer used by `abi.init`/`abi.shutdown`.
- **Components**: Runtime state machine, feature discovery/catalogue,
  configuration parsing, lifecycle management.
- **Dependencies**: Consumes feature registries from `features/mod.zig` and core
  primitives from `shared/core` and `shared/logging`.

### Shared Libraries (`shared/`)
- **Purpose**: Foundation utilities and cross-cutting services shared by both
  framework and features.
- **Components**: Core lifecycle helpers, platform abstractions, logging
  backends, utility collections, SIMD helpers.
- **Dependencies**: Standalone where possible; some modules (e.g. logging)
  depend on `shared/core`.

### Legacy Entrypoints (`mod.zig`, `main.zig`, `root.zig`, `simd.zig`)
- **Purpose**: Provide compatibility layers for existing consumers while the new
  feature-first architecture settles.
- **Components**: Public API surface (`mod.zig`), CLI entry (`main.zig`), legacy
  exports (`root.zig`), SIMD convenience wrapper (`simd.zig`).
- **Dependencies**: Bridge between external callers and the framework/feature
  modules.
## 🔗 Dependencies

```
features/* ─┐
            ├─▶ framework/runtime ─▶ shared/core
shared/utils ─┘                     ├─▶ shared/logging
                                   ├─▶ shared/platform
                                   └─▶ shared/utils & shared/simd
```

- `shared/*` delivers the reusable building blocks consumed across the stack.
- `framework/` activates features based on `FrameworkOptions`, using the plugin system and registry to wire dependencies.
- `features/*` provide vertical capabilities and lean on shared utilities for storage, logging, SIMD, and platform access.
- Tests and examples depend on the same public exports, ensuring parity with consumer usage.

## 🏗️ Build Integration

- Feature families are grouped under `src/features/mod.zig` for easy re-exports.
- `src/mod.zig` exposes `abi.features` and `abi.framework` to callers.
- Shared libraries live under `src/shared/*` and are imported where needed.
- Build orchestration in `build.zig` pulls feature modules via the framework
  runtime.

## 📚 Usage

```zig
const abi = @import("abi");
const framework = try abi.init(allocator, .{ .enable_gpu = true });
defer framework.deinit();

// Opt into a specific feature namespace
const ai = abi.features.ai;
const agent = try ai.Agent.init(allocator, .adaptive);
```

// Opt-in feature modules are available under `abi.features.*`
var agent = try abi.features.ai.enhanced_agent.Agent.init(allocator, .{});
defer agent.deinit();

1. Inspect `src/features/mod.zig` for the list of available feature families.
2. Explore `src/framework/` for runtime orchestration and lifecycle flows.
3. Consult `src/shared/` for reusable utilities and platform abstractions.
4. Browse generated docs in `docs/generated/` for symbol-level details.

## 🎯 Benefits

- Feature-centric layout that mirrors the runtime configuration surface.
- Clear separation between orchestration (`framework/`) and shared utilities.
- Easier discoverability through a consistent namespace (`abi.features.*`).
- Explicit imports make it straightforward to reason about dependencies.
