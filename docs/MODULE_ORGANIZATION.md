# Abi Framework Module Organization

Structured overview of the feature-oriented layout that powers the Abi AI Framework.

## 📁 Module Architecture

```
src/
├── mod.zig              # Primary library surface
├── main.zig             # CLI/bootstrap entrypoint
├── framework/           # Runtime orchestration and feature management
│   ├── config.zig       # Feature toggles and public options
│   ├── runtime.zig      # Framework bootstrap and lifecycle
│   ├── feature_manager.zig  # Feature discovery and activation helpers
│   ├── catalog.zig      # Metadata exposed to tooling and docs
│   └── state.zig        # Long-lived runtime state container
├── features/            # Domain feature families surfaced to users
│   ├── ai/              # Agents, ML pipelines, training/inference helpers
│   ├── database/        # Vector storage, indexing, and query APIs
│   ├── web/             # HTTP servers, gateways, protocol adapters
│   ├── monitoring/      # Metrics, telemetry, and health probes
│   ├── gpu/             # GPU backends, kernels, and device orchestration
│   └── connectors/      # Third-party integrations and adapters
├── shared/              # Cross-cutting subsystems reused everywhere
│   ├── core/            # Configuration, lifecycle, and error types
│   ├── logging/         # Structured logging sinks and formatters
│   ├── platform/        # Host detection and capability probing
│   ├── utils/           # Common helpers (fs/http/json/crypto/math)
│   ├── simd.zig         # SIMD primitives exported to features
│   ├── types.zig        # Shared type aliases and option sets
│   ├── registry.zig     # Service registry used by the framework
│   └── enhanced_plugin_system.zig # Plugin runtime and hot-reload orchestration
├── examples/            # Runnable samples demonstrating the API
├── tests/               # In-tree unit and integration harnesses
└── tools/               # Developer tooling compiled as part of the build
```

## 🔧 Module Details

### Framework Layer (`framework/`)
- **Purpose**: Central orchestration layer that interprets `FrameworkOptions`, derives feature toggles, and coordinates lifecycle hooks.
- **Components**: `config.zig`, `runtime.zig`, `catalog.zig`, `feature_manager.zig`, `state.zig`.
- **Dependencies**: Relies on `shared/core`, `shared/utils`, logging, and the plugin system to manage feature wiring.

### Shared Layer (`shared/`)
- **Purpose**: Provides foundational services that every feature builds upon.
- **Components**:
  - `core/` for configuration management, lifecycle primitives, and error types.
  - `logging/` for structured logging and sinks.
  - `platform/` for OS/runtime detection.
  - `utils/` for reusable helpers (filesystem, HTTP, JSON, crypto, math, encoding, networking).
  - `simd.zig`, `types.zig`, and `registry.zig` for cross-feature data structures.
  - `enhanced_plugin_system.zig` for dynamic plugin discovery, hot reload, and lifecycle management.
- **Dependencies**: Minimal external dependencies; consumed by the framework and every feature module.

### Feature Families (`features/*`)
Each feature family is independent, exporting a `mod.zig` with public APIs and relying on shared utilities for cross-cutting concerns.

- **`features/ai/`**: Agents, neural network layers, reinforcement learning, model registries, and serialization.
- **`features/database/`**: Vector database kernels, sharding, persistence, and query planning.
- **`features/web/`**: HTTP servers, gateway orchestration, WebSocket support, and client adapters.
- **`features/monitoring/`**: Metrics pipelines, telemetry exporters, health checks, and diagnostics.
- **`features/gpu/`**: GPU backend detection, compute kernels, optimization passes, and testing utilities.
- **`features/connectors/`**: Integrations with third-party services, protocol bridges, and connector registries.

### Entry Points (`mod.zig` & `main.zig`)
- **Purpose**: Provide the public ABI surface (`src/mod.zig`) and the CLI runtime (`src/main.zig`) that consumers interact with.
- **Dependencies**: Delegate orchestration to `framework/runtime.zig` and consume shared utilities for bootstrapping.

### Tests & Examples (`src/tests/`, `src/examples/`)
- **Purpose**: House unit/integration harnesses mirroring the feature layout and runnable examples demonstrating best practices.
- **Dependencies**: Import feature modules through `src/mod.zig` and exercise shared infrastructure.

## 🔗 Dependencies

```
shared/* ─┬────────────┬──────────────┐
         │            │              │
         ▼            ▼              ▼
   framework/   features/*      src/tests/, src/examples/
         │            │              │
         └────┬───────┴─────┬────────┘
              ▼             ▼
     enhanced plugin   Framework runtime
     system & registry orchestrate feature lifecycles
```

- `shared/*` delivers the reusable building blocks consumed across the stack.
- `framework/` activates features based on `FrameworkOptions`, using the plugin system and registry to wire dependencies.
- `features/*` provide vertical capabilities and lean on shared utilities for storage, logging, SIMD, and platform access.
- Tests and examples depend on the same public exports, ensuring parity with consumer usage.

## 🏗️ Build Integration

- Module wiring lives in `src/mod.zig`, which re-exports the framework, shared utilities, and feature families.
- `build.zig` compiles the feature modules conditionally based on toggles exposed in `FrameworkOptions`.
- Tests under `src/tests/` and `tests/` mirror the feature hierarchy for clarity.
- Generated documentation references `features/*`, `shared/*`, and `framework/` to match the runtime layout.

## 📚 Usage

```zig
const abi = @import("abi");
const framework = try abi.init(allocator, .{ .enable_gpu = true, .enable_monitoring = true });
defer framework.deinit();

// Opt-in feature modules are available under `abi.features.*`
var agent = try abi.features.ai.enhanced_agent.Agent.init(allocator, .{});
defer agent.deinit();

// Shared utilities remain accessible through `abi.shared.*`
abi.shared.logging.global_logger.info("GPU feature online", .{});
```

## 🎯 Benefits

- Feature-based layout mirrors runtime toggles and clarifies ownership boundaries.
- Shared primitives live in a single place, reducing duplication and easing audits.
- Framework orchestration separates configuration from feature logic, simplifying testing.
- Documentation, examples, and tests align with the in-tree structure, improving discoverability.
