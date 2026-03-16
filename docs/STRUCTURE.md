# ABI Project Structure

Comprehensive directory tree reference for the ABI Zig framework. Covers every major
directory to two or three levels of depth, with annotations on purpose and key files.

---

## Root Directory

```
abi/
  build.zig              # Top-level build script (Zig 0.16)
  build.zig.zon          # Package manifest
  bootstrap_build.zig    # Bootstrap build helper
  bootstrap_root.zig     # Bootstrap root module
  abi.zon                # Package metadata
  .zigversion            # Pinned Zig version (0.16.0-dev.2905+5d71e3051)
  CLAUDE.md              # Agent instructions / project conventions
  AGENTS.md              # Workflow contract for agents
  GEMINI.md              # Gemini agent instructions
  README.md              # Project overview
  LICENSE                # License file
  SECURITY.md            # Security policy
  CODE_OF_CONDUCT.md     # Code of conduct
  swiftly.pkg            # Swift package descriptor
  zig-abi-plugin         # Zig ABI plugin helper
  src/                   # Main source tree
  build/                 # Build system modules
  tools/                 # CLI, gendocs, scripts, dev server
  tests/                 # Integration and wrapper tests
  benchmarks/            # Performance benchmarks (6 domains)
  bindings/              # C and WASM language bindings
  examples/              # Runnable example programs
  docs/                  # Documentation site and API docs
  tasks/                 # Planning files (todo.md, lessons.md)
```

---

## Source Layout (`src/`)

```
src/
  root.zig                        # Public package entrypoint (@import("abi"))
  abi.zig                         # Internal composition layer
  database_fast_tests_root.zig    # Fast-path database test root
  database_wdbx_tests_root.zig    # WDBX database test root
  features/                       # 19 comptime-gated feature modules
  core/                           # Framework internals
  services/                       # Runtime services and connectors
  inference/                      # AI inference engine
  api_server/                     # REST API server
```

### `src/features/` -- 19 Comptime-Gated Feature Modules

Each feature directory contains at minimum `mod.zig` (real implementation) and
`stub.zig` (disabled facade). Features with shared types also have `types.zig`.

| Feature        | `mod.zig` | `stub.zig` | `types.zig` | Notes |
|----------------|:---------:|:----------:|:-----------:|-------|
| `ai`           | Y | Y | Y | AI agents, LLM, embeddings, RAG, vision (36 subdirs) |
| `analytics`    | Y | Y | Y | Analytics and telemetry |
| `auth`         | Y | Y | Y | Authentication and authorization |
| `benchmarks`   | Y | Y | -- | Benchmark feature gate |
| `cache`        | Y | Y | Y | Caching layer |
| `cloud`        | Y | Y | Y | Cloud provider integrations |
| `compute`      | Y | Y | -- | Compute orchestration |
| `database`     | Y | Y | -- | Database feature gate |
| `desktop`      | Y | Y | -- | Desktop platform support |
| `documents`    | Y | Y | -- | Document processing |
| `gateway`      | Y | Y | Y | API gateway |
| `gpu`          | Y | Y | -- | GPU compute (Metal backend) |
| `messaging`    | Y | Y | Y | Message queues and pub/sub |
| `mobile`       | Y | Y | -- | Mobile platform support |
| `network`      | Y | Y | -- | Networking layer |
| `observability`| Y | Y | -- | Metrics, tracing, logging |
| `search`       | Y | Y | Y | Vector and full-text search |
| `storage`      | Y | Y | Y | Persistent storage |
| `web`          | Y | Y | -- | Web server and routing |

Feature flags are all enabled by default. Disable with `-Dfeat-<name>=false`.
GPU backend selection: `-Dgpu-backend=metal`.

### `src/core/` -- Framework Internals

```
src/core/
  mod.zig                # Core module entrypoint
  errors.zig             # Explicit error sets
  feature_catalog.zig    # Feature registry metadata
  comptime_meta.zig      # Comptime metaprogramming utilities
  stub_context.zig       # StubFeature / StubFeatureNoConfig helpers
  framework.zig          # Framework re-export
  database_fast_tests_root.zig  # Fast database test root
  config/                # Configuration loaders (15 files)
    mod.zig              #   Config module entrypoint
    loader.zig           #   Config file loader
    ai.zig, gpu.zig, cloud.zig, network.zig, ...
  database/              # Core database engine (18 subdirs, 60+ files)
    mod.zig              #   Database module entrypoint
    database.zig         #   Main database implementation
    engine.zig           #   Storage engine
    hnsw.zig             #   HNSW index
    simd.zig             #   SIMD-accelerated operations
    wdbx.zig             #   WDBX format support
    api/                 #   Database API layer
    block/               #   Block storage
    cli/                 #   Database CLI integration
    context/             #   Query context
    core/                #   Core data structures
    dist/                #   Distance functions
    distributed/         #   Distributed database support
    formats/             #   Data format handlers
    graph/               #   Graph storage
    index/               #   Index management
    memory/              #   Memory management
    persona/             #   Persona storage
    query/               #   Query processing
    ranking/             #   Result ranking
    semantic_store/      #   Semantic storage
    stubs/               #   Database stubs
    trace/               #   Query tracing
    vector/              #   Vector storage and operations
  framework/             # Framework lifecycle
    builder.zig          #   Framework builder pattern
    context_init.zig     #   Context initialization
    feature_imports.zig  #   Feature import resolution
    lifecycle.zig        #   Startup / shutdown lifecycle
    shutdown.zig         #   Graceful shutdown logic
    state_machine.zig    #   State machine
    state.zig            #   Framework state
  registry/              # Service registry
    mod.zig              #   Registry entrypoint
    registration.zig     #   Service registration
    lifecycle.zig        #   Registry lifecycle
    types.zig            #   Registry types
    stub.zig             #   Registry stub
```

### `src/services/` -- Runtime Services

```
src/services/
  connectors/            # Provider connectors (19 .zig files + 2 subdirs)
    mod.zig              #   Connector module entrypoint
    shared.zig           #   Shared connector utilities
    stub.zig             #   Connector stub
    anthropic.zig        #   Anthropic API
    claude.zig           #   Claude API
    codex.zig            #   Codex API
    cohere.zig           #   Cohere API
    gemini.zig           #   Google Gemini API
    huggingface.zig      #   Hugging Face API
    llama_cpp.zig        #   llama.cpp local inference
    lm_studio.zig        #   LM Studio API
    local_scheduler.zig  #   Local model scheduler
    mistral.zig          #   Mistral API
    mlx.zig              #   Apple MLX framework
    ollama.zig           #   Ollama API
    ollama_passthrough.zig # Ollama passthrough mode
    openai.zig           #   OpenAI API
    opencode.zig         #   OpenCode API
    vllm.zig             #   vLLM serving
    discord/             #   Discord bot connector (REST, types, utils)
    stubs/               #   Connector stubs (19 stub files)
  shared/                # Foundation / shared services module
    mod.zig              #   Shared module entrypoint
    errors.zig           #   Common error types
    io.zig               #   I/O utilities
    logging.zig          #   Logging framework
    matrix.zig           #   Matrix operations
    os.zig               #   OS abstraction
    plugins.zig          #   Plugin system
    signal.zig           #   Signal handling
    sync.zig             #   Synchronization primitives
    tensor.zig           #   Tensor operations
    time.zig             #   Time utilities
    utils.zig            #   General utilities
    app_paths.zig        #   Application path resolution
    stub_common.zig      #   Shared stub helpers
    resilience/          #   Resilience patterns (circuit breaker, retry)
    security/            #   Security services
    simd/                #   SIMD kernel implementations
    utils/               #   Extended utility library
  lsp/                   # Language Server Protocol service
    mod.zig, client.zig, jsonrpc.zig, types.zig
  mcp/                   # Model Context Protocol service
    mod.zig, server.zig, types.zig, zls_bridge.zig
  acp/                   # Agent Communication Protocol
    mod.zig
  ha/                    # High Availability
    mod.zig, backup.zig, pitr.zig, replication.zig, stub.zig
  platform/              # Platform detection
    mod.zig, cpu.zig, detection.zig
  runtime/               # Runtime engine
    mod.zig, workload.zig
    concurrency/         #   Concurrency primitives
    engine/              #   Execution engine
    memory/              #   Memory management
    scheduling/          #   Task scheduling
  tasks/                 # Task management
    mod.zig, types.zig, stub.zig
    lifecycle.zig, persistence.zig, querying.zig
    roadmap.zig, roadmap_catalog.zig
  tests/                 # Service-level tests (6 dirs)
    chaos/               #   Chaos testing
    e2e/                 #   End-to-end tests
    integration/         #   Integration tests
    parity/              #   Parity checks
    property/            #   Property-based tests
    stress/              #   Stress / load tests
```

### `src/inference/` -- AI Inference Engine

```
src/inference/
  engine.zig             # Inference engine core
  kv_cache.zig           # Key-value cache for transformer attention
  sampler.zig            # Token sampling strategies
  scheduler.zig          # Inference request scheduling
```

### `src/api_server/` -- REST API Server

```
src/api_server/
  server.zig             # HTTP server setup
  handlers.zig           # Request handlers
  auth.zig               # API authentication
  metrics.zig            # Request metrics
```

---

## Build System (`build/`)

```
build/
  options.zig            # 25 feature flags and build options
  flags.zig              # 42 flag combination validations
  module_catalog.zig     # Source-of-truth module registry
  modules.zig            # Module dependency wiring
  test_discovery.zig     # Automatic test file discovery
  link.zig               # Linker configuration
  targets.zig            # Cross-compilation target definitions
  gpu.zig                # GPU backend selection
  gpu_policy.zig         # GPU policy enforcement
  mobile.zig             # Mobile target support
  wasm.zig               # WASM target support
  cli_smoke_runner.zig   # CLI smoke test runner
  cli_tests.zig          # CLI test definitions
  cli_tui_test_runner.zig    # TUI test runner
  cli_tui_tests_root.zig    # TUI test root
  gendocs_tests_root.zig    # Gendocs test root
  validate/              # Validation scripts
    stub_surface_check.zig   # Verifies stub/mod signature parity
```

---

## Tools (`tools/`)

```
tools/
  cli/                   # Command-line interface
    main.zig             #   CLI entrypoint
    mod.zig              #   CLI module
    command.zig          #   Command definition
    spec.zig             #   Command specification DSL
    full_matrix_main.zig #   Full feature matrix runner
    launcher_tests_root.zig  # Launcher test root
    tui_tests_root.zig       # TUI test root
    commands/            #   Command implementations
      mod.zig            #     Command registry
      ai/                #     AI commands (agent, chat, embed, model, mcp, ...)
        llm/             #       LLM sub-commands (bench, chat, run, serve, ...)
        ralph/           #       Ralph agent commands (init, run, skills, ...)
        train/           #       Training commands
      core/              #     Core commands (config, init, plugins, profile, update)
        ui/              #       UI helpers
      db/                #     Database commands (db, explore)
      dev/               #     Dev commands (doctor, env, lsp, bench, status, ...)
        bench/           #       Benchmark sub-commands
      infra/             #     Infrastructure commands (gpu, network, simd, system_info)
    framework/           #   CLI framework layer
      mod.zig, router.zig, completion.zig, context.zig, errors.zig, help.zig, types.zig
    generated/           #   Auto-generated files
      cli_registry_snapshot.zig  # Registry snapshot (refresh with `zig build refresh-cli-registry`)
    registry/            #   Command registry
      overrides.zig      #     Registry overrides
    terminal/            #   TUI dashboard and panels
      mod.zig            #     Terminal module
      dashboard.zig      #     Main dashboard
      streaming_dashboard.zig  # Streaming dashboard
      agent_panel.zig, bench_panel.zig, brain_panel.zig, db_panel.zig, ...
      dsl/               #     TUI DSL
      editor/            #     In-terminal editor
      launcher/          #     TUI launcher
      panels/            #     Additional panel implementations
    tests/               #   CLI tests
    utils/               #   CLI utilities (args, output, process, ...)
  gendocs/               # Documentation generator
    main.zig             #   Gendocs entrypoint
    mod.zig              #   Gendocs module
    site_map.zig         #   Site structure definition
    site_builder.zig     #   Static site builder
    model.zig            #   Documentation model
    check.zig            #   Documentation validation
    source_abi.zig       #   ABI source parser
    source_baseline.zig  #   Baseline source parser
    source_cli.zig       #   CLI source parser
    source_features.zig  #   Feature source parser
    source_readme.zig    #   README source parser
    source_roadmap.zig   #   Roadmap source parser
    render_api_app.zig   #   API app renderer
    render_api_md.zig    #   API markdown renderer
    render_guides_md.zig #   Guide markdown renderer
    render_plans_md.zig  #   Plan markdown renderer
    templates/           #   HTML/Markdown templates
    assets/              #   Static assets
    wasm/                #   WASM documentation engine
  scripts/               # Shell scripts and Zig check scripts
    run_build.sh         #   Darwin-safe build wrapper (required on macOS 25+)
    fmt_repo.sh          #   Repository formatter
    use_zvm_master.sh    #   Zig version manager helper
    zig_darwin26_wrapper.sh  # Darwin linker workaround
    baseline.zig         #   Baseline generation
    generate_cli_registry.zig  # CLI registry generator
    toolchain_doctor.zig     # Toolchain diagnostics
    util.zig                 # Script utilities
    check_cli_dsl_consistency.zig       # CLI DSL consistency check
    check_cli_ui_layers.zig             # CLI UI layer check
    check_feature_catalog.zig           # Feature catalog validation
    check_gpu_policy_consistency.zig    # GPU policy consistency check
    check_import_rules.zig              # Import rule enforcement
    check_perf.zig                      # Performance check
    check_ralph_gate.zig                # Ralph gate validation
    check_test_baseline_consistency.zig # Test baseline consistency
    check_workflow_orchestration.zig    # Workflow orchestration check
    check_zig_016_patterns.zig          # Zig 0.16 pattern validation
    check_zig_version_consistency.zig   # Zig version consistency check
    emergency_bootstrap/                # Emergency bootstrap scripts
  server/                # Development server
    main.zig             #   Dev server entrypoint
```

---

## Testing

### `tests/` -- Integration and Wrapper Tests

```
tests/
  zig/                             # Zig test wrappers
    mod.zig                        #   Module entrypoint
    database_fast_tests_root.zig   #   Fast database test root
  distributed_integration.zig      # Distributed system integration tests
  hnsw_test.zig                    # HNSW algorithm tests
  integration_test.zig             # General integration tests
  personas_test.zig                # Persona system tests
  simd_test.zig                    # SIMD correctness tests
```

### `src/services/tests/` -- Service-Level Tests (6 Suites)

| Directory      | Purpose |
|----------------|---------|
| `chaos/`       | Chaos engineering: fault injection, random failures |
| `e2e/`         | End-to-end service tests |
| `integration/` | Cross-service integration tests |
| `parity/`      | Output parity between implementations |
| `property/`    | Property-based / generative tests |
| `stress/`      | Load and stress tests |

### Test Commands

```bash
zig build test --summary all            # Primary unit tests
zig build feature-tests --summary all   # Feature coverage tests
zig build full-check                    # Pre-commit gate (all checks)
zig build validate-flags                # Flag combination validation
```

---

## Benchmarks (`benchmarks/`)

```
benchmarks/
  main.zig               # Benchmark entrypoint
  mod.zig                # Benchmark module
  run.zig                # Benchmark runner
  run_competitive.zig    # Competitive benchmark runner
  baselines/             # Baseline measurements
    main/, branches/, releases/, README.md
  competitive/           # Competitive comparisons
    faiss_comparison.zig, llm_comparison.zig, vector_db_comparison.zig
  core/                  # Core algorithm benchmarks
    distance.zig, vectors.zig, config.zig
  domain/                # Domain-specific benchmarks
    ai/, database/, gpu/, services/
  infrastructure/        # Infrastructure benchmarks
    concurrency.zig, crypto.zig, gpu_backends.zig, memory.zig, simd.zig, network/
  system/                # System-level benchmarks
    baseline_comparator.zig, baseline_store.zig, framework.zig
```

---

## Bindings (`bindings/`)

```
bindings/
  c/                     # C language bindings
    include/
      abi.h              #   Public C header
    src/
      abi_c.zig          #   C binding implementation
  wasm/                  # WebAssembly bindings
    abi_wasm.zig         #   WASM binding implementation
```

---

## Examples (`examples/`)

32 runnable example programs covering major features:

| File | Topic |
|------|-------|
| `hello.zig` | Minimal hello-world |
| `config.zig` | Configuration loading |
| `auth.zig` | Authentication |
| `cache.zig` | Caching |
| `database.zig` | Database operations |
| `search.zig` | Vector / full-text search |
| `embeddings.zig` | Embedding generation |
| `ai_suite.zig` | AI feature suite |
| `llm_real.zig` | Real LLM inference |
| `gpu.zig`, `gpu_training.zig` | GPU compute and training |
| `tensor_ops.zig` | Tensor operations |
| `streaming.zig` | Streaming pipelines |
| `concurrency.zig`, `concurrent_pipeline.zig` | Concurrency patterns |
| `network.zig` | Networking |
| `web.zig`, `web_observability.zig`, `pages.zig` | Web serving |
| `cloud.zig` | Cloud integrations |
| `gateway.zig` | API gateway |
| `compute.zig` | Compute orchestration |
| `storage.zig` | Storage |
| `messaging.zig` | Messaging |
| `analytics.zig` | Analytics |
| `observability.zig` | Observability |
| `mobile.zig` | Mobile platform |
| `discord.zig` | Discord bot |
| `distributed_db.zig` | Distributed database |
| `ha.zig` | High availability |
| `registry.zig` | Service registry |
| `train_ava.zig` | Model training |
| `training/` | Training examples directory |

---

## Documentation (`docs/`)

```
docs/
  index.html             # Documentation site entrypoint
  index.css              # Site stylesheet
  index.js               # Site JavaScript
  README.md              # Docs readme
  STRUCTURE.md           # This file
  ABI_WDBX_ARCHITECTURE.md  # WDBX database architecture
  ZIG_MACOS_LINKER_RESEARCH.md  # macOS linker research notes
  api/                   # Auto-generated API reference (21 pages)
    index.md, v1.md, config.md, errors.md, ...
  data/                  # Data files for documentation engine
    commands.zon, features.zon, guides.zon, modules.zon, plans.zon, roadmap.zon
    docs_engine.wasm     # WASM documentation engine
    docs_engine.wit      # WIT interface definition
```

---

## Key Patterns

### mod/stub Contract

Every feature in `src/features/<name>/` has a `mod.zig` (real implementation) and
a `stub.zig` (disabled facade). The stub must match the public signatures of mod so
that code compiles regardless of which feature flags are enabled.

- Shared types go in `types.zig` -- both `mod.zig` and `stub.zig` import from it.
- Use `StubFeature` or `StubFeatureNoConfig` from `src/core/stub_context.zig` for
  common stub boilerplate.
- Sub-module stubs are not needed; only the top-level feature needs a stub.
- The build system validates stub/mod surface parity via `build/validate/stub_surface_check.zig`.

### Feature Flags

All 25 feature flags (defined in `build/options.zig`) are enabled by default.
Disable individual features at build time:

```bash
zig build -Dfeat-ai=false -Dfeat-gpu=false
```

42 flag combinations are validated in `build/flags.zig`.

### Imports

- Use `@import("abi")` for the framework API.
- Use relative imports within a feature module.
- All `src/` files belong to the single `abi` module (no `shared_services` or `core`
  named modules).
- Explicit `.zig` extensions are required on all path imports.

### Zig 0.16 Conventions

See `CLAUDE.md` for the full list. Key ones affecting file structure:

- `ArrayListUnmanaged` initializer: `.empty` (not `.{}`).
- `root_module` field (not `root_source_file`) in build scripts.
- Explicit `.zig` extensions required on all `@import` path strings.
- Single-module file ownership: every file belongs to exactly one named module.

### Build on macOS (Darwin 25+)

`zig build` fails with linker errors on Darwin 25+. Use the wrapper script:

```bash
./tools/scripts/run_build.sh <args>
```

Format checks (`zig fmt`) work without linking and do not need the wrapper.
