# ABI Framework TODO

This list captures remaining work needed to fully harden the framework for
production use. Items are grouped by subsystem and roughly ordered by
dependency priority (core/platform first, then shared services, then feature
modules).

## Core + Platform
- [ ] Replace hardcoded ANSI detection with a best-effort terminal capability
      check on Windows and non-interactive shells.
- [ ] Implement cross-platform thread priority changes (Windows, Linux, macOS)
      with graceful fallback when permissions are insufficient.
- [ ] Add platform-specific temp directory discovery (use env vars + OS APIs).
- [ ] Replace static CPU frequency/cache info with real OS queries where
      available; keep deterministic fallback for unsupported targets.

## Framework + Feature Wiring
- [ ] Ensure framework init/deinit consistently initializes feature modules
      based on build options (AI/GPU/Web/Database).
- [ ] Validate and document default feature toggles in build options.

## Connectors
- [x] Define `ProviderConfig` and shared connector helpers (requests, auth,
      response parsing, error mapping).
- [x] Implement OpenAI connector with configurable base URL and API key.
- [x] Implement Hugging Face inference connector with bearer token support.
- [x] Implement local scheduler connector with configurable endpoint and health
      check.
- [ ] Add connector unit tests for JSON parsing and error mapping.

## Web
- [x] Replace WDBX agent query placeholder with real connector integration
      (configurable provider + model).
- [x] Wire libcurl-backed HTTP client path when libcurl is available.
- [ ] Provide explicit error responses for malformed JSON in agent endpoints.
- [ ] Decide on Python bindings strategy: implement bindings or gate behind a
      build flag and remove from default `web` module exports.
- [x] Implement WebSocket handshake and message framing utilities in
      `enhanced_web_server.zig`.

## AI
- [ ] Replace placeholder model interface in `agent_subsystem.zig` with a
      concrete interface (or narrow scope to inference-only).
- [ ] Implement missing activation derivatives and add coverage tests.
- [ ] Finalize distributed training batch logic and parameter logging.
- [ ] Complete cache implementation (capacity handling + eviction policy).
- [ ] Finalize serializer metadata round-trip for all model types.

## GPU
- [ ] Implement real GPU detection paths (Vulkan/DX12/Metal/OpenCL) and remove
      synthetic fallback when actual detection is available.
- [ ] Replace GPU backend manager synthetic capability tables with actual
      device queries.
- [ ] Implement shader compilation paths (CUDA/Metal/DX12/OpenGL/WebGPU) or
      gate behind build options.
- [ ] Finish Vulkan and Mach GPU integrations (buffers, pipelines, dispatch).
- [ ] Provide real compute kernels where placeholders remain.

## Database
- [ ] Replace PQC placeholder crypto with verified implementations or remove
      PQC claims from the server module.
- [ ] Implement actual TCP listener for `DatabaseServer.start` and route
      dispatch.
- [ ] Replace CLI placeholder helpers with full config parsing + validation.

## Monitoring
- [ ] Implement Windows/macOS CPU, memory, disk, and process metrics (no fake
      values).
- [ ] Add process uptime based on platform APIs.
- [ ] Include network sampling when enabled.

## Scripts + Tooling
- [x] Replace `TBD` in performance optimization report with measured values or
      an explicit `n/a`.
- [ ] Add CI automation for lint + test + build targets.
