# Inference Engine Connector Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executors-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate real LLM connectors into the ABI inference engine, replacing the current echo-mode fallback.

**Architecture:** Implement robust connector handling within `src/inference/engine/backends.zig`. Each connector (OpenAI, Anthropic, etc.) will be wired to its respective client implementation, using standard env-var configurations.

**Tech Stack:** Zig 0.17, ABI foundation (async HTTP client).

---

### Task 1: Connector Wiring Analysis & Preparation

**Files:**
- Modify: `src/inference/engine/backends.zig`
- Verify: `src/connectors/loaders.zig`

- [ ] **Step 1: Identify required connector-loader signatures**

Review `src/connectors/loaders.zig` and verify each supported provider loader (e.g., `tryLoadOpenAI`, `tryLoadAnthropic`) matches the expected `fn(allocator) !?Config` pattern.

- [ ] **Step 2: Update `generateConnector` to log actual provider load errors**

Modify `src/inference/engine/backends.zig` to log specific errors returned by `loaders` instead of generic `ApiRequestFailed`.

- [ ] **Step 3: Commit**

```bash
git add src/inference/engine/backends.zig
git commit -m "chore: enhance connector loading diagnostics"
```

### Task 2: Implement Real Connector Dispatch

**Files:**
- Modify: `src/inference/engine/backends.zig`

- [ ] **Step 1: Remove echo fallback from `generateConnector`**

Replace `std.log.warn(...)` with actual configuration loading and dispatch logic. Ensure `dispatchToConnector` uses the correctly loaded `config` to make actual API calls.

- [ ] **Step 2: Implement structured error handling for network requests**

Update `callOpenAICompatible` and `callAnthropicNative` to handle network errors specifically (e.g., DNS, connection timeouts) and return proper error types.

- [ ] **Step 3: Verify integration with a real connector (if configured)**

Ensure the engine is correctly using env vars (e.g., `OPENAI_API_KEY`) for authentication if they are present.

- [ ] **Step 4: Commit**

```bash
git add src/inference/engine/backends.zig
git commit -m "feat: wire real connector dispatch"
```

### Task 3: Regression Testing and Validation

**Files:**
- Verify: `src/inference/engine.zig`
- Verify: `test/integration/cognitive_pipeline_test.zig`

- [ ] **Step 1: Run integration tests**

Run: `./build.sh test -- --test-filter "cognitive pipeline"`
Expected: PASS (with real network or mocked connectors)

- [ ] **Step 2: Verify `check-parity`**

Run: `./build.sh check-parity`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git commit -am "test: validate connector integration parity"
```
