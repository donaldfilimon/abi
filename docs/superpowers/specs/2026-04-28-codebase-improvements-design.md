# Design Spec: Codebase Stability & Core Logic Improvements (2026-04-28)

## 1. Objective
Improve the ABI framework codebase by addressing critical technical debt, completing missing feature logic, and establishing a clean test baseline.

## 2. Proposed Changes

### 2.1. Fix ABA Problem in `LockFreeStack`
The current `LockFreeStack` in `src/runtime/concurrency/lockfree.zig` is susceptible to the ABA problem. We will implement **Tagged Pointers** to ensure atomic safety.

- **Implementation**:
    - Define a `TaggedPointer` struct that fits within 128 bits:
      ```zig
      const TaggedPointer = struct {
          ptr: ?*Node,
          tag: usize,
      };
      ```
    - Use `std.atomic.Value(TaggedPointer)` which leverages `cmpxchg16b` on x86_64 or `casp` on AArch64.
    - If the target does not support 128-bit atomics, fall back to a 64-bit packed representation (pointer address bits + generation bits) to ensure portability.
- **Files Affected**: `src/runtime/concurrency/lockfree.zig`

### 2.2. Complete OpenAI Streaming Handler
The `handleOpenAIChatCompletions` function in `src/features/ai/streaming/server/openai.zig` is currently a stub with a `TODO`.

- **Logic**:
    1. Parse the request body into `formats.openai.ChatCompletionRequest`.
    2. Check the `stream` boolean field.
    3. If `stream == true`, call `streamOpenAIResponse`.
    4. If `stream == false`, call `nonStreamingOpenAIResponse`.
    5. Handle parsing errors by returning appropriate HTTP 400 responses.
- **Files Affected**: `src/features/ai/streaming/server/openai.zig`

### 2.3. Resolve Pre-existing Test Failures
Address the 3 known failures mentioned in `GEMINI.md` to achieve a "green" build state.

- **Inference Connectors (2 failures)**:
    - Investigate `src/connectors/` tests.
    - Likely causes: Mock response timing or missing configuration in test environments.
- **Auth Integration (1 failure)**:
    - Investigate `auth_mod_test.zig` or `src/features/auth/`.
    - Likely cause: Default JWT secret mismatch or expired tokens in static tests.
- **Goal**: All tests passing on macOS 26.4+ using `./build.sh test`.

## 3. Data Flow & Architecture
- **Concurrency**: `LockFreeStack` is used by the task scheduler and memory pools. Stability here is critical for the entire runtime.
- **AI Streaming**: Follows the existing SSE (Server-Sent Events) pattern used by the custom ABI endpoint.

## 4. Verification Plan

### 4.1. Concurrency
- Add a high-contention stress test for `LockFreeStack` that specifically attempts to trigger ABA conditions (e.g., rapid push/pop with node reuse).
- Run with `zig build test` across different optimization levels (`-Doptimize=Debug`, `-Doptimize=ReleaseSafe`).

### 4.2. AI Streaming
- Use `curl` or a test script to hit the `/v1/chat/completions` endpoint.
- Verify both streaming and non-streaming responses match the OpenAI API specification.

### 4.3. Baseline
- Run `./build.sh test --summary all` (macOS) or `zig build test` (Linux) and verify 0 failures.
- Run `zig build check-parity` to ensure no API drift.
