# Modernization Plan: The Four Honest-Stub Features

**Scope**: `src/features/accelerator/`, `src/features/shaders/`, `src/features/mlir/`,
`src/features/mobile/`.

**Type**: Planning only. No Zig code changes in this pass.

**Skill applied**: `refactor-strategy` (`.claude/skills/refactor-strategy/SKILL.md`),
Recommended Process steps 1–6.

**Hard constraint carried through this whole plan**: none of the four milestones below
may ship a change that flips `available=true` / `native_dispatch=true` (or removes a
"not linked" message) without an actual linked external toolchain proven by a
`build.zig` link step + a runtime dispatch code path + a passing test that exercises it.
`docs/contracts/external-claims-audit.mdx` explicitly forbids claiming CUDA/Vulkan/ANE
native dispatch, real shader compilation, real MLIR/LLVM lowering, or native mobile
runtime dispatch. Any milestone that only reformats messages or refactors internals
while the underlying capability is still absent is **not** "closing the gap" — it is
housekeeping, and must be labeled as such.

---

## 1. Current behavior and success criteria

### 1.1 Shared contract shape (applies to all four)

Every feature ships a `mod.zig` (real behavior) and a `stub.zig` (feature-off
behavior), selected in `src/features/mod.zig` by a `build_options.feat_*` bool. Both
must expose the identical **public declaration set** — this is enforced two ways:

- `zig build check-parity` — scans column-0 `pub const` / `pub fn` names and diffs
  `mod.zig` vs `stub.zig`. It does not see struct methods, `pub var`, or nested decls.
- `tests/contracts/feature_modules.zig` — a hand-written behavioral contract
  (`test "feature namespaces are stable across flags"` + `test "feature modules expose
  safe runtime contracts"` + the `if (!build_options.feat_X)` blocks starting around
  line 287) that calls into each feature through `abi.features.*` and asserts specific
  field values, not just that the symbol exists.

Both mod and stub already report the honest state today (`available=false` /
`native_dispatch=false` / `accelerated=false` as applicable), and the contract tests
already assert that dishonesty (e.g. `!accelerator_report.native_dispatch`,
`!shader_status.available`) — so the current success criteria for "don't regress
honesty" are already codified as tests, not just prose. Any future work must keep
these specific assertions green:

- `feature_modules.zig:78` `!accelerator_report.native_dispatch`
- `feature_modules.zig:85-86` `shaders.compilerStatus().message.len > 0` / `!available`
- `feature_modules.zig:95-96` `mlir.toolchainStatus().message.len > 0` / `!available`
- `feature_modules.zig:120` `!mobile_runtime.native_dispatch`
- `accelerator/mod.zig` own test `"accelerator never claims native dispatch in this
  module"` (loops all `Workload` values)
- `shaders/mod.zig` / `mlir/mod.zig` own tests asserting `"not linked"` appears in the
  status message

### 1.2 Per-feature current state (verified by reading source, not todo.md)

**`accelerator`** (`src/features/accelerator/mod.zig`, 204 lines):
- `Backend` enum (cpu, gpu_simulated, gpu_metal/vulkan/cuda/webgpu/opengl/webgl2, mlir),
  `Workload` enum (inference, training, shader_compile, graph_lowering).
- `selectionReport(workload)` delegates to `gpu.detectBackend()` +
  `gpu.nativeKernelStatus()`. If Metal (or another) GPU backend is both `available`
  and `accelerated`, it reports that backend as `selected_backend` but **always**
  hardcodes `native_dispatch = false` — even when `native_kernels.linked` is true —
  because "native accelerator dispatch" here means the accelerator module's own
  dispatch layer, distinct from `gpu.vectorOps()`'s Metal kernels. This is a
  deliberate, currently-correct distinction: the accelerator module itself does not
  call into Metal; `gpu.vectorOps()` does that independently.
- No native CUDA/NPU/TPU path exists anywhere in this file — confirmed by reading the
  whole file. The `gpu_cuda`, `gpu_vulkan`, etc. enum variants are declared but
  `gpu.detectBackend()` (in `src/features/gpu/mod.zig`, not read line-by-line here but
  referenced) can only realistically resolve to `.metal` or `.simulated` on this
  codebase's actual linked frameworks (macOS `Metal.framework` only — see
  `build.zig:126-130`).
- Depends on `foundation/validation.zig`? **No** — accelerator has no user-string
  inputs to validate (only enum-typed `Workload`).

**`shaders`** (`src/features/shaders/mod.zig`, 167 lines):
- `Language` enum (zig_kernel, wgsl, msl, spirv_text). `validateDetailed` does
  structural checks only: non-empty name/source, no null bytes, balanced
  `{}/()/[]` delimiters, and a per-language regex-free "entry point" substring probe
  (`"fn main"`, `"kernel"`, `"OpEntryPoint"`). No parsing, no AST, no semantic
  validation, no real compilation.
- `compile()` produces a descriptor string (`shader=...;language=...;backend=...`)
  wrapping the validation report — not actual compiled bytecode/SPIR-V/AIR.
- `compilerStatus().backend = "validated-local"`, `available = false`, explicit "not
  linked" message.
- Depends on `foundation/validation.zig` for the non-empty/no-null-byte checks
  (consistent with `mlir`, unlike `accelerator`/`mobile`).

**`mlir`** (`src/features/mlir/mod.zig`, 165 lines):
- `Dialect` enum (affine, linalg, tensor, gpu). `analyze()` validates the module name
  is a valid symbol (`[a-zA-Z0-9_.-]+`), each operation string is non-empty/no-null,
  and computes a Wyhash checksum over name+dialect+ops.
- `lower()` emits a **textual, hand-rolled** IR-like string
  (`module @name attributes {...} { abi.op @dialect_0 ... }`) — this is ABI's own
  invented pseudo-MLIR syntax, not real MLIR generic/textual format, and it is never
  fed to `mlir-opt`, `mlir-translate`, or any LLVM tool.
- `toolchainStatus().backend = "textual-local"`, `available = false`, explicit "not
  linked" message.
- Depends on `foundation/validation.zig`.

**`mobile`** (`src/features/mobile/mod.zig`, 295 lines — the largest of the four):
- `Platform` enum (ios, android, unknown). `detectPlatform()` uses
  `builtin.target.os.tag` at **comptime** (compile-target based, not runtime device
  probing) to report iOS/Android as "detected" with `available=true` but
  `accelerated=false` and message "native dispatch pending" — i.e. even a real
  iOS/Android cross-compile target only gets a profile report, no runtime dispatch.
  Desktop targets get `unknown` + a simulated profile, with `accelerated` borrowed
  from `gpu.detectBackend().accelerated` (Metal on macOS) — a slightly confusing
  reuse of the GPU accelerated flag for a "mobile" accelerated field, worth flagging
  in gap analysis.
- `profile()` / `deviceProfile()` / `layoutSummary()` / `renderMobileView()` /
  `executeMobileTask()` — a self-contained mock mobile view-rendering pipeline
  (viewport dims per platform, a text "rendered view", a fake "task execution"
  string). All fabricated locally; no UIKit/Android SDK/mobile runtime linkage
  anywhere.
- Has its own private `validateLabel()` doing an inline non-empty/no-null-byte check
  — this duplicates the same two checks `foundation/validation.zig` already
  centralizes for `shaders`/`mlir`. Minor inconsistency, not a honesty problem.
- `native_dispatch` is hardcoded `false` unconditionally in `profile()` regardless of
  platform — accurate today since there is no mobile runtime link at all.

### 1.3 Claims-audit anchoring

`docs/contracts/external-claims-audit.mdx` line 25 explicitly covers shaders+mlir
("Do not claim real shader compilation or MLIR/LLVM lowering") and line 24 covers GPU
("Do not claim general GPU speedup, CUDA/Vulkan dispatch, or ANE execution" — this is
the umbrella `accelerator`'s CUDA/Vulkan enum variants fall under, since accelerator
itself only ever selects them in principle, never dispatches). `mobile` is not named
explicitly in the audit table but is covered by the same "no unproven native
runtime/hardware dispatch" spirit and by its own honest `native_dispatch=false` in
source — this plan does not propose adding a new audit-table row unless a milestone
below actually changes claimable behavior (none currently do).

---

## 2. Ideal modern design (what "real" would require, per feature)

This section sketches what a genuinely non-stub implementation would need — strictly
as a design sketch to size the gap, not a commitment to build it.

### 2.1 `accelerator` — real CUDA/NPU/TPU dispatch

To make `native_dispatch=true` honest for any workload, the module would need:
- A real linked native backend it can call — e.g. an actual CUDA toolkit (`nvcc`
  + `cudart`) or vendor NPU/TPU SDK, wired via `build.zig` `linkSystemLibrary`/
  `addLibraryPath` gated behind a **new** build option (e.g. `-Dfeat-cuda=false`
  default, since CUDA hardware/toolkits are not available on this project's macOS-first
  CI and dev hosts).
- A thin `extern "c" fn` FFI boundary in a new `accelerator/cuda_ffi.zig` (or similar),
  compiled only when the new flag + target triple support it — the same pattern
  `src/features/gpu/` already uses for Metal via pure-Zig objc FFI
  (`build.zig:126-130`, `linkFramework("Metal")` + `linkSystemLibrary("objc")`).
- Dispatch functions that actually submit kernels/buffers and block/poll for
  completion, replacing today's report-only `selectionReport`.
- The `SelectionReport.native_dispatch` field would flip to `true` **only** on hosts
  where the new flag is on and the runtime probe (analogous to
  `gpu.nativeKernelStatus()`) confirms the toolkit initialized — preserving the
  existing honest-degrade-to-report-only behavior on every other host.
- This is the single largest lift of the four: it needs hardware (an NVIDIA GPU) or a
  cloud CI runner with one, a CUDA toolkit install, and ongoing maintenance of a
  second native backend alongside Metal. NPU/TPU dispatch would each be a similar,
  separate lift with different vendor SDKs (unlike Metal/CUDA which at least share
  "GPU compute kernel" shape).

### 2.2 `shaders` — real shader compilation

To make `available=true` honest:
- Link an actual compiler toolchain per target shading language: e.g. `metal`/
  `metallib` CLI (Xcode-bundled on macOS, already partially available since the repo
  builds Metal-linked binaries) for MSL → AIR/metallib, `naga`/`tint`/`dxc` for WGSL →
  SPIR-V, or a SPIR-V assembler for `spirv_text`.
- New `build.zig` step(s) invoking the compiler as a build-time tool (similar to the
  existing `run_gen_plugin_registry` custom `Run` step pattern) or a runtime
  `std.process.Child.exec` shellout to a discovered toolchain path, gated by a new
  flag (e.g. `-Dfeat-shader-compiler=false` default) since these toolchains are
  optional external dependencies the framework should not silently require.
  MSL→metallib is the cheapest of the three (Xcode already present on macOS dev/CI
  hosts per `feat-foundationmodels`'s existing Xcode dependency), WGSL and SPIR-V both
  require adding brand-new third-party toolchain dependencies.
- `ShaderArtifact.bytes` would hold real compiled bytecode instead of a descriptor
  string; `compilerStatus().available` would flip `true` only when the discovered
  compiler binary responds to a version probe at runtime.

### 2.3 `mlir` — real MLIR/LLVM lowering

To make `available=true` honest:
- Vendor or require a system install of LLVM/MLIR (`mlir-opt`, `mlir-translate`),
  which is a large, version-sensitive C++ toolchain with no existing Zig build
  integration in this repo (no LLVM linkage exists anywhere in `build.zig` today).
- Either (a) shell out to `mlir-opt`/`mlir-translate` binaries at runtime (subprocess
  FFI boundary, easiest to gate/disable), or (b) link `libMLIR`/`libLLVM` C API
  bindings via `extern "c" fn` — a much larger and more fragile lift given MLIR's C++
  ABI and lack of a stable C API for most dialects.
- `lower()` would need to emit real MLIR generic/textual syntax (not the current
  invented `abi.op`/`abi.dialect` pseudo-syntax) matching an actual MLIR dialect
  grammar, and `toolchainStatus().backend` would flip from `"textual-local"` to
  something like `"mlir-opt"` only when the subprocess probe succeeds.
- This is the second-largest lift after CUDA: MLIR/LLVM toolchains are large,
  frequently version-pinned, and this repo currently has zero LLVM/MLIR build
  infrastructure to build on.

### 2.4 `mobile` — real iOS/Android runtime dispatch

To make `native_dispatch=true` honest:
- iOS: would require building against UIKit/SwiftUI via an Objective-C/Swift shim
  (the repo already has exactly one precedent for this pattern —
  `feat-foundationmodels`'s Swift shim dylib, `build.zig:133-140`, `-labi_fm_shim`)
  cross-compiled for an actual iOS target triple, run in a simulator or on-device via
  Xcode's iOS toolchain — none of which exists today for `mobile`.
- Android: would require an NDK toolchain, JNI shim, and either an Android emulator
  or device — a full second cross-platform runtime stack, unrelated to the
  Swift/ObjC pattern above.
- Both require actual mobile app packaging (`.app`/`.ipa` or `.apk`), a UI event
  loop, and on-device test execution — this is not a library link, it is a second
  application target with its own build/test/deploy pipeline.
- `RuntimeMode.native_platform` + `native_dispatch=true` would only be honest once a
  real UI renders and a real platform API call succeeds on a real device/simulator,
  which is a materially different kind of artifact than everything else this repo
  produces (a CLI + library, not packaged mobile apps).

### 2.5 What stays identical regardless of "real vs stub" (the interface contract)

Across all four, the **public shape** an eventual real implementation must preserve
so mod/stub parity and the existing contract tests keep working:
- Same enum sets (`Backend`/`Workload`/`Language`/`Dialect`/`Platform`/`RuntimeMode`).
- Same status-report struct shapes (`SelectionReport`/`CompilerStatus`/
  `ToolchainStatus`/`PlatformStatus`/`MobileProfile`/`DeviceProfile`) with the same
  field names — `available`/`native_dispatch`/`accelerated`/`message` are the honesty
  fields every consumer (CLI `backends` handler, contract tests) reads directly.
- Same function signatures (`selectionReport`, `validateDetailed`, `analyze`,
  `detectPlatform`, etc.) — only their *internal* implementation would change to call
  a real backend instead of returning a synthetic report.
- The stub side (`stub.zig`) never needs to change at all for any of these
  milestones — a real backend link only ever changes `mod.zig`; `stub.zig` remains
  the permanently-honest "feature is disabled" path.

---

## 3. Gap analysis (current vs. ideal), per feature

| Feature | Current | Ideal (real) | Gap size | Primary blocker |
|---|---|---|---|---|
| `accelerator` | Report-only backend selection over existing `gpu.detectBackend()` | Real CUDA/NPU/TPU kernel dispatch with FFI + build linkage | **Very large** | No GPU hardware/toolkit beyond Metal in this project's environment; multiple vendor SDKs, each a separate lift |
| `shaders` | Structural source validation + checksum descriptor | Real MSL/WGSL/SPIR-V compilation to bytecode | **Large** (MSL sub-lift is medium; WGSL/SPIR-V large) | External compiler toolchain decision + new build.zig integration per language |
| `mlir` | Textual invented pseudo-IR + symbol/op validation | Real MLIR/LLVM textual lowering via `mlir-opt`/`libMLIR` | **Very large** | No existing LLVM/MLIR build infra in repo; large, fragile, version-sensitive toolchain |
| `mobile` | Comptime target-tag detection + mock view-rendering profile | Real iOS/Android UI runtime with native API dispatch | **Very large, different in kind** | Requires packaged mobile app targets + device/emulator test pipeline, not a library link |

Secondary (non-honesty) gaps worth noting, all small and low-risk if ever picked up:
- `mobile/mod.zig`'s private `validateLabel` duplicates `foundation/validation.zig`'s
  non-empty/no-null-byte checks that `shaders`/`mlir` already use — a small
  consistency cleanup, not a capability change.
- `mobile`'s `accelerated` field borrows `gpu.detectBackend().accelerated` (a Metal
  GPU-acceleration flag) to describe "mobile acceleration," which is a conceptual
  mismatch (desktop Metal acceleration is not mobile acceleration) — worth a comment
  or field rename in a future low-risk pass, not a behavior change.
- `accelerator`'s `Backend` enum declares `gpu_vulkan`/`gpu_cuda`/`gpu_webgpu`/
  `gpu_opengl`/`gpu_webgl2` variants that can never actually be selected today (only
  `.metal`/`.simulated` are reachable via `gpu.detectBackend()` on this codebase's
  linked frameworks) — harmless dead enum space reserved for a future GPU backend,
  not a lie since nothing claims these are selectable now.

---

## 4. Strategy per feature

Using the Strategy Selection Matrix (`references/strategy-guide.md`): all four are
"legacy with poor tests" is **not** the right bucket — tests are actually good
(current behavior is fully pinned by contract + own-module tests). The real
discriminator here is **external dependency risk and disclosed non-goal status**,
not code quality.

| Feature | Strategy | Rationale |
|---|---|---|
| `accelerator` | **Non-goal — do not pursue toolchain linkage.** If ever revisited, phased/strangler-fig (new flag, new FFI module beside existing `mod.zig`, dispatch flips only under flag+probe). | CUDA/NPU/TPU hardware and vendor SDKs are outside this project's environment and stated non-goals (ANE is an explicit non-goal per `AGENTS.md`; CUDA/Vulkan carry the same "not linked" claims-audit language). No safe direct-rewrite path exists because there is nothing to link against. |
| `shaders` | **Conditional / phased, MSL-only, and only with an explicit decision to add an external dependency.** Not a "small direct rewrite" — it requires a toolchain adoption decision first. | MSL compilation via Xcode's bundled `metal`/`metallib` CLI is the cheapest of the three shading languages (toolchain already present on macOS dev/CI hosts) but still requires a new build step, a new flag, and new claims-audit wording. WGSL/SPIR-V stay non-goals unless a third-party dependency is explicitly approved. |
| `mlir` | **Non-goal for now.** If ever revisited: phased, subprocess-shellout first (cheaper/safer than linking `libMLIR`), never linking `libLLVM` C++ ABI directly in a small-risk pass. | No existing LLVM/MLIR build infra; adopting it is a major, version-fragile dependency with no current product requirement pulling it in (nothing in the CLI/MCP frozen surface needs real MLIR lowering). |
| `mobile` | **Non-goal.** Explicitly out of scope — real dispatch requires a second application target (packaged iOS/Android app) and device/emulator CI, categorically different from this repo's CLI+library shape. | Matches the existing ANE non-goal precedent: detection/profile-reporting only, by design, not a gap to close. |

None of the four qualify as "small/low-risk direct rewrite" in the sense of flipping
`available`/`native_dispatch` to `true` — every one requires an external toolchain or
hardware dependency this project does not currently carry. The *only* small/low-risk
direct work available across all four is the housekeeping items in §3 (validation
delegation, field-naming clarity) and doc/message polish — none of which touch the
honesty contract.

---

## 5. Validation criteria per phase (if any milestone is ever picked up)

Applies uniformly to any future work on these four modules:

1. **Before coding**: re-run `./build.sh check` to confirm a clean baseline.
2. **Contract preservation**: `zig build test-contracts` (or targeted
   `zig build test -Dtest-filter="feature modules"`) must keep passing unchanged
   unless the milestone explicitly and honestly changes a reported field — in which
   case the test itself must be updated in the same change, not left stale.
3. **Parity**: `zig build check-parity` after any public declaration added/removed in
   `mod.zig`, with the matching `stub.zig` update landed in the same commit.
4. **Feature-off proof**: build and test with the flag forced off
   (`-Dfeat-accelerator=false` / `-Dfeat-shader=false` / `-Dfeat-mlir=false` /
   `-Dfeat-mobile=false`) to confirm `stub.zig` still compiles and its own
   `test "... stub ..."` still passes untouched.
5. **Claims audit**: any milestone that changes a reported `available`/
   `native_dispatch`/`accelerated` value must have a paired update to
   `docs/contracts/external-claims-audit.mdx` (new row or edited row) in the *same*
   change, proven by pointing at the specific linked framework/toolchain + the
   runtime probe that flips the flag — never by prose alone. Route this update
   through the `external-claims-auditor` sibling agent before merging.
6. **New external dependency decisions** (shader compiler, MLIR/LLVM, CUDA toolkit,
   mobile SDKs) require an explicit human sign-off recorded in a
   `docs/superpowers/specs/*-design.md` before any `build.zig` change that adds a new
   `linkSystemLibrary`/`addLibraryPath`/subprocess dependency — this is a new
   supply-chain and CI-environment commitment, not a routine code change.

---

## 6. Milestones and Definition of Done

### Milestone 0 — Housekeeping (small, low-risk, worth pursuing; no honesty change)

Optional, independent of the toolchain question. Can be picked up any time without
design sign-off since it changes no reported values.

- **DoD**: `mobile/mod.zig`'s `validateLabel` delegates to
  `foundation/validation.zig`'s existing non-empty/no-null-byte helpers (matching
  `shaders`/`mlir`); `mobile`'s `accelerated` field either gets a doc-comment
  clarifying it reflects desktop Metal GPU acceleration (not mobile hardware
  acceleration) or is renamed with a parity-safe mod/stub pair update. `./build.sh
  check` green, `zig build check-parity` green, no test assertions change (same
  behavior, internal-only cleanup).
- **Recommendation**: worth pursuing opportunistically, low priority.

### Milestone 1 — `shaders` MSL real-compile decision gate (large; keep disclosed unless approved)

- **DoD for the decision step only**: a design doc
  (`docs/superpowers/specs/<date>-shader-msl-compile-design.md`) that names the exact
  `metal`/`metallib` CLI invocation, the new build flag, the runtime probe that would
  flip `compilerStatus().available`, and the claims-audit row edit — reviewed and
  explicitly approved by a human before any `build.zig`/`mod.zig` change begins.
- **Recommendation**: **worth pursuing only if there is an actual product need** for
  compiled Metal shader kernels beyond what `src/features/gpu/` already does via its
  existing Metal FFI. Absent such a need, **keep disclosed as a non-goal** — do not
  schedule speculative toolchain work.

### Milestone 2 — `mlir` real lowering (very large; keep disclosed as non-goal)

- **DoD**: N/A unless explicitly re-scoped by a human with a stated product
  requirement; no design doc should be started speculatively.
- **Recommendation**: **keep disclosed as non-goal.** No current CLI/MCP frozen
  surface requires real MLIR/LLVM lowering; the toolchain adoption cost is high and
  open-ended (version pinning, C++ ABI fragility, no existing repo infra).

### Milestone 3 — `accelerator` real CUDA/NPU/TPU dispatch (very large; keep disclosed as non-goal)

- **DoD**: N/A — explicitly matches this project's existing ANE non-goal precedent.
- **Recommendation**: **keep disclosed as non-goal.** No CUDA/NPU/TPU hardware in
  this project's dev/CI environment; would require a categorically new CI runner
  class. The honest report-only behavior already correctly serves the CLI `backends`
  command's purpose (show what *would* be selected, not dispatch).

### Milestone 4 — `mobile` real iOS/Android dispatch (very large, different in kind; keep disclosed as non-goal)

- **DoD**: N/A — this is not a library-linkage gap but a second application-target
  gap (packaged app + device/emulator CI), outside this repo's current product shape
  (CLI + MCP server + library).
- **Recommendation**: **keep disclosed as non-goal**, same category as ANE.

---

## Summary table: pursue vs. disclose

| Feature | Pursue now? | If ever pursued, entry point |
|---|---|---|
| `accelerator` | No — non-goal | New `-Dfeat-cuda`-style flag + FFI module beside `mod.zig`, gated dispatch flip |
| `shaders` | Conditional — only with explicit product need + human sign-off | MSL via Xcode `metal`/`metallib` CLI (cheapest sub-lift); WGSL/SPIR-V stay non-goals |
| `mlir` | No — non-goal | Subprocess shellout to `mlir-opt`/`mlir-translate` (safer than linking `libMLIR`) |
| `mobile` | No — non-goal | Second packaged-app target + device/emulator CI, not a library link |
| Housekeeping (validation delegation, field clarity) | Yes — small, no honesty change | Direct rewrite, no design doc needed |

This plan intentionally recommends **no new toolchain-linking milestones** for three
of the four features and a **conditional, gated** one for `shaders`/MSL only — in
line with the existing ANE/CUDA/MLIR-LLVM non-goal decisions already recorded in
`AGENTS.md`/`CLAUDE.md` and `docs/contracts/external-claims-audit.mdx`. No milestone
in this plan flips any `available`/`native_dispatch` field to `true` without a real
linked toolchain and a runtime probe backing it.
