---
name: rust-webgpu-game-engine
description: Build and verify custom Rust winit and wgpu games across native and wasm WebGPU while preserving deterministic simulation, safe async lifecycle ownership, shader contracts, scenario persistence, and honest evidence boundaries.
---

# Rust WebGPU Game Engine

Use this skill when implementing or debugging a custom Rust game that combines winit, wgpu, native targets, or browser WebGPU.

## Start from authority boundaries

- Keep deterministic game truth independent of windows, GPUs, wall clocks, storage, editor drafts, and advisory output.
- Convert fixed-point state to floating point only in presentation code.
- Let editors produce validated immutable scenarios before constructing a replacement simulation. Build replacements in temporaries and swap once.
- Keep advisory or compute output presentation-only unless a separately designed deterministic contract explicitly authorizes it.

## Research the locked APIs

Read `Cargo.toml`, `Cargo.lock`, and the pinned toolchain before coding. Query documentation for the exact locked winit, wgpu, Naga, and wasm-bindgen versions. Do not copy latest-version examples into an older lock without compiling the target. In particular, verify constructor ownership, enum variants, feature names, surface presentation, and web extension traits against the locked release.

## Validate shaders before pipeline creation

- Parse shipped WGSL with Naga and run full validation before passing it to wgpu.
- Keep host vertex offsets, strides, buffer sizes, binding minimum sizes, and shader packing synchronized and tested.
- Include source labels in parse and validation diagnostics.
- Test shader contracts without requiring a live adapter; keep adapter-backed tests separate.

## Keep coordinate spaces explicit

Use logical pixels for layout, primitive vertices, renderer viewport globals, and hit testing. Use physical pixels only for surface configuration. Convert pointer coordinates once at the platform boundary. Test high-DPI-sensitive transformations rather than inferring correctness from a normal-density display.

## Own surfaces safely

Prefer an owned `Arc<Window>` when creating a `'static` wgpu surface. Create native windows, instances, and surfaces on the winit thread. If native adapter/device acquisition moves to an executor, return it through a generation-checked mailbox and configure the latest surface size back on the winit thread.

Handle every acquisition state deliberately: render and present success; render/present before reconfiguring suboptimal frames; skip timeout and occlusion; reconfigure outdated surfaces; recreate lost surfaces from the retained instance/window; and make validation failures fatal. Never configure a zero extent or reconfigure while a live frame exists.

## Use a separate browser lifecycle

- Create the browser window in `ApplicationHandler::resumed` with winit's append, prevent-default, and focusable web attributes.
- Require an explicit browser WebGPU backend when WebGL fallback is not part of the product.
- Launch GPU creation with `wasm_bindgen_futures::spawn_local`.
- Store `Result<GpuContext, Error>` in a same-thread `Rc<RefCell<Option<_>>>` mailbox and send only a generation marker through `EventLoopProxy`.
- Replace the mailbox when invalidating a generation so a detached stale future cannot overwrite the current result. Do not require `GpuContext: Send` and do not block the browser event loop.
- Use a wasm-compatible wall-clock implementation; a target compile alone will not reveal every unsupported `std` runtime operation.

## Bound persistence and compute

Treat scenario JSON as untrusted. Enforce byte limits before parsing where the platform permits, deny unknown fields, avoid browser-unsafe numeric encodings, and make failed load/save/apply operations preserve active state.

For optional GPU advisory compute, publish the CPU oracle first. Carry epoch/request/model/source metadata, allow one in-flight readback, coalesce newer work, reject stale or non-finite results, compare with an explicit tolerance, and disable the GPU path for the failed device epoch while retaining CPU output. Never claim acceleration without measurements.

## Verify in layers

Run formatting, strict Clippy, all headless tests, native release compilation, wasm target compilation, and the repository-owned pinned bundle script. Add source contract tests for cfg-sensitive browser code where native test runners cannot execute it. Then test native and browser runtime separately.

Report these as distinct evidence:

- headless semantics;
- native compilation;
- wasm compilation;
- bundle generation;
- provider CI execution;
- live native surface and first frame;
- live browser canvas, WebGPU, input, and storage;
- adapter-specific compute parity;
- manual visual acceptance.

Never promote configuration, compilation, generated artifacts, an adapter skip, or a single-platform smoke into broader runtime support.
