---
layout: documentation
title: "zig std Windows Regression"
description: "Impact analysis of the Zig 0.15.x Windows documentation regression and mitigation tactics."
permalink: /zig-std-windows-bug/
---

# Zig `zig std` Windows Regression (0.15.x)

## Issue Summary and Manifestation
- Running `zig std` on Windows 10/11 with Zig 0.15.1 or `0.16.0-dev` nightlies hangs with an endless loading spinner.
- Console output shows `unable to serve /sources.tar: ConnectionResetByPeer` each time the browser requests the archive.
- The embedded documentation server fails to stream `sources.tar`, so the page never loads.

## Affected Versions
- Regression introduced in the 0.15.x line; confirmed on 0.15.1 and 0.16.0-dev (Aug 2025 nightlies).
- Zig 0.14.x (e.g., 0.14.1) behaves normally—`zig std` serves documentation without errors.
- Related chunked-transfer fixes landed during 0.15.0 development, but Windows regressions persist in 0.15.1.

## Official Bug Reports and Discussion
- **#24944** (Aug 21, 2025): “Zig std nonfunctional on Windows after 0.15.1 update.” Labeled `os-windows`, `regression`, and targeted for the 0.15.2 milestone.
- **#24972** (Aug 24, 2025): Duplicate confirmation of the `ConnectionResetByPeer` error while fetching `/sources.tar`; notes 0.14.1 works.
- **#24760 / PR #24864**: Earlier Linux panic traced to `chunkedSendFile`; merged fix (Aug 16, 2025) restored functionality but did not resolve this Windows-specific regression.

## Suspected Root Cause
- Error points to premature TCP connection closure while streaming `sources.tar`.
- On Windows, the HTTP server surfaces this as `std.posix.AcceptError.ConnectionResetByPeer`.
- Likely tied to `std.http.BodyWriter` chunked responses: `std-docs.zig` builds `sources.tar` and streams via chunked encoding.
- Prior bug in `chunkedSendFile` caused mis-sized chunks/incorrect EOF handling; Windows may hit a similar code path leading to resets.

## Workarounds
- Use Zig 0.14.x (0.14.1 confirmed) where `zig std` operates normally.
- Run `zig std` under Linux/macOS or Windows Subsystem for Linux to avoid the Windows TCP behavior.
- Access documentation offline/online via prebuilt HTML or official site until a fix ships.

## Cross-Platform Status
- Regression appears Windows-specific; Linux and macOS users report normal behavior on 0.15.1.
- Linux previously hit an `chunkedSendFile` panic in 0.15.0 but is fixed post-PR #24864, leaving Windows as the remaining platform with failures.

## `zig std` Documentation Server Architecture
1. Zig compiles and executes `lib/compiler/std-docs.zig`, starting a local HTTP server on `127.0.0.1`.
2. The server serves static assets (`index.html`, `main.js`, `main.wasm`) from `lib/docs/`.
3. On request, it dynamically builds `sources.tar` containing all standard library `.zig` sources (~12 MB uncompressed) and streams it via chunked HTTP transfer.
4. Browser loads `main.js` and `main.wasm`; the WASM module fetches `sources.tar`, decompresses, and parses it in memory using `std.tar` for interactive source browsing.

## Role of `sources.tar`
- Essential bundle of standard library `.zig` files generated on demand with each `zig std` run.
- WASM frontend requires the tarball to populate the source browser—any interruption blocks doc rendering.
- Tar approach keeps distribution lightweight while avoiding pre-generated HTML.

## Expected Fix Timeline
- Issue #24944 slated for Zig 0.15.2 patch release; developers acknowledge ongoing investigation.
- Anticipated follow-up to PR #24864 to harden Windows chunked transfer handling for `sources.tar`.
- Users should monitor 0.15.2 release notes and issue tracker updates for confirmation of a resolved Windows experience.

## References
- Zig GitHub issues: #24944, #24972, #24760; PR #24864.
- Zig 0.14.1 vs. 0.15.x behavioral reports from community discussions and changelog notes.
