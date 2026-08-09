---
title: "ABI Multi-Persona Technical Overview"
description: "Current nightly-Rust implementation of Abbey, Aviva, and ABI routing, local generation, policy telemetry, and orchestration boundaries."
---

ABI exposes three deterministic local assistant profiles:

- **Abbey** is the primary empathetic-polymath profile.
- **Aviva** is the direct expert profile.
- **ABI** is the orchestration and governance profile.

The canonical product direction is
[Abbey Core Identity](./abbey-core-identity). This page describes only what the
current Rust source and tests prove.

## Implementation map

| Concern | Source | Current behavior |
| --- | --- | --- |
| Identity | `crates/abi-ai/src/identity.rs` | Profile labels, priors, instructions, and product contracts |
| Routing | `crates/abi-ai/src/router.rs` | Explicit leading selectors plus deterministic keyword weights |
| Adaptive routing | `crates/abi-ai/src/modulator.rs` | EMA smoothing with persisted state in the SEA learn path |
| Local generation | `crates/abi-ai/src/incremental.rs` | Deterministic profile template formatting |
| Completion | `crates/abi-ai/src/completion.rs` | Routing, local generation, policy telemetry, and metadata |
| Constitution | `crates/abi-ai/src/constitution.rs` | Six-principle lexical signal with weighted score and safety/privacy veto fields |
| Orchestration | `crates/abi-ai/src/orchestration.rs` | Local worker specs and deterministic report aggregation |
| CLI | `crates/abi-cli/src/complete.rs`, `agent.rs` | Local/live dispatch, persistence, and user-facing provenance |

## Routing

Routing is local and deterministic. A leading explicit address can select a
profile; otherwise the router starts from an Abbey-favoring prior and adds
weights for recognized tokens. It normalizes the distribution and selects the
largest component with a stable tie-break order.

The ordinary `abi complete` path does not load persisted EMA state. Adaptive
weights are used by the SEA learn loop, where WDBX stores the serialized
modulator state for later learned completions.

Routing is a response-style choice. It is not semantic model routing, a
quality score, or evidence that one profile has greater reasoning ability.

## Generation provenance

Without `--live`, ABI does not call Anthropic or another remote provider. It
formats a deterministic local persona template. Output and persisted metadata
must distinguish:

- the requested model identifier;
- the provider/transport actually used;
- the generation engine, such as `persona-template`;
- the selected profile.

Remote model IDs must not be presented as proof that the corresponding
provider generated an offline response.

Live Anthropic, Apple FoundationModels, and local OpenAI-compatible bridge
paths are separate transports. Availability or a configured model name does
not prove a request succeeded.

## Constitution signal

The constitution module applies a small, case-insensitive lexical rule set to
the generated text. It returns per-principle scores, a weighted E-score,
`passed`, and `vetoed` telemetry.

This signal has strict limits:

- it is not a general-purpose safety classifier;
- absence of a matched phrase is not proof that content is safe;
- paraphrases and obfuscation may evade lexical checks;
- callers must not use a clean lexical result as the sole authorization for
  tool execution or other side effects.

The local completion path may substitute a refusal for a safety/privacy veto.
Other surfaces can expose the signal as telemetry. Those behaviors are not a
universal hard-veto guarantee.

## Local worker orchestration

`abi agent multi` and custom worker specs produce local deterministic reports.
The worker name, instructions, profile override, and tool hints are rendered
as metadata. Instructions do not currently condition generation, and tool
hints do not invoke tools.

Workers are aggregated in specification order. The one-process scheduler can
account for marker tasks, but the CLI drains them before returning. This is
not durable background processing, remote delegation, or distributed
multi-agent execution.

Browser planning is a dry-run plan only. It reports
`embedded_browser=false`; ABI does not embed or control a browser through
this command.

## Persistence

When a durable WDBX path is configured, local completion attempts to store:

1. a query vector;
2. a response vector;
3. metadata under `completion:<query_id>`;
4. a linked audit block.

The status must disclose partial write failures. A set of independent WAL
mutations is not an atomic transaction unless the storage API explicitly
provides one.

SEA learning additionally persists adaptive-router state. Router adaptation
and completion-record persistence are distinct operations and must not be
described as one atomic commit.

## Public and claim boundaries

The frozen CLI and MCP inventories live in
[Public API Contract](../contracts/public-api). Local template generation,
local scheduler accounting, WDBX persistence, and loopback transports do not
prove:

- production distributed agents or multi-host sharding;
- a general safety classifier or audited policy enforcement;
- remote-provider generation without a successful transport call;
- native CUDA/Vulkan/ANE execution;
- production FHE, learned compression, or benchmark superiority.

Run `./tools/check.sh` for the Rust gate and
`.agents/skills/run-abi/smoke.sh` for representative real-binary behavior.
