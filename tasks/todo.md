# TODO — ABI Framework (Rust nightly)

Forward-looking tracker for **incomplete and in-flight** work after the Zig →
Rust rewrite. Completed rewrite history: `RUST-REWRITE-PLAN.md`, `git log`,
`CHANGELOG.md`. Claims gate: `docs/contracts/external-claims-audit.mdx`.

Status legend: `✅ Done` · `🟡 In progress` · `⚪ Not started` · `🔴 Blocked` · `◑ Partial / disclosed`

> Discipline: no Session Summary narratives here. When an item closes, delete
> its row. Source and tests override prose. Gate: `./tools/check.sh`.

---

## Rewrite closeout

| Item | Status | Notes |
| ---- | ------ | ----- |
| Zig teardown | ✅ | 0 tracked `*.zig` / `build.zig*` |
| Frozen CLI (13) + MCP (12) | ✅ | Golden + unit coverage |
| FoundationModels shim | ✅ | `libabi_fm_shim.dylib` on arm64 macOS; honest offline |
| Local OpenAI bridge + MCP HTTP/SSE | ✅ | Loopback; fallback when bridge unusable |
| Land `rust-rewrite` on `main` | ✅ | Squash-merged [#756](https://github.com/donaldfilimon/abi/pull/756) as `34c35d5` |

---

## Disclosed residuals (do NOT fake-complete)

| Item | Status | Constraint |
| ---- | ------ | ---------- |
| Native GPU kernels (Metal/CUDA/Vulkan) | ◑ | `accelerated=false`; CPU SIMD is real |
| Live Discord `wss://` without TLS proxy | ◑ | Offline gateway + framing tested; TLS not linked |
| Live Twilio media WebSocket | ◑ | Local ConversationRelay builder only |
| External shader / MLIR toolchains | ◑ | Validation / textual IR only |
| Mobile `native_dispatch` | ◑ | Simulated desktop profile |
| Production FHE / multi-host sharding | ◑ | Reference demos / ops guidance only |

---

## Candidate next product slices

| Priority | Item | Notes |
| -------- | ---- | ----- |
| 1 | Long-form docs Zig-path scrub | CHANGELOG/threat-model history OK; skills must not teach `zig build` as gate |
| 2 | `nn` checkpoint persist | Demo durability; not production LLM |
| 3 | Live Discord/Twilio TLS clients | Optional product expansion |
| 4 | Metal kernels beyond detection | Large; keep claims honest |
| 5 | Windows runtime CI for ACLs | 🔴 no Windows runner |

---

## Recently landed

- **Rust rewrite on `main`** via [#756](https://github.com/donaldfilimon/abi/pull/756) (`34c35d5`)
- FoundationModels Swift `@c` shim + `complete --live --model apple-fm --confirm`
- shaders / mlir / hash / metrics / mobile report surfaces
- local_bridge + MCP HTTP/SSE; wdbx_stats open-failure disclosure for CI
- Discord gateway/routing/WS framing (offline); Twilio ConversationRelay local path
- Zig one-shot teardown
