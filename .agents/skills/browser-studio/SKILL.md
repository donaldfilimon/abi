---
name: browser-studio
description: Drive ABI's loopback multimodal browser studio — camera/image/video frames, microphone PCM, client-side Web Speech STT/TTS, and deterministic local analysis via `abi agent browser --studio`. Use for vision, voice, image, video, getUserMedia, or browser-studio work. Not a neural vision/STT/TTS model.
---

# browser-studio — loopback multimodal capture

Driver: **`.agents/skills/browser-studio/serve.sh`** (paths relative to repo root).
Builds the CLI, launches `abi agent browser --studio` on `127.0.0.1`, exercises
`GET /health`, `GET /capabilities`, and `POST /analyze`, then tears the server
down. Loopback only.

## What this is

`abi agent browser --studio [--port <port>] [--once]` serves a single-page
studio and JSON routes from `crates/abi-cli/src/browser_studio.rs`. Analysis
runs through `abi_ai::multimodal` (bucket-hash embeddings + classical stats).

Honest boundaries:

- **Not** a neural vision, STT, or TTS model.
- Speech recognition and speech synthesis, when used, are **browser Web Speech
  APIs** in the page. The server only embeds a supplied transcript and/or PCM.
- Navigation planning stays on `abi agent browser <task>` and still reports
  `embedded_browser=false` / `delegation_hint=external-mcp-playwright`.
- The 13-command CLI and 12-tool MCP catalogs are unchanged.

## Run

```bash
./tools/cargo.sh build -p abi-cli
./target/debug/abi agent browser --studio --port 8095
# open http://127.0.0.1:8095/

.agents/skills/browser-studio/serve.sh        # health + analyze smoke
.agents/skills/browser-studio/serve.sh 9105   # custom port
```

`--once` handles a single connection and exits (tests / scripts).

## Routes

| Method | Path | Body |
|--------|------|------|
| GET | `/` | HTML studio (camera, frames, mic, Web Speech, local TTS) |
| GET | `/health` | `{"status":"ok","surface":"browser-studio"}` |
| GET | `/capabilities` | claim-honest capability document |
| POST | `/analyze` | `{"kind":"image\|video\|audio\|voice", ...}` |
| POST | `/fuse` | `{"vision":[32 floats], "audio":..., "video":...}` |

Limits: grayscale `1..=64` side, `<=8` video frames, `<=2048` PCM samples,
sample rate `8000..=48000`. Requests use the foundation 64 KiB cap.

## Auth

`ABI_BROWSER_STUDIO_TOKEN` gates every route except `GET /` and `GET /index.html`
so the page can load. POST analyze/fuse then need
`Authorization: Bearer <token>`. Loopback bind plus Host/Origin allowlists;
this is not a TLS front.

## Gotchas

- Leave the token unset for interactive camera/mic use. A set token makes the
  page's `fetch` calls 401 unless the operator adds a header.
- `createScriptProcessor` is the page's PCM tap — deprecated in browsers but
  available without a worklet build step. Disclosure stays "classical PCM".
- Do not claim Whisper, GPT-4V, or in-process neural TTS in docs or skills.
- Default plan path (`abi agent browser "open docs"`) must keep emitting
  `orchestration=browser-local` and `embedded_browser=false`.
