This file is a compact, actionable guide for automation agents and engineers who need the exact commands and caveats to work in this repository.

Read ONBOARDING.md first for a longer day-one guide. The sections below are intentionally short and contain exact commands and file paths you will need.

Bootstrap (toolchain + Zig)
- Ensure the repository Zig pin (used by CI and tooling): .zigversion
- Install or link Zig and ZLS with the repo helper:
  - tools/zigly --status            # show resolved Zig paths
  - tools/zigly --bootstrap         # install/link Zig + ZLS (recommended)
  - tools/zigly --link              # link an already-installed Zig to ~/.local/bin

Quick local bootstrap
- ./build.sh --bootstrap           # full setup: bootstrap Zig/ZLS, install abi tools, build
- ./build.sh --link                # link Zig + ZLS and install abi tools to ~/.local/bin

Build and sanity commands
- ./build.sh                       # default: build the library
- ./build.sh dev                   # build developer targets (typecheck + cli + mcp)
- ./build.sh quick                 # fast local gate (fmt + typecheck + parity)
- ./build.sh ci                    # runs CI validation locally (lint + tests + parity + MCP)
- ./build.sh cli                   # build CLI binary
- ./build.sh mcp                   # build MCP stdio/SSE server
- ./build.sh check-parity          # verify module/stub declaration parity
- ./build.sh test --summary all    # run tests (add -- to forward zig test args)

MCP (multi-HA) startup and health
- Configure HA instances: mcp/servers.json (examples: abi-mcp-1, abi-mcp-2)
- Start the platform-local launcher (cross-OS): ./mcp/launcher.sh [stdio|sse]
  - launcher resolves the correct binary for the host (abi-mcp or abi-mcp.exe)
- Health checks (automated):
  - scripts/check-mcp-health.sh     # verifies all servers in mcp/servers.json
  - Exit status: 0 = healthy, non-zero = failure (treat as restart/failure)
- Interop and ACP reachability:
  - scripts/check-interop.sh        # runs MCP health check + ACP endpoint discovery
  - scripts/list-acp-endpoints.sh   # requires ACP_ENDPOINTS env var (comma-separated)

CI / Parity notes
- CI pins Zig to the value in .zigversion; see .github/workflows/ci.yml for parity steps.
- Parity tests skip auth-heavy tests when ABI_JWT_SECRET is not set. In CI you may provide ABI_JWT_SECRET to run full parity.

Critical gotchas
- Zig toolchain: this repo expects Zig 0.17-dev.* as pinned in .zigversion. Use tools/zigly to manage it.
- macOS linker quirk: on macOS 26.4+ we prefer ./build.sh (which will add -Dfeat-gpu=false for test stability) due to Zig/LLD changes. See build.sh is_macos_26_4_or_newer logic.
- Launcher: use mcp/launcher.sh to start MCP cross-platform; it selects the correct binary per OS.
- Health scripts assume curl is available; check-mcp-health.sh uses jq to parse mcp/servers.json if present.

Where to read next
- ONBOARDING.md          # day-one, longer bootstrap and commands
- README.md              # repo overview and parity notes
- docs/README.md         # doc map and GitHub Pages material
- CODEBASE_REVIEW.md     # architecture entry points
- SUMMARY.md, GLOSSARY.md

If you'd like, I can run the requested sanity checks now: git status, show the diff for AGENTS.md, verify referenced files exist, and check ONBOARDING.md consistency.
