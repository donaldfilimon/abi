# Repository Guidelines

## Project Structure & Module Organization
```
├─ src/                      # Application source code
│  ├─ core/                  # Core infrastructure + fundamental types
│  ├─ features/              # Feature modules (AI, GPU, database, web, etc.)
│  ├─ framework/             # Orchestration, config, runtime
│  ├─ shared/                # Shared utilities (logging, platform, utils)
│  └─ internal/              # Legacy + experimental modules (not public API)
│     └─ legacy/             # Backward-compat implementations
├─ build.zig                 # Zig build graph + feature flags
├─ build.zig.zon             # Zig package metadata
└─ README.md
```
* Keep new modules under `src/` following the existing package layout.
* Entry points: `src/abi.zig` (public API) and `src/root.zig` (root module).
* Feature modules live under `src/features/` and are re-exported via `src/abi.zig`.
* Prefer `mod.zig` for module barrels; keep files focused and single-purpose.
* Avoid introducing new top-level folders without updating this file and README.

## Build, Test, and Development Commands
| Command | Description |
|---------|-------------|
| `zig build` | Build the core library and CLI (if CLI entrypoint exists). |
| `zig build test` | Run Zig tests (if `tests/mod.zig` exists). |
| `zig build -Doptimize=ReleaseFast` | Optimized build. |
| `zig fmt src/**/*.zig` | Format Zig sources. |
* Build flags are defined in `build.zig`.
* If CLI/tests are removed, update `build.zig` to skip those steps.
* Requires Zig 0.15.2.

## Feature Flags (build.zig)
* `-Denable-ai`, `-Denable-gpu`, `-Denable-web`, `-Denable-database`
* `-Dgpu-cuda`, `-Dgpu-vulkan`, `-Dgpu-metal`, `-Dgpu-webgpu`

## Coding Style & Naming Conventions
* **Indentation:** 4 spaces, no tabs.
* **Zig style:** prefer `snake_case` for functions/vars, `PascalCase` for types.
* **Imports:** group standard lib, internal modules, and external deps.
* **Comments:** keep them short and technical; avoid repeating obvious code.
* Run `zig fmt` before committing.

## Testing Guidelines
* **Frameworks:** Zig built-in tests (`zig build test`).
* **Placement:** co-locate tests near source or under `tests/`.
* **Naming:** use `test "..."` blocks with descriptive names.
* Run `zig build test` locally after behavior changes.

## Module Conventions
* Use `mod.zig` as a small, explicit re-export surface.
* Avoid circular imports between feature modules; prefer shared utilities.
* Shared helpers belong in `src/shared/` rather than duplicated across features.
* Keep legacy compatibility wrappers in `src/internal/legacy/` with clear comments.

## Repo Hygiene
* Keep the tree lean; delete unused modules instead of letting them rot.
* If you remove a top-level directory, update README and AGENTS.
* Prefer small, focused commits with clear intent.

## Commit & Pull Request Guidelines
* **Commit messages:** `<type>(<scope>): <short summary>`
  * Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`.
  * Example: `feat(auth): add OAuth2 login flow`.
* **PR requirements:**
  * Clear description of the change.
  * Reference associated issue (`#123`).
  * Include screenshots or logs when UI or runtime behavior changes.
  * Ensure all checks (`build`, `test`, `lint`) pass.

## Security & Configuration Tips (Optional)
* Store secrets in environment variables; never hard-code them.
* Review third-party dependencies with `npm audit` or `pip-audit` if used.
* Document any required config files in `assets/config.example.*` if added.

---
These guidelines aim to keep the codebase clean, consistent, and easy to contribute to.
When in doubt, follow existing patterns in the repository.
