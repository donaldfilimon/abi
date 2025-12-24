# Repository Guidelines for Agentic Coding Agents

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
├─ tests/mod.zig             # Test runner entry point
└─ README.md
```
* Keep new modules under `src/` following the existing package layout.
* Entry points: `src/abi.zig` (public API) and `src/root.zig` (root module).
* Feature modules live under `src/features/` and are re-exported via `src/abi.zig`.
* Prefer `mod.zig` for module barrels; keep files focused and single-purpose.
* Avoid introducing new top-level folders without updating this file and README.

## Build, Test, and Development Commands

### Core Build Commands
| Command | Description |
|---------|-------------|
| `zig build` | Build the core library and CLI (if CLI entrypoint exists). |
| `zig build test` | Run all Zig tests in the project. |
| `zig build -Doptimize=ReleaseFast` | Build with optimizations for production. |
| `zig build run` | Build and run the CLI (if available). |
| `zig build run -- --help` | Show CLI help information. |

### Testing Commands
| Command | Description |
|---------|-------------|
| `zig build test` | Run all tests (equivalent to `zig test src/ tests/`). |
| `zig test src/core/` | Run tests for core modules only. |
| `zig test src/features/ai/` | Run tests for AI feature modules. |
| `zig test tests/mod.zig` | Run the main test suite. |
| `zig test --test-filter="test name"` | Run specific test by name pattern. |

### Formatting and Linting
| Command | Description |
|---------|-------------|
| `zig fmt .` | Format all Zig source files in the project. |
| `zig fmt src/` | Format only source files. |
| `zig fmt --check .` | Check formatting without modifying files. |

### Feature Flags and Build Options
| Feature Flag | Description | Default |
|-------------|-------------|---------|
| `-Denable-ai` | Enable AI features and modules | `true` |
| `-Denable-gpu` | Enable GPU acceleration features | `true` |
| `-Denable-web` | Enable web utilities and HTTP features | `true` |
| `-Denable-database` | Enable database and vector search features | `true` |
| `-Dgpu-cuda` | Enable CUDA GPU backend | Depends on `-Denable-gpu` |
| `-Dgpu-vulkan` | Enable Vulkan GPU backend | Depends on `-Denable-gpu` |
| `-Dgpu-metal` | Enable Metal GPU backend | Depends on `-Denable-gpu` |
| `-Dgpu-webgpu` | Enable WebGPU backend | Depends on `-Denable-web` |

**Example:** `zig build -Denable-ai=true -Denable-gpu=false -Denable-web=true -Denable-database=true`

* Build flags are defined in `build.zig`.
* If CLI/tests are removed, update `build.zig` to skip those steps.
* Requires Zig 0.15.2 or later.

## Coding Style & Naming Conventions

### Formatting and Layout
* **Indentation:** 4 spaces, no tabs.
* **Line Length:** Keep lines under 100 characters when possible.
* **File Encoding:** UTF-8.
* **Line Endings:** LF (Unix-style).
* Run `zig fmt` before committing all changes.

### Naming Conventions
* **Types:** PascalCase (e.g., `FrameworkOptions`, `AbiError`, `VectorStore`)
* **Functions/Variables:** snake_case (e.g., `init_framework`, `process_input`, `allocator`)
* **Constants:** SCREAMING_SNAKE_CASE (e.g., `MAX_BUFFER_SIZE`)
* **Modules:** snake_case (e.g., `core.zig`, `database.zig`)
* **Test Names:** Descriptive strings in `test "description here"` blocks

### Imports and Dependencies
* **Grouping:** Group imports by: standard library, internal modules, external dependencies.
* **Explicit Imports:** Avoid `usingnamespace`; use explicit imports only.
* **Qualified Access:** Prefer qualified access (e.g., `std.mem.Allocator`) over aliases.
* **Import Style:**
```zig
const std = @import("std");
const abi = @import("abi");

const core = @import("core/mod.zig");
const framework = @import("framework/mod.zig");
```

### Comments and Documentation
* **Function Comments:** Use `//!` for module-level docs, `///` for public APIs.
* **Inline Comments:** Keep short and technical; avoid stating the obvious.
* **TODO/FIXME:** Use `// TODO:` or `// FIXME:` for temporary notes.
* **Example:**
```zig
//! High-level module description.
//!
//! Detailed explanation of module purpose and usage.

/// Performs operation X with input Y.
/// Returns Z on success, error on failure.
pub fn do_operation(input: Input) !Output {
    // Implementation here
}
```

### Error Handling
* **Error Types:** Use specific error enums (e.g., `AbiError`) over generic errors.
* **Return Types:** Use `!T` for fallible operations, `T` for infallible ones.
* **Error Propagation:** Prefer explicit error returns over panics.
* **Cleanup:** Use `defer`/`errdefer` for resource cleanup.
* **Result Types:** Use `Result(T)` for complex error handling patterns.

### Types and Data Structures
* **Structs:** Use structs for complex data; prefer anonymous structs for simple cases.
* **Enums:** Use enums for closed sets of values; consider exhaustive switches.
* **Unions:** Use tagged unions for variant types.
* **Optionals:** Use `?T` for optional values, not zero values.
* **Slices vs Arrays:** Use slices (`[]T`) for variable-length data, arrays (`[N]T`) for fixed-size.

### Memory Management
* **Allocators:** Always pass allocators explicitly; never use global allocators.
* **Ownership:** Clearly document ownership semantics in function docs.
* **Leaks:** Use `defer` for cleanup; run tests with leak detection.
* **Arena Allocators:** Consider arena allocators for scoped memory operations.

### Performance Considerations
* **Comptime:** Use `comptime` for compile-time evaluation when beneficial.
* **SIMD:** Leverage SIMD operations from `src/shared/simd.zig` when appropriate.
* **Memory Layout:** Consider struct field ordering for cache efficiency.
* **Bounds Checking:** Rely on Zig's automatic bounds checking in debug builds.

## Testing Guidelines

### Test Organization
* **Frameworks:** Zig built-in tests (`zig build test`).
* **Placement:** Co-locate tests near source code or in `tests/mod.zig`.
* **Coverage:** New features must include tests or provide clear justification.
* **Naming:** Use descriptive `test "..."` blocks with clear intent.

### Test Patterns
```zig
test "framework initialization succeeds with valid config" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    var framework = try abi.init(gpa.allocator(), abi.FrameworkOptions{});
    defer abi.shutdown(&framework);

    try std.testing.expect(!framework.isRunning());
}

test "error handling for invalid input" {
    const result = try performOperation(null);
    try std.testing.expectError(AbiError.InvalidInput, result);
}
```

### Running Specific Tests
* **Single Test:** `zig test --test-filter="exact test name"`
* **Pattern Match:** `zig test --test-filter="framework.*"`
* **Module Tests:** `zig test src/features/ai/mod.zig`

## Module Conventions

### Module Structure
* Use `mod.zig` as explicit re-export surfaces for submodules.
* Keep modules focused and single-purpose.
* Avoid circular imports between feature modules.
* Shared helpers belong in `src/shared/` rather than duplicated.

### Legacy Code
* Keep legacy compatibility wrappers in `src/internal/legacy/`.
* Add clear deprecation comments for legacy APIs.
* Maintain backward compatibility for public APIs.

## Development Workflow

### Pre-Commit Checklist
1. Run `zig build` to ensure compilation.
2. Run `zig build test` to verify tests pass.
3. Run `zig fmt --check .` to verify formatting.
4. Update documentation for API changes.
5. Test feature flags if modified.

### Debugging Tips
* Use `std.debug.print` for temporary debugging (remove before commit).
* Leverage the monitoring features in `src/features/monitoring/` for profiling.
* Use `std.testing.checkAllAllocationFailures` for memory leak testing.

### Common Patterns
* **Framework Initialization:**
```zig
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
defer _ = gpa.deinit();

var framework = try abi.init(gpa.allocator(), options);
defer abi.shutdown(&framework);
```

* **Error Handling:**
```zig
pub fn riskyOperation(allocator: std.mem.Allocator) !Result {
    errdefer {
        // Cleanup on error
    }

    const resource = try allocateResource(allocator);
    errdefer deallocateResource(resource);

    // ... operation logic ...

    return result;
}
```

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

## Security & Configuration Tips
* Store secrets in environment variables; never hard-code them.
* Review third-party dependencies with `npm audit` or `pip-audit` if used.
* Document any required config files in `assets/config.example.*` if added.
* Use the crypto utilities in `src/shared/utils/crypto/` for security operations.

---
These guidelines aim to keep the codebase clean, consistent, and easy to contribute to.
When in doubt, follow existing patterns in the repository.
