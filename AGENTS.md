## Repository Guidelines

### 1. Project Structure & Module Organization

The repository follows a simple `src/` + `tests/` layout.

```
src/          – Production code (Zig modules)
tests/        – Unit‑and‑integration tests
assets/       – Static files used by the examples
examples/     – runnable demo binaries
```

All library files are located under `src/` and should be imported with
`@import("../src/<module>.zig")`. Test files live next to the module they
exercise, e.g. `src/foo/bar.zig` ⇒ `tests/foo/bar_test.zig`.

### 2. Build, Test, and Development Commands

All development is performed with Zig’s own tooling:

- `zig build` – Compile the library and example binaries.
- `zig build run` – Build and run the default example in `examples/`.
- `zig test` – Execute the test suite defined under `tests/`.
- `zig fmt -w .` – Run the formatter on the whole repository.

The project targets Zig 0.12+.  Ensure your local Zig installation matches
`zig --version | grep 0.12`.

### 3. Coding Style & Naming Conventions

* Indentation: **4 spaces** (no tabs).
* Types: `PascalCase` (e.g. `MyStruct`).
* Functions & variables: `snake_case` (`calculate_sum`).
* Constants: `ALL_CAPS` with underscores.
* Modules: filename matches the module name, all lowercase.
* Formatting: run `zig fmt -w .` to keep code tidy – no `.editorconfig`
  needed.

### 4. Testing Guidelines

The repository uses Zig’s built‑in test framework.  A test file must end
with `_test.zig` and contain `test "description" { … }` blocks.
Run the suite with `zig test`.  Coverage is not enforced during CI but
aim for ~80% for new features.  Each test file should reside in the same
directory as the code it tests.

### 5. Commit & Pull Request Guidelines

Adopt conventional‑commits format:

```
<type>[optional scope]: <short description>

[optional body]
```

Where `<type>` is e.g. `feat`, `fix`, `docs`, `test`, or `style`.

Pull requests should:

* reference a related issue (`Closes #123`).
* contain a clear description of the change.
* include examples or screenshots for UI changes.
* have been formatted (`zig fmt`) and pass `zig test`.

### 6. Security & Configuration Tips

* Keep external dependencies minimal – the project only relies on the Zig
  stdlib.
* All configuration is via a `config.zig` file; do not hard‑code paths.
* Run tests on all supported architectures (`zig test --arch=x86_64`),
  especially when using SIMD or GPU backends.

---

Happy contributing! 🎉

