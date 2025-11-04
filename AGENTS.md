# Repository Guidelines

## 1. Project layout

- **src/** – Zig source modules
- **tests/** – unit tests (filename ends with `_test.zig`)
- **examples/** – runnable demos
- **assets/** – static files used by examples

Every module lives under **src/** and is imported with
```
@import("../src/<module>.zig")
```

## 2. Build & test commands

| Command | Purpose |
|---------|---------|
| `zig build` | Compile library and demos |
| `zig test` | Run all tests |
| `zig fmt -w .` | Apply the standard Zig formatter |

The repo targets Zig 0.12+ – run `zig --version` to confirm.

## 3. Code style

* **Indent**: 4 spaces, no tabs
* **Types**: `PascalCase`
* **Funcs/vars**: `snake_case`
* **Constants**: `ALL_CAPS`
* **Files**: lowercase, match module name
* Run `zig fmt -w .` before committing

## 4. Testing

Use Zig’s built‑in framework.  Test files must end with `_test.zig` and
be in the same directory as the code they exercise.  Run with
`zig test`.  Aim for ~80 % coverage on new features.

## 5. Commit & PR style

Adopt **conventional‑commits**: `type(scope): short subject`.  Typical
types: `feat`, `fix`, `docs`, `test`, `style`.

Pull requests should:

1. Reference a GitHub issue (`Closes #123`).
2. Include a clear description.
3. Pass formatting and tests.

---

Happy hacking! 🎉

