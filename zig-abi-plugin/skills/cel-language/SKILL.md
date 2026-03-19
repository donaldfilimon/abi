---
name: cel-language
description: Use when working with CEL (.cel files), the stage0 compiler, cel.toml, stdlib/cel/, tests/cel/, or asking about CEL syntax and commands
---

> **STATUS: NOT YET IMPLEMENTED** — The CEL language infrastructure described below
> (cel.toml, tools/cel/, stdlib/cel/, tests/cel/) does not yet exist in the repository.
> This skill documents the planned design. Do not attempt to run CEL commands or
> reference CEL files until the infrastructure is created.

# CEL Language

Use this skill for ABI's CEL language work, not for generic Zig bootstrap maintenance.

## Quick Start

CEL is ABI's language surface. The practical surface during stage0 is defined by:

- `./cel` -- the launcher wrapper
- `cel.toml` -- stage0 package metadata and wrapper/compiler paths
- `tools/cel/stage0/main.c` -- lexer, parser, formatter, C emission, and command dispatch
- `examples/cel/` -- runnable CEL examples
- `tests/cel/` -- CEL test files and smoke tests
- `stdlib/cel/` -- CEL standard library modules

Treat `.zig-bootstrap/` as the canonical Zig bridge surface and `.cel/` as the backing implementation layer during the transition. Do not assume features beyond the current stage0 compiler without checking `tools/cel/stage0/main.c`.

## Commands

- `./cel check file.cel`
  - Parse and validate the CEL source.
- `./cel fmt [-w] file.cel`
  - Print formatted CEL, or rewrite in place with `-w`.
- `./cel run file.cel`
  - Emit C, compile it with `cc`, and run `fn main()`.
- `./cel test file.cel`
  - Emit C test code, compile it, and run CEL `test` blocks.
- `./cel emit-c file.cel`
  - Print the generated C backend output.

The `./cel` wrapper rebuilds the stage0 C launcher when `tools/cel/stage0/main.c` or the wrapper itself is newer than `.cel-stage0/bin/cel`.

## Current Syntax Surface

The stage0 compiler currently recognizes:

- `module some.path;`
- `import std.prelude;`
- `fn main() { ... }`
- `test "name" { ... }`
- `let name = "value";`
- `var name = "value";`
- `print("literal");`
- `print(binding);`
- `defer print("later");`
- `return;`

Comments use `//`.

## Key Files

| File | Purpose |
|------|---------|
| `cel.toml` | Stage0 package metadata and wrapper/compiler paths |
| `tools/cel/stage0/main.c` | Lexer, parser, formatter, C emission, and command dispatch |
| `examples/cel/hello.cel` | Small runnable example |
| `tests/cel/stage0_tests.cel` | Canonical test-block example |
| `tests/cel/format_input.cel` | Formatting fixture |
| `stdlib/cel/prelude.cel` | Current CEL prelude module |

## Workflow

1. Confirm the task is really CEL-language scoped.
2. Inspect the touched CEL files plus the stage0 reference content in this skill.
3. Keep changes compatible with the current stage0 surface unless the task explicitly extends the compiler.
4. Validate with the narrowest relevant CEL checks first.

## Decision Rules

- Use this skill when the task mentions CEL modules, CEL formatting, CEL tests, CEL examples, CEL stdlib, or stage0 compilation.
- Do not use this skill for pure Zig toolchain/bootstrap tasks unless the change also affects CEL language UX or CEL-facing guidance.
- If extending syntax or behavior, inspect `tools/cel/stage0/main.c` directly instead of assuming support from future CEL plans.

## Validation

### For normal CEL source edits

1. Run `./cel check` on the touched file.
2. Run `./cel fmt -w` if formatting is part of the change.
3. Run `./cel run` for executable modules with `fn main()`.
4. Run `./cel test` for files containing `test` blocks.

### For stage0 compiler edits

1. Run `./cel check examples/cel/hello.cel`.
2. Run `./cel run examples/cel/hello.cel`.
3. Run `./cel test tests/cel/stage0_tests.cel`.
4. Run `./tests/cel/stage0_smoke.sh` if the change affects multiple commands.

### For repo-wide smoke coverage

Use `./tests/cel/stage0_smoke.sh` when the change touches multiple CEL entrypoints.

If changing stage0 parsing or codegen in `tools/cel/stage0/main.c`, verify at least one example, one test file, and formatting behavior.

## ABI Transition Rules

- CEL is the language direction.
- `.zig-bootstrap/` is the canonical Zig bridge namespace.
- The older `.cel/` tree is still the backing Zig/bootstrap implementation for now, but that is separate from CEL language source files.

When a task is really about Zig bootstrap activation, build failures, or ZLS setup, do not treat that as a CEL-language task just because the path contains `.cel/`.

## When To Inspect The Compiler Directly

Open `tools/cel/stage0/main.c` directly when the task involves:

- Adding syntax
- Changing formatting output
- Changing emitted C
- Changing test naming or test execution
- Changing parse errors or diagnostics

## Output Expectations

- Keep CEL guidance explicit about current stage0 limitations.
- Distinguish between "implemented in stage0 now" and "planned CEL direction".
- When summarizing a change, name the exact CEL commands or files you validated.
