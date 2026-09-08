---
name: zig-build
description: >-
  Use when writing, fixing, or reviewing a build.zig or build.zig.zon on Zig
  master (0.17.0-dev), when addExecutable rejects root_source_file/target/optimize,
  when choosing between .ReleaseFast and .fast optimize modes, on "invalid
  fingerprint" or "missing top-level 'fingerprint' field", when wiring
  b.createModule / b.addModule / b.dependency / module imports, adding build
  options with b.addOptions, defining custom steps with b.step, passing args
  through addRunArtifact, running zig fetch --save, or cross-compiling with
  -Dtarget. Do not use for std.testing assertions (zig-testing) or C interop
  and translate-c (zig-c-interop).
---

# Zig build system on master

Verified against the zvm master toolchain on this Mac:
`/Users/donaldfilimon/.zvm/bin/zig`, version `0.17.0-dev.2018+ab30a0b9a`,
std source under `/Users/donaldfilimon/.zvm/master/lib/`.
Every claim below was either read out of that std source or produced by running
that binary. The build API churns every release, so re-verify against the
installed toolchain rather than trusting recalled idiom.

Scratch work goes in `/private/tmp/zig-build-scratch/`, never iCloud, never home root.

## Changed on master

A future agent reaching for older idiom will hit these. Each row states whether
the old form hard-fails or is a still-resolving deprecated alias; both were
checked by compiling, not assumed.

| Older idiom | Status on this toolchain |
|---|---|
| `b.addExecutable(.{ .name, .root_source_file, .target, .optimize })` | **Hard failure** (no such field). `ExecutableOptions` has only `name`, `root_module`, `version`, `linkage`, `max_rss`, `use_llvm`, `use_lld`, `zig_lib_dir`, `win32_manifest` (`lib/std/Build.zig:557`). Source file, target and optimize live on the module. |
| `.Debug` / `.ReleaseSafe` / `.ReleaseFast` / `.ReleaseSmall` | Canonical names are now `.debug`, `.safe`, `.fast`, `.small` (`lib/std/lang.zig:115`; `std.builtin.Optimize` resolves there, `OptimizeMode` is an alias at `:111`). The old names survive as **deprecated aliases marked for removal after 0.18.0** (`lib/std/lang.zig:127-133`) plus a string map (`:135`), so `-Doptimize=ReleaseFast` and a literal `.ReleaseFast` both still work and both resolve to `.fast`. Verified both on this toolchain. Write the lowercase names anyway: `zig build --help` only advertises `debug safe fast small`, and the aliases are on a removal clock. |
| `b.addStaticLibrary` / `b.addSharedLibrary` | **Removed** (absent from `lib/std/Build.zig`). One `b.addLibrary(.{ .linkage = .static \| .dynamic, .name, .root_module })` (`lib/std/Build.zig:635`). |
| `if (b.args) \|args\| run_cmd.addArgs(args)` | **Hard failure**: `error: no field named 'args' in struct 'Build'`. Use `run_cmd.addPassthruArgs();`, which is what `zig init` generates. |
| `exe.linkLibC()`, `exe.linkSystemLibrary()`, `exe.addIncludePath()`, `exe.addCSourceFiles()` | **Removed**: none of those methods exist on `*Step.Compile` any more. They are module-level: `link_libc: ?bool` in `Module.CreateOptions` (`lib/std/Build/Module.zig:208`), `Module.linkSystemLibrary` (`:341`), `Module.addIncludePath` (`:473`), `Module.addCSourceFiles` (`:388`), `Module.linkLibrary` (`:456`). |
| `exe.addOptions("build_options", opts)` | **Removed from `*Step.Compile`** (no `addOptions` in `Build/Step/Compile.zig`). Use `mod.addOptions("build_options", opts)` on the `*Module` (`lib/std/Build/Module.zig:318`). |
| `.name = "mypkg"` string in `build.zig.zon` | **Hard failure**: `build.zig.zon:2:13: error: expected enum literal`. It must be `.name = .mypkg`, an **enum literal**; `zig init` generates `.name = .init`. |
| no `fingerprint` field | **Hard failure**: `error: missing top-level 'fingerprint' field; suggested value: 0x...`. Mandatory and validated. |
| `@cImport` in build-adjacent code | **Removed**: `error: invalid builtin function: '@cImport'`. See the `zig-c-interop` skill. |

`pub fn main() !void` still compiles: `lib/std/start.zig:786` dispatches on the
parameter count, so zero-parameter `main`, `main(init: std.process.Init.Minimal)`
and `main(init: std.process.Init)` are all accepted. The `zig init` template now
generates the `std.process.Init` form, which is where `init.arena`, `init.io`
and `init.minimal.args` come from.

## The highest-authority example is the compiler's own template

```bash
mkdir -p /private/tmp/zig-build-scratch/scratchpkg
cd /private/tmp/zig-build-scratch/scratchpkg
/Users/donaldfilimon/.zvm/bin/zig init
```

Read the generated `build.zig` and `build.zig.zon`. They are regenerated with
the compiler and settle every question of current idiom. When this skill and the
generated template disagree, the template wins and this skill is stale.

**Trap:** do not name the directory `init`. The template writes
`const init = @import("init");` and `pub fn main(init: std.process.Init)`, so a
package literally named `init` fails to compile with
`error: function parameter shadows declaration of 'init'`. Verified: `zig init`
in `/private/tmp/zig-build-scratch/init` produces a template that does not build.

## Entry point and standard options

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});     // lib/std/Build.zig:1232
    const optimize = b.standardOptimizeOption(.{});  // lib/std/Build.zig:1194
}
```

`standardTargetOptions(args: StandardTargetOptionsArgs) ResolvedTarget` accepts
`whitelist: ?[]const Target.Query` and `default_target: Target.Query`
(`lib/std/Build.zig:1225`). `standardOptimizeOption` accepts
`preferred_optimize_mode: ?std.builtin.Optimize`; when set, it exposes a
`-Drelease` bool instead of `-Doptimize`.

Custom flags come from `b.option(comptime T, name, description) ?T`
(`lib/std/Build.zig:967`). Every declared option shows up in `zig build --help`.

## Modules are the unit of compilation

`Module.CreateOptions` (`lib/std/Build/Module.zig:190`) is where source file,
target, optimize, imports and link behavior live:

```zig
root_source_file: ?LazyPath = null,
imports: []const Import = &.{},        // Import = struct { name, module }
target: ?std.Build.ResolvedTarget = null,
optimize: ?std.builtin.Optimize = null,
link_libc: ?bool = null,
link_libcpp: ?bool = null,
single_threaded: ?bool = null,
strip: ?bool = null,
sanitize_c: ?std.zig.SanitizeC = null,
sanitize_thread: ?bool = null,
fuzz: ?bool = null,
pic: ?bool = null,
// plus code_model, stack_protector, stack_check, unwind_tables, dwarf_format,
// valgrind, red_zone, omit_frame_pointer, error_tracing, no_builtin
```

Two creators, and the difference is visibility to downstream packages:

- `b.addModule(name, options) *Module` (`lib/std/Build.zig:702`) creates a
  **public** module. Consumers reach it through `dep.module(name)`. Adding two
  modules under one name panics.
- `b.createModule(options) *Module` (`lib/std/Build.zig:720`) creates a
  **private** module for this package only. This is what the generated template
  uses for the executable root.

Imports can also be added after creation with `Module.addImport(name, module)`
(`lib/std/Build/Module.zig:303`); cyclical imports are allowed.

## Artifacts

```zig
b.addExecutable(.{ .name = "app", .root_module = mod }) *Step.Compile  // :577
b.addLibrary(.{ .linkage = .static, .name = "x", .root_module = mod }) // :635
b.addObject(.{ .name = "o", .root_module = mod })                      // :601
b.addTest(.{ .root_module = mod })                                     // :674
```

`TestOptions` (`lib/std/Build.zig:651`) carries `name` (default `"test"`),
`root_module`, `filters: []const []const u8`, `test_runner`, `emit_object`.
`addTest` **does not run** the tests; it builds a test binary that you pass to
`addRunArtifact`. That split is deliberate so the two steps cache independently.

`b.installArtifact(compile)` (`lib/std/Build.zig:1460`) declares that the
artifact goes into the install prefix (`zig-out/` by default, `--prefix`/`-p`
overrides). Without it, `zig build` produces nothing in `zig-out`.

## Running and passing args

```zig
const run_cmd = b.addRunArtifact(exe);      // lib/std/Build.zig:760
run_cmd.step.dependOn(b.getInstallStep());  // run from zig-out, not the cache
run_cmd.addPassthruArgs();                  // forwards `zig build run -- a b c`
b.step("run", "Run the app").dependOn(&run_cmd.step);
```

`b.addSystemCommand(argv) *Step.Run` (`lib/std/Build.zig:741`) runs a host
process; it introduces a system dependency and hurts reproducibility, so prefer
`addRunArtifact` where possible. Further args go on with `Step.Run.addArgs`,
`addArtifactArg`, `addFileArg`, `addOutputFileArg`.

## Build options

```zig
const opts = b.addOptions();                          // lib/std/Build.zig:553
opts.addOption([]const u8, "banner", banner);         // Build/Step/Options.zig:43
mod.addOptions("build_options", opts);                // Build/Module.zig:318
```

Then in source: `const build_options = @import("build_options");`.
`Options.addOptionPath(name, LazyPath)` (`Build/Step/Options.zig:424`) embeds a
resolved path. `Options.createModule()` (`:436`) and `getOutput()` (`:444`) exist
if you want the module or the generated `.zig` file directly.

## Custom steps

`b.step(name, description) *Step` (`lib/std/Build.zig:1169`) creates a top-level
step, listed in `zig build --help`. It does nothing until something depends on
it. A duplicate name panics. Wire real work in with `dependOn`:

```zig
const check = b.addSystemCommand(&.{ "sh", "-c", "echo custom-step-ran" });
b.step("check", "Custom step").dependOn(&check.step);
```

## build.zig.zon

Fields, read out of the generated template:

```zig
.{
    .name = .mypkg,               // enum literal, not a string
    .version = "0.1.0",           // semver
    .fingerprint = 0x...,         // mandatory; see below
    .minimum_zig_version = "0.17.0-dev.2018+ab30a0b9a",
    .dependencies = .{
        .other = .{ .url = "...", .hash = "..." },   // or .path = "..."
    },
    .paths = .{ "build.zig", "build.zig.zon", "src" },
}
```

`fingerprint` is a globally unique package id, generated once and then never
changed. It is **validated**, and the low 32 bits derive from `.name`, which is
why every suggestion for a package named `multi` came back as `0xc5914305xxxxxxxx`
while the upper half was fresh random each time.

Getting a valid one, verified: delete the field (or set a wrong value) and run
`zig build`. It **prints a suggestion and refuses to build**; it does not write
the file for you.

```
build.zig.zon:1:2: error: missing top-level 'fingerprint' field; suggested value: 0xc5914305d29ce8df
```

Paste the suggestion in. When forking a maintained upstream, regenerate it;
keeping the upstream fingerprint is a hostile fork claiming their identity, which
is why the template's comment on that line is worth leaving intact.

`paths` is the inclusion list: only what is listed is hashed and only what is
listed survives on disk for consumers. `""` means the whole build root.

## Dependencies

```bash
zig fetch --save=mathdep https://example.com/pkg.tar.gz
zig fetch --save=mathdep /private/tmp/zig-build-scratch/dep   # a local path works too
```

Verified output written into `.dependencies`:

```zig
.mathdep = .{
    .url = "/private/tmp/zig-build-scratch/dep",
    .hash = "mathdep-0.1.0-aTC-0E8CAAA5nnVpn0L10i7JNWf-pqFn3olTnx7uLg5I",
},
```

The hash is `name-version-<base64>` and it, not the url, is the source of truth;
the url is one mirror. When you point a dependency at a new url, delete the hash
too, otherwise you are asserting the old contents live at the new address and
you get a hash mismatch.

Consume it with `b.dependency(name, args) *Dependency` (`lib/std/Build.zig:2019`),
then `dep.module("exported_name")`, `dep.artifact("name")`, or `dep.path(sub)`.
`.lazy = true` in the zon makes the fetch conditional; pair it with
`b.lazyDependency` / `dependencyLazy` (`:1963`). `zig build --fetch` pre-fetches
everything so later builds need no network.

## Cross-compilation

`-Dtarget=<triple>` is wired by `standardTargetOptions`. Verified on this Mac:

```bash
zig build -Dtarget=x86_64-linux-musl -Doptimize=fast
# zig-out/bin/multi: ELF 64-bit LSB executable, x86-64, statically linked

zig build -Dtarget=aarch64-windows
# zig-out/bin/{multi.exe, multi.pdb}
```

`-Dcpu`, `-Dofmt` and `-Ddynamic-linker` come along with the same helper.
`zig targets` lists everything available.

## Gates

```bash
zig build                 # default step: build + install
zig build test            # whatever `test` step you defined
zig build --summary all   # per-step timings, cache hits, test pass counts
zig build --list-steps    # or -l
zig build --help          # options + steps for THIS project
```

`--summary all` output looks like this and is the fastest way to see whether a
step was cached or actually ran:

```
Build Summary: 6/6 steps succeeded; 2/2 tests passed
test success
+- run test cached
+- run test 2 pass (2 total) 4ms MaxRSS:2M
   +- compile test debug native success 1s MaxRSS:256M
      +- options cached
```

**Never pipe a gate into `head` or `tail` and read `$?`.** The shell reports the
pipe tail's status. During this session `zig build --summary all 2>&1 | tail -20`
reported `EXIT: 0` on a build that had actually failed. Redirect to a file, or
echo the command's own status before piping.

## Verified reference build.zig

Multi-module with a package dependency, build options, passthrough args, a run
step and a test step. Built, tested and run on this toolchain:

```
cd /private/tmp/zig-build-scratch/multi
zig build                      # exit 0
zig build test --summary all   # exit 0, 6/6 steps, 2/2 tests passed
zig build run -Dbanner=hello   # exit 0, printed "hello: 42"
```

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const banner = b.option([]const u8, "banner", "Text printed by the app") orelse "multi";
    const opts = b.addOptions();
    opts.addOption([]const u8, "banner", banner);

    const core_mod = b.addModule("core", .{
        .root_source_file = b.path("src/core.zig"),
        .target = target,
        .optimize = optimize,
    });

    const mathdep = b.dependency("mathdep", .{ .target = target, .optimize = optimize });

    const util_mod = b.addModule("util", .{
        .root_source_file = b.path("src/util.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "core", .module = core_mod },
            .{ .name = "mathdep", .module = mathdep.module("mathdep") },
        },
    });
    util_mod.addOptions("build_options", opts);

    const exe = b.addExecutable(.{
        .name = "multi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "util", .module = util_mod }},
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the app").dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run tests");
    for ([_]*std.Build.Module{ core_mod, util_mod }) |m| {
        const t = b.addTest(.{ .root_module = m });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
```

Note the loop at the bottom: a test binary covers exactly one module's root, so
a package with N modules needs N `addTest` calls hung off one `test` step. The
generated template says the same thing in a comment. The two run steps have no
dependency on each other, so the build runner executes them in parallel.

Matching `build.zig.zon`:

```zig
.{
    .name = .multi,
    .version = "0.1.0",
    .fingerprint = 0xc59143051ec36883,
    .minimum_zig_version = "0.17.0-dev.2018+ab30a0b9a",
    .dependencies = .{
        .mathdep = .{
            .url = "/private/tmp/zig-build-scratch/dep",
            .hash = "mathdep-0.1.0-aTC-0E8CAAA5nnVpn0L10i7JNWf-pqFn3olTnx7uLg5I",
        },
    },
    .paths = .{ "build.zig", "build.zig.zon", "src" },
}
```

## Diagnosing

The build system is entirely userland Zig with no private compiler hooks, so
every step ends up as a `zig build-exe` / `zig test` subcommand. On failure the
runner prints the exact invocation:

```
failed command: .../zig build-exe -Odebug --dep init -Mroot=src/main.zig -Minit=src/root.zig ...
```

Run that command by hand to isolate a build-script problem from a compiler
problem. `lib/std/Build.zig` and `lib/std/Build/` are readable and short; when
an option struct is in question, read the struct rather than guessing.
