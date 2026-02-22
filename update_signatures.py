#!/usr/bin/env python3
"""
Migrate CLI command handlers to CommandContext signatures.

This helper rewrites files under tools/cli/commands and tools/cli/command.zig.
It is intentionally conservative and idempotent so it can be re-run safely.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
COMMANDS_ROOT = REPO_ROOT / "tools" / "cli" / "commands"
COMMAND_PROTOCOL = REPO_ROOT / "tools" / "cli" / "command.zig"


FUNC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"((?:pub )?fn \w+)\(allocator: std\.mem\.Allocator, io: std\.Io, (args|_): \[\]const \[:0\]const u8\) !void \{"
        ),
        r"\1(ctx: *const context_mod.CommandContext, \2: []const [:0]const u8) !void {\n    const allocator = ctx.allocator;\n    const io = ctx.io;",
    ),
    (
        re.compile(
            r"((?:pub )?fn \w+)\(allocator: std\.mem\.Allocator, _: std\.Io, (args|_): \[\]const \[:0\]const u8\) !void \{"
        ),
        r"\1(ctx: *const context_mod.CommandContext, \2: []const [:0]const u8) !void {\n    const allocator = ctx.allocator;",
    ),
    (
        re.compile(
            r"((?:pub )?fn \w+)\(_: std\.mem\.Allocator, io: std\.Io, (args|_): \[\]const \[:0\]const u8\) !void \{"
        ),
        r"\1(ctx: *const context_mod.CommandContext, \2: []const [:0]const u8) !void {\n    const io = ctx.io;",
    ),
    (
        re.compile(
            r"((?:pub )?fn \w+)\(_: std\.mem\.Allocator, _: std\.Io, (args|_): \[\]const \[:0\]const u8\) !void \{"
        ),
        r"\1(ctx: *const context_mod.CommandContext, \2: []const [:0]const u8) !void {\n    _ = ctx;",
    ),
    (
        re.compile(
            r"((?:pub )?fn \w+)\(allocator: std\.mem\.Allocator, (args|_): \[\]const \[:0\]const u8\) !void \{"
        ),
        r"\1(ctx: *const context_mod.CommandContext, \2: []const [:0]const u8) !void {\n    const allocator = ctx.allocator;",
    ),
    (
        re.compile(
            r"((?:pub )?fn \w+)\(_: std\.mem\.Allocator, (args|_): \[\]const \[:0\]const u8\) !void \{"
        ),
        r"\1(ctx: *const context_mod.CommandContext, \2: []const [:0]const u8) !void {\n    _ = ctx;",
    ),
]


def context_import_path(path: Path) -> str:
    rel = path.relative_to(COMMANDS_ROOT)
    depth = len(rel.parts) - 1
    return "../framework/context.zig" if depth == 0 else "../../framework/context.zig"


def ensure_context_import(path: Path, content: str) -> str:
    if "context_mod.CommandContext" not in content:
        return content
    if 'const context_mod = @import("' in content:
        return content

    import_line = f'const context_mod = @import("{context_import_path(path)}");\n'

    command_mod_match = re.search(r'const command_mod = @import\([^\n]+\);\n', content)
    if command_mod_match:
        idx = command_mod_match.end()
        return content[:idx] + import_line + content[idx:]

    std_match = re.search(r'const std = @import\("std"\);\n', content)
    if std_match:
        idx = std_match.end()
        return content[:idx] + import_line + content[idx:]

    return import_line + content


def transform_command_file(path: Path) -> bool:
    content = path.read_text()
    original = content

    # Remove legacy mode selectors.
    content = re.sub(r"\n\s*\.io_mode\s*=\s*\.[a-zA-Z_]+,", "", content)

    # Normalize child handler assignment from tagged union to plain fn ptr.
    content = re.sub(
        r"\.handler\s*=\s*\.\{\s*\.basic\s*=\s*([^}]+?)\s*\},",
        r".handler = \1,",
        content,
    )
    content = re.sub(
        r"\.handler\s*=\s*\.\{\s*\.io\s*=\s*([^}]+?)\s*\},",
        r".handler = \1,",
        content,
    )

    for pattern, replacement in FUNC_PATTERNS:
        content = pattern.sub(replacement, content)

    content = ensure_context_import(path, content)

    if content != original:
        path.write_text(content)
        return True
    return False


def transform_command_protocol(path: Path) -> bool:
    content = path.read_text()
    original = content

    content = re.sub(r"\npub const CommandIoMode = types\.CommandIoMode;\n", "\n", content)
    content = re.sub(r"\n\s*io_mode: CommandIoMode = \.basic,\n", "\n", content)
    content = re.sub(
        r"\n\s*\.handler = switch \(m\.io_mode\) \{\n\s*\.basic => \.\{ \.basic = Module\.run \},\n\s*\.io => \.\{ \.io = Module\.run \},\n\s*\},",
        "\n        .handler = Module.run,",
        content,
    )
    content = re.sub(
        r"\n\s*try std\.testing\.expectEqual\(CommandIoMode\.basic, m\.io_mode\);\n",
        "\n",
        content,
    )

    if content != original:
        path.write_text(content)
        return True
    return False


def main() -> None:
    changed = 0
    for file_path in sorted(COMMANDS_ROOT.rglob("*.zig")):
        if transform_command_file(file_path):
            changed += 1

    if COMMAND_PROTOCOL.exists() and transform_command_protocol(COMMAND_PROTOCOL):
        changed += 1

    print(f"Updated {changed} file(s).")


if __name__ == "__main__":
    main()
