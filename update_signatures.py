import os
import glob
import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Skip files that don't look like commands
    if 'pub fn run(' not in content:
        return

    # Determine import path for context_mod based on depth
    depth = filepath.count('/') - filepath.find('tools/cli/commands/')
    if depth == 1:
        context_import = 'const context_mod = @import("../framework/context.zig");'
    elif depth == 2:
        context_import = 'const context_mod = @import("../../framework/context.zig");'
    else:
        context_import = 'const context_mod = @import("../framework/context.zig");' # fallback

    # Add import if missing
    if 'context_mod' not in content:
        # insert after first @import
        content = re.sub(r'(const std = @import\("std"\);)', r'\1
' + context_import, content, count=1)

    # Remove io_mode from meta
    content = re.sub(r'\s*\.io_mode\s*=\s*\.[a-z]+,', '', content)

    # Update pub fn run signatures
    content = re.sub(r'pub fn run\(allocator: std\.mem\.Allocator, args: \[\]const \[:0\]const u8\) !void \{',
                     r'pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;', content)

    content = re.sub(r'pub fn run\(_: std\.mem\.Allocator, args: \[\]const \[:0\]const u8\) !void \{',
                     r'pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = ctx;', content)
                     
    content = re.sub(r'pub fn run\(allocator: std\.mem\.Allocator, io: std\.Io, args: \[\]const \[:0\]const u8\) !void \{',
                     r'pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    const io = ctx.io;', content)
                     
    content = re.sub(r'pub fn run\(_: std\.mem\.Allocator, _: std\.Io, args: \[\]const \[:0\]const u8\) !void \{',
                     r'pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    _ = ctx;', content)

    # For child commands or stubs that might be named differently but have same signature
    content = re.sub(r'fn (\w+)\(allocator: std\.mem\.Allocator, args: \[\]const \[:0\]const u8\) !void \{',
                     r'fn \1(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;', content)
                     
    with open(filepath, 'w') as f:
        f.write(content)

for root, _, files in os.walk('tools/cli/commands'):
    for file in files:
        if file.endswith('.zig'):
            update_file(os.path.join(root, file))

# Also update tools/cli/command.zig
with open('tools/cli/command.zig', 'r') as f:
    content = f.read()

content = re.sub(r'pub const CommandIoMode = types\.CommandIoMode;
', '', content)
content = re.sub(r'\s*\.io_mode\s*=\s*\.[a-z]+,', '', content)
content = re.sub(r'io_mode: CommandIoMode = \.basic,', '', content)

content = re.sub(r'\s*\.handler = switch \(m\.io_mode\) \{\s*\.basic => \.\{ \.basic = Module\.run \},\s*\.io => \.\{ \.io = Module\.run \},\s*\},
',
                 '
        .handler = Module.run,
', content)

content = re.sub(r'pub fn run\(_: std\.mem\.Allocator, _: \[\]const \[:0\]const u8\) !void \{\}',
                 r'pub fn run(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}', content)
                 
content = re.sub(r'pub fn run\(_: std\.mem\.Allocator, _: std\.Io, _: \[\]const \[:0\]const u8\) !void \{\}',
                 r'pub fn run(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}', content)

content = re.sub(r'fn stubChild1\(_: std\.mem\.Allocator, _: \[\]const \[:0\]const u8\) !void \{\}',
                 r'fn stubChild1(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}', content)
                 
content = re.sub(r'fn stubChild2\(_: std\.mem\.Allocator, _: \[\]const \[:0\]const u8\) !void \{\}',
                 r'fn stubChild2(_: *const context_mod.CommandContext, _: []const [:0]const u8) !void {}', content)

# Remove the switch(d.handler) block in tests
content = re.sub(r'\s*// Handler should be \.basic variant\s*switch \(d\.handler\) \{\s*\.basic => \{\},\s*\.io => return error\.TestUnexpectedResult,\s*\}', '', content)
content = re.sub(r'\s*// Handler should be \.io variant\s*switch \(d\.handler\) \{\s*\.io => \{\},\s*\.basic => return error\.TestUnexpectedResult,\s*\}', '', content)
content = re.sub(r'\s*// Verify child handlers are \.basic variant\s*switch \(d\.children\[0\]\.handler\) \{\s*\.basic => \{\},\s*\.io => return error\.TestUnexpectedResult,\s*\}', '', content)
content = re.sub(r'\s*\.\{ \.name = "child-a", \.description = "First child", \.handler = \.\{ \.basic = stubChild1 \} \},',
                 r'            .{ .name = "child-a", .description = "First child", .handler = stubChild1 },', content)
content = re.sub(r'\s*\.\{ \.name = "child-b", \.description = "Second child", \.handler = \.\{ \.basic = stubChild2 \} \},',
                 r'            .{ .name = "child-b", .description = "Second child", .handler = stubChild2 },', content)

if 'context_mod' not in content:
    content = re.sub(r'(const std = @import\("std"\);)', r'\1
const context_mod = @import("framework/context.zig");', content, count=1)

with open('tools/cli/command.zig', 'w') as f:
    f.write(content)

print("Updated signatures!")
