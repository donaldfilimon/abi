//! Shell completion generation for ABI CLI.
//!
//! Generates shell completion scripts for bash, zsh, fish, and powershell.

const std = @import("std");
const command_mod = @import("../command.zig");
const context_mod = @import("../framework/context.zig");
const utils = @import("../utils/mod.zig");
const spec = @import("../spec.zig");
const Feature = @import("abi").config.Feature;

pub const meta: command_mod.Meta = .{
    .name = "completions",
    .description = "Generate shell completions (bash, zsh, fish, powershell)",
    .subcommands = &.{ "bash", "zsh", "fish", "powershell", "help" },
};

/// Entry point for the completions command.
pub fn run(ctx: *const context_mod.CommandContext, args: []const [:0]const u8) !void {
    const allocator = ctx.allocator;
    _ = allocator;
    var parser = utils.args.ArgParser.init(std.heap.page_allocator, args);

    if (parser.wantsHelp()) {
        printHelp();
        return;
    }

    const shell = parser.next() orelse {
        printHelp();
        return;
    };

    if (std.mem.eql(u8, shell, "bash")) {
        generateBash();
    } else if (std.mem.eql(u8, shell, "zsh")) {
        generateZsh();
    } else if (std.mem.eql(u8, shell, "fish")) {
        generateFish();
    } else if (std.mem.eql(u8, shell, "powershell") or std.mem.eql(u8, shell, "pwsh")) {
        generatePowerShell();
    } else {
        utils.output.printError("Unknown shell: {s}", .{shell});
        utils.output.printInfo("Supported shells: bash, zsh, fish, powershell", .{});
    }
}

fn generateBash() void {
    emitScriptLine(
        \\# ABI CLI Bash Completion
        \\# Add to ~/.bashrc: source <(abi completions bash)
        \\
        \\_abi_completions() {
        \\    local cur prev commands subcommands
        \\    COMPREPLY=()
        \\    cur="${COMP_WORDS[$COMP_CWORD]}"
        \\    prev="${COMP_WORDS[$COMP_CWORD-1]}"
        \\
        \\    commands="
    );

    for (spec.command_names_with_aliases, 0..) |name, index| {
        if (index > 0) {
            std.debug.print(" ", .{});
        }
        std.debug.print("{s}", .{name});
    }

    emitScriptLine(
        \\"
        \\    case "$prev" in
        \\        abi)
        \\            COMPREPLY=( $(compgen -W "$commands" -- $cur) )
        \\            return 0
        \\            ;;
        \\
    );

    for (spec.command_subcommands) |entry| {
        std.debug.print("        {s})\n", .{entry.command});
        std.debug.print("            COMPREPLY=( $(compgen -W \"", .{});
        printSpaceSeparated(entry.subcommands);
        std.debug.print("\" -- $cur) )\n", .{});
        std.debug.print("            return 0\n            ;;\n", .{});
    }

    emitScriptLine(
        \\        *)
        \\            ;;
        \\    esac
        \\
    );

    emitScriptLine(
        \\    if [[ $cur == --* ]]; then
    );
    std.debug.print(
        "        COMPREPLY=( $(compgen -W \"--help --list-features --help-features --no-color ",
        .{},
    );
    printFeatureGlobalFlags();
    std.debug.print("\" -- $cur) )\n", .{});
    emitScriptLine(
        \\        return 0
        \\    fi
        \\}
        \\
        \\complete -F _abi_completions abi
        \\
    );
}

fn generateZsh() void {
    std.debug.print(
        \\#compdef abi
        \\# ABI CLI Zsh Completion
        \\# Add to ~/.zshrc: source <(abi completions zsh)
        \\
        \\_abi() {{
        \\    local -a commands
        \\    local -a subcommands
        \\
        \\    commands=(
    , .{});

    for (spec.command_infos) |info| {
        std.debug.print("        '{s}:{s}'\n", .{ info.name, info.description });
    }
    for (spec.aliases) |alias| {
        std.debug.print("        '{s}:Alias for {s}'\n", .{ alias.alias, alias.target });
    }

    std.debug.print(
        \\    )
        \\
        \\    _arguments -C \
        \\        '1: :->command' \
        \\        '*: :->args'
        \\
        \\    case $state in
        \\        command)
        \\            _describe -t commands 'abi commands' commands
        \\            ;;
        \\        args)
        \\            case $words[2] in
    , .{});

    for (spec.command_subcommands) |entry| {
        std.debug.print("\\                ", .{});
        printCommandMatchers(entry.command);
        std.debug.print(")\n", .{});
        std.debug.print("\\                subcommands=(", .{});
        for (entry.subcommands) |subcommand| {
            std.debug.print("'{s}' ", .{subcommand});
        }
        std.debug.print(
            \\)
            \\                _describe -t subcommands 'subcommands' subcommands
            \\                ;;
        , .{});
    }

    std.debug.print(
        \\                *)
        \\                    ;;
        \\            esac
        \\            ;;
        \\    esac
        \\}}
        \\
        \\_abi "$@"
        \\
    , .{});
}

fn generateFish() void {
    std.debug.print(
        \\# ABI CLI Fish Completion
        \\# Add to ~/.config/fish/completions/abi.fish
        \\
        \\# Disable file completion by default
        \\complete -c abi -f
        \\
        \\# Main commands
        \\
    , .{});

    for (spec.command_infos) |info| {
        std.debug.print(
            \\complete -c abi -n "__fish_use_subcommand" -a "{s}" -d "{s}"
            \\
        , .{ info.name, info.description });
    }
    for (spec.aliases) |alias| {
        std.debug.print(
            \\complete -c abi -n "__fish_use_subcommand" -a "{s}" -d "Alias for {s}"
            \\
        , .{ alias.alias, alias.target });
    }

    for (spec.command_subcommands) |entry| {
        emitFishSubcommands(entry.command, entry.subcommands);
    }

    std.debug.print(
        \\# Global flags
        \\complete -c abi -l help -d "Show help"
        \\complete -c abi -l enable -d "Enable feature (use --enable-<feature>)"
        \\complete -c abi -l disable -d "Disable feature (use --disable-<feature>)"
        \\complete -c abi -l list-features -d "List features"
        \\complete -c abi -l help-features -d "List features"
        \\complete -c abi -l no-color -d "Disable colored output"
        \\
    , .{});

    emitFishFlagCompletions();
}

fn generatePowerShell() void {
    std.debug.print(
        \\# ABI CLI PowerShell Completion
        \\# Add to your PowerShell profile: abi completions powershell | Out-String | Invoke-Expression
        \\# Or save to a file and dot-source it: . $HOME\abi-completions.ps1
        \\
        \\$script:AbiCommands = @(
    , .{});

    for (spec.command_names_with_aliases) |name| {
        std.debug.print("    '{s}',\n", .{name});
    }

    std.debug.print(
        \\)
        \\
        \\$script:AbiSubcommands = @{{
    , .{});

    for (spec.command_subcommands) |entry| {
        std.debug.print("    '{s}' = @(", .{entry.command});
        for (entry.subcommands, 0..) |subcommand, index| {
            if (index > 0) std.debug.print(", ", .{});
            std.debug.print("'{s}'", .{subcommand});
        }
        std.debug.print(
            \\),
            \\
        , .{});
    }

    emitPowerShellAliasSubcommands();

    std.debug.print(
        \\}}
        \\
        \\$script:AbiGlobalFlags = @(
        \\    '--help',
        \\    '--list-features',
        \\    '--help-features',
        \\    '--no-color',
    , .{});
    emitPowerShellFeatureFlags();
    std.debug.print(
        \\)
        \\
        \\Register-ArgumentCompleter -Native -CommandName abi -ScriptBlock {{
        \\    param($wordToComplete, $commandAst, $cursorPosition)
        \\
        \\    $tokens = $commandAst.CommandElements
        \\    $command = $null
        \\
        \\    # Find the command (skip 'abi' and any global flags)
        \\    for ($i = 1; $i -lt $tokens.Count; $i++) {{
        \\        $token = $tokens[$i].Extent.Text
        \\        if (-not $token.StartsWith('-')) {{
        \\            $command = $token
        \\            break
        \\        }}
        \\    }}
        \\
        \\    # Completing global flags
        \\    if ($wordToComplete.StartsWith('-')) {{
        \\        $script:AbiGlobalFlags | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
        \\            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterName', $_)
        \\        }}
        \\        return
        \\    }}
        \\
        \\    # Completing commands
        \\    if (-not $command -or $command -eq $wordToComplete) {{
        \\        $script:AbiCommands | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
        \\            [System.Management.Automation.CompletionResult]::new($_, $_, 'Command', $_)
        \\        }}
        \\        return
        \\    }}
        \\
        \\    # Completing subcommands
        \\    if ($script:AbiSubcommands.ContainsKey($command)) {{
        \\        $script:AbiSubcommands[$command] | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
        \\            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        \\        }}
        \\    }}
        \\}}
        \\
        \\Write-Host "ABI CLI completions loaded. Press Tab after 'abi' to see commands."
        \\
    , .{});
}

fn emitScriptLine(text: []const u8) void {
    std.debug.print("{s}", .{text});
}

fn printHelp() void {
    const help =
        \\Usage: abi completions <shell>
        \\
        \\Generate shell completion scripts.
        \\
        \\Shells:
        \\  bash        Generate Bash completion script
        \\  zsh         Generate Zsh completion script
        \\  fish        Generate Fish completion script
        \\  powershell  Generate PowerShell completion script
        \\
        \\Installation:
        \\  Bash:  source <(abi completions bash)
        \\         Or add to ~/.bashrc
        \\
        \\  Zsh:   source <(abi completions zsh)
        \\         Or add to ~/.zshrc
        \\
        \\  Fish:  abi completions fish > ~/.config/fish/completions/abi.fish
        \\
        \\  PowerShell:
        \\         # Add to your profile ($PROFILE):
        \\         abi completions powershell | Out-String | Invoke-Expression
        \\
        \\         # Or save to file and dot-source:
        \\         abi completions powershell > $HOME\abi-completions.ps1
        \\         . $HOME\abi-completions.ps1
        \\
    ;
    std.debug.print("{s}", .{help});
}

fn printFeatureGlobalFlags() void {
    const fields = std.meta.fields(Feature);
    inline for (fields, 0..) |field, index| {
        if (index > 0) {
            std.debug.print(" ", .{});
        }
        std.debug.print("--enable-{s} --disable-{s}", .{ field.name, field.name });
    }
}

fn emitPowerShellFeatureFlags() void {
    const fields = std.meta.fields(Feature);
    inline for (fields) |field| {
        std.debug.print("    '--enable-{s}',\n", .{field.name});
        std.debug.print("    '--disable-{s}',\n", .{field.name});
    }
}

fn emitPowerShellAliasSubcommands() void {
    for (spec.aliases) |alias| {
        if (spec.findSubcommands(alias.target) != null) {
            std.debug.print("    '{s}' = $script:AbiSubcommands['{s}']\n", .{ alias.alias, alias.target });
        }
    }
}

fn emitFishFlagCompletions() void {
    const fields = std.meta.fields(Feature);
    inline for (fields) |field| {
        std.debug.print(
            \\complete -c abi -l "enable-{s}" -d "Enable {s} feature at runtime"
            \\
        , .{ field.name, field.name });
        std.debug.print(
            \\complete -c abi -l "disable-{s}" -d "Disable {s} feature at runtime"
            \\
        , .{ field.name, field.name });
    }
}

fn printCommandMatchers(command: []const u8) void {
    std.debug.print("{s}", .{command});
    for (spec.aliases) |alias| {
        if (std.mem.eql(u8, alias.target, command)) {
            std.debug.print("|{s}", .{alias.alias});
        }
    }
}

fn emitFishSubcommands(target: []const u8, subcommands: []const []const u8) void {
    emitFishSubcommandsForCommand(target, subcommands);
    for (spec.aliases) |alias| {
        if (std.mem.eql(u8, alias.target, target)) {
            emitFishSubcommandsForCommand(alias.alias, subcommands);
        }
    }
}

fn emitFishSubcommandsForCommand(command: []const u8, subcommands: []const []const u8) void {
    for (subcommands) |subcommand| {
        std.debug.print(
            \\complete -c abi -n "__fish_seen_subcommand_from {s}" -a "{s}" -d "{s} subcommand"
            \\
        , .{ command, subcommand, subcommand });
    }
}

fn printSpaceSeparated(words: []const []const u8) void {
    for (words, 0..) |word, index| {
        if (index > 0) {
            std.debug.print(" ", .{});
        }
        std.debug.print("{s}", .{word});
    }
}
