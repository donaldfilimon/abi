//! Shell completion generation for ABI CLI.
//!
//! Generates shell completion scripts for bash, zsh, fish, and powershell.
//! Includes:
//! - Main command names and aliases
//! - Subcommand completion with descriptions for group commands
//! - Feature flag completion (--enable-*/--disable-*)
//! - Command-specific option completion (--model, --theme, etc.)
//! - Theme name completion for ui commands

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
    var parser = utils.args.ArgParser.init(ctx.allocator, args);

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

// ═══════════════════════════════════════════════════════════════════════════
// Bash
// ═══════════════════════════════════════════════════════════════════════════

fn generateBash() void {
    emitScriptLine(
        \\# ABI CLI Bash Completion
        \\# Add to ~/.bashrc: source <(abi completions bash)
        \\
        \\_abi_completions() {
        \\    local cur prev words cword commands subcommands cmd_opts
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
        \\
        \\    # Theme names for --theme completion
        \\    local themes="
    );
    for (spec.theme_names, 0..) |name, index| {
        if (index > 0) std.debug.print(" ", .{});
        std.debug.print("{s}", .{name});
    }
    emitScriptLine(
        \\"
        \\
        \\    # Complete theme name after --theme
        \\    if [[ "$prev" == "--theme" ]]; then
        \\        COMPREPLY=( $(compgen -W "$themes" -- $cur) )
        \\        return 0
        \\    fi
        \\
        \\    case "$prev" in
        \\        abi)
        \\            COMPREPLY=( $(compgen -W "$commands" -- $cur) )
        \\            return 0
        \\            ;;
        \\
    );

    // Subcommand completion for group commands
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

    // Command-specific option completion
    emitScriptLine(
        \\    if [[ $cur == --* ]]; then
        \\        # Find the command word (first non-flag after 'abi')
        \\        local cmd=""
        \\        for ((i=1; i < $COMP_CWORD; i++)); do
        \\            if [[ "${COMP_WORDS[$i]}" != --* ]]; then
        \\                cmd="${COMP_WORDS[$i]}"
        \\                break
        \\            fi
        \\        done
        \\
        \\        cmd_opts=""
        \\        case "$cmd" in
    );

    for (spec.command_options) |entry| {
        std.debug.print("            {s})\n", .{entry.command});
        std.debug.print("                cmd_opts=\"", .{});
        for (entry.options, 0..) |opt, i| {
            if (i > 0) std.debug.print(" ", .{});
            std.debug.print("{s}", .{opt.flag});
        }
        std.debug.print("\"\n                ;;\n", .{});
    }

    emitScriptLine(
        \\        esac
        \\
    );

    std.debug.print(
        "        COMPREPLY=( $(compgen -W \"--help --list-features --help-features --no-color $cmd_opts ",
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

// ═══════════════════════════════════════════════════════════════════════════
// Zsh
// ═══════════════════════════════════════════════════════════════════════════

fn generateZsh() void {
    std.debug.print(
        \\#compdef abi
        \\# ABI CLI Zsh Completion
        \\# Add to ~/.zshrc: source <(abi completions zsh)
        \\
        \\_abi() {{
        \\    local -a commands
        \\    local -a subcommands
        \\    local -a cmd_options
        \\
        \\    commands=(
    , .{});

    for (spec.command_infos) |info| {
        std.debug.print("        '{s}:{s}'\n", .{ info.name, escapeZshDescription(info.description) });
    }
    for (spec.aliases) |alias| {
        std.debug.print("        '{s}:Alias for {s}'\n", .{ alias.alias, alias.target });
    }

    std.debug.print(
        \\    )
        \\
        \\    # Theme names for --theme completion
        \\    local -a themes
        \\    themes=(
    , .{});
    for (spec.theme_names) |name| {
        std.debug.print("        '{s}'\n", .{name});
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
        \\            # Complete theme name after --theme
        \\            if [[ "$words[$CURRENT-1]" == "--theme" ]]; then
        \\                _describe -t themes 'themes' themes
        \\                return
        \\            fi
        \\
        \\            case $words[2] in
    , .{});

    // Subcommands with descriptions (using children metadata when available)
    for (spec.command_subcommands) |entry| {
        std.debug.print("                ", .{});
        printCommandMatchers(entry.command);
        std.debug.print(")\n", .{});
        std.debug.print("                    subcommands=(\n", .{});

        // Try to find rich descriptions from children metadata
        if (spec.findSubcommandInfos(entry.command)) |infos| {
            emitZshSubcommandsWithDescriptions(entry.subcommands, infos);
        } else {
            // Fall back to plain subcommand names
            for (entry.subcommands) |subcommand| {
                std.debug.print("                        '{s}'\n", .{subcommand});
            }
        }

        std.debug.print("                    )\n", .{});

        // Also emit command-specific options if available
        if (spec.findCommandOptions(entry.command)) |opts| {
            std.debug.print("                    cmd_options=(\n", .{});
            for (opts) |opt| {
                std.debug.print("                        '{s}[{s}]'\n", .{ opt.flag, escapeZshDescription(opt.description) });
            }
            std.debug.print("                    )\n", .{});
            std.debug.print("                    _describe -t subcommands 'subcommands' subcommands\n", .{});
            std.debug.print("                    _describe -t options 'options' cmd_options\n", .{});
        } else {
            std.debug.print("                    _describe -t subcommands 'subcommands' subcommands\n", .{});
        }
        std.debug.print("                    ;;\n", .{});
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

/// Emit zsh subcommand entries, enriching plain names with descriptions
/// from children metadata where a match exists.
fn emitZshSubcommandsWithDescriptions(
    subcommands: []const []const u8,
    infos: []const spec.SubcommandInfo,
) void {
    for (subcommands) |subcommand| {
        // Try to find a matching description
        var found_desc: ?[]const u8 = null;
        for (infos) |info| {
            if (std.mem.eql(u8, info.name, subcommand)) {
                found_desc = info.description;
                break;
            }
        }
        if (found_desc) |desc| {
            std.debug.print("                        '{s}:{s}'\n", .{ subcommand, escapeZshDescription(desc) });
        } else {
            std.debug.print("                        '{s}'\n", .{subcommand});
        }
    }
}

/// Escape single quotes and colons in zsh description strings.
fn escapeZshDescription(desc: []const u8) []const u8 {
    // For comptime string slices, we cannot allocate -- just return as-is.
    // Descriptions from our spec do not contain problematic characters.
    return desc;
}

// ═══════════════════════════════════════════════════════════════════════════
// Fish
// ═══════════════════════════════════════════════════════════════════════════

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

    // Subcommands with descriptions
    std.debug.print(
        \\
        \\# Subcommands
        \\
    , .{});
    for (spec.command_subcommands) |entry| {
        emitFishSubcommands(entry.command, entry.subcommands);
    }

    // Command-specific options
    std.debug.print(
        \\
        \\# Command-specific options
        \\
    , .{});
    for (spec.command_options) |entry| {
        emitFishCommandOptions(entry.command, entry.options);
    }

    // Theme completion for --theme flag
    std.debug.print(
        \\
        \\# Theme completion (for ui commands that accept --theme)
        \\
    , .{});
    for (spec.theme_names) |name| {
        std.debug.print(
            \\complete -c abi -n "__fish_seen_subcommand_from ui launch start" -a "{s}" -d "{s} theme"
            \\
        , .{ name, name });
    }

    // Global flags
    std.debug.print(
        \\
        \\# Global flags
        \\complete -c abi -l help -d "Show help"
        \\complete -c abi -l list-features -d "List available features"
        \\complete -c abi -l help-features -d "Show feature help"
        \\complete -c abi -l no-color -d "Disable colored output"
        \\
    , .{});

    emitFishFlagCompletions();
}

/// Emit fish subcommand completions, using children descriptions when available.
fn emitFishSubcommands(target: []const u8, subcommands: []const []const u8) void {
    emitFishSubcommandsForCommand(target, subcommands);
    for (spec.aliases) |alias| {
        if (std.mem.eql(u8, alias.target, target)) {
            emitFishSubcommandsForCommand(alias.alias, subcommands);
        }
    }
}

fn emitFishSubcommandsForCommand(command: []const u8, subcommands: []const []const u8) void {
    // Look up rich descriptions from children metadata
    const infos = spec.findSubcommandInfos(command);

    for (subcommands) |subcommand| {
        var desc: []const u8 = subcommand;
        if (infos) |info_list| {
            for (info_list) |info| {
                if (std.mem.eql(u8, info.name, subcommand)) {
                    desc = info.description;
                    break;
                }
            }
        }
        std.debug.print(
            \\complete -c abi -n "__fish_seen_subcommand_from {s}" -a "{s}" -d "{s}"
            \\
        , .{ command, subcommand, desc });
    }
}

/// Emit fish completions for command-specific option flags.
fn emitFishCommandOptions(command: []const u8, options: []const spec.OptionInfo) void {
    for (options) |opt| {
        // Strip leading -- for fish long option format
        const flag_name = if (std.mem.startsWith(u8, opt.flag, "--"))
            opt.flag[2..]
        else
            opt.flag;
        std.debug.print(
            \\complete -c abi -n "__fish_seen_subcommand_from {s}" -l "{s}" -d "{s}"
            \\
        , .{ command, flag_name, opt.description });
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PowerShell
// ═══════════════════════════════════════════════════════════════════════════

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
        \\$script:AbiCommandOptions = @{{
    , .{});

    for (spec.command_options) |entry| {
        std.debug.print("    '{s}' = @(", .{entry.command});
        for (entry.options, 0..) |opt, index| {
            if (index > 0) std.debug.print(", ", .{});
            std.debug.print("'{s}'", .{opt.flag});
        }
        std.debug.print(
            \\),
            \\
        , .{});
    }

    std.debug.print(
        \\}}
        \\
        \\$script:AbiThemes = @(
    , .{});

    for (spec.theme_names) |name| {
        std.debug.print("    '{s}',\n", .{name});
    }

    std.debug.print(
        \\)
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
        \\    $prevToken = $null
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
        \\    # Get previous token for context-sensitive completion
        \\    if ($tokens.Count -ge 2) {{
        \\        $prevToken = $tokens[$tokens.Count - 2].Extent.Text
        \\    }}
        \\
        \\    # Complete theme names after --theme
        \\    if ($prevToken -eq '--theme') {{
        \\        $script:AbiThemes | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
        \\            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        \\        }}
        \\        return
        \\    }}
        \\
        \\    # Completing flags (global + command-specific)
        \\    if ($wordToComplete.StartsWith('-')) {{
        \\        $allFlags = @() + $script:AbiGlobalFlags
        \\        if ($command -and $script:AbiCommandOptions.ContainsKey($command)) {{
        \\            $allFlags += $script:AbiCommandOptions[$command]
        \\        }}
        \\        $allFlags | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
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

// ═══════════════════════════════════════════════════════════════════════════
// Shared helpers
// ═══════════════════════════════════════════════════════════════════════════

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
    utils.output.print("{s}", .{help});
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

fn printSpaceSeparated(words: []const []const u8) void {
    for (words, 0..) |word, index| {
        if (index > 0) {
            std.debug.print(" ", .{});
        }
        std.debug.print("{s}", .{word});
    }
}

test {
    std.testing.refAllDecls(@This());
}
