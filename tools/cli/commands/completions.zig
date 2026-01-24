//! Shell completion generation for ABI CLI.
//!
//! Generates shell completion scripts for bash, zsh, and fish.

const std = @import("std");
const utils = @import("../utils/mod.zig");

/// Entry point for the completions command.
pub fn run(allocator: std.mem.Allocator, args: []const [:0]const u8) !void {
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
    const script =
        \\# ABI CLI Bash Completion
        \\# Add to ~/.bashrc: source <(abi completions bash)
        \\
        \\_abi_completions() {
        \\    local cur prev commands subcommands
        \\    COMPREPLY=()
        \\    cur="${COMP_WORDS[COMP_CWORD]}"
        \\    prev="${COMP_WORDS[COMP_CWORD-1]}"
        \\
        \\    commands="db agent bench config discord embed explore gpu llm network simd system-info train task tui completions version help"
        \\
        \\    case "${prev}" in
        \\        abi)
        \\            COMPREPLY=( $(compgen -W "${commands}" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        db)
        \\            COMPREPLY=( $(compgen -W "add query stats optimize backup restore" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        agent)
        \\            COMPREPLY=( $(compgen -W "--message --persona --debug help" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        gpu)
        \\            COMPREPLY=( $(compgen -W "backends devices summary default" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        llm)
        \\            COMPREPLY=( $(compgen -W "chat generate info bench download" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        train)
        \\            COMPREPLY=( $(compgen -W "run resume info" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        network)
        \\            COMPREPLY=( $(compgen -W "list register status" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        config)
        \\            COMPREPLY=( $(compgen -W "init show validate" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        bench)
        \\            COMPREPLY=( $(compgen -W "all simd memory ai quick" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        completions)
        \\            COMPREPLY=( $(compgen -W "bash zsh fish" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        task)
        \\            COMPREPLY=( $(compgen -W "add list done stats" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        discord)
        \\            COMPREPLY=( $(compgen -W "status guilds send commands" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        embed)
        \\            COMPREPLY=( $(compgen -W "--provider openai mistral cohere ollama" -- ${cur}) )
        \\            return 0
        \\            ;;
        \\        *)
        \\            ;;
        \\    esac
        \\
        \\    # Global flags
        \\    if [[ ${cur} == --* ]]; then
        \\        COMPREPLY=( $(compgen -W "--help --enable-gpu --disable-gpu --enable-ai --disable-ai --enable-database --disable-database --list-features" -- ${cur}) )
        \\        return 0
        \\    fi
        \\}
        \\
        \\complete -F _abi_completions abi
        \\
    ;
    std.debug.print("{s}", .{script});
}

fn generateZsh() void {
    const script =
        \\#compdef abi
        \\# ABI CLI Zsh Completion
        \\# Add to ~/.zshrc: source <(abi completions zsh)
        \\
        \\_abi() {
        \\    local -a commands
        \\    local -a subcommands
        \\
        \\    commands=(
        \\        'db:Database operations'
        \\        'agent:Interactive AI assistant'
        \\        'bench:Performance benchmarks'
        \\        'config:Configuration management'
        \\        'discord:Discord bot integration'
        \\        'embed:Generate embeddings'
        \\        'explore:Search the codebase'
        \\        'gpu:GPU devices and backends'
        \\        'llm:Local LLM inference'
        \\        'network:Cluster management'
        \\        'simd:SIMD performance demo'
        \\        'system-info:System and framework status'
        \\        'train:Training pipeline'
        \\        'task:Task management'
        \\        'tui:Interactive terminal UI'
        \\        'completions:Generate shell completions'
        \\        'version:Show version'
        \\        'help:Show help'
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
        \\                db)
        \\                    subcommands=('add' 'query' 'stats' 'optimize' 'backup' 'restore')
        \\                    _describe -t subcommands 'db subcommands' subcommands
        \\                    ;;
        \\                gpu)
        \\                    subcommands=('backends' 'devices' 'summary' 'default')
        \\                    _describe -t subcommands 'gpu subcommands' subcommands
        \\                    ;;
        \\                llm)
        \\                    subcommands=('chat' 'generate' 'info' 'bench' 'download')
        \\                    _describe -t subcommands 'llm subcommands' subcommands
        \\                    ;;
        \\                train)
        \\                    subcommands=('run' 'resume' 'info')
        \\                    _describe -t subcommands 'train subcommands' subcommands
        \\                    ;;
        \\                network)
        \\                    subcommands=('list' 'register' 'status')
        \\                    _describe -t subcommands 'network subcommands' subcommands
        \\                    ;;
        \\                config)
        \\                    subcommands=('init' 'show' 'validate')
        \\                    _describe -t subcommands 'config subcommands' subcommands
        \\                    ;;
        \\                bench)
        \\                    subcommands=('all' 'simd' 'memory' 'ai' 'quick')
        \\                    _describe -t subcommands 'bench subcommands' subcommands
        \\                    ;;
        \\                task)
        \\                    subcommands=('add' 'list' 'done' 'stats')
        \\                    _describe -t subcommands 'task subcommands' subcommands
        \\                    ;;
        \\                discord)
        \\                    subcommands=('status' 'guilds' 'send' 'commands')
        \\                    _describe -t subcommands 'discord subcommands' subcommands
        \\                    ;;
        \\                completions)
        \\                    subcommands=('bash' 'zsh' 'fish')
        \\                    _describe -t subcommands 'shell types' subcommands
        \\                    ;;
        \\            esac
        \\            ;;
        \\    esac
        \\}
        \\
        \\_abi "$@"
        \\
    ;
    std.debug.print("{s}", .{script});
}

fn generateFish() void {
    const script =
        \\# ABI CLI Fish Completion
        \\# Add to ~/.config/fish/completions/abi.fish
        \\
        \\# Disable file completion by default
        \\complete -c abi -f
        \\
        \\# Main commands
        \\complete -c abi -n "__fish_use_subcommand" -a "db" -d "Database operations"
        \\complete -c abi -n "__fish_use_subcommand" -a "agent" -d "Interactive AI assistant"
        \\complete -c abi -n "__fish_use_subcommand" -a "bench" -d "Performance benchmarks"
        \\complete -c abi -n "__fish_use_subcommand" -a "config" -d "Configuration management"
        \\complete -c abi -n "__fish_use_subcommand" -a "discord" -d "Discord bot integration"
        \\complete -c abi -n "__fish_use_subcommand" -a "embed" -d "Generate embeddings"
        \\complete -c abi -n "__fish_use_subcommand" -a "explore" -d "Search the codebase"
        \\complete -c abi -n "__fish_use_subcommand" -a "gpu" -d "GPU devices and backends"
        \\complete -c abi -n "__fish_use_subcommand" -a "llm" -d "Local LLM inference"
        \\complete -c abi -n "__fish_use_subcommand" -a "network" -d "Cluster management"
        \\complete -c abi -n "__fish_use_subcommand" -a "simd" -d "SIMD performance demo"
        \\complete -c abi -n "__fish_use_subcommand" -a "system-info" -d "System and framework status"
        \\complete -c abi -n "__fish_use_subcommand" -a "train" -d "Training pipeline"
        \\complete -c abi -n "__fish_use_subcommand" -a "task" -d "Task management"
        \\complete -c abi -n "__fish_use_subcommand" -a "tui" -d "Interactive terminal UI"
        \\complete -c abi -n "__fish_use_subcommand" -a "completions" -d "Generate shell completions"
        \\complete -c abi -n "__fish_use_subcommand" -a "version" -d "Show version"
        \\complete -c abi -n "__fish_use_subcommand" -a "help" -d "Show help"
        \\
        \\# db subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from db" -a "add" -d "Add data"
        \\complete -c abi -n "__fish_seen_subcommand_from db" -a "query" -d "Query data"
        \\complete -c abi -n "__fish_seen_subcommand_from db" -a "stats" -d "Show statistics"
        \\complete -c abi -n "__fish_seen_subcommand_from db" -a "optimize" -d "Optimize database"
        \\complete -c abi -n "__fish_seen_subcommand_from db" -a "backup" -d "Backup database"
        \\complete -c abi -n "__fish_seen_subcommand_from db" -a "restore" -d "Restore database"
        \\
        \\# gpu subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from gpu" -a "backends" -d "List GPU backends"
        \\complete -c abi -n "__fish_seen_subcommand_from gpu" -a "devices" -d "List GPU devices"
        \\complete -c abi -n "__fish_seen_subcommand_from gpu" -a "summary" -d "GPU summary"
        \\complete -c abi -n "__fish_seen_subcommand_from gpu" -a "default" -d "Show default device"
        \\
        \\# llm subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from llm" -a "chat" -d "Interactive chat"
        \\complete -c abi -n "__fish_seen_subcommand_from llm" -a "generate" -d "Generate text"
        \\complete -c abi -n "__fish_seen_subcommand_from llm" -a "info" -d "Model info"
        \\complete -c abi -n "__fish_seen_subcommand_from llm" -a "bench" -d "Benchmark"
        \\complete -c abi -n "__fish_seen_subcommand_from llm" -a "download" -d "Download model"
        \\
        \\# train subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from train" -a "run" -d "Run training"
        \\complete -c abi -n "__fish_seen_subcommand_from train" -a "resume" -d "Resume training"
        \\complete -c abi -n "__fish_seen_subcommand_from train" -a "info" -d "Training info"
        \\
        \\# network subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from network" -a "list" -d "List nodes"
        \\complete -c abi -n "__fish_seen_subcommand_from network" -a "register" -d "Register node"
        \\complete -c abi -n "__fish_seen_subcommand_from network" -a "status" -d "Network status"
        \\
        \\# config subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from config" -a "init" -d "Initialize config"
        \\complete -c abi -n "__fish_seen_subcommand_from config" -a "show" -d "Show config"
        \\complete -c abi -n "__fish_seen_subcommand_from config" -a "validate" -d "Validate config"
        \\
        \\# bench subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from bench" -a "all" -d "Run all benchmarks"
        \\complete -c abi -n "__fish_seen_subcommand_from bench" -a "simd" -d "SIMD benchmarks"
        \\complete -c abi -n "__fish_seen_subcommand_from bench" -a "memory" -d "Memory benchmarks"
        \\complete -c abi -n "__fish_seen_subcommand_from bench" -a "ai" -d "AI benchmarks"
        \\complete -c abi -n "__fish_seen_subcommand_from bench" -a "quick" -d "Quick benchmarks"
        \\
        \\# task subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from task" -a "add" -d "Add task"
        \\complete -c abi -n "__fish_seen_subcommand_from task" -a "list" -d "List tasks"
        \\complete -c abi -n "__fish_seen_subcommand_from task" -a "done" -d "Mark done"
        \\complete -c abi -n "__fish_seen_subcommand_from task" -a "stats" -d "Task stats"
        \\
        \\# discord subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from discord" -a "status" -d "Bot status"
        \\complete -c abi -n "__fish_seen_subcommand_from discord" -a "guilds" -d "List guilds"
        \\complete -c abi -n "__fish_seen_subcommand_from discord" -a "send" -d "Send message"
        \\complete -c abi -n "__fish_seen_subcommand_from discord" -a "commands" -d "Bot commands"
        \\
        \\# completions subcommands
        \\complete -c abi -n "__fish_seen_subcommand_from completions" -a "bash" -d "Bash completions"
        \\complete -c abi -n "__fish_seen_subcommand_from completions" -a "zsh" -d "Zsh completions"
        \\complete -c abi -n "__fish_seen_subcommand_from completions" -a "fish" -d "Fish completions"
        \\
        \\# Global flags
        \\complete -c abi -l help -d "Show help"
        \\complete -c abi -l enable-gpu -d "Enable GPU feature"
        \\complete -c abi -l disable-gpu -d "Disable GPU feature"
        \\complete -c abi -l enable-ai -d "Enable AI feature"
        \\complete -c abi -l disable-ai -d "Disable AI feature"
        \\complete -c abi -l enable-database -d "Enable database feature"
        \\complete -c abi -l disable-database -d "Disable database feature"
        \\complete -c abi -l list-features -d "List features"
        \\
    ;
    std.debug.print("{s}", .{script});
}

fn generatePowerShell() void {
    const script =
        \\# ABI CLI PowerShell Completion
        \\# Add to your PowerShell profile: abi completions powershell | Out-String | Invoke-Expression
        \\# Or save to a file and dot-source it: . $HOME\abi-completions.ps1
        \\
        \\$script:AbiCommands = @(
        \\    'db', 'agent', 'bench', 'config', 'discord', 'embed', 'explore',
        \\    'gpu', 'gpu-dashboard', 'llm', 'network', 'simd', 'system-info',
        \\    'multi-agent', 'train', 'convert', 'task', 'tui', 'plugins',
        \\    'profile', 'completions', 'version', 'help'
        \\)
        \\
        \\$script:AbiSubcommands = @{
        \\    'db' = @('add', 'query', 'stats', 'optimize', 'backup', 'restore')
        \\    'agent' = @('--message', '--persona', '--debug', 'help')
        \\    'gpu' = @('backends', 'devices', 'summary', 'default')
        \\    'llm' = @('chat', 'generate', 'info', 'bench', 'download')
        \\    'train' = @('run', 'resume', 'info')
        \\    'network' = @('list', 'register', 'status')
        \\    'config' = @('init', 'show', 'validate')
        \\    'bench' = @('all', 'simd', 'memory', 'ai', 'quick')
        \\    'completions' = @('bash', 'zsh', 'fish', 'powershell')
        \\    'task' = @('add', 'list', 'done', 'stats')
        \\    'discord' = @('status', 'guilds', 'send', 'commands')
        \\    'embed' = @('--provider', 'openai', 'mistral', 'cohere', 'ollama')
        \\    'plugins' = @('list', 'info', 'enable', 'disable', 'search')
        \\    'profile' = @('show', 'list', 'create', 'switch', 'delete', 'set', 'get', 'api-key', 'export', 'import')
        \\}
        \\
        \\$script:AbiGlobalFlags = @(
        \\    '--help', '--list-features',
        \\    '--enable-gpu', '--disable-gpu',
        \\    '--enable-ai', '--disable-ai',
        \\    '--enable-database', '--disable-database',
        \\    '--enable-network', '--disable-network'
        \\)
        \\
        \\Register-ArgumentCompleter -Native -CommandName abi -ScriptBlock {
        \\    param($wordToComplete, $commandAst, $cursorPosition)
        \\
        \\    $tokens = $commandAst.CommandElements
        \\    $command = $null
        \\
        \\    # Find the command (skip 'abi' and any global flags)
        \\    for ($i = 1; $i -lt $tokens.Count; $i++) {
        \\        $token = $tokens[$i].Extent.Text
        \\        if (-not $token.StartsWith('-')) {
        \\            $command = $token
        \\            break
        \\        }
        \\    }
        \\
        \\    # Completing global flags
        \\    if ($wordToComplete.StartsWith('-')) {
        \\        $script:AbiGlobalFlags | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        \\            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterName', $_)
        \\        }
        \\        return
        \\    }
        \\
        \\    # Completing commands
        \\    if (-not $command -or $command -eq $wordToComplete) {
        \\        $script:AbiCommands | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        \\            [System.Management.Automation.CompletionResult]::new($_, $_, 'Command', $_)
        \\        }
        \\        return
        \\    }
        \\
        \\    # Completing subcommands
        \\    if ($script:AbiSubcommands.ContainsKey($command)) {
        \\        $script:AbiSubcommands[$command] | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
        \\            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        \\        }
        \\    }
        \\}
        \\
        \\Write-Host "ABI CLI completions loaded. Press Tab after 'abi' to see commands."
        \\
    ;
    std.debug.print("{s}", .{script});
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
