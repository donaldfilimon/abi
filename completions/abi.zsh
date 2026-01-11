#compdef abi
# Zsh completion script for ABI framework CLI
# Installation:
#   fpath=(/path/to/completions $fpath)
#   autoload -Uz compinit && compinit

_abi() {
    local -a commands
    local -a db_commands
    local -a gpu_commands
    local -a network_commands
    local -a config_commands

    commands=(
        'db:Database operations (add, query, stats, optimize, backup)'
        'agent:Run AI agent (interactive or one-shot)'
        'config:Configuration management (init, show, validate)'
        'explore:Search and explore codebase'
        'gpu:GPU commands (backends, devices, summary)'
        'network:Manage network registry (list, register, status)'
        'simd:Run SIMD performance demo'
        'system-info:Show system and framework status'
        'version:Show framework version'
        'help:Show help message'
    )

    db_commands=(
        'add:Add a vector to the database'
        'query:Query vectors by similarity'
        'stats:Show database statistics'
        'optimize:Optimize database performance'
        'backup:Backup database to file'
        'restore:Restore database from file'
        'index:Manage vector indexes'
        'help:Show database help'
    )

    gpu_commands=(
        'backends:List GPU backends and build flags'
        'devices:List detected GPU devices'
        'list:Alias for devices'
        'summary:Show GPU module summary'
        'default:Show default GPU device'
        'status:Show CUDA native/fallback status'
        'help:Show GPU help'
    )

    network_commands=(
        'status:Show network config and node count'
        'list:List registered nodes'
        'nodes:List registered nodes (alias)'
        'register:Register or update a node'
        'unregister:Remove a node'
        'touch:Update node heartbeat timestamp'
        'set-status:Set node status (healthy, degraded, offline)'
        'help:Show network help'
    )

    config_commands=(
        'init:Generate a default configuration file'
        'show:Display configuration (default or from file)'
        'validate:Validate a configuration file'
        'env:List environment variables'
        'help:Show config help'
    )

    _arguments -C \
        '1: :->command' \
        '*:: :->args'

    case $state in
        command)
            _describe -t commands 'abi commands' commands
            ;;
        args)
            case $words[1] in
                db)
                    if (( CURRENT == 2 )); then
                        _describe -t db-commands 'db commands' db_commands
                    else
                        case $words[2] in
                            backup|restore)
                                _files
                                ;;
                        esac
                    fi
                    ;;
                gpu)
                    if (( CURRENT == 2 )); then
                        _describe -t gpu-commands 'gpu commands' gpu_commands
                    fi
                    ;;
                network)
                    if (( CURRENT == 2 )); then
                        _describe -t network-commands 'network commands' network_commands
                    else
                        case $words[2] in
                            set-status)
                                if (( CURRENT == 4 )); then
                                    local -a statuses
                                    statuses=('healthy' 'degraded' 'offline')
                                    _describe -t statuses 'node status' statuses
                                fi
                                ;;
                        esac
                    fi
                    ;;
                config)
                    if (( CURRENT == 2 )); then
                        _describe -t config-commands 'config commands' config_commands
                    else
                        case $words[2] in
                            init)
                                _arguments \
                                    '(-o --output)'{-o,--output}'[Output file path]:file:_files -g "*.json"'
                                ;;
                            show)
                                _arguments \
                                    '(-f --format)'{-f,--format}'[Output format]:format:(human json)' \
                                    '*:config file:_files -g "*.json"'
                                ;;
                            validate)
                                _files -g "*.json"
                                ;;
                        esac
                    fi
                    ;;
                explore)
                    _arguments \
                        '(-l --level)'{-l,--level}'[Exploration depth]:level:(quick medium thorough deep)' \
                        '(-f --format)'{-f,--format}'[Output format]:format:(human json compact yaml)' \
                        '(-i --include)'{-i,--include}'[Include files matching pattern]:pattern:' \
                        '(-e --exclude)'{-e,--exclude}'[Exclude files matching pattern]:pattern:' \
                        '(-c --case-sensitive)'{-c,--case-sensitive}'[Match case sensitively]' \
                        '(-r --regex)'{-r,--regex}'[Treat query as regex pattern]' \
                        '--path[Root directory to search]:directory:_directories' \
                        '--max-files[Maximum files to scan]:number:' \
                        '--max-depth[Maximum directory depth]:number:' \
                        '--timeout[Timeout in milliseconds]:number:' \
                        '(-h --help)'{-h,--help}'[Show help message]' \
                        '*:query:'
                    ;;
                agent)
                    _arguments \
                        '--name[Agent name]:name:' \
                        '(-m --message)'{-m,--message}'[Message to send]:message:' \
                        '--interactive[Run in interactive mode]' \
                        '(-h --help)'{-h,--help}'[Show help message]'
                    ;;
            esac
            ;;
    esac
}

_abi "$@"
