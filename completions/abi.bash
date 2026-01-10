#!/bin/bash
# Bash completion script for ABI framework CLI
# Installation:
#   source /path/to/abi.bash
#   or copy to /etc/bash_completion.d/abi

_abi_completions() {
    local cur prev words cword
    _init_completion || return

    local commands="db agent config explore gpu network simd system-info version help"
    local db_commands="add query stats optimize backup restore index help"
    local gpu_commands="backends devices list summary default status help"
    local network_commands="status list nodes register unregister touch set-status help"
    local config_commands="init show validate env help"
    local explore_options="--level --format --include --exclude --case-sensitive --regex --path --max-files --max-depth --timeout --help"
    local agent_options="--name --message --interactive --help"

    case "${cword}" in
        1)
            COMPREPLY=($(compgen -W "${commands}" -- "${cur}"))
            ;;
        2)
            case "${prev}" in
                db)
                    COMPREPLY=($(compgen -W "${db_commands}" -- "${cur}"))
                    ;;
                gpu)
                    COMPREPLY=($(compgen -W "${gpu_commands}" -- "${cur}"))
                    ;;
                network)
                    COMPREPLY=($(compgen -W "${network_commands}" -- "${cur}"))
                    ;;
                config)
                    COMPREPLY=($(compgen -W "${config_commands}" -- "${cur}"))
                    ;;
                explore)
                    COMPREPLY=($(compgen -W "${explore_options}" -- "${cur}"))
                    ;;
                agent)
                    COMPREPLY=($(compgen -W "${agent_options}" -- "${cur}"))
                    ;;
            esac
            ;;
        *)
            case "${words[1]}" in
                explore)
                    case "${prev}" in
                        --level|-l)
                            COMPREPLY=($(compgen -W "quick medium thorough deep" -- "${cur}"))
                            ;;
                        --format|-f)
                            COMPREPLY=($(compgen -W "human json compact yaml" -- "${cur}"))
                            ;;
                        --path)
                            COMPREPLY=($(compgen -d -- "${cur}"))
                            ;;
                        --include|--exclude|-i|-e)
                            COMPREPLY=($(compgen -W "*.zig *.json *.md *.txt" -- "${cur}"))
                            ;;
                        *)
                            COMPREPLY=($(compgen -W "${explore_options}" -- "${cur}"))
                            ;;
                    esac
                    ;;
                config)
                    case "${words[2]}" in
                        init)
                            case "${prev}" in
                                --output|-o)
                                    COMPREPLY=($(compgen -f -X '!*.json' -- "${cur}"))
                                    ;;
                                *)
                                    COMPREPLY=($(compgen -W "--output -o" -- "${cur}"))
                                    ;;
                            esac
                            ;;
                        show)
                            case "${prev}" in
                                --format|-f)
                                    COMPREPLY=($(compgen -W "human json" -- "${cur}"))
                                    ;;
                                *)
                                    COMPREPLY=($(compgen -f -X '!*.json' -- "${cur}"))
                                    ;;
                            esac
                            ;;
                        validate)
                            COMPREPLY=($(compgen -f -X '!*.json' -- "${cur}"))
                            ;;
                    esac
                    ;;
                db)
                    case "${words[2]}" in
                        add)
                            # Vector input expected
                            ;;
                        query)
                            # Query vector expected
                            ;;
                        backup|restore)
                            COMPREPLY=($(compgen -f -- "${cur}"))
                            ;;
                    esac
                    ;;
                network)
                    case "${words[2]}" in
                        set-status)
                            if [[ "${cword}" -eq 4 ]]; then
                                COMPREPLY=($(compgen -W "healthy degraded offline" -- "${cur}"))
                            fi
                            ;;
                    esac
                    ;;
                agent)
                    case "${prev}" in
                        --name)
                            COMPREPLY=()
                            ;;
                        --message|-m)
                            COMPREPLY=()
                            ;;
                        *)
                            COMPREPLY=($(compgen -W "${agent_options}" -- "${cur}"))
                            ;;
                    esac
                    ;;
            esac
            ;;
    esac
}

complete -F _abi_completions abi
