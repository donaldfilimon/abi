# Fish completion script for ABI framework CLI
# Installation:
#   cp abi.fish ~/.config/fish/completions/

# Disable file completions for the main command
complete -c abi -f

# Main commands
complete -c abi -n '__fish_use_subcommand' -a 'db' -d 'Database operations'
complete -c abi -n '__fish_use_subcommand' -a 'agent' -d 'Run AI agent'
complete -c abi -n '__fish_use_subcommand' -a 'config' -d 'Configuration management'
complete -c abi -n '__fish_use_subcommand' -a 'explore' -d 'Search and explore codebase'
complete -c abi -n '__fish_use_subcommand' -a 'gpu' -d 'GPU commands'
complete -c abi -n '__fish_use_subcommand' -a 'network' -d 'Manage network registry'
complete -c abi -n '__fish_use_subcommand' -a 'simd' -d 'Run SIMD performance demo'
complete -c abi -n '__fish_use_subcommand' -a 'system-info' -d 'Show system status'
complete -c abi -n '__fish_use_subcommand' -a 'version' -d 'Show framework version'
complete -c abi -n '__fish_use_subcommand' -a 'help' -d 'Show help message'

# db subcommands
complete -c abi -n '__fish_seen_subcommand_from db' -a 'add' -d 'Add a vector'
complete -c abi -n '__fish_seen_subcommand_from db' -a 'query' -d 'Query vectors'
complete -c abi -n '__fish_seen_subcommand_from db' -a 'stats' -d 'Show statistics'
complete -c abi -n '__fish_seen_subcommand_from db' -a 'optimize' -d 'Optimize database'
complete -c abi -n '__fish_seen_subcommand_from db' -a 'backup' -d 'Backup database'
complete -c abi -n '__fish_seen_subcommand_from db' -a 'restore' -d 'Restore database'
complete -c abi -n '__fish_seen_subcommand_from db' -a 'index' -d 'Manage indexes'
complete -c abi -n '__fish_seen_subcommand_from db' -a 'help' -d 'Show help'

# gpu subcommands
complete -c abi -n '__fish_seen_subcommand_from gpu' -a 'backends' -d 'List GPU backends'
complete -c abi -n '__fish_seen_subcommand_from gpu' -a 'devices' -d 'List GPU devices'
complete -c abi -n '__fish_seen_subcommand_from gpu' -a 'list' -d 'List GPU devices'
complete -c abi -n '__fish_seen_subcommand_from gpu' -a 'summary' -d 'Show GPU summary'
complete -c abi -n '__fish_seen_subcommand_from gpu' -a 'default' -d 'Show default device'
complete -c abi -n '__fish_seen_subcommand_from gpu' -a 'status' -d 'Show GPU status'
complete -c abi -n '__fish_seen_subcommand_from gpu' -a 'help' -d 'Show help'

# network subcommands
complete -c abi -n '__fish_seen_subcommand_from network' -a 'status' -d 'Show network status'
complete -c abi -n '__fish_seen_subcommand_from network' -a 'list' -d 'List nodes'
complete -c abi -n '__fish_seen_subcommand_from network' -a 'nodes' -d 'List nodes'
complete -c abi -n '__fish_seen_subcommand_from network' -a 'register' -d 'Register a node'
complete -c abi -n '__fish_seen_subcommand_from network' -a 'unregister' -d 'Remove a node'
complete -c abi -n '__fish_seen_subcommand_from network' -a 'touch' -d 'Update heartbeat'
complete -c abi -n '__fish_seen_subcommand_from network' -a 'set-status' -d 'Set node status'
complete -c abi -n '__fish_seen_subcommand_from network' -a 'help' -d 'Show help'

# config subcommands
complete -c abi -n '__fish_seen_subcommand_from config' -a 'init' -d 'Generate config file'
complete -c abi -n '__fish_seen_subcommand_from config' -a 'show' -d 'Display configuration'
complete -c abi -n '__fish_seen_subcommand_from config' -a 'validate' -d 'Validate config file'
complete -c abi -n '__fish_seen_subcommand_from config' -a 'env' -d 'List environment variables'
complete -c abi -n '__fish_seen_subcommand_from config' -a 'help' -d 'Show help'

# config init options
complete -c abi -n '__fish_seen_subcommand_from config; and __fish_seen_subcommand_from init' -s o -l output -d 'Output file path' -r

# config show options
complete -c abi -n '__fish_seen_subcommand_from config; and __fish_seen_subcommand_from show' -s f -l format -d 'Output format' -xa 'human json'

# explore options
complete -c abi -n '__fish_seen_subcommand_from explore' -s l -l level -d 'Exploration depth' -xa 'quick medium thorough deep'
complete -c abi -n '__fish_seen_subcommand_from explore' -s f -l format -d 'Output format' -xa 'human json compact yaml'
complete -c abi -n '__fish_seen_subcommand_from explore' -s i -l include -d 'Include pattern' -r
complete -c abi -n '__fish_seen_subcommand_from explore' -s e -l exclude -d 'Exclude pattern' -r
complete -c abi -n '__fish_seen_subcommand_from explore' -s c -l case-sensitive -d 'Case sensitive'
complete -c abi -n '__fish_seen_subcommand_from explore' -s r -l regex -d 'Use regex'
complete -c abi -n '__fish_seen_subcommand_from explore' -l path -d 'Root directory' -xa '(__fish_complete_directories)'
complete -c abi -n '__fish_seen_subcommand_from explore' -l max-files -d 'Max files to scan' -r
complete -c abi -n '__fish_seen_subcommand_from explore' -l max-depth -d 'Max directory depth' -r
complete -c abi -n '__fish_seen_subcommand_from explore' -l timeout -d 'Timeout in ms' -r
complete -c abi -n '__fish_seen_subcommand_from explore' -s h -l help -d 'Show help'

# agent options
complete -c abi -n '__fish_seen_subcommand_from agent' -l name -d 'Agent name' -r
complete -c abi -n '__fish_seen_subcommand_from agent' -s m -l message -d 'Message to send' -r
complete -c abi -n '__fish_seen_subcommand_from agent' -l interactive -d 'Interactive mode'
complete -c abi -n '__fish_seen_subcommand_from agent' -s h -l help -d 'Show help'

# network set-status values
complete -c abi -n '__fish_seen_subcommand_from network; and __fish_seen_subcommand_from set-status' -a 'healthy degraded offline' -d 'Node status'
