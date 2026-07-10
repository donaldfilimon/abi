# Agent multi/spawn/browser claim-honest contract smoke
browser_out="$("$ABI" agent browser "contract smoke" 2>&1)"
require_substring "$browser_out" "embedded_browser=false"
require_substring "$browser_out" "delegation_hint=external-mcp-playwright"

set +e
execute_out="$("$ABI" agent browser --execute "task" 2>&1)"
execute_rc=$?
set -e
expect_exit_2 "abi agent browser --execute without --confirm" "$execute_rc" "$execute_out"

spawn_out="$("$ABI" agent spawn "contract worker smoke" 2>&1)"
require_substring "$spawn_out" "CUSTOM MULTI-AGENT RESULTS"

multi_out="$("$ABI" agent multi "trio smoke" 2>&1)"
require_substring "$multi_out" "MULTI-AGENT RESULTS"