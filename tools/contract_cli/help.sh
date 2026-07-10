help_out="$("$ABI" help 2>&1)"
require_substring "$help_out" 'abi help [--json|--completion <shell>] <command> [subcommand]'

help_json_out="$("$ABI" help --json 2>&1)"
printf '%s' "$help_json_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); assert data["type"] == "abi.cli.help"; assert data["completion"]["usage"] == "abi help --completion <bash|zsh|fish>"; assert data["completion"]["shells"] == ["bash","zsh","fish"]; names=[cmd["name"] for cmd in data["commands"]]; assert "complete" in names and "dashboard" in names and "wdbx" in names; help_cmd=next(cmd for cmd in data["commands"] if cmd["name"]=="help"); assert "--completion" in help_cmd["usage"]; shortcuts=data["shortcuts"]; assert any(s["token"]=="--tui" and s["command"]=="tui" for s in shortcuts)'
if grep -Fq -- "Usage: abi" <<<"$help_json_out"; then
  echo "expected 'abi help --json' to emit JSON only, not text help" >&2
  echo "$help_json_out" >&2
  exit 1
fi

completion_bash_out="$("$ABI" help --completion bash 2>&1)"
require_substring "$completion_bash_out" '_abi_complete()'
require_substring "$completion_bash_out" 'tui|--tui)'
require_substring "$completion_bash_out" '--list-panes'
require_substring "$completion_bash_out" 'words="list run "'
reject_substring "$completion_bash_out" 'words="list run list run '
printf '%s' "$completion_bash_out" > zig-out/abi-completion-contract.bash
bash -n zig-out/abi-completion-contract.bash
if grep -Fq -- "Usage: abi" <<<"$completion_bash_out"; then
  echo "expected 'abi help --completion bash' to emit completion only, not text help" >&2
  echo "$completion_bash_out" >&2
  exit 1
fi

completion_zsh_out="$("$ABI" help --completion zsh 2>&1)"
require_substring "$completion_zsh_out" '#compdef abi'
require_substring "$completion_zsh_out" 'compadd -- --pane'
require_substring "$completion_zsh_out" 'bash zsh fish'
require_substring "$completion_zsh_out" 'compadd -- list run '
reject_substring "$completion_zsh_out" 'compadd -- list run list run '

completion_fish_out="$("$ABI" help --completion fish 2>&1)"
require_substring "$completion_fish_out" 'complete -c abi -f'
require_substring "$completion_fish_out" "__fish_seen_subcommand_from tui --tui"
require_substring "$completion_fish_out" '-l list-panes'
require_substring "$completion_fish_out" "' -a 'list run '"
reject_substring "$completion_fish_out" "' -a 'list run list run "

set +e
completion_bad_out="$("$ABI" help --completion powershell 2>&1)"
completion_bad_rc=$?
set -e
if [[ "$completion_bad_rc" -ne 2 ]]; then
  echo "expected 'abi help --completion powershell' to exit 2, got $completion_bad_rc" >&2
  echo "$completion_bad_out" >&2
  exit 1
fi
require_substring "$completion_bad_out" "abi help [--json|--completion <bash|zsh|fish>] [command] [subcommand]"

dashboard_help_json_out="$("$ABI" help --json dashboard 2>&1)"
printf '%s' "$dashboard_help_json_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); command=data["command_detail"]; names={arg["name"] for arg in command["args"]}; assert data["command"] == "dashboard"; assert command["name"] == "dashboard"; assert "json" in names and "compact" in names'

tui_shortcut_help_json_out="$("$ABI" help --json --tui 2>&1)"
printf '%s' "$tui_shortcut_help_json_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); assert data["command"] == "tui"; shortcuts=data["shortcuts"]; assert any(s["token"]=="--tui" and s["command"]=="tui" for s in shortcuts)'

wdbx_cluster_help_json_out="$("$ABI" help wdbx cluster --json 2>&1)"
printf '%s' "$wdbx_cluster_help_json_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); sub=data["subcommand"]; assert data["command"] == "wdbx"; assert sub["name"] == "cluster"; assert "cluster serve" in sub["usage"]'
