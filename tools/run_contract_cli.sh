#!/usr/bin/env bash
set -euo pipefail

# Keep the CLI contract hermetic: the default-ON durable WDBX store would
# otherwise persist into ~/.abi/wdbx during the run. Force in-memory.
export ABI_WDBX_PERSIST=0

ABI="${ABI_EXE:-zig-out/bin/abi}"

if [[ ! -x "$ABI" ]]; then
  echo "error: $ABI not found; run 'zig build cli' first" >&2
  exit 1
fi

require_substring() {
  local output="$1"
  local needle="$2"
  if ! grep -Fq -- "$needle" <<<"$output"; then
    echo "expected substring missing: $needle" >&2
    echo "output:" >&2
    echo "$output" >&2
    exit 1
  fi
}

reject_substring() {
  local output="$1"
  local needle="$2"
  if grep -Fq -- "$needle" <<<"$output"; then
    echo "unexpected substring present: $needle" >&2
    echo "output:" >&2
    echo "$output" >&2
    exit 1
  fi
}

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

complete_out="$("$ABI" complete "hello world" 2>&1)"
require_substring "$complete_out" "model=claude-fable-5"
require_substring "$complete_out" "audit_passed="
require_substring "$complete_out" "persisted="
require_substring "$complete_out" "wdbx kv_entries="

complete_help_out="$("$ABI" complete --help 2>&1)"
require_substring "$complete_help_out" "abi complete [--live] [--confirm] [--learn] [--model <id>] <input>"
require_substring "$complete_help_out" "Arguments:"
require_substring "$complete_help_out" "--model <value>"
if grep -Fq -- "model=claude-fable-5" <<<"$complete_help_out"; then
  echo "expected 'abi complete --help' to render help, not run completion" >&2
  echo "$complete_help_out" >&2
  exit 1
fi

set +e
complete_bad_out="$("$ABI" complete --bogus 2>&1)"
complete_bad_rc=$?
set -e
if [[ "$complete_bad_rc" -ne 2 ]]; then
  echo "expected 'abi complete --bogus' to exit 2, got $complete_bad_rc" >&2
  echo "$complete_bad_out" >&2
  exit 1
fi
require_substring "$complete_bad_out" "abi complete [--live]"

set +e
complete_typo_out="$("$ABI" complte 2>&1)"
complete_typo_rc=$?
set -e
if [[ "$complete_typo_rc" -ne 2 ]]; then
  echo "expected 'abi complte' to exit 2, got $complete_typo_rc" >&2
  echo "$complete_typo_out" >&2
  exit 1
fi
require_substring "$complete_typo_out" "unknown command 'complte'"
require_substring "$complete_typo_out" 'hint: did you mean `complete`?'

set +e
complete_help_typo_out="$("$ABI" help complte 2>&1)"
complete_help_typo_rc=$?
set -e
if [[ "$complete_help_typo_rc" -ne 2 ]]; then
  echo "expected 'abi help complte' to exit 2, got $complete_help_typo_rc" >&2
  echo "$complete_help_typo_out" >&2
  exit 1
fi
require_substring "$complete_help_typo_out" "unknown command 'complte'"
require_substring "$complete_help_typo_out" 'hint: did you mean `complete`?'

complete_literal_out="$("$ABI" complete -- --literal-leading-dash 2>&1)"
require_substring "$complete_literal_out" "model=claude-fable-5"
require_substring "$complete_literal_out" "audit_passed="

backends_out="$("$ABI" backends 2>&1)"
require_substring "$backends_out" "GPU:"

plugins_out="$("$ABI" plugin list 2>&1)"
require_substring "$plugins_out" "Installed Plugins ("
require_substring "$plugins_out" "example-plugin"
require_substring "$plugins_out" "(mod.zig)"

plugin_help_out="$("$ABI" plugin --help 2>&1)"
require_substring "$plugin_help_out" "abi plugin list | run <name> [input]"
plugin_run_help_out="$("$ABI" plugin run --help 2>&1)"
require_substring "$plugin_run_help_out" "usage: abi plugin run <name> [input]"
plugin_run_help_alias_out="$("$ABI" help plugin run 2>&1)"
require_substring "$plugin_run_help_alias_out" "usage: abi plugin run <name> [input]"

plugin_run_out="$("$ABI" plugin run example-plugin hello plugin 2>&1)"
require_substring "$plugin_run_out" "example-plugin received input (len=12)"

set +e
plugin_bad_out="$("$ABI" plugin bogus 2>&1)"
plugin_bad_rc=$?
set -e
if [[ "$plugin_bad_rc" -ne 2 ]]; then
  echo "expected 'abi plugin bogus' to exit 2, got $plugin_bad_rc" >&2
  echo "$plugin_bad_out" >&2
  exit 1
fi
require_substring "$plugin_bad_out" "abi plugin list | run <name> [input]"

set +e
plugin_typo_out="$("$ABI" plugin rn 2>&1)"
plugin_typo_rc=$?
set -e
if [[ "$plugin_typo_rc" -ne 2 ]]; then
  echo "expected 'abi plugin rn' to exit 2, got $plugin_typo_rc" >&2
  echo "$plugin_typo_out" >&2
  exit 1
fi
require_substring "$plugin_typo_out" "abi plugin list | run <name> [input]"
require_substring "$plugin_typo_out" 'hint: did you mean `run`?'

set +e
plugin_run_missing_out="$("$ABI" plugin run 2>&1)"
plugin_run_missing_rc=$?
set -e
if [[ "$plugin_run_missing_rc" -ne 2 ]]; then
  echo "expected 'abi plugin run' to exit 2, got $plugin_run_missing_rc" >&2
  echo "$plugin_run_missing_out" >&2
  exit 1
fi
require_substring "$plugin_run_missing_out" "abi plugin run <name> [input]"

auth_out="$("$ABI" auth status 2>&1)"
require_substring "$auth_out" "Authentication Status:"
require_substring "$auth_out" "OpenAI:"

auth_help_out="$("$ABI" auth --help 2>&1)"
require_substring "$auth_help_out" "abi auth <signin|logout|status>"
auth_signin_help_out="$("$ABI" auth signin --help 2>&1)"
require_substring "$auth_signin_help_out" "usage: abi auth signin"

set +e
auth_bad_out="$("$ABI" auth bogus 2>&1)"
auth_bad_rc=$?
set -e
if [[ "$auth_bad_rc" -ne 2 ]]; then
  echo "expected 'abi auth bogus' to exit 2, got $auth_bad_rc" >&2
  echo "$auth_bad_out" >&2
  exit 1
fi
require_substring "$auth_bad_out" "abi auth <signin|logout|status>"

set +e
auth_signin_missing_out="$("$ABI" auth signin 2>&1)"
auth_signin_missing_rc=$?
set -e
if [[ "$auth_signin_missing_rc" -ne 2 ]]; then
  echo "expected 'abi auth signin' to exit 2, got $auth_signin_missing_rc" >&2
  echo "$auth_signin_missing_out" >&2
  exit 1
fi
require_substring "$auth_signin_missing_out" "abi auth signin <openai|anthropic|discord|grok|twilio>"

set +e
auth_signin_bad_out="$("$ABI" auth signin notaservice 2>&1)"
auth_signin_bad_rc=$?
set -e
if [[ "$auth_signin_bad_rc" -ne 2 ]]; then
  echo "expected 'abi auth signin notaservice' to exit 2, got $auth_signin_bad_rc" >&2
  echo "$auth_signin_bad_out" >&2
  exit 1
fi
require_substring "$auth_signin_bad_out" "abi auth signin <openai|anthropic|discord|grok|twilio>"

agent_plan_help_out="$("$ABI" agent plan --help 2>&1)"
require_substring "$agent_plan_help_out" "usage: abi agent plan <input>"
agent_plan_help_alias_out="$("$ABI" help agent plan 2>&1)"
require_substring "$agent_plan_help_alias_out" "usage: abi agent plan <input>"

twilio_help_out="$("$ABI" twilio simulate --help 2>&1)"
require_substring "$twilio_help_out" "usage: abi twilio simulate <input>"

twilio_top_help_out="$("$ABI" twilio --help 2>&1)"
require_substring "$twilio_top_help_out" "abi twilio simulate <input>"
require_substring "$twilio_top_help_out" "Arguments:"

set +e
twilio_bad_out="$("$ABI" twilio bogus hi 2>&1)"
twilio_bad_rc=$?
set -e
if [[ "$twilio_bad_rc" -ne 2 ]]; then
  echo "expected 'abi twilio bogus hi' to exit 2, got $twilio_bad_rc" >&2
  echo "$twilio_bad_out" >&2
  exit 1
fi
require_substring "$twilio_bad_out" "abi twilio simulate <input>"

set +e
twilio_missing_out="$("$ABI" twilio simulate 2>&1)"
twilio_missing_rc=$?
set -e
if [[ "$twilio_missing_rc" -ne 2 ]]; then
  echo "expected 'abi twilio simulate' to exit 2, got $twilio_missing_rc" >&2
  echo "$twilio_missing_out" >&2
  exit 1
fi
require_substring "$twilio_missing_out" "abi twilio simulate <input>"

scheduler_out="$("$ABI" scheduler status 2>&1)"
require_substring "$scheduler_out" "scheduler status"
require_substring "$scheduler_out" "source=cli-scheduler-status"
require_substring "$scheduler_out" "completed=1"

scheduler_help_out="$("$ABI" scheduler --help 2>&1)"
require_substring "$scheduler_help_out" "abi scheduler status"
require_substring "$scheduler_help_out" "Arguments:"
require_substring "$scheduler_help_out" "choices=status"

set +e
scheduler_bad_out="$("$ABI" scheduler bogus 2>&1)"
scheduler_bad_rc=$?
set -e
if [[ "$scheduler_bad_rc" -ne 2 ]]; then
  echo "expected 'abi scheduler bogus' to exit 2, got $scheduler_bad_rc" >&2
  echo "$scheduler_bad_out" >&2
  exit 1
fi
require_substring "$scheduler_bad_out" "abi scheduler status"

wdbx_help_out="$("$ABI" wdbx --help 2>&1)"
require_substring "$wdbx_help_out" "abi wdbx <db|block|query|benchmark|cluster|compute|secure|gpu|api> ..."
require_substring "$wdbx_help_out" "Subcommands:"
require_substring "$wdbx_help_out" "db"
require_substring "$wdbx_help_out" "cluster serve <port>"
require_substring "$wdbx_help_out" "api serve [port]"

dashboard_help_out="$("$ABI" dashboard --help 2>&1)"
require_substring "$dashboard_help_out" "abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]"
require_substring "$dashboard_help_out" "--pane <value>"
require_substring "$dashboard_help_out" "--plain"
require_substring "$dashboard_help_out" "--no-color"
require_substring "$dashboard_help_out" "--compact"
require_substring "$dashboard_help_out" "--once"
require_substring "$dashboard_help_out" "--interval <value>"
require_substring "$dashboard_help_out" "--json"
require_substring "$dashboard_help_out" "--list-panes"
require_substring "$dashboard_help_out" "choices=1|2|3|4|5|system|plugins|storage|wdbx|scheduler|memory"

tui_shortcut_help_out="$("$ABI" help --tui 2>&1)"
require_substring "$tui_shortcut_help_out" "abi tui [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]"

dashboard_memory_out="$("$ABI" dashboard --pane memory 2>&1)"
require_substring "$dashboard_memory_out" "ABI Diagnostics Dashboard"
require_substring "$dashboard_memory_out" "Memory"

dashboard_plain_out="$("$ABI" dashboard --plain 2>&1)"
require_substring "$dashboard_plain_out" "ABI Diagnostics Dashboard"
if grep -Fq -- $'\033[' <<<"$dashboard_plain_out"; then
  echo "expected 'abi dashboard --plain' to omit ANSI styling" >&2
  echo "$dashboard_plain_out" >&2
  exit 1
fi

dashboard_compact_out="$("$ABI" dashboard --compact --pane scheduler --plain 2>&1)"
require_substring "$dashboard_compact_out" "ABI Diagnostics Dashboard"
require_substring "$dashboard_compact_out" "Scheduler"
if grep -Fq -- "WDBX Storage" <<<"$dashboard_compact_out" || grep -Fq -- "Memory" <<<"$dashboard_compact_out"; then
  echo "expected 'abi dashboard --compact --pane scheduler --plain' to render only the selected pane" >&2
  echo "$dashboard_compact_out" >&2
  exit 1
fi

dashboard_interval_out="$("$ABI" dashboard --once --interval 250 2>&1)"
require_substring "$dashboard_interval_out" "ABI Diagnostics Dashboard"
require_substring "$dashboard_interval_out" "live snapshot every 250ms"

dashboard_panes_out="$("$ABI" dashboard --list-panes --pane scheduler 2>&1)"
require_substring "$dashboard_panes_out" "Dashboard panes:"
require_substring "$dashboard_panes_out" "* scheduler (Scheduler) hotkey=4"
if grep -Fq -- "ABI Diagnostics Dashboard" <<<"$dashboard_panes_out"; then
  echo "expected 'abi dashboard --list-panes' to omit dashboard frame" >&2
  echo "$dashboard_panes_out" >&2
  exit 1
fi

dashboard_json_out="$("$ABI" dashboard --json --pane plugins --interval 250 2>&1)"
require_substring "$dashboard_json_out" '"type":"abi.dashboard"'
require_substring "$dashboard_json_out" '"selected_pane":"plugins"'
require_substring "$dashboard_json_out" '"refresh_interval_ms":250'
require_substring "$dashboard_json_out" '"layout"'
printf '%s' "$dashboard_json_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); layout=data["layout"]; assert layout["format"] == "json"; assert layout["compact"] is False; assert "plugins" in layout["visible_panes"]; panes=layout["panes"]; assert any(p["name"]=="plugins" and p["title"]=="Plugins" and p["hotkey"]=="2" and p["selected"] and p["visible"] for p in panes)'
if grep -Fq -- "ABI Diagnostics Dashboard" <<<"$dashboard_json_out"; then
  echo "expected 'abi dashboard --json' to emit JSON only, not the text dashboard" >&2
  echo "$dashboard_json_out" >&2
  exit 1
fi
if grep -Fq -- $'\033[' <<<"$dashboard_json_out"; then
  echo "expected 'abi dashboard --json' to omit ANSI controls" >&2
  echo "$dashboard_json_out" >&2
  exit 1
fi

dashboard_json_compact_out="$("$ABI" dashboard --json --compact --pane scheduler --plain 2>&1)"
printf '%s' "$dashboard_json_compact_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); layout=data["layout"]; assert data["selected_pane"] == "scheduler"; assert layout["format"] == "json"; assert layout["compact"] is True; assert layout["color"] is False; assert layout["visible_panes"] == ["scheduler"]; panes=layout["panes"]; assert any(p["name"]=="scheduler" and p["selected"] and p["visible"] for p in panes); assert any(p["name"]=="memory" and not p["visible"] for p in panes)'

dashboard_json_panes_out="$("$ABI" dashboard --json --list-panes --pane memory 2>&1)"
printf '%s' "$dashboard_json_panes_out" | python3 -c 'import json,sys; data=json.load(sys.stdin); assert data["type"] == "abi.dashboard.panes"; assert data["selected_pane"] == "memory"; panes=data["panes"]; assert len(panes) == 5; assert any(p["name"]=="memory" and p["hotkey"]=="5" and p["selected"] for p in panes)'

set +e
dashboard_bad_out="$("$ABI" dashboard --pane nope 2>&1)"
dashboard_bad_rc=$?
set -e
if [[ "$dashboard_bad_rc" -ne 2 ]]; then
  echo "expected 'abi dashboard --pane nope' to exit 2, got $dashboard_bad_rc" >&2
  echo "$dashboard_bad_out" >&2
  exit 1
fi
require_substring "$dashboard_bad_out" "abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]"

set +e
dashboard_pane_typo_out="$("$ABI" dashboard --pane memry 2>&1)"
dashboard_pane_typo_rc=$?
set -e
if [[ "$dashboard_pane_typo_rc" -ne 2 ]]; then
  echo "expected 'abi dashboard --pane memry' to exit 2, got $dashboard_pane_typo_rc" >&2
  echo "$dashboard_pane_typo_out" >&2
  exit 1
fi
require_substring "$dashboard_pane_typo_out" "abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]"
require_substring "$dashboard_pane_typo_out" 'hint: did you mean `memory`?'

set +e
dashboard_flag_typo_out="$("$ABI" dashboard --plian 2>&1)"
dashboard_flag_typo_rc=$?
set -e
if [[ "$dashboard_flag_typo_rc" -ne 2 ]]; then
  echo "expected 'abi dashboard --plian' to exit 2, got $dashboard_flag_typo_rc" >&2
  echo "$dashboard_flag_typo_out" >&2
  exit 1
fi
require_substring "$dashboard_flag_typo_out" "abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]"
require_substring "$dashboard_flag_typo_out" 'hint: did you mean `--plain`?'

set +e
dashboard_interval_bad_out="$("$ABI" dashboard --interval 99 2>&1)"
dashboard_interval_bad_rc=$?
set -e
if [[ "$dashboard_interval_bad_rc" -ne 2 ]]; then
  echo "expected 'abi dashboard --interval 99' to exit 2, got $dashboard_interval_bad_rc" >&2
  echo "$dashboard_interval_bad_out" >&2
  exit 1
fi
require_substring "$dashboard_interval_bad_out" "abi dashboard [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]"

tui_shortcut_out="$("$ABI" --tui --compact --pane scheduler --plain 2>&1)"
require_substring "$tui_shortcut_out" "ABI Diagnostics Dashboard"
require_substring "$tui_shortcut_out" "Scheduler"
if grep -Fq -- "WDBX Storage" <<<"$tui_shortcut_out" || grep -Fq -- "Memory" <<<"$tui_shortcut_out"; then
  echo "expected 'abi --tui --compact --pane scheduler --plain' to render only the selected pane" >&2
  echo "$tui_shortcut_out" >&2
  exit 1
fi

set +e
tui_extra_out="$("$ABI" --tui extra 2>&1)"
tui_extra_rc=$?
set -e
if [[ "$tui_extra_rc" -ne 2 ]]; then
  echo "expected 'abi --tui extra' to exit 2, got $tui_extra_rc" >&2
  echo "$tui_extra_out" >&2
  exit 1
fi
require_substring "$tui_extra_out" "abi tui [--pane <pane>] [--plain|--no-color] [--compact] [--once] [--interval <ms>] [--json] [--list-panes]"

# `abi nn` is the miniature character-level demo trainer. Assert the honest
# help framing and a real inline training run (stable label substrings only,
# never float loss values).
nn_help_out="$("$ABI" help nn 2>&1)"
require_substring "$nn_help_out" "character-level demo trainer"
require_substring "$nn_help_out" "not a production/LLM/distributed trainer"

nn_local_help_out="$("$ABI" nn --help 2>&1)"
require_substring "$nn_local_help_out" "train --jsonl <path>"
require_substring "$nn_local_help_out" "sample --text \"<corpus>\""

nn_train_help_out="$("$ABI" nn train --help 2>&1)"
require_substring "$nn_train_help_out" "usage: abi nn train"
if grep -Fq -- "nn train:" <<<"$nn_train_help_out"; then
  echo "expected 'abi nn train --help' to render help, not train" >&2
  echo "$nn_train_help_out" >&2
  exit 1
fi

nn_sample_help_out="$("$ABI" nn sample --help 2>&1)"
require_substring "$nn_sample_help_out" "usage: abi nn sample"

set +e
nn_bad_out="$("$ABI" nn bogus 2>&1)"
nn_bad_rc=$?
set -e
if [[ "$nn_bad_rc" -ne 2 ]]; then
  echo "expected 'abi nn bogus' to exit 2, got $nn_bad_rc" >&2
  echo "$nn_bad_out" >&2
  exit 1
fi
require_substring "$nn_bad_out" "abi nn train"

set +e
nn_sample_missing_out="$("$ABI" nn sample --text hello 2>&1)"
nn_sample_missing_rc=$?
set -e
if [[ "$nn_sample_missing_rc" -ne 2 ]]; then
  echo "expected 'abi nn sample --text hello' to exit 2, got $nn_sample_missing_rc" >&2
  echo "$nn_sample_missing_out" >&2
  exit 1
fi
require_substring "$nn_sample_missing_out" "abi nn sample --text"

wdbx_cluster_help_out="$("$ABI" wdbx cluster --help 2>&1)"
require_substring "$wdbx_cluster_help_out" "usage: abi wdbx cluster"
wdbx_cluster_help_alias_out="$("$ABI" help wdbx cluster 2>&1)"
require_substring "$wdbx_cluster_help_alias_out" "usage: abi wdbx cluster"

nn_train_out="$("$ABI" nn train "hello hello hello " 2>&1)"
require_substring "$nn_train_out" "nn train: initial_loss="
require_substring "$nn_train_out" "improved="

echo "run_contract_cli: ok"
