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
