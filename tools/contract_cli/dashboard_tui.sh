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
