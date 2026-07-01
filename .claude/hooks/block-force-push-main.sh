#!/usr/bin/env bash
# PreToolUse(Bash) guard: block `git push --force` / `-f` targeting `main`.
#
# Rationale (tasks/todo.md): origin/main shares no common ancestor with local
# main (different roots) — "Never force-push to reconcile." A force-push to main
# is near-irreversible and can fuse/clobber unrelated history. This hook blocks
# only that case; ordinary pushes and force-pushes of feature branches pass.
#
# Reads the tool-call JSON on stdin. exit 2 = block (message on stderr); exit 0 = allow.
#
# Known limitation: matches `git push` only at a real command boundary (start, or
# after ; && || | ( ), so it ignores mere mentions in commit messages/greps. It
# CANNOT distinguish a real chained push from `&& git push --force … main` text
# inside a quoted string/heredoc body, so such meta-commands are blocked too
# (fail-safe). Run those yourself if needed.
set -uo pipefail

c="$(jq -r '.tool_input.command // empty' 2>/dev/null)"

# Must be an ACTUAL `git push` invocation — at the start of the command or right
# after a shell separator (; && || | ( newline). This avoids false positives on
# commands that merely *mention* the strings: `git commit -m "...git push..."`,
# `echo`, `grep`, heredoc PR bodies, etc.
printf '%s' "$c" | grep -Eq '(^|[;&|(]|&&|\|\|)[[:space:]]*git[[:space:]]+push([[:space:]]|$)' || exit 0
# Not a force push → allow (normal pushes to main are fine).
case "$c" in *"--force"*|*" -f"*) ;; *) exit 0 ;; esac

# It IS a force push. Block if it targets main explicitly, or is a bare
# force-push while the checked-out branch is main.
blocked=0
case "$c" in
  *main*) blocked=1 ;;
  *)
    br="$(git -C "${CLAUDE_PROJECT_DIR:-.}" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    [ "$br" = "main" ] && blocked=1
    ;;
esac

if [ "$blocked" = "1" ]; then
  echo "BLOCKED: force-push to main is disallowed. See tasks/todo.md — origin/main shares no common ancestor with local main; never force-push to reconcile. If this is genuinely intended, run the push yourself outside Claude." >&2
  exit 2
fi
exit 0
