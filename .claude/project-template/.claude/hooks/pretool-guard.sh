#!/usr/bin/env bash
# PreToolUse (Bash): deny destructive command patterns via permissionDecision.
# Local models under an iterate-until-done loop will happily nuke things;
# this is the hard floor. Extend BLOCK as needed.
set -u

CMD=$(jq -r '.tool_input.command // empty')
[ -z "$CMD" ] && exit 0

BLOCK='rm -rf +/([^a-zA-Z]|$)|rm -rf +~|rm -rf +"?\$HOME|mkfs\.|dd +if=.* +of=/dev/|git +push +[^ ]*--force|git +reset +--hard +origin|chmod +-R +777 +/|:> */dev/sd|shutdown|reboot'

if printf '%s' "$CMD" | grep -qE "$BLOCK"; then
  jq -n --arg cmd "$CMD" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: ("pretool-guard: destructive pattern blocked: " + $cmd + ". Use a targeted, reversible alternative.")
    }
  }'
  exit 0
fi
exit 0
