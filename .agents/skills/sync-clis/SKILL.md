---
name: sync-clis
description: Sync canonical skills/plugins/commands/experiences from central (.grok + abi-mega) to all CLIs (grok,claude,codex,opencode,abi,cursor+). Idempotent. Launch with /sync-clis or the launch.sh .
---
# /sync-clis

This skill is backed by executable launcher at `launch.sh` (run via Grok skill system or directly).

It calls the central sync script, propagating to all configured targets.
