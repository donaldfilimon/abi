---
name: codebase-analysis
description: This skill should be used when the user asks to find legacy patterns, technical debt, or modernization targets in ABI — e.g. 'where should we start modernizing this module', 'find outdated patterns in src/', 'audit this file for tech debt' — before any refactor begins.
---

# Codebase Analysis

Systematic techniques to discover technical debt, outdated patterns, and high-value modernization targets.

## Process

1. Identify boundaries and modules.
2. Look for:
   - Stringly-typed state and magic values.
   - Broad exception catching or error ignoring.
   - God classes / large files.
   - Tight coupling to outdated frameworks or stdlib workarounds.
   - Missing or weak typing.
   - Manual resource management that modern constructs solve.
3. Prioritize by impact and risk (use the strategy skill).
4. Produce a modernization opportunity report with evidence (code locations + rationale).

## Heuristics

- Any place using "TODO: modernize" or historical comments.
- Code that duplicates what the language now provides natively.
- Areas with high churn + high bug rate.
- Modules that are hard to test in isolation.

## Additional Resources

- `.agents/skills/codebase-analysis/references/analysis-checklist.md` — structured scan + prioritization table
- Use Grep + Glob heavily for pattern hunting.

Output should be actionable for the planner agent and strategy skill.

Base directory for this skill: /Users/donaldfilimon/abi/.claude/skills/codebase-analysis
Relative paths in this skill (e.g., references/) are relative to this base directory.
