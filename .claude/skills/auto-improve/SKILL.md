---
name: auto-improve
description: Run the propose-then-apply self-improvement loops in abi (`abi improve`) and abbey (`abbey learn improve`). Use when asked to "self-improve", "auto-improve", or preview/apply what abi or abbey would learn from recent activity. Both default to a dry run; applying is an explicit, gated step. Not a trainer, not LoRA, and not skill-loop.
---

# auto-improve — preview, then apply, what abi and abbey would learn

Two local CLIs, one contract: **a dry run is the default, and `--apply` performs
only the steps its gate allows.** Neither command edits code, skills, or prompts.

## abi — `abi improve [--apply] [--model <id>] <input>`

Run from `~/dev/active/abi`. It wraps the SEA learn loop (`abi-sea` `run_learn_loop`).

- **Dry run** runs the loop with `persist` and `adapt_router` off. It recalls
  evidence and reports `task=`, `profile=`, `evidence_count=`, `audit_passed=`,
  `audit_vetoed=`, and `apply_allowed=`, and it writes nothing.
- **`--apply`** re-runs the loop with the default config only when
  `audit_passed=true` and `audit_vetoed=false`. Otherwise it prints `skipped:`
  and persists nothing, so a rejected completion never becomes evidence.
- It needs a persistent WDBX store and exits 1 without one. For experiments,
  point `ABI_WDBX_PATH` at a scratch directory, never the live `~/.abi/wdbx`.

```bash
./tools/cargo.sh build -p abi-cli
ABI_WDBX_PATH=target/skill-scratch/wdbx target/debug/abi improve "plan next repair"
ABI_WDBX_PATH=target/skill-scratch/wdbx target/debug/abi improve --apply "plan next repair"
```

## abbey — `abbey learn improve [n] [--apply]`

Run from `~/dev/active/abbey`. It reads the last `n` route records (default 20),
the memory `reflect` report, and `train_candidate` curation counts.

- Output rows tagged `[auto]` are applied by `--apply`. Today the only auto step
  is promoting recent routes into the activity layer.
- Rows tagged `[review]` are for a human: routes below 0.6 confidence, duplicate
  or low-confidence or superseded memories, and candidates missing provenance.
  `--apply` never touches them.
- A dry run with no memory store does not create one.

## Gotchas

- In abi, run tests and `./tools/check.sh` with `< /dev/null` (an auth test reads stdin).
- abi's `improve` is the 14th frozen command. Its help, `help.json`, completion
  files, and `site/index.html` chip list are golden-tested, so a change to its
  wording means updating those files together.
- The abbey help text calls LoRA "Proposed". Neither loop trains weights beyond
  abi's adaptive router, so do not describe either as fine-tuning.
- skill-loop's `review`/`fix` flags are unrelated and unreliable; do not use them
  as the improvement signal.
