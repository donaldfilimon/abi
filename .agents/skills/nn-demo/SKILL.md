---
name: nn-demo
description: Build the abi CLI and run the miniature character-level neural-net demo — `nn train` (real manual-backprop char-LM) and `nn sample` (train-then-generate). Use to train, sample, or smoke-test the nn demo trainer, or verify feat-nn after touching src/features/nn/. This is a demo trainer, NOT a production/LLM/distributed trainer.
---

# nn-demo — drive abi's character-level demo trainer

Driver: **`.agents/skills/nn-demo/nn.sh`** (paths relative to repo root).
Builds the CLI and drives `abi nn train` + `abi nn sample`. Evidence is the
`RESULT:` line. Fully local, no network.

## Run (agent path)
```bash
.agents/skills/nn-demo/nn.sh                                   # default corpus, seed 't', n=20
.agents/skills/nn-demo/nn.sh "hello world hello there" h 24    # custom corpus / seed / n
```
- `nn train "<corpus>"` — asserts `nn train:`, `final_loss=`, `steps=`.
- `nn sample --text "<corpus>" --seed <char> --n <k>` — trains, then generates;
  asserts `nn sample:`.

Prints `RESULT: PASS` (exit 0) or a FAIL count.

Verified this session: **PASS** on Zig master `0.17.0-dev.1099` — training reports
`initial_loss`/`final_loss`/`steps` and `nn sample` emits a generated string.

## Gotchas
- ⚠️ **Demo, not a production trainer.** `nn` is a pure-Zig hand-derived-backprop
  char-LM for demonstration — it is not an LLM, not distributed, not a real
  training stack. Loss values vary run-to-run; the driver asserts structural
  markers (`final_loss=`, `steps=`), not a specific loss.
- `nn sample`'s `--seed` is the **seed character** for generation, not an RNG
  seed. `nn sample` trains first, so its output has a leading `nn train:` line.
- `nn train --jsonl <path> [--field <name>]` trains from a JSONL corpus; the
  driver exercises the inline-text path. For source analysis, read
  `src/features/nn/`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| `build` FAIL | Check `zig version` (see `/zig-pin`), then `./build.sh check`. |
| missing `nn train:` / `nn sample:` | Output grammar drift — check the `nn` handler in `src/cli/handlers/` and `src/features/nn/`. |
