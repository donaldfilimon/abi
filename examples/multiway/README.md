# Multiway simulator examples

Example configurations for `abi wdbx simulate` — the bounded multiway
string-rewriting simulator. See `docs/spec/wdbx-multiway.mdx` for the full
scope, scientific-boundary discipline, and metric definitions.

> These are bounded slices of computational rule space. Nothing here simulates
> "the ruliad" or makes any physics claim.

| File | System | Notes |
|------|--------|-------|
| `reference.rules` | `A->AB`, `A->BA`, `BB->A` | The reference regression rule set (rules-file form). |
| `branching.json`  | Same rules, JSON config | Growing/branching with a shrinking rule; rule families + weights. |
| `convergent.json` | `A->C`, `B->C` | Distinct branches converge onto shared canonical states. |
| `cyclic.json`     | `A->B`, `B->A` | Two-state cycle; converges to 2 states, frontier exhausts. |

## Running

```bash
# Reference experiment (canonical JSON export)
abi wdbx simulate --initial A --rules-file examples/multiway/reference.rules \
  --depth 5 --max-states 500 --max-events 5000 \
  --format json --output experiment.json --verbose

# From a JSON config; flags override file values
abi wdbx simulate --config examples/multiway/branching.json

# Persist into a WDBX checkpoint, then resume deeper
abi wdbx simulate --config examples/multiway/convergent.json --store convergent.wdbx.jsonl
abi wdbx simulate --config examples/multiway/convergent.json --depth 8 --resume-wdbx convergent.wdbx.jsonl

# Graphviz DOT
abi wdbx simulate --config examples/multiway/cyclic.json --format dot --output cyclic.dot
```
