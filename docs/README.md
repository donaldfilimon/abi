# Docs layout (Mintlify / GitHub Pages hub)

ABI's published documentation lives under this directory and is configured by
[`docs.json`](docs.json). The public site is served by **Mintlify** (GitHub app /
linked project that publishes from this tree). It is **not** a custom static
`gh-pages` tree for the main docs. Separately, optional benchmark dashboards may
use the `gh-pages` branch via [`.github/workflows/benchmarks-gh-pages.yml`](../.github/workflows/benchmarks-gh-pages.yml).
GitHub Actions validates the Mintlify site on every push/PR (`docs-validate`
job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).

## Tree

| Path | Role | In Mintlify nav? |
| ---- | ---- | ---------------- |
| [`index.mdx`](index.mdx) | Hub (Cards → contracts / identity / architecture) | Yes |
| [`contributing.mdx`](contributing.mdx) | Doc contribution + claim rules | Yes |
| [`contracts/*.mdx`](contracts/) | Frozen public surfaces + external claims audit | Yes |
| [`spec/abbey-core-identity.mdx`](spec/abbey-core-identity.mdx) | Canonical Abbey / Aviva / ABI / WDBX identity + evidence map | Yes |
| [`spec/*.mdx`](spec/) | Architecture / north-star / ops (Current/Partial/Proposed) | Yes |
| [`superpowers/`](superpowers/) | Working plans/specs + archive — **not** public contracts | No |
| [`tutorials/`](tutorials/) | How-to guides (not in Mintlify nav) | No |
| [`research/`](research/) | Research notes and PoC plans | No |

## Source of truth

When prose disagrees with code, trust `build.zig`, `src/`, `tests/contracts/`, and
`./build.sh check`. Claim boundaries:
[`contracts/external-claims-audit.mdx`](contracts/external-claims-audit.mdx).
Persona / product identity direction:
[`spec/abbey-core-identity.mdx`](spec/abbey-core-identity.mdx) and
`src/features/ai/identity.zig`.

## Local validation

```bash
.agents/skills/docs-validate/validate.sh
```

Requires an LTS Node (22.x; Mintlify rejects Node 25+). Preview: `cd docs && npx mint@latest dev`.

## Do not

- Add `superpowers/archive/` (or draft `.md` plans) to `docs.json` navigation.
- Promote Proposed items (ANE, CUDA/Vulkan native, sharding, audited FHE, SOTA codecs)
  to Current without source + tests.
- Re-list the frozen 13 CLI commands / 12 MCP tools — link [`contracts/public-api.mdx`](contracts/public-api.mdx).
