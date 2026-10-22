# Security Policy

## Scope

ABI is a local-first Rust framework: a CLI (`abi`), an MCP server (`abi-mcp`),
and the WDBX vector/key-value/block store it uses. The threat model, the
trust boundaries, and the claim boundaries for every surface live in
[`abi-threat-model.md`](abi-threat-model.md) and
[`docs/contracts/external-claims-audit.mdx`](docs/contracts/external-claims-audit.mdx).
Read those before reporting: several surfaces are reference-grade or
single-host by design, and a report that treats them as production
multi-host, audited-cryptography, or verified-accelerator behavior may
describe a documented boundary rather than a defect.

## Supported Versions

The workspace version is `0.1.0` (see `Cargo.toml`). The project is
pre-1.0 and has no published release train, so there is no version matrix
with distinct support windows.

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |
| < 0.1.0 | :x:                |

Fixes land on `main`; there are no backported patch releases.

## Reporting a Vulnerability

Open a **private** GitHub security advisory on this repository
(<https://github.com/donaldfilimon/abi/security/advisories/new>) rather than
a public issue. Do not include credentials, live store contents
(`~/.abi/`), or model artifacts in a report.

Please include:

- the affected crate and version or commit,
- a minimal reproduction (ideally a test that fails),
- the expected versus observed behaviour, and
- whether the issue crosses a documented trust boundary (e.g. the loopback
  HTTP listener, the OS-control allowlist, the WDBX writer lock, or the
  plugin/connector dispatch paths).

Expect an acknowledgement within a few days. Because this is a
single-maintainer project, triage and remediation timing depends on
availability; there is no guaranteed SLA, and no bounty program.
