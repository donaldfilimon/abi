# Latest canonical qualification — 2026-10-20

Status: in progress. Local qualification targets stable Rust 1.99.1 across ABI,
Abbey and sibling WDBX. CI remains paused. No commits or pushes are authorized.
Existing work and earlier qualification receipts are preserved.

Evidence: `/Users/donald/.codex/verification/abi-abbey-canonical-20261020/latest-canonical/`.

## Current observations

- The latest baseline stopped at formatting while ABI host and WDBX decoder
  sources changed. Receipt: `../abi-latest-baseline-receipt.json`. This is a
  failed gate, not acceptance of the latest tree.
- Independent ABI review found no actionable issue in the runtime final-stop
  and buffered tool-admission repairs. The first focused run passed 128 Rust
  executions. Later test additions received scoped review without new findings.
- A subsequent focused run exited 101: 57 runtime tests passed and two new
  capture-prefix regressions failed. An external writer added capture changes
  during the run. Preserve that red evidence; the earlier pass does not cover
  those later bytes. Log: `abi-focused-final.log`.
- The writer then applied the capture-prefix repair. A component rerun passed
  72 runtime test executions with the three inspected runtime files unchanged.
  Receipt: `runtime-green-result.json`. The whole host/cohort still requires
  its final gate, including the newer authority-expiry edits.
- Abbey's owner repaired config initialization falsely reporting success for a
  directory or special-file destination. RED unit and CLI cases were retained;
  18 filtered unit cases plus two CLI cases in each of default and personal
  editions passed after repair. Independent scoped review found no blocker.
  Receipt: `/Users/donald/.codex/verification/abbey-latest-canonical-20261020/config-ready.json`.
- WDBX's owner reports 53 focused decoder and 64 legacy-selection passes;
  independent decoder review and complete current-source gate remain separate.
  External episode-store/filesystem/type edits stopped root formatting and
  continue to change. The initial macOS ancestor-symlink review finding was
  superseded during inspection and withdrawn for the later bytes.

## Aligned producer qualification

The coherent producer cohort is
`/Users/donald/.codex/verification/abbey-latest-cohort-20261020T114706Z/`.
Its capture verified all three repositories' bytes, modes and file sets before
and after copying. Private Git indexes support the hook and size gates; they
created no commits and changed no canonical Git metadata.

| Gate | Actual result |
| --- | --- |
| ABI `./tools/check.sh`, Rust 1.99.1 | Exit 0 in 182.12 seconds; 1,082 Rust executions, zero failures/ignored, 173 Python cases with eight unavailable cases skipped; zero stripping warnings |
| WDBX `bash tools/check.sh`, same producer cohort | Exit 0; 776 passed, zero failed/ignored across 32 summaries; snapshot sources unchanged and current canonical Rust/contract/gate inputs matched |
| Whole captured source after ABI gate | All three byte/mode/file-set comparisons unchanged |

ABI receipt: `../abi-final-114706-receipt.json`; log adjacent. WDBX receipt:
`wdbx-final-gate-result.json`. Full copied-source equality:
`abi-final-snapshot-equality.json`. ABI's earlier 113854 cohort also passed the
complete gate (1,082 executions); that result remains historical.

Abbey's final consumer snapshot is
`/Users/donald/.codex/verification/abbey-final-consumer-20261020T115249Z/`.
Its producer Rust/manifests/locks match the qualified cohort. It includes the
later provider tests and isolated Python metadata fixture; complete serialized
root/desktop qualification is in progress. The prior metadata-control failure
and ten unchanged passing focused reruns are retained, with no proven cause.
The fixture now skips unrelated Python site initialization while retaining the
actual 500 ms deadline and execution assertions; independent review accepted it.

## Dependency identity audit

`dependency-alignment.json` records exactly one local identity each for
abi-wdbx, abi-compute, abi-core, abi-foundation and abi-telemetry. ABI and Abbey
consume the intended adjacent canonical WDBX sources, with no duplicate Git
identity. ABI's lock remains SHA-256
`1dbbba40d78c3a321953b6fb3380dd2cbc5ef98c3ac87b50bc39915015c0e35f`.
WDBX's jsonschema 0.58 requirement is dev-only in this consumer graph; ABI's
runtime validators independently use 0.52. The ABI workspace TFHE declaration
is unused by ABI members; the actual optional sibling dependency and lock use
1.8.1. No manifest or lock change was needed to align current local sources.

## Retained decisions and limits

- Multiple writers are supported through coherent snapshots plus final
  canonical comparison. Later executable changes require further qualification.
- Preserve intended dependency constraints unless the resolved graph shows a
  defect; speculative validator/API upgrades would add unnecessary risk.
- Give the final consumer gate a quiet compiler window rather than weakening
  its metadata deadline or assertions. The earlier failures remain evidence.
- Eight personal central-sync policy cases lack the local central script;
  CUDA compilation lacks nvcc. Windows/Linux and GUI/installed-package
  acceptance remain separate from local macOS source qualification.
- The private complete-episode decoder remains unratified as a shared wire
  contract. Local source gates do not establish production encryption,
  multi-host operation, useful trained weights or deployed acceptance.

## Required closure

Obtain stable source handoffs, review the final diffs, and run complete ABI,
WDBX, and Abbey root/desktop gates against those exact current inputs. Record
source/lock hashes, file modes, exit statuses, unavailable checks and artifact
acceptance. A passing older or frozen cohort does not qualify subsequent edits.
No complete latest-canonical success is claimed here.
