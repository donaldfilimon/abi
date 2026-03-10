# .zig-bootstrap

Canonical user-facing wrapper namespace for ABI's temporary Zig bootstrap
infrastructure during the CEL language transition.

Current state:
- `.zig-bootstrap/build.sh` delegates to the existing `.cel/build.sh`.
- `.zig-bootstrap/bin/zig` and `.zig-bootstrap/bin/zls` delegate to `.cel/bin/*`.
- `abi bootstrap-zig ...` is the preferred CLI surface; `abi toolchain ...` stays
  as a compatibility alias for one migration wave.

The underlying `.cel/` implementation remains in place only as a transitional
compatibility layer while CEL language tooling is brought up.
