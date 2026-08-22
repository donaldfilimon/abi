# Abbey contract compatibility

Contract major, additive revision, and capability version are independent.
Removing or renaming a field, narrowing an accepted value, changing a required
field, or changing existing semantics requires a new schema identifier and
contract major. An additive revision may add a new optional field only where
the schema already defines an explicit compatible extension point. Capability
versions change the meaning or availability of one capability without silently
reinterpreting an existing wire shape.

Authority-bearing envelopes reject unknown fields. Tolerant content or event
metadata may preserve only a bounded `extensions` object and never consult it
to widen authority, open consent, authorize execution, or establish evidence.
Unknown top-level members are never treated as extensions.

Every consumer vendors exact corpus bytes and verifies the aggregate and
per-file SHA-256 commitments before generating bindings, compiling a release,
starting a production profile, or accepting a negotiated peer. A mismatch
disables authorization, consent opening, execution, and durable writes. A
developer profile may expose read-only diagnostics with a loud mismatch status.
It may not weaken that fail-closed boundary.

Rollback returns the consumer to the last qualified corpus digest. Failed
versions remain in compatibility history rather than being silently rewritten.
The corpus may later be extracted without path or byte changes when independent
release cadence, multiple non-ABI consumers, separate governance, or package
distribution makes an independent repository necessary. Extraction is a move
of the same committed artifact set, not an opportunity to change the contract.

## Deterministic vendoring

Consumers vendor with an immutable 40-character lower-hex ABI revision:

```sh
python3 tools/vendor_abbey_contracts.py \
  --source contracts/abbey \
  --destination <consumer-contract-directory> \
  --source-revision <abi-commit-sha> \
  --write
```

The destination contains only `abbey-contracts.lock.json` and `corpus/`.
`corpus/` contains the exact source `manifest.json` plus every artifact listed
by it, with unchanged bytes and line endings. The closed lock contains only
`source_repository`, `source_revision`, `contract_major`, `contract_revision`,
and `aggregate_digest`. A branch name, tag, mutable URL, or generated binding is
not a source identity.

`--check` verifies the source, lock, destination inventory, manifest, digests,
and byte equality without rewriting files or changing their modification times.
`--write` stages a complete mode-0700 sibling tree and publishes it by rename.
It creates an absent destination, but replaces an existing destination only
after proving that its current lock and corpus form a valid managed tree. It
never adopts or deletes an unmanaged destination.

Consumers run `--check` in their repository gate and also implement native
schema/digest validation. The Python tool establishes byte provenance only; it
does not qualify a native decoder, authorize an action, open consent, or write
durable state.
