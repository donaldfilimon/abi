# Abbey v1 contract and immutable ChangeSet evidence

Date: 2026-08-24  
Evidence ceiling: C1 (source and contract)  
Contract major: 2  
Contract revision: 2  
Corpus digest: `3ffd487bdc497b7ce54b8c29978a3686dcbffdb66a85957a0ee4f99ba576cdfd`

## Proven in this slice

- The ABI corpus contains a closed `abbey.v1` request envelope with all 19
  ratified method names, exact contract and capability-manifest commitments,
  bounded request identifiers, and a content commitment for parameters.
- The normative C6 local transport is a Unix-domain socket with a four-byte
  big-endian frame length, UTF-8 JSON, a 1 MiB frame ceiling, and a JSON
  container-depth ceiling of 32. Authority-bearing downgrade, legacy retry,
  and backend fallback are forbidden.
- Immutable ChangeSets bind operation, requester, Abbey proposal author,
  guild/capability/package, compensation class, risk, approval floor,
  precondition, expected postcondition, rollback, snapshot, generator, and
  expiry fields into one domain-separated digest.
- `ExactRestore`, `BestEffort`, and `None` map exactly to the existing
  `Reversible`, `ReversibleWithLoss`, and `Irreversible` vocabulary.
- An approval binds one exact ChangeSet digest and requires an authorized human
  identity distinct from both the requester and Abbey proposal author. A
  changed proposal produces a new digest and cannot reuse that approval.
- Sanitized receipts contain only request/operation identifiers, relevant
  digests, a closed policy decision, compensation class, evidence level,
  terminal state, and an explicit redaction marker.

## Evidence commands

```sh
python3 -m unittest tools.tests.test_abbey_contracts
python3 tools/abbey_contracts.py verify contracts/abbey
./tools/cargo.sh test -p abi-capability
./tools/cargo.sh clippy -p abi-capability --all-targets -- -D warnings
```

## Explicitly not proven

This slice does not implement the Abbey daemon service, a native Rust-bot or
Swift decoder, WDBX episode persistence, a Discord actuator, live consent,
deployment installation, or any C2-C6 witness. The existing ABI actuator
remains recording-only. Best-effort production execution remains unavailable,
and non-loopback federation remains unshipped.
