# Abbey contract corpus

This directory is the canonical, language-neutral Program 1
`abbey-contracts` corpus. It contains UTF-8 JSON Schema Draft 2020-12 documents
and synthetic golden fixtures. The corpus is data-only: it has no Cargo feature,
generated binding, runtime listener, authorization actuator, Discord behavior,
model call, storage write, or production-deployment effect.

The source corpus is qualified as one exact byte set. Contract major 2 retains
every v1 artifact byte and adds closed v2 authority-envelope identifiers rather
than changing v1 required fields or semantics. `manifest.json` lists
every normative schema, fixture, and policy document once; its aggregate digest
also commits to a normalized copy of the manifest itself. Consumers vendor the
listed bytes unchanged, pin that aggregate digest, and validate real inputs with
their native schema implementation. Generated types are projections, not a
replacement source of truth.

Contract revision 2 adds the local `abbey.v1` federation envelope, its complete
19-method registry, sanitized federation receipts, an immutable digest-bound
ChangeSet plus distinct human approval, and a normative Unix-socket transport
policy. The C6 transport is four-byte big-endian length framing followed by
UTF-8 JSON, capped at 1 MiB and 32 JSON containers deep. Contract, revision,
corpus-digest, or capability-manifest mismatch is refused before authorization;
authority-bearing traffic never downgrades, retries through a legacy protocol,
or falls back to another backend. Non-loopback federation remains unshipped.

The corpus can establish C1 source and contract evidence only. Its fixtures are
synthetic and intentionally exclude message content, prompts, responses,
transcripts, audio, participant identities, credentials, private paths, vectors,
and WDBX payloads. Transport JSON is never a canonical WDBX episode commitment.

Verify the checked-in bytes with:

```sh
python3 tools/abbey_contracts.py verify contracts/abbey
```

Regenerate the reviewable manifest only after intentional corpus edits:

```sh
python3 tools/abbey_contracts.py build-manifest contracts/abbey --write
```

Verification never rewrites the corpus. CI, release, and production consumers
must require exact digest equality before consequential work.

The exact corpus establishes source-corpus evidence at C1 only. The separate
recording-kernel replay may establish C2 for that narrow local path. Neither is
evidence of an installed consumer artifact, runtime integration, authorization,
production federation, durable WDBX behavior, or participant-consented Discord
operation. Those claims require their own program gates and evidence.
