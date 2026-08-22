# Abbey contract corpus

This directory is the canonical, language-neutral Program 1
`abbey-contracts` corpus. It contains UTF-8 JSON Schema Draft 2020-12 documents
and synthetic golden fixtures. The corpus is data-only: it has no Cargo feature,
generated binding, runtime listener, authorization actuator, Discord behavior,
model call, storage write, or production-deployment effect.

The source corpus is qualified as one exact byte set. `manifest.json` lists
every normative schema, fixture, and policy document once; its aggregate digest
also commits to a normalized copy of the manifest itself. Consumers vendor the
listed bytes unchanged, pin that aggregate digest, and validate real inputs with
their native schema implementation. Generated types are projections, not a
replacement source of truth.

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

This implementation establishes source-corpus evidence at C1 only. It is not
evidence of an installed consumer artifact, runtime integration, authorization,
production federation, durable WDBX behavior, or participant-consented Discord
operation. Those claims require their own program gates and evidence.
