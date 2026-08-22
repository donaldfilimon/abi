# Abbey contract corpus

This directory is the canonical, language-neutral Program 1
`abbey-contracts` corpus. It contains UTF-8 JSON Schema Draft 2020-12 documents
and synthetic golden fixtures. The corpus is data-only: it has no Cargo feature,
generated binding, runtime listener, authorization actuator, Discord behavior,
model call, storage write, or production-deployment effect.

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
