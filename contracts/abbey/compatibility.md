# Abbey contract compatibility

Contract major, additive revision, and capability version are independent.
Breaking semantics require a new schema identifier or contract major; an
existing wire shape is never silently reinterpreted. Authority-bearing
envelopes reject unknown fields. Tolerant metadata may preserve only a bounded
`extensions` object and never use it to widen authority or establish evidence.

Every consumer vendors exact corpus bytes and verifies the aggregate and
per-file SHA-256 commitments. A mismatch disables authorization, consent
opening, execution, and durable writes. A developer profile may expose
read-only diagnostics with a loud mismatch status. It may not weaken that
fail-closed boundary.

Rollback returns the consumer to the last qualified corpus digest. Failed
versions remain in compatibility history rather than being silently rewritten.
The corpus may later be extracted without path or byte changes when one of the
approved extraction triggers requires an independent release.
