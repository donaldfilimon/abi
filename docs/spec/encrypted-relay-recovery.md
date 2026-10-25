# Encrypted relay recovery specification

Status: implemented local source slice; final whole-workspace qualification is
recorded separately in the execution ledger. Scope: durable output redelivery
and reconnect recovery for enrolled private inference workers. The approved
[ecosystem specification](../superpowers/plans/2026-10-22-abi-production-ecosystem.md)
continues to govern ownership and release boundaries.

## Required outcome

An enrolled worker can reconnect after lost requests, lost acknowledgements,
control-plane restarts or its own process restart and redeliver its saved encrypted
stream/result packets. Recorded inference is never executed a second time to
repair delivery. Interrupted inference returns a verified partial prefix and an
unresolved execution outcome; this slice does not resume generation.

Durability belongs to user infrastructure. The control plane continues to persist
only account, scope, lifecycle, quota, consent, ciphertext commitments and delivery
metadata. Neither plaintext nor encrypted application payloads enter its database,
journal, backups, logs or operator-managed object storage.

## Local implementation and evidence

The 2026-10-22 continuation implements this contract in `abi-worker` and
`abi-control-plane`. `WorkerManager::recorded` reads exact historical outcomes
without admission or provider invocation; `recover_interrupted` writes an
unresolved tombstone only for replayed prior-process admission. The private
`FileRelayOutbox`, `DurableRelayRuntime` and reconnect driver save sealed output
before upload and restore transient relay buffers using replay-only authority.
`VerifiedDeliveryStore` commits verified encrypted packets and its contiguous
cursor atomically. Native and browser helpers support the durable opt-in.

Source and test entry points:

- `crates/abi-worker/tests/recovery.rs`: historical lookup, exact identity,
  prior-process abandonment and permanent refusal after unresolved recovery.
- `crates/abi-control-plane/src/delivery_outbox/`: private synced storage,
  reservations, immutable bytes, cancellation observation and real SQLite VFS
  full/partial-write/sync failure tests.
- `crates/abi-control-plane/src/relay/tests/recovery.rs`: schema migration,
  replay grants, current ownership/revocation/cancellation, post-deadline replay,
  capacity and actual encoded page bounds.
- `crates/abi-control-plane/src/service/tests/reconnect_process.rs`: real local
  child-process kills/restarts and TCP response loss with durable provider counts.
- `crates/abi-control-plane/src/delivery_consumer.rs` and `web/tests/recovery.cjs`:
  verified prefix/terminal import, cursor failure/reopen and strict browser files.

The [execution ledger](../verification/2026-10-22-production-ecosystem.md) records
current test receipts and the final gate separately from implementation. These
are local Unix/macOS fixture proofs, not public deployment, power-loss,
separate-host or Windows/Linux runtime qualification. Version-1 jobs remain
ephemeral; upgrading metadata does not grant them durable delivery.

## Invariants

1. Only a fresh worker admission may call the provider. Recovery APIs have no
   provider parameter and cannot admit, renew or resume execution.
2. A node journals a sealed packet and successfully syncs it before first upload.
   Every retry reuses the exact saved packet bytes and packet identity. It signs a
   fresh action proof; it never reuses a single-use node proof.
3. A server acknowledgement proves a ciphertext commitment was recorded, not that
   the packet will survive server restart or that the client consumed it.
4. Existing scope, signatures, tenant/node ownership, ordered 16-frame ceiling,
   result bounds, cancellation, active/queued limits and replay protection remain.
5. Payload delivery and execution have separate clocks and independent states.
   A delivery grant never supplies or extends a dispatch permit.
6. Admission uncertainty is fail-closed. Missing or contradictory history never
   causes automatic provider invocation, new job identity or empty-log creation.
7. Cancellation is sticky. Recovery may deliver an authorized prefix and report
   a stopped outcome, but cannot turn cancellation into successful inference.
8. Owner deletion or node revocation blocks future replay. Expired credentials
   require an actual renewal mechanism; this slice never bypasses expiry.
9. A verified client imports each `(job, packet kind, sequence, digest)` once.
   Conflicting duplicates refuse. Rendering is derived from the saved prefix;
   human-visible stdout or external effects do not receive an exactly-once claim.
10. A corrupt or uncertain persistence write poisons the relevant handle and stops
    fresh dispatch. Checksums are corruption detection, not rollback protection.

## Storage ownership and budgets

The worker creates `relay-outbox.sqlite` in the explicit user-node private directory, beside
`worker-control.log`. Use the existing private-path policy, a stable exclusive
lock, SQLite DELETE journaling, `synchronous=FULL`, foreign keys and explicit
create/reopen. All database/journal files remain owner-only. Never replace a live
locked file, follow a symlink, reset a missing existing outbox or use `~/.abi/`.
Windows continues to refuse until equivalent ACL acceptance exists.

Outbox defaults: 64 MiB allocated database ceiling, 1 MiB encoded artifacts per
job, 256 retained jobs, 16 delta frames, one terminal packet per job. Reserve a
full 1 MiB job allowance before admission; do not accept inference if reservation
fails. Enforce encoded record/SQL page ceilings as well as raw ciphertext limits.
Delta raw plaintext total and terminal plaintext each remain bounded by the
original `JobBinding.max_result_bytes` (at most 65,536 bytes). Public relay
request/response bodies remain capped at 384 KiB; recovery pages stop before the
actual encoded response exceeds that ceiling.

The outbox stores the exact signed dispatch, immutable delivery policy, packet
bytes, ordered ciphertext commitments, observed cancellation, terminal facts and
transport acknowledgements. It stores no plaintext prompt, provider credentials or decryption keys.
Private local delta hashes support comparison with recorded worker results;
those hashes are not published centrally. Existing control history can still contain private plaintext
results on the node; this is not a universal encrypted-at-rest claim.

Retention defaults to 24 hours after original submission, configurable from
60 seconds to seven days, fixed at admission. It is a transport-artifact policy,
not a second WDBX memory/backup erasure implementation. A server upload ack never
deletes the only durable copy. Retain until the immutable delivery deadline or
explicit local owner discard, even after client receipt; retain identity/digest
and terminal tombstones so deletion cannot permit job re-execution. Expiry purges
transport ciphertext, not the private worker control log. Secure physical erasure
of SQLite free pages, storage snapshots and backups remains an explicit limitation.

## Interfaces and compatibility

Keep v1 `Submission`, `Dispatch`, `JobBinding`, `StreamFrame`, signing bytes and
HPKE associated-data encodings byte-compatible. Add opted-in durable jobs through:

- `POST /v2/jobs`: session/Origin/CSRF-authenticated
  `DurableSubmission { submission: Submission, delivery: DeliveryPolicy }`.
- `GET /v2/jobs/{id}/delivery?after=N`: owning-account authenticated, bounded
  encrypted packet page and authoritative delivery status. `after` is the next
  zero-based delta sequence, 0 through 16, never a caller-supplied tenant.
- `POST /v2/jobs/{id}/delivery/ack`: owning-account/Origin/CSRF, commits exact
  imported manifest root/contiguous cursor and optional terminal digest. A future
  reconnect still checks ownership and revocation before returning packets.
- Existing `/v1/relay/node`: additive signed
  `NodeAction::Recovery { request: RecoveryRequest }`, with inner protocol
  `abi.relay-recovery.v1`. Reuse the complete action-bound node-proof machinery.
  New replies add an optional `recovery` member; existing v1 behavior remains.

`DeliveryPolicy` fixes `until_unix_s`, original binding digest and submission
ciphertext digest. The authority signs that policy with the dispatch; caller
policy values are validated and cannot outlive the configured seven-day ceiling.
A public delivery deadline is not a credential lifetime extension: current node
credentials and account authorization must still be valid on every reconnect.
Current one-hour credentials can therefore make delivery unavailable before a
24-hour retention deadline; automatic renewal is not supplied by this slice.

`RecoveryAction` has `Inspect { job_id, manifest }` and
`Publish { grant, packet }`. An immutable terminal manifest binds exact job scope,
submission digest, ordered frame digests, terminal ciphertext digest (if any),
recorded outcome and completion timestamp. Partial/unknown manifests contain the
saved prefix and no successful terminal assertion. Manifest revisions may only
extend a previously committed prefix; terminal manifests are closed forever.

A signed `ReplayGrant` binds job/account/tenant/node, policy and manifest digest,
current service epoch, allowed packet identities/digests, current cancellation
state, issuance and exclusive expiry (30 seconds, no later than delivery expiry).
It contains no inference capability/permit. Uploads recheck current ownership,
revocation, cancellation and policy, even when the grant is still cryptographically
valid. Stale-epoch grants refuse and require a fresh inspection.

`RecoveryReply` distinguishes authentication/refusal, unavailable/expired,
capacity-full, recorded commitment, missing transient packet, exact packet cached
and contradictory metadata. Inspect returns both the durable commitment cursor
and the packet identities currently present in this relay epoch; those are not
the same cursor. An empty memory buffer must never be mistaken for client receipt.
Do not infer retryability from the current generic `Error::Denied` alone.

The authority schema migrates transactionally from relay version 1 to 2, adding
immutable delivery policy, cancellation marker, execution outcome/recovery reason,
delivery state, generation and commitment metadata. Existing v1 jobs stay ephemeral;
no retroactive durable guarantee. Old binaries refuse schema version 2. Automatic downgrade is unsupported. Restoring a pre-upgrade metadata backup for
operation is allowed only before any post-upgrade mutation, with stopped writers
and an exact source/state check. Once jobs, cancellation, enrollment or other
metadata have changed, preserve the new database and forward-fix; do not restore
an older authority state and resurrect revoked permissions or forgotten jobs. CLI 14 and MCP 12 stay frozen.

## Write ordering and reconnect behavior

1. Reserve outbox capacity; journal exact dispatch/policy as `Prepared`; sync.
2. Existing worker manager admits and syncs before provider invocation. Persisted
   outbox preparation alone is never execution authority.
3. Seal each delta once; journal packet/digest/sequence; sync; then enqueue upload.
   One packet in flight provides bounded transport backpressure. Network failure
   still cancels generation cooperatively; saved frames remain deliverable.
4. Worker manager commits terminal result before publishing success. Then seal and
   journal the terminal packet/manifest; sync before upload. If it crashes in the
   gap, read the recorded result through a new read-only worker lookup, verify
   the saved frame prefix, seal the unpublished terminal once, and persist it.
5. On reconnect, validate both stores and current enrollment; reconcile all
   retained jobs before polling for new work. Obtain a replay-only grant and send
   missing exact packets in order, including a cached terminal packet.
6. Record upload acknowledgements locally. A changed service epoch or a report of
   missing transient packets replays retained packets despite prior upload acks.
7. Valid terminal evidence may resolve an earlier restart/transport-uncertain
   central outcome; it cannot override cancellation, expiry-of-execution outcome,
   revoked authority or a contradictory committed terminal. Completion must have
   been durably recorded within the original execution deadline to report success.
8. History containing only admission never resumes inference. Reconnect can deliver
   its saved partial prefix and report unresolved. `Prepared` plus absent worker
   admission is also abandoned without execution on the recovery path; a small
   availability loss is accepted in exchange for a simple dispatch boundary.

For a previously admitted result lookup, validate exact envelope signature,
principal, namespace, audience, request/envelope digests and nonce without applying
execution-time admission. Return Missing/Admitted/Completed/Cancelled as recorded.
It must not append events, create leases, invoke authority guards or call providers.
Historical lookup compares the originally recorded envelope/capability identity;
a newly configured model revision must not invalidate a previously committed
result's delivery.

A separate, explicitly mutating `recover_interrupted` operation can durably mark
an admission replayed from a prior process as `Unresolved`, releasing its active
quota while retaining job/nonce/request tombstones. It refuses admissions created
by the current manager. The exclusive control-history lock and completed startup
replay are required. It cannot claim success/cancellation, create a new lease or
make the same job admissible again. Add a tagged worker control event for this
transition; new readers accept old logs, old readers encountering that event fail
closed. Never roll a worker log back to an earlier backup to make old code run.

On central restart preserve irreversible execution/cancellation facts and set
delivery to awaiting replay. Outstanding v1 jobs retain their old conservative
unresolved behavior. All v2 jobs with uncertain execution remain non-dispatchable.
Replayed packets restore buffers only, without recharging active/queued job quota
or appending duplicate frame/audit commitments. Current memory limits still apply;
429/full does not erase packets from the worker outbox.

## Failure and acceptance matrix

| Failure boundary | Required recovery | Provider invocation ceiling |
| --- | --- | --- |
| Prepared outbox before admission | Mark abandoned/unknown; no recovery dispatch | 0 |
| Admission sync before provider | Unresolved; optional saved prefix only | 0 or 1; never another call |
| Packet sync before first upload | Publish exact saved packet | 1 |
| Upload commits, response is lost | Fresh proof; same packet/digest, idempotent acknowledgement | 1 |
| Central restart after upload ack | New epoch; restore missing buffers from outbox | 1 |
| Local completion before terminal sealing | Read-only recorded-result lookup, persist terminal once | 1 |
| Terminal sync before upload or after lost terminal ack | Replay exact terminal ciphertext | 1 |
| Worker crash during provider work | Unresolved prefix; no resumed generation | 1 |
| Cancellation during disconnect/restart | Sticky cancellation; never success or renewed dispatch | 1 |
| Corrupt, missing, truncated or contradictory stores | Explicit refusal; no reset, reseal or new execution | No new calls |
| Revoked/deleted/expired tenant/node or changed scope | Refuse replay; retain local evidence until retention policy | No new calls |
| Server/client/disk quota exhausted | Explicit bounded refusal; preserve durable packets | No new calls |

Real local TCP acceptance uses two independently verified tenants, separate worker
processes/stores, distinct client reply keys and a provider counter durable across
child process restart. Kill and restart actual processes and drop HTTP responses
at each boundary; do not substitute only reopening mocks. Client-import tests prove
identical packet bytes, contiguous verified output and conflict refusal. Central
DB/journal/backups/log checks search both plaintext canaries and exact ciphertext
bytes. Filesystem fault injection covers short write, sync failure, disk full,
corruption, lock contention and permission replacement. Local process-kill tests
are not a claim of power-loss, separate-host, Windows or cloud-operation acceptance.
