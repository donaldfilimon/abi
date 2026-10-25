# Encrypted enrolled-tenant inference relay

This is a local research implementation with actual outbound HTTP acceptance.
Only a verified account's enrolled private node receives its jobs. Requests and
results use RFC 9180 HPKE (X25519, HKDF-SHA256, ChaCha20-Poly1305); the complete
service/account/tenant/node/job context and request/result direction are
associated data. Ed25519 signatures authenticate node actions, handoffs and
fresh dispatch permits separately from HPKE base-mode confidentiality.

The control plane stores job scope, ciphertext commitments, status and permit
timestamps. It never has the node or client decryption keys, and neither
plaintext nor ciphertext application payloads are written to its database.
Ciphertext is buffered only in bounded process memory. User-node control logs
are separate private files and can contain plaintext results for exact retries.

## Local workflow

First follow [account setup and pairing](README.md). Use the current enrollment
example: the local approval must include `encryption_public_key`. It saves
`node-key.json`, `encryption-key.json`, `authority-key.json` and
`node-certificate.json` with owner-only permissions. Old enrollment credentials
without an encryption key cannot run relay jobs; enroll into a new private
node directory. Application credentials currently last one hour, with no
automatic renewal.

Build the service and examples:

```sh
./tools/cargo.sh build --locked -p abi-control-plane --all-targets
```

Start the account service with the private configuration from the setup guide.
In another terminal, start the user-node probe worker **before** preparing a job:

```sh
./target/debug/examples/relay_worker /tmp/abi-enrolled-node --initialize --loopback-development
```

`--initialize` explicitly creates both a new private worker-control log and
`relay-outbox.sqlite` (with a stable `.lock` sidecar). It refuses existing state.
On subsequent launches, omit it:

```sh
./target/debug/examples/relay_worker /tmp/abi-enrolled-node --loopback-development
```

For a stopped older installation that already has `worker-control.log` but has
never had an outbox, use the explicit outbox-only migration:

```sh
./target/debug/examples/relay_worker /tmp/abi-enrolled-node --initialize-outbox --loopback-development
```

This validates and preserves the existing worker history and exclusively creates
the new outbox. It refuses an existing outbox. Use it only for that deliberate
upgrade after verifying the outbox was never initialized; it is not recovery for
a lost/deleted outbox. Existing v1 jobs do not acquire durable delivery through
this operation. Never remove or truncate either journal to make startup pass,
and never retry with `--initialize` after an ordinary reopen failure.

Missing, corrupt or private-permission-invalid history refuses startup; it
never falls back to empty history. `--loopback-development` permits plain HTTP
only for a literal loopback origin. Use HTTPS and omit this flag outside local
development. The worker pins the authority key saved during local enrollment,
opens only outbound connections, and prints job IDs/status rather than content.

The included worker runs `local-echo-v1`, a deterministic transport probe. It
has no trained model, shell, file/process tools, provider credentials or model
download path. Library `RelayRuntime` accepts a startup-bound `ModelProvider`
and checks its requested revision. Native provider cancellation is cooperative;
this is not an OS sandbox for untrusted providers.

Prepare a request locally, keeping the response key on the submitting client:

```sh
printf 'private relay probe\n' > /tmp/abi-relay-prompt.txt
./target/debug/examples/relay_job prepare /tmp/abi-enrolled-node /tmp/abi-relay-prompt.txt /tmp/abi-private-job local-echo-v1
```

The job directory must be new. Only `node-certificate.json` and
`authority-key.json` from the node directory are needed for preparation; the
client does not use the node's private signing/decryption keys. For clients on
another machine, transfer these public artifacts through a trusted channel,
keep them in an owner-private directory, and retain the authority-key pin.

In the owning account's browser, choose the generated `submission.json` under
**Encrypted jobs**, then submit it. The browser sends sealed bytes, never the
prompt. Check job status; when complete, download the encrypted result. Open
that downloaded file locally with the matching private reply key:

```sh
./target/debug/examples/relay_job open /tmp/abi-private-job /path/to/abi-result-JOB_ID.json
```

This verifies the exact job binding, decrypts locally and checks the worker
result digest before displaying output. The browser never receives the private
reply key. Prepared jobs expire after at most two minutes (or earlier credential
expiry), so upload promptly. Encrypted results are transient and must be fetched
before the job deadline or a service restart. Keep clocks synchronized; time
checks fail closed.

## Admission and execution boundaries

| Phase | Durable state | Meaning |
| --- | --- | --- |
| Session-authenticated submission | `queued` | Node, account, tenant, capability, deadline, ciphertext digest and finite budgets validated |
| Signed outbound node poll | `offered` | Exact ciphertext handoff recorded before delivery; this offer is never sent again |
| Fresh signed node inspection | `dispatched` | Enrollment/revocation/ownership rechecked; a signed permit is recorded before issuance |
| Local worker admission | User-node FileControlStore | Stable signed worker envelope admitted once; permit rechecked after durable sync and before provider invocation |
| Signed encrypted result | `completed` | Ciphertext commitment recorded; response packet held transiently for the owner |
| Lost handoff, restart or uncertain execution | `unresolved` | No automatic redispatch or repeat; this does not prove an effect did or did not happen |

Permits are bound to the exact encrypted submission and expire within five
seconds. Re-inspection returns the same issuance window; it does not renew
execution authority. A revocation before permit issuance blocks execution.
Once a permit has been issued, the job has crossed the dispatch boundary;
revocation/cancellation cannot prove that a local run did not start. The worker
checks permit freshness again after its admission write. Stable worker IDs,
namespace, nonce, principal, revision and request identity make local retries
return durable cached results or existing unresolved admissions.

Cancellation of `queued`/`offered` work is definite. Cancellation after dispatch
records `cancel_requested` and continues charging active capacity. The streaming
worker polls authority every 250 ms when transport is responsive, signals its
cooperative cancellation token, and acknowledges `Stopped` only after its provider
returns. Only that signed acknowledgement records `cancelled`. A lost worker,
revocation, expired credential, deadline or service restart leaves uncertain work
`unresolved`. A provider panic likewise leaves local admission unresolved. An in-flight request may take up to the client's 15-second transport
timeout; no hard real-time cancellation bound or preemptive OS termination is
claimed. The local inference history records a stopped run as a failed result,
not a fabricated signed worker `CancellationRequest` or successful completion.
No arbitrary tool effects or automatic retry are offered. A caller must not
create a new job ID merely to hide an unresolved outcome.

After a control-plane restart, outstanding queued/offered/dispatched/cancel_requested jobs are
marked unresolved because application payloads were never persisted there.
Completed metadata survives, but buffered output may be unavailable. The
worker's private cached results survive its own restart. For opted-in v2 jobs, the private durable outbox and replay-only protocol below
restore exact saved output. Version-1 frames still disappear on restart; only
their ciphertext commitments and strict sequence survive. Metadata is returned
rather than authorizing another run.

## Encrypted streaming and cancellation

The native worker example uses `RelayClient::run_durable_once`, which reconciles
retained output before polling for fresh work. Legacy `run_streaming_once` and
`run_once` remain library APIs for v1 jobs. Synchronous
provider work runs in a blocking task while the async client delivers encrypted
deltas and inspects cancellation concurrently. A one-frame channel and per-frame
acknowledgement provide backpressure. An upload/inspection failure cancels local
execution cooperatively; dropping the runner also signals cancellation. Unknown
outcomes never authorize another provider call. Noncooperative providers still
require a separate process sandbox and are outside this acceptance.

`NodeAction::Stream` carries an `abi.relay-stream.v1` frame: strict zero-based
sequence (at most 16), HPKE ciphertext, and an Ed25519 signature from the enrolled
worker. Both encryption and signature bind the complete job scope and sequence.
The relay persists only ciphertext commitments and byte counts. Identical frame
retries return recorded state; changed duplicates, gaps, foreign workers, extra
frames and quota exhaustion refuse without admitting another frame.

`GET /v1/jobs/{id}` includes ordered transient `frames`, even before terminal
completion. This is a bounded snapshot polling interface, not SSE/WebSocket;
clients can display each newly verified sequence once. Browser **Watch encrypted
stream** polls counts/status every second through cancellation acknowledgement.
**Download encrypted output** saves the current encrypted snapshot. Decrypt its
signed prefix on the original client (new prepared jobs retain the node key):

```sh
target/debug/examples/relay_job open-stream PRIVATE_JOB_DIR DOWNLOADED_OUTPUT.json
```

A partial prefix does not prove successful completion. The existing `open`
command still verifies the terminal result digest. Neither browser watching nor
local decryption automatically retries execution.

## Durable delivery and reconnect

Durable delivery is an explicit v2 opt-in. Prepare it locally instead of using
the legacy `prepare` command:

```sh
./target/debug/examples/relay_job prepare-durable /tmp/abi-enrolled-node /tmp/abi-relay-prompt.txt /tmp/abi-durable-job local-echo-v1
```

The new private job directory contains the reply key, original binding, enrolled
node public key, immutable delivery policy, encrypted `submission.json` wrapper
and a private verified-import journal. Upload that wrapper through **Encrypted
jobs**. The browser validates its exact schema before posting to `/v2/jobs`;
plaintext or unexpected fields refuse locally. The browser never receives the
reply private key.

The helper fixes retention at 24 hours from submission. Library callers may
choose 60 seconds through seven days; the signed policy is immutable after
admission. Retention does not extend execution time, credential lifetime or
account authorization. The current one-hour node credential can therefore make
replay unavailable before the default retention deadline; automatic renewal is
not implemented.

The user node reserves a full encoded 1 MiB per job before worker admission.
Its outbox has a 64 MiB allocated-database ceiling, at most 256 retained identity
rows, at most 16 deltas plus one terminal packet, and the original 65,536-byte
maximum decrypted result. Physical database overhead and reservations may refuse
new work before all 256 rows can be active. Full storage refuses admission;
there is no eviction of another job's only durable output. Ciphertext and its
local prefix commitments are synced before notification/upload. Recovery never
calls a provider, resumes generation or renews a dispatch permit.

After disconnect/restart, the worker reads recorded history and obtains a
short-lived replay-only grant under current ownership, enrollment, credential,
capability, retention and cancellation checks. Grants last at most 30 seconds,
bind the current service epoch and exact packet digests, and cannot authorize
execution. Every retry signs a fresh node proof and reuses the saved ciphertext.
A central upload acknowledgement records a commitment; it is neither durable
central payload storage nor a client receipt. A new epoch can require replay
even after a prior upload acknowledgement. Retryable network/full responses
back off from 250 ms to five seconds plus up to 250 ms of random jitter, with
one packet in flight and at most 16 publications per recovery round.

Missing prior admission is abandoned; prior-process active admission becomes
unresolved and releases its worker quota. A recorded terminal result can fill
the crash gap before terminal sealing after the saved delta prefix is checked.
An existing packet is never resealed. Interrupted work remains partial/unresolved
rather than calling the provider again. Cancellation observations are committed
locally before signalling a stop; success replay cannot override cancellation.

Browser **Watch encrypted stream** reports running/recovering, partial/unresolved,
unavailable output and retention expiry using bounded snapshots. **Download
encrypted output** saves one bounded encrypted delivery page; repeated clicks
advance the page cursor. Import each downloaded page locally:

```sh
./target/debug/examples/relay_job import /tmp/abi-durable-job /path/to/abi-result-JOB_ID.json
./target/debug/examples/relay_job open-durable /tmp/abi-durable-job
```

Import verifies node signatures, AEAD, exact scope, contiguous prefix and terminal
result commitments before atomically saving encrypted packets and cursor. It
refuses conflicting duplicates and accepts identical pages after restart. No
plaintext or reply private key is written into that journal. `open-durable`
renders the saved verified view; printing stdout is replayable, not an
exactly-once external effect.

Import writes `delivery-ack.json`. Upload it through **Local verified receipt
acknowledgement** to record explicit owning-client receipt. Browser metadata does
not claim browser decryption. Neither server upload nor client receipt deletes
the outbox early. At the immutable deadline transport ciphertext is purged and
identity/digest/outcome tombstones remain; worker control history and storage
snapshots/backups have their own retention requirements. This is not secure
physical erasure or WDBX retained-memory deletion.

A corrupt/uncertain write poisons the affected handle and stops fresh dispatch.
Keep both user-node journals and the authority's current metadata for diagnosis;
no automatic reset, rollback or ciphertext reconstruction from a missing store
is supplied. Do not restore old worker history to make an older binary run.
Authority schema v2 rejects old readers; once post-upgrade mutations occur,
forward-fix the current database rather than restoring a stale backup that could
resurrect jobs or revoked permissions. Pre-mutation rollback requires stopped
writers and an exact source/state check.

## Versioned HTTP interface

| Request | Authority |
| --- | --- |
| `POST /v1/jobs` | Browser session, exact Origin, CSRF; `Submission` metadata must match its account and an enrolled node |
| `GET /v1/jobs/{id}` | Owning account only; returns `JobView`, including optional encrypted result |
| `POST /v1/jobs/{id}/cancel` | Owning session plus Origin/CSRF |
| `POST /v1/relay/node` | Fresh single-use Ed25519 proof bound to the complete action, including additive `Recovery` inspect/publish |
| `POST /v2/jobs` | Owning session plus Origin/CSRF; sealed `DurableSubmission` with immutable delivery policy |
| `GET /v2/jobs/{id}/delivery?after=N` | Owning account; bounded encrypted packet page after the next zero-based delta cursor |
| `POST /v2/jobs/{id}/delivery/ack` | Owning session plus Origin/CSRF; exact verified manifest/cursor/optional terminal receipt |

Unknown fields, forged signatures, action mutation, tenant crossing, operator
training capabilities and unsupported job types are refused. An encrypted
request must decode locally as `InferenceJob` with the configured model revision,
at most 8 KiB of prompt and 1–1,024 requested output tokens. Provider tool calls
are refused by the bounded inference executor. Result serialization, chunks,
tokens, deadline and ciphertext lengths are bounded and digests checked.

Optional startup configuration:

```json
"relay_limits": {
  "active_jobs": 2,
  "queued_jobs": 10,
  "job_records": 10000,
  "memory_bytes": 16777216
}
```

These are per-account active/queue limits and global retained-record/ciphertext
byte limits. Job metadata tombstones are retained rather than silently erased
for capacity; full queues/buffers/record budgets return 429. Account deletion
removes its metadata and buffered packets. Global ciphertext byte accounting
bounds packet contents, not total process resident memory. JSON relay bodies
are capped at 384 KiB and individual packets at 64 KiB plus HPKE overhead. An
exclusive private relay lock permits one relay process per database. Never
replace/unlink live database, lock or worker-history files.

## Acceptance and remaining release work

Focused tests exercise two verified mock-issuer tenants over actual local TCP,
separate user-worker control files, separate client response keys, cross-tenant
refusals, action mutation, replay, full queues, restart ambiguity, cached worker
reopen, delayed admission/expired permits and absence of application payloads
from central storage. Recovery tests additionally kill/restart real worker and authority children,
drop committed-upload responses, compare exact saved ciphertext, and retain a
durable provider invocation marker across restarts. SQLite VFS tests inject
actual partial writes, full-storage and sync failures, while client journal tests
cover atomic import/cursor failure and restart before rendering. These are
single-host local proofs; the execution ledger records final gate receipts.

Public deployment/IdP acceptance, separate-host partitions, real model artifacts,
preemptive provider termination, separate-host delivery qualification, Abbey durable approvals
and general governed platform adapters, key renewal/rotation, backup/audit archival, independent security
assessment and native Windows/Linux acceptance remain open. No public service,
production cryptography assessment, availability SLA or useful model quality is
implied by the probe.

Final recovery hardening: native action proofs last five seconds, while replay
grants last at most thirty seconds. Reconnect rotates through retained jobs with
at most sixteen total network exchanges per round, and invalidates cached
confirmations when the relay epoch changes. Network delays beyond proof expiry
refuse that attempt; retained ciphertext remains available for a fresh attempt.
Consumer journal format 2 pins both node and authority keys and verifies the
signed delivery policy before advancing its cursor. Previously imported output
remains locally readable after cancellation; cancellation refuses new terminal
imports and publication. Contradictory control/outbox history refuses startup,
and an uncertain execution or cancellation persistence failure blocks fresh work
until the existing journals are reopened and reconciled.

## Locally approved tool proposals

The separate tool-only path uses durable v2 submissions with `tool.execute`.
Prepare an encrypted `ToolJob` with `relay_job prepare-tool`; use an enrollment
created with `enroll ... --tools`. Sibling Abbey's
`examples/governed_relay_worker.rs` demonstrates exact local-terminal approval
and a fixed create-new scratch-file effect. Its startup-owned configuration
supplies all authority, resources and adapter selection. The browser has no
approval endpoint or plaintext preview.

The approved local decision is consumed atomically with Abbey's durable intent.
Current service ownership and full local authority are rechecked before effects.
Worker admission precedes the effect so interrupted control/approval histories
remain unresolved instead of dispatching again. Pending local previews survive
user-node reopen under current service authority; an ambiguous service handoff
can still close unresolved under the best-effort policy. Approval persistence is
not permission to recreate missing journals or repeat an uncertain effect.

The local native TCP/PTY acceptance tests enrollment, explicit decisions,
private client import and identical ciphertext after server/worker restart.
Its account/session rows are seeded fixtures; it does not qualify public OIDC
login, separate hosts, other platforms or general desktop/messaging effects.
