# Account and private-node enrollment authority

Local research implementation of the ecosystem plan's account/enrollment
boundary. This Rust service and browser client are separate from the frozen
14-command ABI CLI and 12-tool MCP surface. They relay encrypted inference and durable governed-tool proposals to enrolled tenant nodes. The service
never decrypts or persists application payloads and does not execute models,
tools or memory retrieval. See [encrypted relay workflow](RELAY.md).

The browser authenticates with one operator-configured OIDC issuer using code
flow, PKCE S256, a browser-bound state cookie and nonce. JWT verification uses
`jsonwebtoken` with AWS-LC: RS256, ES256 or EdDSA only, trusted discovery/JWKS,
matching issuer/audience/authorized party, expiry, not-before, issuance time,
nonce, optional access-token hash, and a verified nonempty email claim.
Email is not an account key and is not persisted. Accounts are unique by
issuer/subject and receive separate random tenant identities. The configured
provider owns signup and email verification; there is no local password API.

## Run locally

Use an existing development OIDC provider; this repository does not provision
an issuer or register a real account. Register the exact redirect URI
`http://127.0.0.1:3000/auth/callback` for the following loopback setup.

```sh
./tools/cargo.sh build --locked -p abi-control-plane --all-targets
mkdir -m 700 /tmp/abi-account-demo
./target/debug/abi-control-plane init /tmp/abi-account-demo/authority.sqlite http://127.0.0.1:3000
```

Save the following configuration as `/tmp/abi-account-demo/service.json`, replace
issuer/client values with your development provider's settings, and set its
permissions to `0600`. Keep any confidential-client secret in this private
file. Do not place model-provider or user-node credentials in it.

```json
{
  "listen": "127.0.0.1:3000",
  "database": "/tmp/abi-account-demo/authority.sqlite",
  "oidc": {
    "issuer": "http://127.0.0.1:8080/realms/abi-dev",
    "client_id": "abi-browser",
    "client_secret": null,
    "origin": "http://127.0.0.1:3000",
    "loopback_development": true
  }
}
```

```sh
chmod 600 /tmp/abi-account-demo/service.json
./target/debug/abi-control-plane serve /tmp/abi-account-demo/service.json
```

Open `http://127.0.0.1:3000`, sign in, then start the node example in a second
terminal (the node directory must not already exist):

```sh
./tools/cargo.sh run --locked -p abi-control-plane --example enroll -- http://127.0.0.1:3000 /tmp/abi-enrolled-node
```

Paste the displayed code into the browser. Compare the exact account, tenant,
public key and capabilities in both places. In the node terminal, type the
account UUID to confirm or anything else to refuse. The example stores owner-only signing/encryption keys, the pinned authority
public key and signed application credential, then proves possession
once. EOF refuses approval. It does not run a worker or infer a model.
An interrupted/expired example leaves its private directory for inspection;
use a new directory for another attempt. Production recovery/key custody is
separate work.

## Health and operator metrics

- `GET /healthz` returns `{"status":"ok"}` when the authority database answers,
  and `503 {"status":"unavailable"}` otherwise. The response has no counts and no
  identifiers.
- `GET /v1/operator/metrics` is off (`404`) unless the configuration sets
  `"operator_token_sha256"` to the 64-digit hex SHA-256 of an operator bearer
  token, for example from
  `printf %s "$TOKEN" | shasum -a 256`. Requests without
  `Authorization: Bearer <token>`, or with the wrong token, get `401`. The
  `200` body holds only fixed keys and integers: accounts, active sessions,
  active/revoked nodes, pending pairings, relay jobs by state, audit events by
  kind over the last 24 hours, and `auth_denied_total`, a process-local count of
  `401` responses since start. Job states and audit kinds outside the closed
  internal lists count as `other`.
- `abi-control-plane metrics DATABASE ORIGIN` prints the same snapshot offline,
  without `auth_denied_total`.

The metrics never contain account, tenant, issuer, subject, session, node or job
identifiers, tokens, digests or payloads. Alerting, export and audit retention
remain operator work (C38 is not qualified for staging or public operation).

Operator procedures (revocation, authority replacement, backup and restore,
upgrade and rollback, shutdown, health and metrics) and their local drills are
in [RUNBOOKS.md](RUNBOOKS.md).

## Protocol v1

All requests and responses are bounded metadata JSON. Unknown mutation fields
are rejected. Browser writes require the account session, exact configured
Origin, and matching `X-CSRF-Token`. Cookies use Secure, HttpOnly (except the
CSRF cookie), SameSite=Lax and `__Host-` names outside explicit literal-loopback
development. No forwarded identity headers, CORS, user-selected issuer URLs,
provider redirects, or callback return URLs are trusted.

| Endpoint | Authority and result |
| --- | --- |
| `GET /auth/login`, `GET /auth/callback` | Provider code flow; ephemeral one-use state, 5-minute expiry |
| `GET /v1/authority` | Public credential verification key, trusted through the configured HTTPS origin |
| `GET /v1/account` | Session-scoped account and enrolled-node metadata |
| `DELETE /v1/account`, `POST /v1/logout` | Session plus CSRF; revoke/delete account or session |
| `POST /v1/enrollments` | Node public key and sorted unique capabilities; returns enrollment ID, one-use browser code and separate node polling secret |
| `POST /v1/enrollments/claim` | Verified browser session plus CSRF; atomically binds a code to its tenant |
| `GET /v1/enrollments/{id}` | Node-only polling bearer secret; returns the immutable local preview or null |
| `POST /v1/enrollments/{id}/confirm` | Poll secret plus node Ed25519 signature over `PairClaim::signing_bytes()`; exact retry returns the recorded credential |
| `POST /v1/nodes/{id}/revoke` | Owning account plus CSRF; other tenants are refused |
| `POST /v1/nodes/authenticate` | Node-signed `abi.node-proof.v1` nonce preview; current account/credential/revocation checked transactionally, then nonce use recorded |

Pairings expire after 5 minutes. Capabilities are a private-workload allowlist
(`memory.retrieve`, `model.infer`, `tool.execute`, `tool.script`), not operator training-pool
membership. Enrollment approval does not approve any future tool execution.
Node nonces are unpredictable, at least 32 characters, expire within 60 seconds,
and are signed over the protocol, audience, node, nonce and expiry. At most
64 successful nonces per node may be retained during that window. Unverified
requests reserve no per-node slots. A successful proof returns metadata only;
it is not a bearer session or permission to dispatch work.

`NodeCertificate` is a versioned Ed25519 **application credential, not an X.509
TLS certificate**. It binds the service audience, tenant, account, node public
key and capabilities. Lifetime is one hour. Signature validity alone never
replaces current authority checks. New enrollments optionally bind an X25519 relay key inside the signed local
approval. Older nodes without this key remain usable for account management
but refuse encrypted jobs; enroll a new node to enable relay. The outbound
relay uses fresh signed permits and durable worker admission. Credential
renewal, rotation and mTLS connections remain open.

## Storage and operational limits

The private SQLite database contains account/tenant identifiers, OIDC
issuer/subject, hashed session/code/poll secrets, node public keys and signed
credentials, used nonces, and the service signing key. It contains no prompts, conversation results, encrypted application payloads,
memory, user-node private keys or model-provider credentials. Relay ciphertext
is retained only in bounded process memory. Job scope/status/digests and used
permit timestamps are metadata; user workers retain their own durable results.
Opted-in v2 jobs additionally retain exact encrypted transport packets in a
private user-node outbox; their delivery policy, manifest and commitments remain
metadata centrally. Recovery restores bounded transient buffers without invoking
inference. See [durable delivery and upgrade instructions](RELAY.md).
Administrative audit rows retain opaque actor/resource IDs, time and operation;
no emails, provider tokens or raw secrets are written to the audit table.
Anonymous pairing and nonce requests do not append administrative audit rows.

Transactions use SQLite FULL synchronization and a rollback journal; audit
failure rolls back its authority mutation. Triggers forbid audit UPDATE/DELETE.
`secure_delete=ON` overwrites deleted records in the active database. Deleting
an account removes its sessions, pairings, nodes and nonce records, so previously
issued credentials fail fresh checks. Opaque administrative audit IDs remain.
This does not promise physical media erasure or deletion from operator backups.

Initialization is explicit and never overwrites or resets state. Database and
configuration files must be private regular owner files, with no symlinks or
hard links; the database directory must be owner-private, and its canonical
ancestors must prevent replacement by another user. Unix is locally
tested; other platforms refuse until equivalent ACL validation is implemented.
The node example uses private Unix files, not platform credential facilities.

Current fixed bounds: 16 KiB account HTTP bodies (384 KiB relay bodies),
1 MiB provider responses, 1,024 pending
logins and pairings each, 10,000 accounts, 16 live sessions per account,
32 unrevoked node registrations per account, and a 64 MiB database page budget.
The relay enforces configurable active/queued quotas (defaults 2/10),
10,000 retained job metadata entries and 16 MiB of buffered ciphertext bytes.
Full queues/buffers return explicit 429 responses; no fallback scheduler runs.
Expired sessions/pairings/nonces are reclaimed on relevant admission paths.
Used nonces expire before removal, so removing them cannot permit valid replay.
Administrative audit is append-only with no automatic archive/rotation yet;
monitor capacity and stop admission before the database is full. Genuine write
failure refuses all authority mutations rather than losing their audit record.

The binary binds loopback only. For an eventual public deployment, configure an
HTTPS origin, disable loopback development, and terminate HTTPS at a controlled
reverse proxy. Proxy rate/concurrency/header/time limits, certificate/key and
JWKS rotation, audit archive/recovery, account abuse controls, container delivery,
backup/restore acceptance, and independent security assessment remain release
work. Discovery/JWKS are validated at startup; restart after provider key changes.
Existing browser sessions expire after eight hours; IdP logout does not remotely
revoke them. No public deployment or real-provider acceptance is claimed.

## Verification

```sh
./tools/cargo.sh test --locked -p abi-control-plane --all-targets < /dev/null
./tools/cargo.sh clippy --locked -p abi-control-plane --all-targets -- -D warnings
./tools/check.sh
```

Tests use an isolated private database and a local mock issuer that actually
signs JWTs and validates PKCE during token exchange. They cover isolated
accounts with the same email, forged/expired/unverified tokens, CSRF/Origin,
single-use pairing and local approval mutation, concurrent claims, audit
failure, restart/revocation/deletion, signed-nonce replay and metadata budgets.
They do not substitute for separate-host execution, actual IdP acceptance,
Windows/Linux runtime qualification, separate-host partitions, useful model
quality, general governed platform adapters or public deployment. The inference relay acceptance test uses
actual local TCP requests, two verified mock-issuer tenants, separate worker
histories, local deterministic providers and separately encrypted client results.


The recovery continuation adds real local worker/authority child-process
kill/restart and committed-response-loss acceptance, a durable provider invocation
marker, exact encrypted packet comparisons, replay after the execution deadline,
current credential/cancellation refusal, private outbox failure/retention tests
and atomic verified client-import tests. These extend the local source boundary;
they do not qualify power loss or remote hosts. Final source/gate receipts are
recorded in `docs/verification/2026-10-22-production-ecosystem.md`.

## Governed tool continuation

Enroll a separate tool-only node with the enrollment example's `--tools` flag.
Use `relay_job prepare-tool NODE_DIRECTORY TOOL_JOB.json NEW_JOB_DIRECTORY
TOOL_REVISION` to encrypt a versioned proposal; the browser accepts it only in a
durable v2 wrapper. A proposal contains `protocol: "abi.governed-tool.v1"`,
`tool`, `revision`, and exact JSON `parameters`. It cannot choose grants,
approvers, an executable, credentials, workspace or adapter implementation.

The user node composes `DurableRelayRuntime::execute_governed` with Abbey's
`AbbeyLocalApprovalAuthority`. Full private previews stay on the user node;
explicit decisions bind tenant/account/node/user, request, resource, parameters,
expected effects and immutable expiry. The control plane exposes no approval
endpoint. Current tool inspection permits may refresh for the same original job
while local durable worker and Abbey intents still prohibit duplicate dispatch.

Sibling Abbey's `examples/governed_relay_worker.rs` assembles a local-terminal
scratch-file demonstration. General file/process/desktop/messaging adapters and
cross-platform/live-account acceptance remain separate. Follow the
[governance contract](../../docs/spec/durable-governed-tools.md) and execution
ledger for current validation rather than treating example compilation as full
product acceptance.
