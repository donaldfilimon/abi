# Control-plane operator runbooks

These runbooks cover the local `abi-control-plane` service described in
[README.md](README.md) and [RELAY.md](RELAY.md). Each one gives preconditions,
commands, the signal that shows it worked, and the test that exercises the
procedure. The drills are in `src/service/tests/runbook_drills.rs`.

**Limits.** The drills run on one Mac against a temporary private database, a
local mock OIDC issuer and in-process or loopback HTTP. They show what the
service code does. They are not staging, separate-host or public-deployment
evidence: executing these procedures on staging and on separate hosts (C36/C41)
stays external work, as do live identity-provider acceptance and Linux/Windows
runtime qualification.

Run every drill:

```sh
./tools/cargo.sh test --locked -p abi-control-plane --lib -- runbook_drills < /dev/null
```

The examples below use the README's demo paths: database
`/tmp/abi-account-demo/authority.sqlite`, configuration
`/tmp/abi-account-demo/service.json` and origin `http://127.0.0.1:3000`.
Substitute your own.

## 1. Node compromise or tenant revocation

**Preconditions.** The account owner can sign in to the browser client. The
compromised node's ID is visible in the account's node list.

**Procedure.**

1. Sign in as the owning account and revoke the node from the browser client.
   This calls `POST /v1/nodes/{id}/revoke`, which needs the session cookie, the
   exact configured `Origin` and a matching `X-CSRF-Token`. Other tenants get
   `401`.
2. To revoke every node and session of a tenant at once, delete the account
   (`DELETE /v1/account`). This removes its sessions, pairings, nodes and nonce
   records. Opaque administrative audit IDs remain.
3. Re-enroll a replacement node with a new key directory (README, "Run locally").

**Verification signal.** `GET /v1/account` lists the node with
`"revoked": true`. The next node proof at `POST /v1/nodes/authenticate`
returns `401`, and an existing one-hour credential stops working at once,
because every authentication rechecks current revocation state. The
`node.revoked` audit count rises in `/v1/operator/metrics`.

**Drill.** `drill_node_revocation_refuses_authentication`: pair a node, show a
signed proof returns `200`, revoke it over HTTP, and show the next proof
returns `401` and the account view marks the node revoked.

## 2. Authority signing-key compromise

**Key rotation is not supported.** The README lists credential renewal and
rotation as open work. There is no rotation command, and this runbook does not
describe one. The supported procedure replaces the authority.

**Preconditions.** You can stop the service, you hold the configuration file,
and you can reach every enrolled node's owner to re-pair it.

**Procedure.**

1. Stop the service (runbook 5).
2. Keep the compromised database for investigation. Move it aside; do not
   delete it, and do not serve from it again.
3. Initialize a new authority at a new path. `init` never overwrites an existing
   file:

   ```sh
   ./target/debug/abi-control-plane init /tmp/abi-account-demo/authority-new.sqlite http://127.0.0.1:3000
   ```

4. Point `"database"` in the configuration at the new file and start the
   service again.
5. Every user signs in again (accounts are recreated by issuer/subject with new
   tenant IDs) and re-pairs every node from a new node directory.
6. Do not copy account or node rows from the old database into the new one.

**Verification signal.** `GET /v1/authority` returns a different public key.
Nodes still holding credentials from the old authority get `401` from
`POST /v1/nodes/authenticate`. Re-paired nodes authenticate with `200`.

**Drill.** `drill_new_authority_refuses_old_node_certificates`: a credential
signed by authority A fails verification against freshly initialized
authority B, and B refuses the node's proof with `401`. This holds even when
A's account and node rows are inserted into B by mistake, because the
credential signature no longer verifies. Re-pairing the same node key with B
then authenticates.

## 3. Backup and restore

**Preconditions.** The service is stopped. The database uses a rollback
journal (`journal_mode=DELETE`) with FULL synchronization, so a stopped
database is one consistent file. Never copy a live database.

**Procedure.**

1. Stop the service (runbook 5).
2. Copy the database file into an owner-only directory (`chmod 700`) and keep
   the copy owner-only (`chmod 600`). Backups contain issuer/subject,
   hashed secrets and the authority signing key, so protect them like the live
   database.
3. To restore, stop the service, copy the backup to the configured
   `"database"` path with the same private permissions, and start the service.
   Startup refuses a file or directory that is not private, a symlink, a hard
   link, or a database initialized for a different origin.

**Restore caution.** A backup cannot undo later revocations or deletions it
does not contain. Restoring a backup older than a revocation or account
deletion would bring that node or account back. Revoke again after any such
restore. RELAY.md says to forward-fix after post-upgrade mutations rather than
restore a stale backup.

**Verification signal.** `GET /healthz` returns `{"status":"ok"}`. The offline
`metrics` command reports the expected account and node counts:

```sh
./target/debug/abi-control-plane metrics /tmp/abi-account-demo/authority.sqlite http://127.0.0.1:3000
```

**Drill.** `drill_restore_preserves_revocation_and_deletion`: revoke one node
and delete one account, stop the service, copy the closed database file, and
reopen from the copy. The revoked node and the deleted account's node are still
refused, the deleted account is absent, and an unaffected node still
authenticates.

## 4. Binary upgrade and rollback

**Preconditions.** You have a backup taken while the service was stopped
(runbook 3) and the previous binary.

**Procedure.**

1. Stop the service and back up the database.
2. Replace the binary and start the service. The relay store migrates its schema
   on first open. A database from a newer, unknown schema version is refused.
3. To roll back **before any post-upgrade mutation**, stop the service, put the
   previous binary back, and restore the pre-upgrade backup.
4. After post-upgrade mutations, do not roll back. Forward-fix the current
   database instead (RELAY.md, "Durable delivery and reconnect"), because an
   older backup could resurrect jobs or revoked permissions.

**Verification signal.** `GET /healthz` returns `200` after start, and the
startup line `ABI account authority listening on ...` appears.

**Tests.** `relay::tests::recovery::schema_migration_retains_v1_ephemeral_and_refuses_future_versions`
and `relay::tests::recovery::populated_version_one_migrates_without_retroactive_durability`
cover migration and refusal of a future version. Pre-mutation rollback is a
restore, covered by `drill_restore_preserves_revocation_and_deletion`. No drill
swaps real binaries; packaged install/upgrade is tracked separately (C32).

## 5. Emergency shutdown

**Procedure.** Send `SIGINT` (Ctrl-C in the terminal, or `kill -INT <pid>`) to
`abi-control-plane serve`. The server stops accepting connections, lets
in-flight requests finish, closes idle keep-alive connections and exits.
For a hard stop, `kill -TERM` or `kill -KILL` also works. Every authority
mutation commits in one SQLite transaction, so an interrupted write rolls back.
Relay recovery is described in RELAY.md.

**Verification signal.** The process exits, and new connections to the listen
address are refused.

**Drill.** `drill_graceful_shutdown_completes_inflight_health`: the router,
served with graceful shutdown, answers one `/healthz` request on a keep-alive
connection. After the shutdown signal the server stops within 5 s while that
connection is still open, and new connections are refused. The drill uses a
oneshot channel in place of the `SIGINT` handler.

## 6. Metrics and health checks

**Procedure.**

1. Probe liveness with `curl -fsS http://127.0.0.1:3000/healthz`. It returns
   `{"status":"ok"}` or `503 {"status":"unavailable"}`.
2. To enable operator metrics, set `"operator_token_sha256"` in the
   configuration to the hex SHA-256 of a private token and restart:

   ```sh
   printf %s "$TOKEN" | shasum -a 256
   curl -fsS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:3000/v1/operator/metrics
   ```

3. Without the service running, use the offline `metrics` command (runbook 3).

**Verification signal.** Metrics return `404` when unconfigured and `401` with
a missing or wrong token. The `200` body holds only fixed keys and integers,
with no identifiers. A rising `auth_denied_total` means more `401` responses.

**Tests.** `healthz_reports_ok_without_content`,
`operator_metrics_is_absent_when_unconfigured`,
`operator_metrics_refuses_missing_and_wrong_token` and
`operator_metrics_counts_denied_requests` in `src/service/tests/operator.rs`.
Alerting, export and audit retention remain operator work.
