# ABI WDBX gateway

`abi-wdbx-gateway` is a bounded network adapter around the synchronous WDBX v2
product facade. It deliberately uses two explicit listeners:

- gRPC (`--grpc`, loopback `127.0.0.1:50051` by default) implements the ten
  RPCs in `proto/gateway.proto`: the eight WDBX v2 facade methods plus the
  canonical episode gate `ProposeEpisodeWrite` / `VerifyEpisode`, which is live
  only when `--episode-policy <json StorePolicy>` is configured (otherwise both
  answer `FAILED_PRECONDITION`). `VerifyEpisode` answers from the guild's whole
  ledger (`EpisodeStore::find_receipt`, not the windowed `retrieve`).
- HTTP/WebSocket (`--events`, loopback `127.0.0.1:50052` by default) exposes
  only `/v1/events`.

Both listeners require the same bearer token. Any non-loopback address also
requires a server certificate and owner-protected private key. Supplying a
client CA makes client certificates mandatory on both listeners.

Mutation and query-result events contain only operation kind, transaction ID,
item count, sequence, and time. They never include vectors, keys, or KV values.
Queues, requests, batches, values, rates, blocking jobs, idle time, and streams
are bounded.

`MembershipChange` uses WDBX's canonical signed membership records and a
store-local owner-protected signing keypair. The bounded lineage is replayed
and signature-verified on restart; responses include its generation, head
digest, and tombstone state. It remains **gateway-local**: this is durable
authenticated local state, not separate-host consensus, production cluster
membership, or operator-managed signing-key lifecycle evidence.
