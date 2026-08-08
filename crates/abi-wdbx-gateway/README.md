# ABI WDBX gateway

`abi-wdbx-gateway` is a bounded network adapter around the synchronous WDBX v2
product facade. It deliberately uses two explicit listeners:

- gRPC (`--grpc`, loopback `127.0.0.1:50051` by default) implements the eight
  RPCs in `proto/gateway.proto`.
- HTTP/WebSocket (`--events`, loopback `127.0.0.1:50052` by default) exposes
  only `/v1/events`.

Both listeners require the same bearer token. Any non-loopback address also
requires a server certificate and owner-protected private key. Supplying a
client CA makes client certificates mandatory on both listeners.

Mutation and query-result events contain only operation kind, transaction ID,
item count, sequence, and time. They never include vectors, keys, or KV values.
Queues, requests, batches, values, rates, blocking jobs, idle time, and streams
are bounded.

`MembershipChange` is intentionally **gateway-local reference state**. It is
not the WDBX signed membership ledger, is not durable, and does not claim a
consensus or authenticated cluster-membership change.
