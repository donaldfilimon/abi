# Connector Guidelines

Provider connectors expose a uniform interface defined in
`src/connectors/mod.zig`. Each connector must implement:
- `init(allocator)` for configuration.
- `call(allocator, CallRequest)` returning `CallResult` with structured
  success/failure data.
- `health()` for readiness probes.

Real connectors (OpenAI, Hugging Face, local schedulers) must load credentials
from environment variables or CI secrets. Do not print secrets or write them to
logs. Timeouts, retry behaviour, and rate limits should honour the policies
configured in the agent controller. The mock connector ships as the CI default
and should remain deterministic.
