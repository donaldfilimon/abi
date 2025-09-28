# Persona Manifests

The persona manifests describe runtime personas and environment profiles for the ABI agent.
They are compatible with `abi chat --persona-manifest` and can be loaded in tests or during
runtime orchestration.

## Schema overview

- Each `[[persona]]` entry defines the runtime characteristics for a persona, including
  its system prompt, enabled tools, safety filters, and sampling parameters.
- `[[profile]]` entries capture deployment profiles (dev/test/prod) that toggle
  streaming, function calling, and log sinks.
- Optional `[persona.rate_limits]` tables set request budgets for each persona.

Manifests can also be expressed as JSON using the same field names. See
`deploy/personas/dev.toml` for a complete reference manifest.
