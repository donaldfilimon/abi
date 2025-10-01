# Prompt Guidelines

Agent prompts should follow the schemas described in `src/agent/schema.zig`.
When emitting model instructions use explicit JSON envelopes to maintain
structured outputs. Implement correction loops where the controller re-validates
responses and requests fixes when validation fails. Keep system prompts concise
and reference policy constraints directly.
