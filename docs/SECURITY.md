# Security Guidelines

- Secrets must be loaded from environment variables or secret managers.
- Do not hardcode API keys or tokens in source files or tests.
- Follow least privilege in CI by scoping tokens to the minimal repositories and
  permissions required.
- Document any new external dependencies in the changelog and review licences
  before adoption.
