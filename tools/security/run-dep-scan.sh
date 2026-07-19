#!/usr/bin/env sh
set -e

echo "Dependency scan helper - prints recommended commands (no API keys or secrets included)"

echo "# 1. GitHub CodeQL (recommended):"
echo "# - https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors"
echo "# Example: use actions/codeql-analysis in GitHub Actions"

echo "# 2. OSSF Scorecard (quick hardening checks)"
echo "# Example local run: scorecard --repo=<repo>"

echo "# 3. Dependabot: enable in repository settings for dependency updates"

echo "Done. Please enable the preferred scanners in repository settings or in CI."
