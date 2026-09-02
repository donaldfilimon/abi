"""Behavioral validation for ABI's executable GitHub Actions trust boundary."""

from __future__ import annotations

import re


WDBX_REVISION = "8ceca077e1d888c2955a0aa52bcbb278c01967a5"
WDBX_REPOSITORY = "donaldfilimon/wdbx"


def _job_sections(workflow: str) -> dict[str, str]:
    marker = "\njobs:\n"
    if marker not in workflow:
        return {}
    body = workflow.split(marker, 1)[1]
    starts = list(re.finditer(r"(?m)^  ([a-zA-Z0-9_-]+):\s*$", body))
    sections: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(body)
        sections[match.group(1)] = body[match.start() : end]
    return sections


def validate_workflow(workflow: str) -> tuple[str, ...]:
    """Return stable safety failures for the concrete ABI workflow.

    The validator treats the workflow as an operational policy artifact. It
    checks the checkout and event-routing outcomes that would otherwise expose
    a trusted runner or make public-fork builds depend on an unavailable secret.
    """

    failures: list[str] = []
    revision = re.search(r"(?m)^  WDBX_REVISION:\s*([^\s#]+)", workflow)
    if revision is None or revision.group(1) != WDBX_REVISION:
        failures.append("WDBX revision must be the reviewed immutable commit")

    if "WDBX_CHECKOUT_TOKEN" in workflow or re.search(
        r"(?m)^\s*token:\s*\$\{\{\s*secrets\.", workflow
    ):
        failures.append("wdbx checkout must not use a secret")

    sections = _job_sections(workflow)
    required = ("check", "check-hosted", "windows-acl")
    if any(name not in sections for name in required):
        failures.append("required ABI CI jobs are missing")
        return tuple(failures)

    checkout = f"repository: {WDBX_REPOSITORY}"
    if sum(section.count(checkout) for section in sections.values()) != len(required):
        failures.append("every ABI CI job must check out the reviewed WDBX repository once")
    for name in required:
        section = sections[name]
        if section.count(checkout) != 1 or "ref: ${{ env.WDBX_REVISION }}" not in section:
            failures.append(f"{name} must use the immutable WDBX checkout")
        if "path: wdbx" not in section:
            failures.append(f"{name} must place WDBX at the sibling path")

    trusted = sections["check"]
    if (
        "runs-on: [self-hosted" not in trusted
        or "github.event.pull_request.head.repo.full_name == github.repository" not in trusted
    ):
        failures.append(
            "trusted self-hosted job must require a same-repository pull request"
        )

    hosted = sections["check-hosted"]
    hosted_runner = re.search(r"(?m)^    runs-on:\s*([^\n#]+)", hosted)
    if (
        "github.event.pull_request.head.repo.full_name != github.repository" not in hosted
        or hosted_runner is None
        or hosted_runner.group(1).strip()
        not in {"macos-latest", "ubuntu-latest", "windows-latest"}
    ):
        failures.append("fork pull requests must run on a GitHub-hosted runner")

    return tuple(dict.fromkeys(failures))
