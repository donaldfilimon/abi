"""Behavioral validation for ABI's executable GitHub Actions trust boundary."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]


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


def sibling_dependency_requirements(cargo_toml: str) -> dict[str, tuple[str, ...]]:
    """Return parent-sibling checkout roots and their manifest paths."""

    manifest = tomllib.loads(cargo_toml)
    dependencies = manifest["workspace"]["dependencies"]
    requirements: dict[str, list[str]] = {}
    for spec in dependencies.values():
        if not isinstance(spec, dict) or "path" not in spec:
            continue
        path = PurePosixPath(str(spec["path"]))
        if len(path.parts) < 3 or path.parts[0] != "..":
            continue
        requirements.setdefault(path.parts[1], []).append(path.as_posix())
    return {name: tuple(sorted(paths)) for name, paths in sorted(requirements.items())}


def _repository_owner(cargo_toml: str) -> str:
    manifest = tomllib.loads(cargo_toml)
    repository = manifest["workspace"]["package"]["repository"]
    match = re.fullmatch(r"https://github\.com/([^/]+)/[^/]+", repository)
    if match is None:
        raise ValueError("workspace repository must be a GitHub URL")
    return match.group(1)


def _checkout_steps(workflow: str) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """Return every Actions checkout step with its YAML indentation and lines.

    This deliberately parses only the small step boundary needed by the policy
    instead of pretending a regular expression is a YAML parser. A step begins
    at a sequence item and ends at the next item at the same indentation (or a
    line dedented beyond it).
    """

    lines = workflow.splitlines()
    steps: list[tuple[int, tuple[str, ...]]] = []
    index = 0
    while index < len(lines):
        match = re.match(r"^(\s*)-(?:\s+.*)?$", lines[index])
        if match is None:
            index += 1
            continue
        indent = len(match.group(1))
        end = index + 1
        while end < len(lines):
            candidate = lines[end]
            if candidate.strip():
                candidate_indent = len(candidate) - len(candidate.lstrip())
                if candidate_indent < indent:
                    break
                if candidate_indent == indent and re.match(r"^\s*-(?:\s+.*)?$", candidate):
                    break
            end += 1
        block = tuple(lines[index:end])
        if any(
            not line.lstrip().startswith("#")
            and re.search(r"actions/checkout@", line, re.IGNORECASE)
            for line in block
        ):
            steps.append((indent, block))
        index = max(end, index + 1)
    return tuple(steps)


def _checkout_disables_persisted_credentials(
    indent: int, step: tuple[str, ...]
) -> bool:
    with_line = re.compile(rf"^\s{{{indent + 2}}}with:\s*(?:#.*)?$")
    value_line = re.compile(
        rf"^\s{{{indent + 4}}}persist-credentials:\s*([^#]+?)(?:\s+#.*)?$"
    )
    in_with = False
    for line in step:
        line_indent = len(line) - len(line.lstrip())
        if with_line.fullmatch(line):
            in_with = True
            continue
        if not in_with:
            continue
        if line.strip() and line_indent <= indent + 2:
            break
        match = value_line.fullmatch(line)
        if match is not None:
            return match.group(1).strip() in {"false", '"false"', "'false'"}
    return False


def validate_checkout_credentials(workflow: str) -> tuple[str, ...]:
    """Return the repository-wide checkout credential policy failures."""

    # This dependency-free policy checker deliberately rejects YAML
    # indirection instead of attempting to resolve anchors and aliases. It also
    # rejects YAML escapes and block-style `uses` scalars, which could otherwise
    # hide an executable checkout spelling from a textual safety audit.
    yaml_indirection = re.compile(r'''(?<![\w])[&*](?![&*])[^\s\[\]{},'"&*]+''')
    uses_key = re.compile(r"(?:^|[-{,]\s*)(?:[\"']uses[\"']|uses)\s*:", re.IGNORECASE)
    for line in workflow.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if yaml_indirection.search(line) or "\\" in line:
            return ("every checkout must disable persisted credentials",)
        if uses_key.search(line) and re.search(r":\s*[>|][-+]?\s*$", line):
            return ("every checkout must disable persisted credentials",)

    checkout_steps = _checkout_steps(workflow)
    checkout_mentions = sum(
        len(re.findall(r"actions/checkout@", line, re.IGNORECASE))
        for line in workflow.splitlines()
        if not line.lstrip().startswith("#")
    )
    if checkout_mentions != len(checkout_steps):
        return ("every checkout must disable persisted credentials",)
    if any(
        not _checkout_disables_persisted_credentials(indent, step)
        for indent, step in checkout_steps
    ):
        return ("every checkout must disable persisted credentials",)
    return ()


def validate_workflow(workflow: str, cargo_toml: str | None = None) -> tuple[str, ...]:
    """Return stable safety failures for the concrete ABI workflow.

    The validator treats the workflow as an operational policy artifact. It
    checks the checkout and event-routing outcomes that would otherwise expose
    a trusted runner or make public-fork builds depend on an unavailable secret.
    """

    cargo_toml = cargo_toml or (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    siblings = sibling_dependency_requirements(cargo_toml)
    owner = _repository_owner(cargo_toml)
    failures: list[str] = []

    for sibling in siblings:
        env_name = f"{sibling.upper().replace('-', '_')}_REVISION"
        revision = re.search(rf"(?m)^  {re.escape(env_name)}:\s*([^\s#]+)", workflow)
        if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision.group(1)) is None:
            failures.append(f"{sibling} revision must be an immutable commit")

    if "WDBX_CHECKOUT_TOKEN" in workflow or re.search(
        r"(?m)^\s*token:\s*\$\{\{\s*secrets\.", workflow
    ):
        failures.append("wdbx checkout must not use a secret")

    sections = _job_sections(workflow)
    required = ("check", "check-hosted", "windows-acl")
    if any(name not in sections for name in required):
        failures.append("required ABI CI jobs are missing")
        return tuple(failures)

    for sibling in siblings:
        checkout = f"repository: {owner}/{sibling}"
        env_name = f"{sibling.upper().replace('-', '_')}_REVISION"
        if sum(section.count(checkout) for section in sections.values()) != len(required):
            failures.append(
                f"every ABI CI job must check out the required {sibling} repository once"
            )
        for name in required:
            section = sections[name]
            if section.count(checkout) != 1 or f"ref: ${{{{ env.{env_name} }}}}" not in section:
                failures.append(f"{name} must use the immutable {sibling} checkout")
            if f"path: {sibling}" not in section:
                failures.append(f"{name} must place {sibling} at the sibling path")

    failures.extend(validate_checkout_credentials(workflow))

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
