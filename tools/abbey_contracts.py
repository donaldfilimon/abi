#!/usr/bin/env python3
"""Verify and review-build the language-neutral Abbey contract corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MAX_ARTIFACT_BYTES = 1024 * 1024
MAX_CORPUS_BYTES = 16 * 1024 * 1024
AGGREGATE_DOMAIN = b"abbey-contract-corpus-v1\0"
MANIFEST_NAME = "manifest.json"
MANIFEST_KEYS = {
    "contract_major",
    "contract_revision",
    "algorithm",
    "redaction_profile",
    "artifacts",
    "aggregate_digest",
}


class ContractError(Exception):
    """A closed corpus-verification failure without artifact content."""

    def __init__(self, code: str, path: str | Path | None = None) -> None:
        self.code = code
        self.path = _normalize_display_path(path)
        super().__init__(f"{code}: {self.path}" if self.path else code)


@dataclass(frozen=True)
class VerificationReport:
    """A successful or inventory-incomplete manifest comparison."""

    aggregate_digest: str
    artifact_count: int
    total_bytes: int
    unlisted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixtureOutcome:
    """Closed validation result for one synthetic fixture document."""

    code: str
    path: str


def _normalize_display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = Path(path).as_posix() if isinstance(path, Path) else str(path).replace("\\", "/")
    return text if len(text) <= 256 else text[:253] + "..."


def _reject_constant(value: str) -> Any:
    raise ContractError("non_finite_number")


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate_member")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    """Load one bounded UTF-8 JSON file with duplicate/non-finite rejection."""

    try:
        stat = path.lstat()
    except OSError as exc:
        raise ContractError("artifact_unreadable", path.name) from exc
    if path.is_symlink():
        raise ContractError("symlink_forbidden", path.name)
    if not path.is_file():
        raise ContractError("artifact_not_regular", path.name)
    if stat.st_size > MAX_ARTIFACT_BYTES:
        raise ContractError("artifact_too_large", path.name)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("invalid_utf8", path.name) from exc
    except OSError as exc:
        raise ContractError("artifact_unreadable", path.name) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_constant,
        )
    except ContractError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ContractError("invalid_json", path.name) from exc


def _validate_relative_path(path: str) -> PurePosixPath:
    if not path or "\\" in path:
        raise ContractError("invalid_artifact_path", path)
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ContractError("invalid_artifact_path", path)
    return candidate


def discover_artifacts(root: Path) -> tuple[Path, ...]:
    """Return sorted corpus artifacts without following any symbolic links."""

    root = root.resolve(strict=True)
    discovered: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in tuple(directory_names):
            candidate = current / name
            if candidate.is_symlink():
                raise ContractError("symlink_forbidden", candidate.relative_to(root))
        for name in file_names:
            candidate = current / name
            relative = candidate.relative_to(root)
            if candidate.is_symlink():
                raise ContractError("symlink_forbidden", relative)
            if not candidate.is_file():
                raise ContractError("artifact_not_regular", relative)
            if relative.as_posix() == MANIFEST_NAME:
                continue
            _validate_relative_path(relative.as_posix())
            discovered.append(relative)
    return tuple(sorted(discovered, key=lambda item: item.as_posix().encode("utf-8")))


def _fixed_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid_manifest_value") from exc


def _zeroed_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    zeroed = dict(manifest)
    zeroed["aggregate_digest"] = "0" * 64
    return _fixed_json_bytes(zeroed)


def _aggregate_digest(rows: Iterable[dict[str, Any]], manifest: dict[str, Any]) -> str:
    entries = [dict(row) for row in rows]
    manifest_bytes = _zeroed_manifest_bytes(manifest)
    entries.append(
        {
            "path": MANIFEST_NAME,
            "bytes": len(manifest_bytes),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    )
    entries.sort(key=lambda row: row["path"].encode("utf-8"))
    digest = hashlib.sha256()
    digest.update(AGGREGATE_DOMAIN)
    for row in entries:
        digest.update(row["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(row["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/schema+json" if path.name.endswith(".schema.json") else "application/json"
    if path.suffix == ".md":
        return "text/markdown; charset=utf-8"
    raise ContractError("unsupported_media_type", path)


def _artifact_row(root: Path, relative: Path) -> dict[str, Any]:
    absolute = root / relative
    raw = absolute.read_bytes()
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ContractError("artifact_too_large", relative)
    row: dict[str, Any] = {
        "path": relative.as_posix(),
        "bytes": len(raw),
        "media_type": _media_type(relative),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if relative.name.endswith(".schema.json"):
        document = load_json_strict(absolute)
        schema_id = document.get("$id") if isinstance(document, dict) else None
        if not isinstance(schema_id, str):
            raise ContractError("schema_id_missing", relative)
        row["schema_id"] = schema_id
    return row


def _schema_registry(root: Path) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    schema_root = root / "v1" / "schemas"
    if not schema_root.is_dir():
        raise ContractError("schema_directory_missing", "v1/schemas")
    for path in sorted(schema_root.rglob("*.schema.json")):
        if path.is_symlink():
            raise ContractError("symlink_forbidden", path.relative_to(root))
        schema = load_json_strict(path)
        relative = path.relative_to(root)
        if not isinstance(schema, dict):
            raise ContractError("schema_shape", relative)
        required_metadata = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "x-abbey-data-class": None,
            "x-abbey-max-bytes": None,
            "x-abbey-unknown-fields": None,
        }
        for key, expected in required_metadata.items():
            if key not in schema or (expected is not None and schema[key] != expected):
                raise ContractError("schema_metadata_missing", relative)
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id.startswith(
            "https://abbey.local/contracts/abbey/v1/schemas/"
        ):
            raise ContractError("schema_id_invalid", relative)
        if schema_id in registry:
            raise ContractError("schema_id_duplicate", relative)
        max_bytes = schema["x-abbey-max-bytes"]
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or not 1 <= max_bytes <= MAX_ARTIFACT_BYTES:
            raise ContractError("schema_max_bytes_invalid", relative)
        if schema["x-abbey-unknown-fields"] not in {"reject", "extensions-only"}:
            raise ContractError("schema_unknown_policy_invalid", relative)
        registry[schema_id] = schema
    for schema_id, schema in registry.items():
        _check_schema_references(schema, schema_id, registry)
    return registry


def _check_schema_references(value: Any, owner: str, registry: dict[str, dict[str, Any]]) -> None:
    if isinstance(value, dict):
        reference = value.get("$ref")
        if reference is not None:
            if not isinstance(reference, str):
                raise ContractError("schema_ref_invalid", owner)
            base = reference.split("#", 1)[0]
            target = owner if not base else base
            if target not in registry:
                raise ContractError("schema_ref_external", owner)
        for nested in value.values():
            _check_schema_references(nested, owner, registry)
    elif isinstance(value, list):
        for nested in value:
            _check_schema_references(nested, owner, registry)


def _resolve_reference(
    reference: str,
    owner_id: str,
    registry: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    base, separator, fragment = reference.partition("#")
    target_id = base or owner_id
    target: Any = registry.get(target_id)
    if target is None:
        raise ContractError("schema_ref_external", owner_id)
    if separator and fragment:
        if not fragment.startswith("/"):
            raise ContractError("schema_ref_invalid", owner_id)
        for token in fragment[1:].split("/"):
            token = token.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or token not in target:
                raise ContractError("schema_ref_invalid", owner_id)
            target = target[token]
    if not isinstance(target, dict):
        raise ContractError("schema_ref_invalid", owner_id)
    return target, target_id


def _is_json_type(value: Any, expected: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected, False)


def _validate_schema(
    value: Any,
    schema: dict[str, Any],
    owner_id: str,
    registry: dict[str, dict[str, Any]],
) -> bool:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        target, target_id = _resolve_reference(reference, owner_id, registry)
        return _validate_schema(value, target, target_id, registry)
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    for subschema in schema.get("allOf", []):
        if not _validate_schema(value, subschema, owner_id, registry):
            return False
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        if sum(_validate_schema(value, option, owner_id, registry) for option in one_of) != 1:
            return False
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and not any(
        _validate_schema(value, option, owner_id, registry) for option in any_of
    ):
        return False
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _is_json_type(value, expected_type):
        return False
    if isinstance(expected_type, list) and not any(_is_json_type(value, item) for item in expected_type):
        return False
    if isinstance(value, str):
        import re

        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", sys.maxsize):
            return False
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, value) is None:
            return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", sys.maxsize):
            return False
        if schema.get("uniqueItems"):
            encoded = [_fixed_json_bytes(item) for item in value]
            if len(set(encoded)) != len(encoded):
                return False
        item_schema = schema.get("items")
        if isinstance(item_schema, dict) and any(
            not _validate_schema(item, item_schema, owner_id, registry) for item in value
        ):
            return False
    if isinstance(value, dict):
        required = schema.get("required", [])
        if any(key not in value for key in required):
            return False
        if len(value) < schema.get("minProperties", 0) or len(value) > schema.get(
            "maxProperties", sys.maxsize
        ):
            return False
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                if not _validate_schema(item, properties[key], owner_id, registry):
                    return False
            else:
                additional = schema.get("additionalProperties", True)
                if additional is False:
                    return False
                if isinstance(additional, dict) and not _validate_schema(item, additional, owner_id, registry):
                    return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", value) or value > schema.get("maximum", value):
            return False
    return True


def _walk_values(value: Any) -> Iterable[tuple[str | None, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk_values(item)


def _privacy_safe(value: Any) -> bool:
    forbidden_keys = {
        "audio",
        "transcript",
        "message",
        "prompt",
        "response_text",
        "credential",
        "token",
        "password",
        "username",
        "display_name",
        "filesystem_path",
        "participant_identity",
    }
    for key, item in _walk_values(value):
        if key is not None and key.lower() in forbidden_keys:
            return False
        if isinstance(item, str):
            if item.isdecimal() and 17 <= len(item) <= 20:
                return False
            if item.startswith(("/Users/", "/home/", "C:\\", "sk-", "ghp_")):
                return False
    return True


def _semantic_code(schema_id: str, document: Any) -> str | None:
    if schema_id.endswith("/identity/delegation-chain.schema.json") and isinstance(document, dict):
        hops = document.get("hops", [])
        for left, right in zip(hops, hops[1:]):
            if left.get("delegatee_principal_id") != right.get("delegator_principal_id"):
                return "delegation_chain_broken"
        if hops:
            seen = {hops[0].get("delegator_principal_id")}
            for hop in hops:
                delegatee = hop.get("delegatee_principal_id")
                if delegatee in seen:
                    return "delegation_cycle"
                seen.add(delegatee)
    if schema_id.endswith("/authorization/approval.schema.json") and isinstance(document, dict):
        if document.get("approver_principal_id") == document.get("request_subject_principal_id"):
            return "self_approval"
    if schema_id.endswith("/authorization/policy-decision.schema.json") and isinstance(document, dict):
        if document.get("reason_code") == "dependency_unavailable" and document.get("decision") != "deny":
            return "degraded_authority"
    return None


def validate_fixture(root: Path, fixture_path: Path) -> FixtureOutcome:
    """Decode and validate one fixture without trusting its expected outcome."""

    root = root.resolve(strict=True)
    try:
        relative = fixture_path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise ContractError("fixture_path_invalid") from exc
    fixture = load_json_strict(fixture_path)
    if not isinstance(fixture, dict) or set(fixture) != {"case_id", "schema", "expect", "document"}:
        return FixtureOutcome("fixture_shape", relative.as_posix())
    schema_id = fixture.get("schema")
    if not isinstance(schema_id, str):
        return FixtureOutcome("fixture_shape", relative.as_posix())
    document = fixture.get("document")
    if not _privacy_safe(document):
        return FixtureOutcome("forbidden_content", relative.as_posix())
    registry = _schema_registry(root)
    schema = registry.get(schema_id)
    if schema is None:
        return FixtureOutcome("schema_unknown", relative.as_posix())
    encoded = _fixed_json_bytes(document)
    if len(encoded) > schema["x-abbey-max-bytes"]:
        return FixtureOutcome("document_too_large", relative.as_posix())
    if not _validate_schema(document, schema, schema_id, registry):
        return FixtureOutcome("schema_invalid", relative.as_posix())
    semantic = _semantic_code(schema_id, document)
    return FixtureOutcome(semantic or "valid", relative.as_posix())


def build_manifest(root: Path) -> dict[str, Any]:
    """Build deterministic manifest content without writing it."""

    root = root.resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    existing = load_json_strict(manifest_path)
    if not isinstance(existing, dict) or set(existing) != MANIFEST_KEYS:
        raise ContractError("manifest_shape", MANIFEST_NAME)
    rows = [_artifact_row(root, path) for path in discover_artifacts(root)]
    total = sum(int(row["bytes"]) for row in rows)
    if total > MAX_CORPUS_BYTES:
        raise ContractError("corpus_too_large")
    manifest = {
        "contract_major": existing["contract_major"],
        "contract_revision": existing["contract_revision"],
        "algorithm": existing["algorithm"],
        "redaction_profile": existing["redaction_profile"],
        "artifacts": rows,
        "aggregate_digest": "0" * 64,
    }
    manifest["aggregate_digest"] = _aggregate_digest(rows, manifest)
    return manifest


def verify_manifest(root: Path) -> VerificationReport:
    """Verify inventory, per-file commitments, and the aggregate commitment."""

    root = root.resolve(strict=True)
    manifest = load_json_strict(root / MANIFEST_NAME)
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ContractError("manifest_shape", MANIFEST_NAME)
    if manifest.get("algorithm") != "abbey-contract-corpus-sha256-v1":
        raise ContractError("algorithm_mismatch", MANIFEST_NAME)
    digest_text = manifest.get("aggregate_digest")
    if not isinstance(digest_text, str) or len(digest_text) != 64:
        raise ContractError("aggregate_digest_shape", MANIFEST_NAME)
    artifact_rows = manifest.get("artifacts")
    if not isinstance(artifact_rows, list):
        raise ContractError("manifest_shape", MANIFEST_NAME)

    indexed: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    total = 0
    for row in artifact_rows:
        if not isinstance(row, dict):
            raise ContractError("artifact_entry_shape", MANIFEST_NAME)
        allowed = {"path", "bytes", "media_type", "sha256", "schema_id"}
        if not set(row).issubset(allowed) or not {"path", "bytes", "media_type", "sha256"}.issubset(row):
            raise ContractError("artifact_entry_shape", MANIFEST_NAME)
        path_value = row.get("path")
        if not isinstance(path_value, str):
            raise ContractError("invalid_artifact_path", MANIFEST_NAME)
        _validate_relative_path(path_value)
        if path_value in indexed:
            duplicates.append(path_value)
        else:
            indexed[path_value] = row

    actual_paths = tuple(path.as_posix() for path in discover_artifacts(root))
    actual_set = set(actual_paths)
    listed_set = set(indexed)
    unlisted = tuple(sorted(actual_set - listed_set))
    missing = tuple(sorted(listed_set - actual_set))

    for path_text, row in indexed.items():
        if path_text in missing:
            continue
        relative = Path(path_text)
        raw = (root / relative).read_bytes()
        total += len(raw)
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ContractError("artifact_too_large", path_text)
        if row["bytes"] != len(raw):
            raise ContractError("artifact_length_mismatch", path_text)
        if row["media_type"] != _media_type(relative):
            raise ContractError("media_type_mismatch", path_text)
        if row["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ContractError("artifact_digest_mismatch", path_text)
    if total > MAX_CORPUS_BYTES:
        raise ContractError("corpus_too_large")
    if not unlisted and not missing and not duplicates:
        expected = _aggregate_digest(artifact_rows, manifest)
        if digest_text != expected:
            raise ContractError("aggregate_digest_mismatch", MANIFEST_NAME)
    return VerificationReport(
        aggregate_digest=digest_text,
        artifact_count=len(indexed),
        total_bytes=total,
        unlisted=unlisted,
        missing=missing,
        duplicates=tuple(sorted(set(duplicates))),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("root", type=Path)
    build = subparsers.add_parser("build-manifest")
    build.add_argument("root", type=Path)
    build.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.command == "verify":
            report = verify_manifest(args.root)
            if report.unlisted or report.missing or report.duplicates:
                raise ContractError("manifest_inventory_mismatch")
            print(
                f"abbey-contracts: verified {report.artifact_count} artifacts "
                f"({report.total_bytes} bytes), digest={report.aggregate_digest}"
            )
            return 0
        manifest = build_manifest(args.root)
        if not args.write:
            raise ContractError("write_flag_required")
        (args.root / MANIFEST_NAME).write_bytes(_fixed_json_bytes(manifest))
        print(f"abbey-contracts: wrote {args.root / MANIFEST_NAME}")
        return 0
    except ContractError as exc:
        print(f"abbey-contracts: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
