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
