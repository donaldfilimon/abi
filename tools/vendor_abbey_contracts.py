#!/usr/bin/env python3
"""Vendor the exact qualified Abbey contract corpus into a consumer tree."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.abbey_contracts import (
    ContractError,
    VerificationReport,
    load_json_strict,
    verify_manifest,
)


LOCK_NAME = "abbey-contracts.lock.json"
CORPUS_DIRECTORY = "corpus"
SOURCE_REPOSITORY = "https://github.com/donaldfilimon/abi"
REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
LOCK_KEYS = {
    "source_repository",
    "source_revision",
    "contract_major",
    "contract_revision",
    "aggregate_digest",
}


class VendorError(Exception):
    """A closed vendoring failure that never includes artifact content."""

    def __init__(self, code: str, path: str | Path | None = None) -> None:
        self.code = code
        self.path = _display_path(path)
        super().__init__(f"{code}: {self.path}" if self.path else code)


@dataclass(frozen=True)
class VendorReport:
    """Evidence returned by a successful write or read-only check."""

    aggregate_digest: str
    artifact_count: int
    total_bytes: int
    wrote: bool


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    text = Path(path).as_posix() if isinstance(path, Path) else str(path).replace("\\", "/")
    return text if len(text) <= 256 else text[:253] + "..."


def _json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as exc:
        raise VendorError("lock_value_invalid") from exc


def _validate_revision(source_revision: str) -> None:
    if not isinstance(source_revision, str) or REVISION_PATTERN.fullmatch(source_revision) is None:
        raise VendorError("source_revision_invalid")


def _closed_manifest(source: Path) -> tuple[dict[str, Any], VerificationReport]:
    if source.is_symlink():
        raise VendorError("source_symlink")
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise VendorError("source_unreadable") from exc
    if not source.is_dir():
        raise VendorError("source_not_directory")
    try:
        report = verify_manifest(source)
        manifest = load_json_strict(source / "manifest.json")
    except ContractError as exc:
        raise VendorError("source_corpus_invalid", exc.path) from exc
    if report.unlisted or report.missing or report.duplicates:
        raise VendorError("source_inventory_mismatch")
    if not isinstance(manifest, dict):
        raise VendorError("source_corpus_invalid", "manifest.json")
    return manifest, report


def _artifact_paths(manifest: dict[str, Any]) -> tuple[str, ...]:
    rows = manifest.get("artifacts")
    if not isinstance(rows, list):
        raise VendorError("source_corpus_invalid", "manifest.json")
    paths: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise VendorError("source_corpus_invalid", "manifest.json")
        paths.append(row["path"])
    return tuple(paths)


def _expected_lock(manifest: dict[str, Any], source_revision: str) -> dict[str, Any]:
    return {
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": source_revision,
        "contract_major": manifest["contract_major"],
        "contract_revision": manifest["contract_revision"],
        "aggregate_digest": manifest["aggregate_digest"],
    }


def _load_lock(destination: Path) -> dict[str, Any]:
    lock_path = destination / LOCK_NAME
    try:
        lock = load_json_strict(lock_path)
    except ContractError as exc:
        raise VendorError("destination_lock_invalid", LOCK_NAME) from exc
    if not isinstance(lock, dict) or set(lock) != LOCK_KEYS:
        raise VendorError("destination_lock_invalid", LOCK_NAME)
    if lock.get("source_repository") != SOURCE_REPOSITORY:
        raise VendorError("destination_lock_invalid", LOCK_NAME)
    revision = lock.get("source_revision")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise VendorError("destination_lock_invalid", LOCK_NAME)
    for key in ("contract_major", "contract_revision"):
        if isinstance(lock.get(key), bool) or not isinstance(lock.get(key), int):
            raise VendorError("destination_lock_invalid", LOCK_NAME)
    digest = lock.get("aggregate_digest")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise VendorError("destination_lock_invalid", LOCK_NAME)
    return lock


def _validate_managed_destination(destination: Path) -> tuple[dict[str, Any], VerificationReport]:
    if destination.is_symlink():
        raise VendorError("destination_symlink")
    if not destination.exists() or not destination.is_dir():
        raise VendorError("destination_missing")
    entries = {entry.name for entry in destination.iterdir()}
    expected_entries = {LOCK_NAME, CORPUS_DIRECTORY}
    if not expected_entries.issubset(entries):
        raise VendorError("unmanaged_destination")
    if entries != expected_entries:
        raise VendorError("destination_inventory_mismatch")
    corpus = destination / CORPUS_DIRECTORY
    if corpus.is_symlink():
        raise VendorError("destination_symlink", CORPUS_DIRECTORY)
    lock = _load_lock(destination)
    try:
        report = verify_manifest(corpus)
        manifest = load_json_strict(corpus / "manifest.json")
    except ContractError as exc:
        raise VendorError("destination_corpus_invalid", exc.path) from exc
    if report.unlisted or report.missing or report.duplicates:
        raise VendorError("destination_inventory_mismatch")
    if not isinstance(manifest, dict):
        raise VendorError("destination_corpus_invalid", "corpus/manifest.json")
    if (
        lock["contract_major"] != manifest.get("contract_major")
        or lock["contract_revision"] != manifest.get("contract_revision")
        or lock["aggregate_digest"] != report.aggregate_digest
    ):
        raise VendorError("destination_lock_mismatch", LOCK_NAME)
    return lock, report


def _compare_bytes(source: Path, destination: Path, manifest: dict[str, Any]) -> None:
    for relative in ("manifest.json", *_artifact_paths(manifest)):
        try:
            source_bytes = (source / relative).read_bytes()
            destination_bytes = (destination / CORPUS_DIRECTORY / relative).read_bytes()
        except OSError as exc:
            raise VendorError("destination_byte_unreadable", relative) from exc
        if source_bytes != destination_bytes:
            raise VendorError("destination_byte_mismatch", relative)


def _validate_expected_destination(
    source: Path,
    destination: Path,
    manifest: dict[str, Any],
    expected_lock: dict[str, Any],
) -> VerificationReport:
    lock, report = _validate_managed_destination(destination)
    if lock != expected_lock:
        raise VendorError("destination_lock_mismatch", LOCK_NAME)
    _compare_bytes(source, destination, manifest)
    return report


def _copy_corpus(source: Path, destination: Path, manifest: dict[str, Any]) -> None:
    corpus = destination / CORPUS_DIRECTORY
    corpus.mkdir(mode=0o700)
    for relative in ("manifest.json", *_artifact_paths(manifest)):
        source_path = source / relative
        destination_path = corpus / relative
        destination_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            destination_path.write_bytes(source_path.read_bytes())
        except OSError as exc:
            raise VendorError("copy_failed", relative) from exc


def _publish(staged: Path, destination: Path) -> None:
    if not destination.exists():
        staged.rename(destination)
        return
    backup = destination.parent / f".{destination.name}.backup-{secrets.token_hex(8)}"
    destination.rename(backup)
    try:
        staged.rename(destination)
    except OSError:
        backup.rename(destination)
        raise
    shutil.rmtree(backup)


def vendor(source: Path, destination: Path, source_revision: str, check: bool) -> VendorReport:
    """Write or verify an exact, digest-pinned Abbey corpus vendor tree."""

    _validate_revision(source_revision)
    if source.is_symlink():
        raise VendorError("source_symlink")
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise VendorError("source_unreadable") from exc
    destination = destination.absolute()
    if destination == source or source in destination.parents:
        raise VendorError("destination_inside_source")
    parent = destination.parent
    if not parent.exists() or not parent.is_dir():
        raise VendorError("destination_parent_invalid")

    manifest, source_report = _closed_manifest(source)
    expected_lock = _expected_lock(manifest, source_revision)
    if check:
        report = _validate_expected_destination(source, destination, manifest, expected_lock)
        return VendorReport(report.aggregate_digest, report.artifact_count, report.total_bytes, False)

    if destination.is_symlink():
        raise VendorError("destination_symlink")
    if destination.exists():
        _validate_managed_destination(destination)

    staged = Path(tempfile.mkdtemp(prefix=f".{destination.name}.vendor-", dir=parent))
    staged.chmod(0o700)
    try:
        _copy_corpus(source, staged, manifest)
        (staged / LOCK_NAME).write_bytes(_json_bytes(expected_lock))
        _validate_expected_destination(source, staged, manifest, expected_lock)
        refreshed_manifest, refreshed_report = _closed_manifest(source)
        if refreshed_manifest != manifest or refreshed_report != source_report:
            raise VendorError("source_changed_during_copy")
        _publish(staged, destination)
    except VendorError:
        if staged.exists():
            shutil.rmtree(staged)
        raise
    except OSError as exc:
        if staged.exists():
            shutil.rmtree(staged)
        raise VendorError("publication_failed") from exc

    return VendorReport(
        source_report.aggregate_digest,
        source_report.artifact_count,
        source_report.total_bytes,
        True,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = vendor(
            args.source,
            args.destination,
            args.source_revision,
            check=args.check,
        )
    except VendorError as exc:
        print(f"abbey-contracts-vendor: {exc}", file=sys.stderr)
        return 1
    action = "verified" if args.check else "wrote"
    print(
        f"abbey-contracts-vendor: {action} {report.artifact_count} artifacts "
        f"({report.total_bytes} bytes), digest={report.aggregate_digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
