"""Policy tests for the unsigned macOS package installer (C32).

The archives here are synthetic: a fake `bin/abi` shell script plus a matching
`MANIFEST.json`, built in a temporary directory. No cargo build runs. The
manifest uses the same one-entry-per-line layout `tools/package/build_macos.sh`
writes, because `tools/package/install.sh` parses it without a JSON library.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = ROOT / "tools/package/install.sh"
BUILD_SH = ROOT / "tools/package/build_macos.sh"
TARGET = "aarch64-apple-darwin"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_text(version: str, revision: str, files: dict[str, str]) -> str:
    lines = [
        "{",
        '  "name": "abi",',
        f'  "version": "{version}",',
        f'  "revision": "{revision}",',
        f'  "target": "{TARGET}",',
        '  "rustc": "rustc 1.99.1 (synthetic)",',
        '  "files": {',
    ]
    entries = sorted(files.items())
    for index, (relpath, digest) in enumerate(entries):
        comma = "," if index + 1 < len(entries) else ""
        lines.append(f'    "{relpath}": "{digest}"{comma}')
    lines += ["  }", "}", ""]
    return "\n".join(lines)


def make_archive(out: Path, version: str, revision: str, *, tamper: bool = False) -> Path:
    """Write `abi-<version>-<revision>-<target>.tar.gz` and its `.sha256` into `out`."""
    package_id = f"abi-{version}-{revision}-{TARGET}"
    stage = out / "stage" / package_id
    (stage / "bin").mkdir(parents=True)
    abi = stage / "bin/abi"
    abi.write_text(f"#!/bin/sh\necho fake-abi-{version}\n")
    abi.chmod(0o755)
    dylib = stage / "bin/libabi_fake.dylib"
    dylib.write_bytes(b"not a real dylib " + version.encode())
    files = {rel: sha256(stage / rel) for rel in ("bin/abi", "bin/libabi_fake.dylib")}
    (stage / "MANIFEST.json").write_text(manifest_text(version, revision, files))
    if tamper:
        # Payload changes after the manifest was written: the archive digest is
        # still correct, but a manifest hash no longer matches.
        abi.write_text("#!/bin/sh\necho tampered\n")
    archive = out / f"{package_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(stage, arcname=package_id)
    (out / f"{package_id}.tar.gz.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
    return archive


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(INSTALL_SH), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )


@unittest.skipUnless(sys.platform == "darwin", "the package and its installer are macOS-only")
class PackageInstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.prefix = self.tmp / "prefix"
        self.archives = self.tmp / "archives"
        self.archives.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def archive(self, version: str, revision: str, **kwargs) -> Path:
        out = self.archives / f"{version}-{revision}"
        out.mkdir()
        return make_archive(out, version, revision, **kwargs)

    def install(self, archive: Path) -> subprocess.CompletedProcess:
        result = run("install", str(self.prefix), str(archive))
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def current_id(self) -> str:
        return os.readlink(self.prefix / "current")

    def run_current_abi(self) -> str:
        result = subprocess.run(
            [str(self.prefix / "current/bin/abi")], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def test_install_then_status_reports_version(self) -> None:
        self.install(self.archive("0.1.0", "aaaaaaa"))
        self.assertEqual(self.current_id(), f"versions/abi-0.1.0-aaaaaaa-{TARGET}")
        self.assertEqual(self.run_current_abi(), "fake-abi-0.1.0")
        status = run("status", str(self.prefix))
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertIn(f"current: abi-0.1.0-aaaaaaa-{TARGET} (version 0.1.0)", status.stdout)
        self.assertIn("previous: none", status.stdout)

    def test_upgrade_keeps_previous_and_rollback_restores_it(self) -> None:
        self.install(self.archive("0.1.0", "aaaaaaa"))
        self.install(self.archive("0.2.0", "bbbbbbb"))
        self.assertEqual(self.run_current_abi(), "fake-abi-0.2.0")
        self.assertEqual(
            os.readlink(self.prefix / "previous"), f"versions/abi-0.1.0-aaaaaaa-{TARGET}"
        )
        rollback = run("rollback", str(self.prefix))
        self.assertEqual(rollback.returncode, 0, rollback.stderr)
        self.assertEqual(self.run_current_abi(), "fake-abi-0.1.0")
        self.assertEqual(
            os.readlink(self.prefix / "previous"), f"versions/abi-0.2.0-bbbbbbb-{TARGET}"
        )
        # Rolling back again returns to the upgrade: rollback is a swap.
        self.assertEqual(run("rollback", str(self.prefix)).returncode, 0)
        self.assertEqual(self.run_current_abi(), "fake-abi-0.2.0")

    def test_rollback_without_previous_refuses(self) -> None:
        self.install(self.archive("0.1.0", "aaaaaaa"))
        result = run("rollback", str(self.prefix))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.run_current_abi(), "fake-abi-0.1.0")

    def test_checksum_mismatch_refuses_and_leaves_current(self) -> None:
        self.install(self.archive("0.1.0", "aaaaaaa"))
        # 1. Archive digest mismatch: refused before anything is extracted.
        bad = self.archive("0.2.0", "bbbbbbb")
        Path(f"{bad}.sha256").write_text(f"{'0' * 64}  {bad.name}\n")
        result = run("install", str(self.prefix), str(bad))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum mismatch", result.stderr)
        # 2. Archive digest fine, but a manifest hash does not match its file.
        tampered = self.archive("0.3.0", "ccccccc", tamper=True)
        result = run("install", str(self.prefix), str(tampered))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest hash mismatch", result.stderr)
        self.assertEqual(self.run_current_abi(), "fake-abi-0.1.0")
        self.assertFalse((self.prefix / "previous").exists())
        self.assertEqual(
            sorted(p.name for p in (self.prefix / "versions").iterdir()),
            [f"abi-0.1.0-aaaaaaa-{TARGET}"],
        )

    def test_interrupted_upgrade_keeps_previous_current(self) -> None:
        self.install(self.archive("0.1.0", "aaaaaaa"))
        # An upgrade that died mid-extraction leaves only a staging directory.
        staging = self.prefix / "versions/.staging-x"
        (staging / "bin").mkdir(parents=True)
        (staging / "bin/abi").write_text("#!/bin/sh\necho partial\n")
        self.assertEqual(self.current_id(), f"versions/abi-0.1.0-aaaaaaa-{TARGET}")
        self.assertEqual(self.run_current_abi(), "fake-abi-0.1.0")
        status = run("status", str(self.prefix))
        self.assertIn(f"current: abi-0.1.0-aaaaaaa-{TARGET}", status.stdout)
        self.install(self.archive("0.2.0", "bbbbbbb"))
        self.assertFalse(staging.exists())
        self.assertEqual(self.run_current_abi(), "fake-abi-0.2.0")

    def test_reinstalling_an_existing_version_refuses(self) -> None:
        archive = self.archive("0.1.0", "aaaaaaa")
        self.install(archive)
        result = run("install", str(self.prefix), str(archive))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.run_current_abi(), "fake-abi-0.1.0")

    def test_remove_requires_yes_and_removes_only_owned_paths(self) -> None:
        self.install(self.archive("0.1.0", "aaaaaaa"))
        self.install(self.archive("0.2.0", "bbbbbbb"))
        sibling = self.prefix / "keep-me.txt"
        sibling.write_text("user data\n")
        refused = run("remove", str(self.prefix))
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(self.run_current_abi(), "fake-abi-0.2.0")
        removed = run("remove", str(self.prefix), "--yes")
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.assertFalse((self.prefix / "versions").exists())
        self.assertFalse(os.path.lexists(self.prefix / "current"))
        self.assertFalse(os.path.lexists(self.prefix / "previous"))
        self.assertEqual(sibling.read_text(), "user data\n")


class PackageBuildScriptTests(unittest.TestCase):
    def test_build_script_refuses_without_out_dir(self) -> None:
        result = subprocess.run(
            ["sh", str(BUILD_SH)], capture_output=True, text=True, stdin=subprocess.DEVNULL
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("usage:", result.stderr)

    def test_build_script_checks_tracked_changes_and_uses_locked_wrapper(self) -> None:
        text = BUILD_SH.read_text()
        self.assertIn("git status --porcelain --untracked-files=no", text)
        self.assertIn(
            "./tools/cargo.sh build --locked --release -p abi-cli -p abi-mcp -p abi-control-plane",
            text,
        )


if __name__ == "__main__":
    unittest.main()
