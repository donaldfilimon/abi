"""Exercise the installed central driver without synchronizing the user's home.

The driver remains authoritative at ~/.grok/scripts/sync-clis.py. Hosts without
that personal tool explicitly skip these integration tests; no driver is vendored.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


DRIVER = Path.home() / ".grok/scripts/sync-clis.py"
SUPPORT_DIRS = ("references", "scripts", "examples", "assets")


def isolated_run(driver_path: Path, root: Path, arguments: list[str]) -> int:
    """Run the actual entrypoint with fixture configuration and bounded writes."""
    root = root.resolve()

    def check_write(path: object) -> None:
        if isinstance(path, int):
            return
        if not Path(os.fsdecode(path)).resolve().is_relative_to(root):
            raise RuntimeError(f"sync-clis test refused write outside fixture: {path}")

    def audit(event: str, args: tuple) -> None:
        if event == "open":
            _, mode, flags = args
            if (mode and any(char in mode for char in "wax+")) or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            ):
                check_write(args[0])
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"}:
            check_write(args[0])
        elif event == "os.rename":
            check_write(args[0])
            check_write(args[1])
        elif event in {"os.symlink", "os.link"}:
            check_write(args[1])

    # Install before importing, so import-time writes cannot reach the real home.
    sys.addaudithook(audit)
    module = types.ModuleType("sync_clis_under_test")
    module.__file__ = str(driver_path)
    exec(compile(driver_path.read_bytes(), str(driver_path), "exec"), module.__dict__)
    module.SCRATCH = root / "scratch"
    module.LOG_FILE = module.SCRATCH / "sync.log"
    load_manifest = module.load_manifest
    module.load_manifest = lambda: load_manifest(root / "manifest.json")
    sys.argv = [str(driver_path), *arguments]
    return module.main()


def snapshot(root: Path) -> dict:
    """Include entry types, empty directories and metadata, excluding access time."""
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == "scratch":
            continue  # The central driver intentionally refreshes its run logs.
        info = path.lstat()
        content = (
            os.readlink(path) if path.is_symlink()
            else path.read_bytes() if path.is_file()
            else None
        )
        result[str(relative)] = (
            info.st_mode, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, content
        )
    return result


@unittest.skipUnless(DRIVER.is_file(), "central ~/.grok/scripts/sync-clis.py is not installed")
class CentralSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="abi-sync-clis-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.central = self.root / "home/.grok"
        self.skills = self.central / "skills"
        self.personas = self.central / "bundled/personas"
        self.roles = self.central / "bundled/roles"
        for directory in (self.skills, self.personas, self.roles):
            directory.mkdir(parents=True)
        for name in ("sl", "swift"):
            self.write(self.skills / name / "SKILL.md", f"---\nname: {name}\n---\nCanonical {name}.\n")
            for subdir in SUPPORT_DIRS:
                self.write(self.skills / name / subdir / "nested/guide.txt", "canonical\n")
                (self.skills / name / subdir / "empty").mkdir()
        self.write(self.personas / "reviewer.toml", 'description = "Review from the persona source"\ninstructions = "Read and report."\n')
        self.write(self.roles / "reviewer.toml", 'description = "Role fallback must not override persona"\n')
        self.write(self.roles / "implementer.toml", 'description = "Implement from the role source"\ninstructions = "Implement and verify."\n')
        self.targets = [
            {"name": "grok", "skillsDir": str(self.skills)},
            {"name": "codex", "skillsDir": str(self.root / "home/.codex/skills"),
             "agentsDir": str(self.root / "home/.codex/agents"), "agentsAdapter": "codex-toml"},
            {"name": "opencode", "skillsDir": str(self.root / "home/.config/opencode/skills"),
             "commandsDirs": [str(self.root / "home/.config/opencode" / name)
                              for name in ("command", "commands")]},
        ]
        self.manifest = {
            "central": {"skills": str(self.skills), "personas": str(self.personas), "roles": str(self.roles)},
            "catalog": {"portableSkills": ["sl", "swift"], "personas": ["reviewer"],
                        "roles": ["reviewer", "implementer"], "taskAgents": ["reviewer", "implementer"]},
            "targets": self.targets,
        }
        self.write(self.root / "manifest.json", json.dumps(self.manifest))
        self.write(self.root / "home/.codex/memories/keep.md", "unrelated memory\n")
        self.write(self.root / "home/.claude/CLAUDE.md", "unrelated charter\n")
        self.write(self.root / "home/.codex/skills/unmanaged/SKILL.md", "unmanaged skill\n")

    @staticmethod
    def write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def run_sync(self, *arguments: str) -> tuple[dict, str]:
        environment = dict(os.environ, HOME=str(self.root / "home"), TMPDIR=str(self.root))
        completed = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
             "--isolated-run", str(DRIVER), str(self.root), *arguments],
            cwd=self.root, env=environment, stdin=subprocess.DEVNULL,
            text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads((self.root / "scratch/sync-summary.json").read_text())
        self.assertEqual(summary["targets"], [target["name"] for target in self.targets])
        self.assertEqual(summary["actions_count"], len(summary["actions"]))
        return summary, completed.stdout

    def test_complete_second_run_reports_zero_and_leaves_managed_files_untouched(self) -> None:
        first, _ = self.run_sync()
        self.assertGreater(first["actions_count"], 0)
        for target in self.targets:
            destination = Path(target["skillsDir"])
            self.assertTrue((destination / ".plugins-synced-from-central").is_file())
            for name in ("sl", "swift"):
                self.assertEqual((destination / name / "SKILL.md").read_bytes(), (self.skills / name / "SKILL.md").read_bytes())
                for subdir in SUPPORT_DIRS:
                    self.assertEqual((destination / name / subdir / "nested/guide.txt").read_text(), "canonical\n")
                    self.assertTrue((destination / name / subdir / "empty").is_dir())
        before = snapshot(self.root)
        second, output = self.run_sync()
        self.assertEqual(second["actions_count"], 0)
        self.assertEqual(second["actions"], [])
        self.assertIn("Done. 0 actions/changes.", output)
        self.assertEqual(snapshot(self.root), before)

    def test_dry_run_reports_pending_changes_without_writing_managed_targets(self) -> None:
        before = snapshot(self.root)
        preview, _ = self.run_sync("--dry-run")
        self.assertTrue(preview["dry_run"])
        self.assertGreater(preview["actions_count"], 0)
        self.assertEqual(snapshot(self.root), before)
        applied, _ = self.run_sync()
        self.assertEqual(applied["actions"], preview["actions"])
        before = snapshot(self.root)
        clean, _ = self.run_sync("--dry-run")
        self.assertEqual(clean["actions_count"], 0)
        self.assertEqual(snapshot(self.root), before)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--isolated-run":
        sys.exit(isolated_run(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]))
    unittest.main()
