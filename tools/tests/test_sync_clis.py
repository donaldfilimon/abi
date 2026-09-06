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
import tomllib
import types
import unittest


DRIVER = Path.home() / ".grok/scripts/sync-clis.py"
MANIFEST = Path.home() / ".grok/sync-targets.json"
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
        self.codex = self.root / "home/.codex"
        self.claude = self.root / "home/.claude"
        self.opencode = self.root / "home/.config/opencode"
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
            {"name": "codex", "skillsDir": str(self.codex / "skills"),
             "agentsDir": str(self.codex / "agents"), "agentsAdapter": "codex-toml"},
            {"name": "claude", "skillsDir": str(self.claude / "skills"),
             "agentsDir": str(self.claude / "agents"), "agentsAdapter": "claude-markdown"},
            {"name": "opencode", "skillsDir": str(self.opencode / "skills"),
             "agentsDir": str(self.opencode / "agents"), "agentsAdapter": "opencode-markdown",
             "commandsDirs": [str(self.opencode / name)
                              for name in ("command", "commands")]},
        ]
        self.targets.extend(
            {"name": name, "skillsDir": str(self.root / "targets" / name / "skills")}
            for name in ("abi", "agents", "cursor", "hermes", "openclaw", "factory", "coreai", "gemini")
        )
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
        self.write(self.codex / "agents/unmanaged.toml", 'name = "unmanaged"\n')

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

    def test_destination_only_support_entries_do_not_trigger_replacement(self) -> None:
        self.run_sync()
        for subdir in SUPPORT_DIRS:
            self.write(self.codex / "skills/sl" / subdir / "local-only.txt", "keep me\n")
            (self.codex / "skills/sl" / subdir / "local-empty").mkdir()
        before = snapshot(self.root)
        result, _ = self.run_sync()
        self.assertEqual(result["actions_count"], 0)
        self.assertEqual(snapshot(self.root), before)

    def test_divergent_support_files_are_repaired_without_deleting_extras(self) -> None:
        self.run_sync()
        for subdir in SUPPORT_DIRS:
            with self.subTest(subdir=subdir):
                destination = self.codex / "skills/sl" / subdir
                # Equal-length bytes catch shallow size-only comparisons.
                self.write(destination / "nested/guide.txt", "divergent\n")
                self.write(destination / "local-only.txt", "keep me\n")
                before = snapshot(self.root)
                preview, _ = self.run_sync("--dry-run")
                self.assertEqual(preview["actions"], ["codex:skill:sl"])
                self.assertEqual(snapshot(self.root), before)
                result, _ = self.run_sync()
                self.assertEqual(result["actions"], preview["actions"])
                self.assertEqual((destination / "nested/guide.txt").read_text(), "canonical\n")
                self.assertEqual((destination / "local-only.txt").read_text(), "keep me\n")
                before = snapshot(self.root)
                result, _ = self.run_sync()
                self.assertEqual(result["actions_count"], 0)
                self.assertEqual(snapshot(self.root), before)

    def test_source_entry_type_conflict_replaces_only_the_conflicting_entry(self) -> None:
        self.run_sync()
        destination = self.codex / "skills/sl/references"
        (destination / "nested/guide.txt").unlink()
        self.write(destination / "nested/guide.txt/removed-with-conflict.txt", "conflict\n")
        self.write(destination / "local-only.txt", "keep me\n")
        result, _ = self.run_sync()
        self.assertEqual(result["actions"], ["codex:skill:sl"])
        self.assertEqual((destination / "nested/guide.txt").read_text(), "canonical\n")
        self.assertEqual((destination / "local-only.txt").read_text(), "keep me\n")
        before = snapshot(self.root)
        result, _ = self.run_sync()
        self.assertEqual(result["actions_count"], 0)
        self.assertEqual(snapshot(self.root), before)

    def test_task_agents_use_native_destinations_and_preserve_sources_and_unmanaged_files(self) -> None:
        protected = {
            "personas": snapshot(self.personas),
            "roles": snapshot(self.roles),
            "memories": snapshot(self.codex / "memories"),
            "unmanaged_skill": snapshot(self.codex / "skills/unmanaged"),
        }
        self.run_sync()
        for name, description in (
            ("reviewer", "Review from the persona source"),
            ("implementer", "Implement from the role source"),
        ):
            codex_agent = tomllib.loads((self.codex / "agents" / f"{name}.toml").read_text())
            self.assertEqual(codex_agent["name"], name)
            self.assertEqual(codex_agent["description"], description)
            self.assertTrue(codex_agent["developer_instructions"])
            for runtime, frontmatter in ((self.claude, "model: inherit"), (self.opencode, "mode: subagent")):
                agent = (runtime / "agents" / f"{name}.md").read_text()
                self.assertEqual(agent.splitlines().count("---"), 2)
                self.assertIn(frontmatter, agent)
                self.assertIn(description, agent)
        self.assertEqual(snapshot(self.personas), protected["personas"])
        self.assertEqual(snapshot(self.roles), protected["roles"])
        self.assertEqual(snapshot(self.codex / "memories"), protected["memories"])
        self.assertEqual(snapshot(self.codex / "skills/unmanaged"), protected["unmanaged_skill"])
        self.assertEqual((self.claude / "CLAUDE.md").read_text(), "unrelated charter\n")
        self.assertEqual((self.codex / "agents/unmanaged.toml").read_text(), 'name = "unmanaged"\n')
        for name in ("abi", "agents", "cursor", "hermes", "openclaw", "factory", "coreai", "gemini"):
            self.assertEqual(list((self.root / "targets" / name).rglob("*.toml")), [])
            self.assertFalse((self.root / "targets" / name / "agents").exists())

    def test_divergent_agents_commands_and_markers_are_repaired_then_unchanged(self) -> None:
        self.run_sync()
        expected = {}
        for runtime, suffix in ((self.codex, ".toml"), (self.claude, ".md"), (self.opencode, ".md")):
            path = runtime / "agents" / f"reviewer{suffix}"
            expected[path] = path.read_bytes()
        for directory in ("command", "commands"):
            path = self.opencode / directory / "sl.md"
            expected[path] = path.read_bytes()
            self.assertEqual(path.read_text().splitlines().count("---"), 2)
            self.assertEqual(path.read_text().count("<!-- synced from central:"), 1)
        marker = self.codex / "skills/.plugins-synced-from-central"
        expected[marker] = marker.read_bytes()
        for path in expected:
            path.write_text("divergent\n")
        before = snapshot(self.root)
        preview, _ = self.run_sync("--dry-run")
        self.assertEqual(preview["actions_count"], len(expected))
        self.assertEqual(snapshot(self.root), before)
        result, _ = self.run_sync()
        self.assertEqual(result["actions"], preview["actions"])
        for path, content in expected.items():
            self.assertEqual(path.read_bytes(), content)
        before = snapshot(self.root)
        result, _ = self.run_sync()
        self.assertEqual(result["actions_count"], 0)
        self.assertEqual(snapshot(self.root), before)

    @unittest.skipUnless(MANIFEST.is_file(), "central sync-targets.json is not installed")
    def test_installed_manifest_keeps_grok_source_and_native_agent_destinations(self) -> None:
        # Read only configuration, never load real skills, personas or memories.
        manifest = json.loads(MANIFEST.read_text())
        self.assertNotIn("abi-skills", manifest["catalog"]["portableSkills"])
        self.assertEqual(Path(manifest["central"]["skills"]), Path.home() / ".grok/skills")
        self.assertEqual(Path(manifest["central"]["personas"]), Path.home() / ".grok/bundled/personas")
        self.assertEqual(Path(manifest["central"]["roles"]), Path.home() / ".grok/bundled/roles")
        destinations = {
            target["name"]: (Path(target["agentsDir"]), target["agentsAdapter"])
            for target in manifest["targets"] if target.get("agentsDir")
        }
        self.assertEqual(destinations, {
            "codex": (Path.home() / ".codex/agents", "codex-toml"),
            "claude": (Path.home() / ".claude/agents", "claude-markdown"),
            "opencode": (Path.home() / ".config/opencode/agents", "opencode-markdown"),
        })


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--isolated-run":
        sys.exit(isolated_run(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]))
    unittest.main()
