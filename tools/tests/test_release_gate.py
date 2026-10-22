"""Execute release scripts against scratch transports and fake build tools."""

import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]


def executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/bash\nset -eu\n" + body)
    path.chmod(0o755)


class BenchmarkTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory(prefix="abi-release-test-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        (self.root / "tools").mkdir()
        shutil.copy2(ROOT / "tools/bench_regress.sh", self.root / "tools/bench_regress.sh")
        baseline = json.loads((ROOT / "tools/bench_baseline.json").read_text())
        baseline.update(system=platform.system(), machine=platform.machine())
        (self.root / "tools/bench_baseline.json").write_text(json.dumps(baseline, indent=2))
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("ABI_BENCH_") and key != "CARGO_TARGET_DIR"
        }

    def run_benchmark(self, target: str | None) -> subprocess.CompletedProcess:
        env = dict(self.env)
        if target is not None:
            env["CARGO_TARGET_DIR"] = target
        return subprocess.run(
            ["bash", "tools/bench_regress.sh"], cwd=self.root, env=env,
            text=True, capture_output=True, timeout=15,
        )

    def test_selected_absolute_and_relative_targets_with_spaces(self) -> None:
        executable(self.root / "target/debug/abi", "echo stale-default-binary >&2; exit 77\n")
        selected = self.root / "selected build"
        executable(selected / "debug/abi", "printf '  p50=1\\n  p50=1\\n'\n")
        for target in (str(selected), "selected build"):
            with self.subTest(target=target):
                result = self.run_benchmark(target)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                self.assertIn("benchmark regression gate: PASS", result.stdout)

    def test_missing_selected_binary_never_falls_back_to_stale_default(self) -> None:
        executable(self.root / "target/debug/abi", "printf '  p50=1\\n  p50=1\\n'\n")
        result = self.run_benchmark("missing build")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing build/debug/abi is missing", result.stderr)

    def test_unset_or_empty_target_uses_repository_default(self) -> None:
        executable(self.root / "target/debug/abi", "printf '  p50=1\\n  p50=1\\n'\n")
        for target in (None, ""):
            with self.subTest(target=target):
                result = self.run_benchmark(target)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


class LockedReleaseGateTests(unittest.TestCase):
    def test_gate_locks_resolution_requires_audit_and_builds_release(self) -> None:
        with tempfile.TemporaryDirectory(prefix="abi-gate-test-") as directory:
            root = Path(directory)
            (root / "tools/tests").mkdir(parents=True)
            shutil.copy2(ROOT / "tools/check.sh", root / "tools/check.sh")
            # Python 3.12+ unittest exits 5 when discovery finds no tests, so
            # the fixture carries one real test to prove the policy-test step
            # actually executes rather than silently passing on an empty dir.
            (root / "tools/tests/test_fixture_policy.py").write_text(
                "import unittest\n\n\n"
                "class FixturePolicyTests(unittest.TestCase):\n"
                "    def test_policy_step_runs(self) -> None:\n"
                "        self.assertTrue(True)\n"
            )
            executable(root / "tools/cargo.sh", """
printf '%s\\n' "$*" >> "$GATE_COMMANDS"
case "$1" in
  run|clippy|build|test|check|doc)
    case " $* " in *' --locked '*) ;; *) echo 'unlocked dependency resolution' >&2; exit 44;; esac
    ;;
esac
""")
            executable(root / "tools/check_rust_sizes.sh", "exit 0\n")
            executable(root / "tools/bench_regress.sh", "exit 0\n")
            executable(root / "tools/security/run-dep-scan.sh", """
test "${ABI_DEP_SCAN_REQUIRE:-0}" = 1
echo audit-required >> "$GATE_COMMANDS"
""")
            (root / "tools/abbey_contracts.py").write_text("")
            executable(root / "bin/rustup", 'echo "$GATE_RUSTC"\n')
            executable(root / "bin/rustc", "echo fixture-rustc\n")
            executable(root / "bin/uname", "echo Linux\n")
            commands = root / "commands.txt"
            # The gate requires a Python 3.11+ interpreter (tomllib). This
            # fixture exercises gate wiring, not interpreter selection, so the
            # running interpreter's directory is put on the minimal PATH.
            env = {**os.environ, "PATH": f"{root / 'bin'}:{Path(sys.executable).parent}:/usr/bin:/bin",
                   "GATE_COMMANDS": str(commands), "GATE_RUSTC": str(root / "bin/rustc")}
            result = subprocess.run(
                ["bash", "tools/check.sh"], cwd=root, env=env,
                text=True, capture_output=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = commands.read_text().splitlines()
            self.assertIn("audit-required", calls)
            self.assertTrue(any(line.startswith("build ") and "--release" in line for line in calls))
            self.assertTrue(any("--manifest-path ../wdbx/Cargo.toml" in line and
                                "--test abbey_contracts" in line and
                                "--test v3_cross_language_commitment" in line and
                                "--test v3_cross_language_episode" in line for line in calls))

    def test_gate_requires_a_capable_python_interpreter_and_says_so(self) -> None:
        # The Python policy tooling needs 3.11+ (tomllib). macOS still ships
        # /usr/bin/python3 at 3.9, and a bare `python3` can resolve to it. The
        # gate must fail closed with an actionable diagnostic rather than
        # dying inside the interpreter with a ModuleNotFoundError traceback.
        with tempfile.TemporaryDirectory(prefix="abi-gate-python-") as directory:
            root = Path(directory)
            (root / "tools").mkdir(parents=True)
            shutil.copy2(ROOT / "tools/check.sh", root / "tools/check.sh")
            # A python3 that satisfies `python3 --version` but cannot import
            # tomllib, standing in for the macOS system interpreter.
            stale = root / "bin" / "python3"
            stale.parent.mkdir(parents=True)
            stale.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = '-c' ]; then exit 1; fi\n"
                "echo 'Python 3.9.6'\n"
            )
            stale.chmod(0o755)
            # The PATH is self-contained: the resolver discovers versioned
            # interpreters by scanning PATH, so including /usr/bin would let a
            # host-supplied python3.X satisfy the resolver and make this test
            # fail for an unrelated reason. Only the required shell utilities
            # are linked in.
            for tool in ("bash", "dirname", "grep", "sort", "sed", "cat"):
                source = shutil.which(tool)
                if source is not None:
                    (root / "bin" / tool).symlink_to(source)
            env = {**os.environ, "PATH": f"{root / 'bin'}"}
            result = subprocess.run(
                ["bash", "tools/check.sh"], cwd=root, env=env,
                text=True, capture_output=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Python 3.11+", result.stderr)
            self.assertIn("ABI_PYTHON", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_gate_rejects_a_non_interpreter_as_the_python_oracle(self) -> None:
        # The probe must use a distinguishing exit status, not "exit 0": a
        # program that ignores its `-c` argument and exits 0 would otherwise
        # be accepted, silently turning the policy tests and the Abbey Python
        # oracle into no-ops while the gate still reported green.
        with tempfile.TemporaryDirectory(prefix="abi-gate-python-") as directory:
            root = Path(directory)
            (root / "tools/tests").mkdir(parents=True)
            shutil.copy2(ROOT / "tools/check.sh", root / "tools/check.sh")
            (root / "tools/tests/test_fixture_policy.py").write_text(
                "import unittest\n\n\n"
                "class FixturePolicyTests(unittest.TestCase):\n"
                "    def test_policy_step_runs(self) -> None:\n"
                "        self.assertTrue(True)\n"
            )
            (root / "tools/abbey_contracts.py").write_text("")
            executable(root / "tools/cargo.sh", "exit 0\n")
            executable(root / "tools/check_rust_sizes.sh", "exit 0\n")
            executable(root / "tools/bench_regress.sh", "exit 0\n")
            executable(root / "tools/security/run-dep-scan.sh", "exit 0\n")
            # A stand-in that ignores `-c` and exits 0, like /bin/echo.
            executable(root / "bin/python3", "exit 0\n")
            for tool in ("bash", "dirname", "grep", "sort", "sed", "cat"):
                source = shutil.which(tool)
                if source is not None:
                    (root / "bin" / tool).symlink_to(source)
            env = {**os.environ, "PATH": f"{root / 'bin'}"}
            result = subprocess.run(
                ["bash", "tools/check.sh"], cwd=root, env=env,
                text=True, capture_output=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Python 3.11+", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


class IsolatedCiTargetTests(unittest.TestCase):
    def step(self, name: str) -> str:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        block = workflow.split(f"- name: {name}\n", 1)[1].split("\n      - ", 1)[0]
        return textwrap.dedent(block.split("run: |\n", 1)[1])

    def test_each_allocation_is_fresh_and_cleanup_is_confined(self) -> None:
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        gate = workflow.split("- name: ./tools/check.sh\n", 1)[1].split("\n      - ", 1)[0]
        self.assertIn("run: |\n          ABI_WDBX_PATH=:memory: ./tools/check.sh", gate)
        with tempfile.TemporaryDirectory(prefix="abi ci fixture ") as directory:
            root = Path(directory)
            env_file = root / "github-env"
            env = {**os.environ, "RUNNER_TEMP": str(root), "GITHUB_ENV": str(env_file)}
            allocate = self.step("Allocate isolated build directory")
            cleanup = self.step("Remove isolated build directory")
            for _ in range(2):
                subprocess.run(["bash", "-euc", allocate], env=env, check=True, timeout=10)
            targets = [Path(line.split("=", 1)[1]) for line in env_file.read_text().splitlines()
                       if line.startswith("CARGO_TARGET_DIR=")]
            self.assertEqual(len(targets), 2)
            self.assertNotEqual(*targets)
            self.assertTrue(all(path.is_dir() and path.parent == root for path in targets))
            subprocess.run(["bash", "-euc", cleanup], env={**env, "CARGO_TARGET_DIR": str(targets[0])},
                           check=True, timeout=10)
            self.assertFalse(targets[0].exists())
            self.assertTrue(targets[1].is_dir())
            protected = root / "unrelated"
            protected.mkdir()
            subprocess.run(["bash", "-euc", cleanup], env={**env, "CARGO_TARGET_DIR": str(protected)},
                           check=True, timeout=10)
            self.assertTrue(protected.is_dir())

    def test_failed_allocation_exits_without_export(self) -> None:
        with tempfile.TemporaryDirectory(prefix="abi ci failure ") as directory:
            root = Path(directory)
            env_file = root / "github-env"
            env = {**os.environ, "RUNNER_TEMP": str(root), "GITHUB_ENV": str(env_file)}
            failed_allocator = "mktemp() { return 41; }\n" + self.step("Allocate isolated build directory")
            result = subprocess.run(["bash", "-euc", failed_allocator], env=env,
                                    text=True, capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(env_file.exists(), env_file.read_text() if env_file.exists() else "")


if __name__ == "__main__":
    unittest.main()
