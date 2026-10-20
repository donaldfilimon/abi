"""Exercise smoke orchestration without opening stores or launching real MCP."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class RunAbiSmokeTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="abi smoke fixture ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        script = self.root / ".agents/skills/run-abi/smoke.sh"
        script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / ".agents/skills/run-abi/smoke.sh", script)
        self.script = script
        self.env = dict(os.environ, ABI_WDBX_PATH="/must-not-open/live-store",
                        ABI_WDBX_PERSIST="1", SMOKE_CALLS=str(self.root / "calls"))
        self.write_executable("tools/cargo.sh", """
printf 'build %s\n' "$*" >> "$SMOKE_CALLS"
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    case "$*" in *"--target-dir $PWD/target"*) ;; *) exit 49;; esac
fi
exit "${SMOKE_BUILD_EXIT:-0}"
""")
        self.write_executable("target/debug/abi", """
printf 'cli %s %s\n' "${ABI_WDBX_PATH:-unset}" "${ABI_WDBX_PERSIST:-unset}" >> "$SMOKE_CALLS"
printf '%s\n' 'Usage: abi' 'Compute Backends:' 'source=cli-scheduler-status' \
 'model=claude-fable-5' 'Installed Plugins (' 'MULTI-AGENT RESULTS' \
 'orchestration=browser-local' 'initialized empty WDBX' 'appended block' 'blocks'
""")
        for profile in ("debug", "release"):
            self.write_executable(f"target/{profile}/abi-mcp", """
echo direct-mcp >> "$SMOKE_CALLS"
exit 88
""")
        self.write_executable("mcp/launcher.sh", """
printf 'launcher %s %s %s\n' "$*" "${ABI_WDBX_PATH:-unset}" "${ABI_WDBX_PERSIST:-unset}" >> "$SMOKE_CALLS"
cat >/dev/null
printf '%s\n' '\"serverInfo\" \"name\":\"gpu_status\" scheduler running= backend= plugins count=16'
""")

    def write_executable(self, relative, body):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)

    def run_smoke(self, **overrides):
        return subprocess.run(["bash", str(self.script)], cwd=self.root,
                              env={**self.env, **overrides}, text=True,
                              capture_output=True, timeout=15)

    def test_inherited_live_store_is_overridden_and_mcp_uses_launcher(self):
        result = self.run_smoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = (self.root / "calls").read_text().splitlines()
        cli = [line for line in calls if line.startswith("cli ")]
        self.assertEqual(len(cli), 10)
        self.assertTrue(all(line == "cli :memory: 0" for line in cli), calls)
        self.assertIn("launcher stdio :memory: 0", calls)
        self.assertNotIn("direct-mcp", calls)
        self.assertTrue(any(line.startswith("build build --locked --release -p abi-mcp ")
                            for line in calls), calls)

    def test_inherited_target_does_not_build_elsewhere_and_run_stale_binaries(self):
        result = self.run_smoke(CARGO_TARGET_DIR=str(self.root / "other target"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_build_does_not_exercise_stale_binaries(self):
        result = self.run_smoke(SMOKE_BUILD_EXIT="41")
        self.assertNotEqual(result.returncode, 0)
        calls = (self.root / "calls").read_text().splitlines()
        self.assertTrue(calls)
        self.assertTrue(all(line.startswith("build ") for line in calls), calls)


if __name__ == "__main__":
    unittest.main()
