"""Run the MCP smoke driver against scratch binaries and bounded responses."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ("ai_run ai_complete ai_learn ai_train wdbx_query scheduler_stats "
         "scheduler_info connector_test gpu_status plugin_list wdbx_stats plugin_run").split()


class McpSmokeTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="abi mcp smoke ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.script = self.root / ".agents/skills/mcp-smoke/smoke.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / ".agents/skills/mcp-smoke/smoke.sh", self.script)
        (self.root / "mcp").mkdir()
        shutil.copy2(ROOT / "mcp/launcher.sh", self.root / "mcp/launcher.sh")
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.executable("tools/cargo.sh", """
printf '%s\n' "$*" > "$SMOKE_BUILD_LOG"
exit "${SMOKE_BUILD_RC:-0}"
""")
        for profile in ("debug", "release"):
            self.executable(f"target/{profile}/abi-mcp", """
printf '%s %s %s\n' "$0" "${ABI_WDBX_PATH:-unset}" "${ABI_WDBX_PERSIST:-unset}" > "$SMOKE_CALL_LOG"
cat >/dev/null
cat "$SMOKE_RESPONSE"
exit "${SMOKE_SERVER_RC:-0}"
""")
        self.response = {"jsonrpc": "2.0", "id": 1,
                         "result": {"tools": [{"name": name} for name in TOOLS]}}

    def executable(self, relative, body):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)

    def run_smoke(self, **overrides):
        response = self.root / "response"
        response.write_text(json.dumps(self.response) + "\n")
        env = dict(os.environ, ABI_WDBX_PATH="/must-not-open/live-store",
                   ABI_WDBX_PERSIST="1", SMOKE_RESPONSE=str(response),
                   SMOKE_CALL_LOG=str(self.root / "calls"),
                   SMOKE_BUILD_LOG=str(self.root / "build"), **overrides)
        return subprocess.run(["bash", str(self.script)], cwd=self.root,
                              env=env, text=True, capture_output=True, timeout=15)

    def test_valid_response_uses_fresh_launcher_profile_and_safe_store(self):
        result = self.run_smoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("target/release/abi-mcp :memory: 0", (self.root / "calls").read_text())
        build = (self.root / "build").read_text()
        self.assertIn("--locked", build)
        self.assertIn("--release", build)
        self.assertIn(f"--target-dir {self.root}/target", build)

    def test_nonzero_server_exit_cannot_pass_with_valid_tool_names(self):
        result = self.run_smoke(SMOKE_SERVER_RC="17")
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_response_must_match_request_identity_and_success_envelope(self):
        for changes in ({"id": 2}, {"id": True}, {"jsonrpc": "1.0"},
                        {"error": {"code": -32603, "message": "failed"}}):
            original = dict(self.response)
            with self.subTest(changes=changes):
                self.response.update(changes)
                result = self.run_smoke()
                self.assertNotEqual(result.returncode, 0, result.stdout)
            self.response = original

    def test_duplicate_tool_cannot_replace_missing_tool(self):
        self.response["result"]["tools"][-1] = {"name": TOOLS[0]}
        result = self.run_smoke()
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_build_failure_does_not_launch_existing_binary(self):
        result = self.run_smoke(SMOKE_BUILD_RC="41")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "calls").exists())


if __name__ == "__main__":
    unittest.main()
