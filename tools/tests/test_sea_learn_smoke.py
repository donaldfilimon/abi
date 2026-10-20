"""Self-learning smoke must prove persistence and recall, using only scratch state."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SeaLearnSmokeTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="abi sea fixture ")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.script = self.root / ".agents/skills/sea-learn-loop/learn.sh"
        self.script.parent.mkdir(parents=True)
        shutil.copy2(ROOT / ".agents/skills/sea-learn-loop/learn.sh", self.script)
        self.env = dict(os.environ, ABI_WDBX_PATH="/must-not-open/live-store",
                        ABI_WDBX_PERSIST="0", SEA_CALLS=str(self.root / "calls"))
        self.executable("tools/cargo.sh", """
printf 'build %s\n' "$*" >> "$SEA_CALLS"
if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    case "$*" in *"--target-dir $PWD/target"*) ;; *) exit 49;; esac
fi
exit "${SEA_BUILD_EXIT:-0}"
""")
        self.executable("target/debug/abi", """
printf 'cli %s %s %s\n' "$ABI_WDBX_PATH" "$ABI_WDBX_PERSIST" "$*" >> "$SEA_CALLS"
mkdir -p "$ABI_WDBX_PATH"
count=0
[[ ! -f "$ABI_WDBX_PATH/count" ]] || count=$(cat "$ABI_WDBX_PATH/count")
printf '%s' "$((count + 1))" > "$ABI_WDBX_PATH/count"
if [[ "$count" -eq 0 ]]; then evidence=0; else evidence=${SEA_RECALL:-1}; fi
printf 'requested_model=local provider=local generation_engine=persona-template learn=true persisted=%s adapted=%s evidence_count=%s\n' "${SEA_PERSISTED:-true}" "${SEA_ADAPTED:-true}" "$evidence"
printf '%s\n' 'untrusted response: persisted=true evidence_count=99'
exit "${SEA_CLI_EXIT:-0}"
""")

    def executable(self, relative, body):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/bash\nset -eu\n" + body)
        path.chmod(0o755)

    def run_smoke(self, **overrides):
        return subprocess.run(["bash", str(self.script), "--leading-dash prompt"],
                              cwd=self.root, env={**self.env, **overrides},
                              capture_output=True, text=True, timeout=15)

    def test_two_passes_use_enabled_unique_scratch_store_and_prompt_separator(self):
        result = self.run_smoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = (self.root / "calls").read_text().splitlines()
        cli = [line for line in calls if line.startswith("cli ")]
        self.assertEqual(len(cli), 2, calls)
        self.assertTrue(all(" 1 complete --learn -- --leading-dash prompt" in line
                            for line in cli), cli)
        self.assertTrue(all("/must-not-open" not in line for line in cli), cli)
        self.assertTrue(any("build --locked" in line for line in calls), calls)
        self.assertEqual(list((self.root / "target/skill-scratch").iterdir()), [])

    def test_disabled_persistence_is_a_failure_even_when_response_claims_success(self):
        result = self.run_smoke(SEA_PERSISTED="false")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_missing_second_pass_recall_is_a_failure(self):
        result = self.run_smoke(SEA_RECALL="0")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_router_adaptation_is_a_failure(self):
        result = self.run_smoke(SEA_ADAPTED="false")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_recall_counter_is_a_failure(self):
        result = self.run_smoke(SEA_RECALL="garbage")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_inherited_target_cannot_execute_stale_binary(self):
        result = self.run_smoke(CARGO_TARGET_DIR=str(self.root / "other target"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_build_does_not_run_existing_cli(self):
        result = self.run_smoke(SEA_BUILD_EXIT="41")
        self.assertNotEqual(result.returncode, 0)
        calls = (self.root / "calls").read_text().splitlines()
        self.assertTrue(all(line.startswith("build ") for line in calls), calls)

    def test_failed_cli_is_not_accepted_based_on_output_markers(self):
        result = self.run_smoke(SEA_CLI_EXIT="42")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
