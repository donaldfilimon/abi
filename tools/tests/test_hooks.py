"""Execute repository hooks with isolated tools; never invoke a real push/network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class HookTests(unittest.TestCase):
    def configs(self):
        return [json.loads((ROOT / p).read_text())["hooks"] for p in
                (".claude/settings.json", ".codex/hooks.json")]

    def test_no_implicit_package_download(self):
        for config in self.configs():
            self.assertNotIn("npx", json.dumps(config))

    def test_formatter_uses_repository_pin_and_supports_response_path(self):
        for payload in ({"tool_input": {"file_path": "sample file.rs"}},
                        {"tool_response": {"filePath": "sample file.rs"}},
                        {"tool_input": {"file_path": str(ROOT / "absolute file.rs")}},
                        {"tool_input": {"file_path": "-sample '$(literal)'.rs"}}):
            for config in self.configs():
                command = config["PostToolUse"][0]["hooks"][0]["command"]
                with tempfile.TemporaryDirectory(prefix="hook tools ") as tmp:
                    path = Path(tmp)
                    (path / "rustup").write_text(
                        '#!/bin/sh\nprintf "%s\\n" "$PWD" "$@" > "$HOOK_LOG"\n'
                        'printf "%s/rustfmt\\n" "$HOOK_BIN"\n')
                    (path / "rustfmt").write_text(
                        f'#!{sys.executable}\nimport json, os, sys\n'
                        'with open(os.environ["HOOK_ARGS"], "w") as log:\n'
                        '    json.dump(sys.argv[1:], log)\n')
                    for name in ("rustup", "rustfmt"):
                        (path / name).chmod(0o755)
                    env = dict(os.environ, PATH=tmp + ":" + os.environ["PATH"],
                               HOOK_LOG=str(path / "log"), HOOK_BIN=tmp,
                               HOOK_ARGS=str(path / "args"),
                               CLAUDE_PROJECT_DIR=str(ROOT))
                    result = subprocess.run(command, shell=True, cwd=ROOT, env=env,
                                            input=json.dumps(payload), text=True,
                                            capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    log = (path / "log").read_text()
                    self.assertEqual(log.splitlines(), [str(ROOT), "which", "rustfmt"])
                    self.assertNotIn("--toolchain nightly", log)
                    filename = (payload.get("tool_input", {}).get("file_path") or
                                payload["tool_response"]["filePath"])
                    self.assertEqual(json.loads((path / "args").read_text()),
                                     ["--edition", "2024", str(ROOT / filename)])

    def test_skill_loop_configs_forward_payload_mode_and_exit_from_subdirectory(self):
        for config in self.configs():
            for event, mode in (("PreToolUse", "pre-hook"), ("PostToolUse", "post-hook")):
                command = config[event][1]["hooks"][0]["command"]
                with tempfile.TemporaryDirectory(prefix="hook tools ") as tmp:
                    executable = Path(tmp) / "skill-loop-claude"
                    executable.write_text(
                        '#!/bin/sh\n/bin/cat > "$HOOK_LOG"\n'
                        'printf "\\n%s\\n%s\\n" "$#" "$1" >> "$HOOK_LOG"\nexit 2\n')
                    executable.chmod(0o755)
                    env = dict(os.environ, PATH=tmp + ":" + os.environ["PATH"],
                               CLAUDE_PROJECT_DIR=str(ROOT), HOOK_LOG=str(Path(tmp) / "log"))
                    result = subprocess.run(command, shell=True, cwd=ROOT / "tools",
                                            env=env, input='{"payload":"unchanged"}',
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual((Path(tmp) / "log").read_text(),
                                     '{"payload":"unchanged"}\n1\n' + mode + '\n')

    def test_formatter_surfaces_rustfmt_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "rustup").write_text(
                '#!/bin/sh\nprintf "%s/rustfmt\\n" "$HOOK_BIN"\n')
            (path / "rustfmt").write_text('#!/bin/sh\necho formatter-failed >&2\nexit 7\n')
            for name in ("rustup", "rustfmt"):
                (path / name).chmod(0o755)
            result = subprocess.run(
                ["/bin/bash", str(ROOT / ".claude/hooks/format-rust.sh")],
                env=dict(os.environ, PATH=tmp + ":" + os.environ["PATH"], HOOK_BIN=tmp),
                input='{"tool_input":{"file_path":"x.rs"}}', text=True, capture_output=True)
            self.assertEqual(result.returncode, 7)
            self.assertIn("formatter-failed", result.stderr)

    def test_formatter_preserves_relative_path_from_subdirectory(self):
        for config in self.configs():
            command = config["PostToolUse"][0]["hooks"][0]["command"]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                (path / "rustup").write_text(
                    '#!/bin/sh\nprintf "%s/rustfmt\\n" "$HOOK_BIN"\n')
                (path / "rustfmt").write_text(
                    '#!/bin/sh\ntest -f "$3" || exit 9\n'
                    '/bin/cat "$3" > "$HOOK_LOG"\n')
                for name in ("rustup", "rustfmt"):
                    (path / name).chmod(0o755)
                with tempfile.TemporaryDirectory(dir=ROOT / "tools") as work:
                    filename = "sample file.rs"
                    (Path(work) / filename).write_text("subdirectory input")
                    env = dict(os.environ, PATH=tmp + ":" + os.environ["PATH"],
                               HOOK_BIN=tmp, HOOK_LOG=str(path / "log"),
                               CLAUDE_PROJECT_DIR=str(ROOT))
                    result = subprocess.run(
                        command, shell=True, cwd=work, env=env,
                        input=json.dumps({"tool_input": {"file_path": filename}}),
                        text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual((path / "log").read_text(), "subdirectory input")

    def test_non_rust_and_missing_paths_do_not_resolve_toolchain(self):
        for config in self.configs():
            for payload in ({}, {"tool_input": {"file_path": "readme.md"}}):
                with tempfile.TemporaryDirectory() as tmp:
                    rustup = Path(tmp) / "rustup"
                    rustup.write_text('#!/bin/sh\necho unexpected-rustup >&2\nexit 99\n')
                    rustup.chmod(0o755)
                    env = dict(os.environ, PATH=tmp + ":" + os.environ["PATH"],
                               CLAUDE_PROJECT_DIR=str(ROOT))
                    result = subprocess.run(
                        config["PostToolUse"][0]["hooks"][0]["command"],
                        shell=True, cwd=ROOT / "tools", env=env,
                        input=json.dumps(payload), text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, "")

    def test_skill_loop_rejects_invalid_modes_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "skill-loop-claude"
            executable.write_text('#!/bin/sh\necho unexpected-dispatch >&2\nexit 99\n')
            executable.chmod(0o755)
            for args in ([], ["invalid"], ["pre-hook", "extra"]):
                result = subprocess.run(
                    ["/bin/bash", str(ROOT / ".claude/hooks/skill-loop.sh"), *args],
                    env=dict(os.environ, PATH=tmp), input="{}",
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Expected pre-hook or post-hook", result.stderr)
                self.assertNotIn("unexpected-dispatch", result.stderr)

    def test_optional_skill_loop_has_no_install_fallback(self):
        hook = ROOT / ".claude/hooks/skill-loop.sh"
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PATH=tmp, HOOK_LOG=str(Path(tmp) / "log"))
            result = subprocess.run(["/bin/bash", str(hook), "pre-hook"],
                                    env=env, input="{}", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            executable = Path(tmp) / "skill-loop-claude"
            executable.write_text('#!/bin/sh\n/bin/cat > "$HOOK_LOG"\nprintf "%s" "$1" >> "$HOOK_LOG"\nexit 2\n')
            executable.chmod(0o755)
            result = subprocess.run(["/bin/bash", str(hook), "post-hook"],
                                    env=env, input="payload", capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual((Path(tmp) / "log").read_text(), "payloadpost-hook")

    def test_formatter_surfaces_toolchain_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            rustup = Path(tmp) / "rustup"
            rustup.write_text("#!/bin/sh\necho missing-toolchain >&2\nexit 1\n")
            rustup.chmod(0o755)
            env = dict(os.environ, PATH=tmp + ":" + os.environ["PATH"])
            result = subprocess.run(["bash", str(ROOT / ".claude/hooks/format-rust.sh")],
                                    env=env, input='{"tool_input":{"file_path":"x.rs"}}',
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing-toolchain", result.stderr)

    def test_guard_command_and_cmd_payloads(self):
        hook = ROOT / ".claude/hooks/block-force-push-main.sh"
        for key in ("command", "cmd"):
            for command, expected in (("git push --force origin main", 2),
                                      ("git push origin main", 0),
                                      ('echo "git push --force origin main"', 0)):
                result = subprocess.run(["bash", str(hook)], cwd=ROOT,
                                        input=json.dumps({"tool_input": {key: command}}),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == "__main__":
    unittest.main()
