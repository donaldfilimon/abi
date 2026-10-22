"""Existing runner state must never be replaced by the install helper."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'github-runner/setup-macos-arm64.sh'
HARDEN = Path(__file__).resolve().parents[1] / 'github-runner/harden-macos.sh'


class RunnerHardeningToolchainTests(unittest.TestCase):
    """The hardening check must track the pinned channel, not a floating one."""

    def stage(self, root: Path, manifest: str | None) -> Path:
        """Copy the script so its REPO_ROOT is the fixture, not the real repo."""
        script = root / 'tools' / 'github-runner' / 'harden-macos.sh'
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(HARDEN.read_text(encoding='utf-8'), encoding='utf-8')
        (root / 'tools' / 'cargo.sh').write_text('#!/bin/sh\nexit 0\n', encoding='utf-8')
        (root / 'tools' / 'cargo.sh').chmod(0o755)
        if manifest is not None:
            (root / 'rust-toolchain.toml').write_text(manifest, encoding='utf-8')
        return script

    def test_pinned_channel_is_read_from_the_repository_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = self.stage(root, '[toolchain]\nchannel = "1.99.1"\n')
            binaries = root / 'bin'
            binaries.mkdir()
            # A rustup shim that only answers for the pinned channel.
            rustup = binaries / 'rustup'
            rustup.write_text(
                '#!/bin/sh\n'
                'case "$2" in\n'
                '  1.99.1) echo "rustc 1.99.1 fixture" ;;\n'
                '  nightly) echo "unexpected nightly selection" >&2; exit 9 ;;\n'
                '  *) exit 8 ;;\n'
                'esac\n',
                encoding='utf-8')
            rustup.chmod(0o755)
            env = dict(os.environ, PATH=f'{binaries}:/usr/bin:/bin')
            result = subprocess.run(
                ['bash', str(script)], cwd=root, env=env,
                capture_output=True, text=True, timeout=20)
            self.assertNotIn('unexpected nightly selection', result.stderr)
            self.assertNotIn('unexpected nightly selection', result.stdout)
            # The pinned channel is reported, never a floating one.
            self.assertIn('1.99.1', result.stdout)
            self.assertNotIn('nightly', result.stdout)

    def test_unreadable_manifest_reports_a_diagnostic_instead_of_aborting(self) -> None:
        # `set -euo pipefail` must not kill the checklist before the
        # diagnostic path when rust-toolchain.toml is missing.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = self.stage(root, None)
            binaries = root / 'bin'
            binaries.mkdir()
            env = dict(os.environ, PATH=f'{binaries}:/usr/bin:/bin')
            result = subprocess.run(
                ['bash', str(script)], cwd=root, env=env,
                capture_output=True, text=True, timeout=20)
            self.assertIn('cannot read the pinned channel', result.stdout)
            # The script still completes its checklist and reports a summary.
            self.assertIn('RESULT:', result.stdout)


class RunnerSetupTests(unittest.TestCase):
    def assert_refused(self, marker, override):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            binaries = home / 'test-bin'
            binaries.mkdir()
            uname = binaries / 'uname'
            uname.write_text('#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
            uname.chmod(0o755)
            gh = binaries / 'gh'
            gh.write_text('#!/bin/sh\necho unexpected-network-call >&2\nexit 99\n')
            gh.chmod(0o755)
            installation = home / ('explicit-runner' if override else 'dev/ci/runners/abi')
            installation.mkdir(parents=True)
            state = installation / marker
            state.write_text('existing-state\n')
            environment = dict(os.environ, HOME=str(home), PATH=f'{binaries}:/usr/bin:/bin')
            environment.pop('GITHUB_RUNNER_DIR', None)
            if override:
                environment['GITHUB_RUNNER_DIR'] = str(installation)
            result = subprocess.run(['bash', str(SCRIPT)], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('existing runner/service', result.stderr)
            self.assertNotIn('unexpected-network-call', result.stderr)
            self.assertEqual(state.read_text(), 'existing-state\n')
            self.assertEqual(sorted(p.name for p in installation.iterdir()), [marker])

    def test_default_consolidated_installation_is_preserved(self):
        self.assert_refused('.runner', override=False)

    def test_explicit_service_only_installation_is_preserved(self):
        self.assert_refused('.service', override=True)


if __name__ == '__main__':
    unittest.main()
