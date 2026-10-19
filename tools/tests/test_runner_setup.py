"""Existing runner state must never be replaced by the install helper."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'github-runner/setup-macos-arm64.sh'


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
