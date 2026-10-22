"""Executable wrapper fixtures, including one installed-stable offline regression."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ToolchainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / 'repo'
        (self.repo / 'tools').mkdir(parents=True)
        shutil.copy(ROOT / 'tools/cargo.sh', self.repo / 'tools/cargo.sh')
        shutil.copy(ROOT / 'tools/toolchain_policy.py', self.repo / 'tools/toolchain_policy.py')
        shutil.copy(ROOT / 'rust-toolchain.toml', self.repo / 'rust-toolchain.toml')
        self.bin = self.root / 'selected/bin'
        self.bin.mkdir(parents=True)
        self.hostile = self.root / 'hostile'
        self.hostile.mkdir()
        self.script(self.hostile / 'cargo', "raise SystemExit('hostile cargo selected')")
        self.script(self.hostile / 'rustup', '''import os, pathlib
repo = pathlib.Path.cwd()
if not (repo / 'rust-toolchain.toml').exists():
    raise SystemExit('rustup resolved outside repository')
print(pathlib.Path(os.environ['SELECTED_BIN']) / sys.argv[-1])
''')
        self.script(self.bin / 'cargo', '''import os, shutil, subprocess
if sys.argv[1:] == ['--version']:
    print('cargo 1.99.1 fixture')
else:
    for tool in ['rustc','rustdoc','rustfmt','clippy-driver']:
        subprocess.run([tool, '--version'], check=True, stdout=subprocess.DEVNULL)
    print(json.dumps(dict(cwd=os.getcwd(), args=sys.argv[1:],
        tools={t: shutil.which(t) for t in ['rustc','rustdoc','rustfmt','clippy-driver']},
        cc=os.environ.get('CC'), cxx=os.environ.get('CXX'),
        path=os.environ.get('PATH', ''),
        dyld=os.environ['DYLD_FALLBACK_LIBRARY_PATH'], override=os.environ.get('RUSTUP_TOOLCHAIN'))))
''')
        for tool in ('rustc', 'rustdoc', 'rustfmt', 'clippy-driver'):
            self.script(self.bin / tool, "import os\nprint('rustc fixture\\nrelease: ' + os.environ.get('FAKE_RELEASE', '1.99.1') + '\\nhost: fixture-host')")
            self.script(self.hostile / tool, "raise SystemExit('hostile tool selected')")
        self.env = {k: v for k, v in os.environ.items() if k not in (
            'RUSTC', 'RUSTDOC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER',
            'RUSTC_BOOTSTRAP', 'RUSTUP_TOOLCHAIN', 'CC', 'CXX', 'DYLD_FALLBACK_LIBRARY_PATH',
            'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTDOC', 'CARGO_BUILD_RUSTC_WRAPPER',
            'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER')}
        self.env.update(PATH=f'{self.hostile}:{Path(sys.executable).parent}:/usr/bin:/bin',
                        SELECTED_BIN=str(self.bin), CARGO_HOME=str(self.root / 'cargo-home'))

    def script(self, path, body):
        path.write_text(f'#!{sys.executable}\nimport sys, json\n' + body + '\n')
        path.chmod(0o755)

    def run_wrapper(self, *args, **env):
        command = [str(self.repo / 'tools/cargo.sh'), *args]
        # macOS SIP strips DYLD variables at /bin/bash exec. Set the fixture
        # inside that shell, then source the executable wrapper without reexec.
        if 'DYLD_FALLBACK_LIBRARY_PATH' in env:
            command = ['/bin/bash', '-c',
                       'export DYLD_FALLBACK_LIBRARY_PATH=/existing; source "$@"',
                       'fixture', *command]
        return subprocess.run(command, cwd=self.root,
                              env=self.env | env, text=True, capture_output=True)

    def test_missing_home_does_not_abort_the_wrapper(self):
        # A minimal container or an `env -i` job has no HOME. The wrapper runs
        # under `set -u`, so an unguarded `${HOME}` would abort it before any
        # work ("HOME: unbound variable") and take every build/gate entrypoint
        # with it. The `~/.cargo/bin` scanner entry and rustup's `~/lib` dyld
        # fallback are both skipped instead.
        for label, home in (('unset', None), ('empty', '')):
            with self.subTest(home=label):
                env = {k: v for k, v in self.env.items() if k != 'HOME'}
                if home is not None:
                    env['HOME'] = home
                result = subprocess.run(
                    [str(self.repo / 'tools/cargo.sh'), 'check', '--locked'],
                    cwd=self.root, env=env, text=True, capture_output=True)
                self.assertNotIn('unbound variable', result.stderr)
                self.assertEqual(result.returncode, 0, result.stderr)
                observed = json.loads(result.stdout)
                # The scanner entry is skipped, not fabricated from an empty HOME.
                self.assertNotIn('.cargo/bin', observed['path'])
                self.assertNotIn('//lib', observed['dyld'])

    def test_outside_repository_uses_matching_tools_and_preserves_cwd(self):
        result = self.run_wrapper('check', '--locked')
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        # The required dependency scanner lives in ~/.cargo/bin; the wrapper
        # must put it on PATH for the gate to find it.
        self.assertIn(f'{Path.home()}/.cargo/bin', observed['path'])
        self.assertEqual(observed['cwd'], str(self.root))
        self.assertEqual(observed['args'], ['check', '--locked'])
        self.assertEqual(set(observed['tools'].values()), {str(self.bin / t) for t in observed['tools']})
        self.assertEqual(observed['cc'], '/usr/bin/cc')
        self.assertEqual(observed['cxx'], '/usr/bin/c++')
        self.assertEqual(observed['dyld'], f'{self.bin.parent}/lib:{Path.home()}/lib:/usr/local/lib:/usr/lib')

    def test_developer_override_and_existing_native_environment_survive(self):
        result = self.run_wrapper('check', RUSTUP_TOOLCHAIN='developer', CC='/custom/cc',
                                  CXX='/custom/cxx', DYLD_FALLBACK_LIBRARY_PATH='/existing')
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        self.assertEqual(observed['override'], 'developer')
        self.assertEqual(observed['cc'], '/custom/cc')
        self.assertEqual(observed['cxx'], '/custom/cxx')
        self.assertEqual(observed['dyld'], f'{self.bin.parent}/lib:/existing')

    def test_qualification_records_full_compiler_and_rejects_other_release(self):
        good = self.run_wrapper('--abi-qualify-toolchain')
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertIn('release: 1.99.1', good.stdout)
        self.assertIn(str(self.bin / 'rustc'), good.stdout)
        bad = self.run_wrapper('--abi-qualify-toolchain', FAKE_RELEASE='1.100.0-nightly')
        self.assertNotEqual(bad.returncode, 0)

    def test_qualification_rejects_bootstrap_and_compiler_substitution(self):
        for variable in ('RUSTC_BOOTSTRAP', 'RUSTC', 'RUSTDOC', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTDOC',
                         'CARGO_BUILD_RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER'):
            with self.subTest(variable=variable):
                result = self.run_wrapper('--abi-qualify-toolchain', **{variable: 'substitute'})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(variable, result.stderr)

    def test_qualification_rejects_cargo_config_substitutions(self):
        for location in (self.root / '.cargo', self.root / 'cargo-home'):
            location.mkdir(exist_ok=True)
            config = location / 'config.toml'
            for content in ('[build]\nrustc = "/substitute"\n',
                            '[build]\nrustdoc = "/substitute"\n',
                            '[build]\nrustc-wrapper = "/substitute"\n',
                            '[build]\nrustc-workspace-wrapper = "/substitute"\n',
                            '[env]\nRUSTC_BOOTSTRAP = "1"\n'):
                with self.subTest(location=location, content=content):
                    config.write_text(content)
                    result = self.run_wrapper('--abi-qualify-toolchain')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('stable qualification rejects', result.stderr)
            config.write_text('[build]\njobs = 2\n')
            self.assertEqual(self.run_wrapper('--abi-qualify-toolchain').returncode, 0)
            config.unlink()

    def test_repository_pin_and_components_are_exact(self):
        toolchain = tomllib.loads((ROOT / 'rust-toolchain.toml').read_text())['toolchain']
        self.assertEqual(toolchain['channel'], '1.99.1')
        self.assertEqual(toolchain['components'], ['rustfmt', 'clippy', 'rust-src'])

    def test_resolution_failure_stops_before_cargo(self):
        self.script(self.hostile / 'rustup', "raise SystemExit('resolution refused')")
        result = self.run_wrapper('check')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('resolution refused', result.stderr)
        self.assertEqual(result.stdout, '')

    def test_forced_cargo_path_cannot_qualify_a_substituted_real_compiler(self):
        # Real Cargo applies [env] after the wrapper has established PATH.
        # The sentinel proves that this is a reachable substitution, then the
        # qualification admission must stop it before Cargo is allowed to run.
        rustup = shutil.which('rustup')
        selected = subprocess.run(
            [rustup, 'which', '--toolchain', '1.99.1', 'cargo'],
            cwd=ROOT, check=True, text=True, capture_output=True,
        ).stdout.strip()
        self.env['SELECTED_BIN'] = str(Path(selected).parent)
        self.env['RUSTUP_TOOLCHAIN'] = '1.99.1'
        marker = self.root / 'compiler-was-substituted'
        self.script(self.hostile / 'rustc',
                    f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
                    "print('SUBSTITUTED_RUSTC', file=sys.stderr)\nraise SystemExit(81)")
        (self.root / 'Cargo.toml').write_text(
            '[package]\nname="compiler-fixture"\nversion="0.1.0"\nedition="2024"\n'
            '[lib]\npath="lib.rs"\n')
        (self.root / 'lib.rs').write_text('pub fn fixture() {}\n')
        config_dir = self.root / '.cargo'
        config_dir.mkdir()
        (config_dir / 'config.toml').write_text(
            '[env]\nPATH = { value = ' + json.dumps(f'{self.hostile}:/usr/bin:/bin')
            + ', force = true }\n')
        build = self.run_wrapper('check', '--offline', '--locked',
                                 CARGO_TARGET_DIR=str(self.root / 'target'))
        self.assertNotEqual(build.returncode, 0)
        self.assertIn('SUBSTITUTED_RUSTC', build.stderr)
        self.assertTrue(marker.exists())
        marker.unlink()
        qualified = self.run_wrapper('--abi-qualify-toolchain')
        if qualified.returncode == 0:
            self.run_wrapper('check', '--offline', '--locked',
                             CARGO_TARGET_DIR=str(self.root / 'target'))
        self.assertFalse(marker.exists(), 'qualification admitted substituted rustc')
        self.assertNotEqual(qualified.returncode, 0)
        self.assertIn('env.PATH', qualified.stderr)

    def test_invalid_compiler_configuration_fails_closed_with_diagnostic(self):
        location = self.root / '.cargo'
        location.mkdir()
        for content in ('[build]\nrustc = false\n',
                        '[build]\nrustdoc = ""\n',
                        'build = false\n', 'env = false\n',
                        '[env]\nRUSTC_BOOTSTRAP = false\n'):
            with self.subTest(content=content):
                (location / 'config.toml').write_text(content)
                result = self.run_wrapper('--abi-qualify-toolchain')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('stable qualification', result.stderr)
                self.assertNotIn('Traceback', result.stderr)
