#!/usr/bin/env python3
"""Reject Cargo configuration that would substitute the qualified compiler."""
import os
from pathlib import Path
import sys
import tomllib

COMPILER_ENV = (
    'RUSTC_BOOTSTRAP', 'RUSTC', 'RUSTDOC', 'RUSTC_WRAPPER',
    'RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_RUSTC', 'CARGO_BUILD_RUSTDOC',
    'CARGO_BUILD_RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER',
)
COMPILER_KEYS = ('rustc', 'rustdoc', 'rustc-wrapper', 'rustc-workspace-wrapper')


def configuration_paths(cwd: Path, environ: dict[str, str]) -> list[Path]:
    """Cargo searches the invocation directory's ancestors and Cargo home."""
    cwd = cwd.resolve()
    directories = [parent / '.cargo' for parent in (cwd, *cwd.parents)]
    directories.append(Path(environ.get('CARGO_HOME') or Path.home() / '.cargo'))
    paths = []
    for directory in directories:
        # Cargo chooses the extensionless legacy file if both exist.
        legacy = directory / 'config'
        path = legacy if legacy.is_file() else directory / 'config.toml'
        if path.is_file() and path not in paths:
            paths.append(path)
    return paths


def violations(cwd: Path, environ: dict[str, str]) -> list[str]:
    errors = [f'environment {key}' for key in COMPILER_ENV if environ.get(key)]
    for path in configuration_paths(cwd, environ):
        config = tomllib.loads(path.read_text(encoding='utf-8'))
        build = config.get('build', {})
        environment = config.get('env', {})
        if not isinstance(build, dict) or not isinstance(environment, dict):
            raise ValueError(f'{path}: build and env must be tables')
        for key in COMPILER_KEYS:
            # Presence is prohibited even if Cargo would reject the value's type.
            if key in build:
                errors.append(f'{path}: build.{key}')
        # Cargo applies configured env after the wrapper establishes PATH. A
        # forced PATH can therefore select another rustc/rustdoc/tool process.
        # Reject configuration rather than overwrite and hide that substitution.
        for key in (*COMPILER_ENV, 'PATH'):
            if key in environment:
                errors.append(f'{path}: env.{key}')
    return errors


if __name__ == '__main__':
    try:
        errors = violations(Path.cwd(), dict(os.environ))
    except (OSError, ValueError) as error:
        raise SystemExit(f'stable qualification cannot inspect Cargo configuration: {error}')
    if errors:
        raise SystemExit('stable qualification rejects compiler substitution: ' + ', '.join(errors))
