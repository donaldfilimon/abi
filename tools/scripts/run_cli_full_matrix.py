#!/usr/bin/env python3
"""Run exhaustive ABI CLI command matrix with isolation and strict pass criteria."""

from __future__ import annotations

import argparse
import json
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PRECHECK_JSON = Path("/tmp/abi-cli-full-preflight.json")
REPORT_JSON = Path("/tmp/abi-cli-full-report.json")
REPORT_MD = Path("/tmp/abi-cli-full-report.md")
LOG_ROOT = Path("/tmp/abi-cli-full-logs")
MATRIX_JSON = Path("/tmp/abi-cli-full-matrix.json")
PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


@dataclass
class ExecResult:
    exit_code: int
    output: str
    elapsed_ms: int
    started: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full CLI matrix")
    parser.add_argument("--repo", required=True, help="ABI repository root")
    parser.add_argument("--env-file", help="Path to KEY=VALUE env file")
    parser.add_argument(
        "--id-prefix",
        action="append",
        default=[],
        help="Limit run to matrix IDs starting with this prefix. Repeatable.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Continue when preflight checks fail and mark blocked vectors in the report.",
    )
    parser.add_argument(
        "--pty-probe-window",
        type=float,
        default=8.0,
        help="Seconds to wait for PTY sessions before forcing shutdown.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep isolated temp workspace for debugging.",
    )
    parser.add_argument("--timeout-scale", type=float, default=1.0)
    return parser.parse_args()


def load_env_file(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        if key:
            result[key] = value
    return result


def run_checked(cmd: List[str], cwd: Path, env: Dict[str, str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def run_oneshot(cmd: List[str], cwd: Path, env: Dict[str, str], timeout_s: float) -> ExecResult:
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    elapsed = int((time.time() - start) * 1000)
    output = (proc.stdout or "") + (proc.stderr or "")
    return ExecResult(exit_code=proc.returncode, output=output, elapsed_ms=elapsed)


def timeout_output(exc: subprocess.TimeoutExpired) -> str:
    parts: List[str] = []
    for payload in (exc.stdout, exc.stderr):
        if payload is None:
            continue
        if isinstance(payload, bytes):
            parts.append(payload.decode("utf-8", errors="replace"))
        else:
            parts.append(payload)
    return "".join(parts)


def run_probe(cmd: List[str], cwd: Path, env: Dict[str, str], timeout_s: float, startup_s: float) -> ExecResult:
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    started = False
    output_parts: List[str] = []

    try:
        proc.wait(timeout=startup_s)
        started = False
    except subprocess.TimeoutExpired:
        started = True

    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=min(5.0, timeout_s))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    if proc.stdout is not None:
        try:
            remainder = proc.stdout.read()
            if remainder:
                output_parts.append(remainder)
        except Exception:
            pass

    elapsed = int((time.time() - start) * 1000)
    return ExecResult(
        exit_code=proc.returncode if proc.returncode is not None else -9,
        output="".join(output_parts),
        elapsed_ms=elapsed,
        started=started,
    )


def pty_script_for(args: List[str]) -> bytes:
    if not args:
        return b"q"
    if args[0] == "agent":
        return b"exit\n"
    if len(args) >= 2 and args[0] == "llm" and args[1] == "chat":
        return b"/exit\n"
    return b"q"


def run_pty(
    cmd: List[str],
    args: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout_s: float,
    probe_window_s: float,
) -> ExecResult:
    start = time.time()
    master_fd, slave_fd = pty.openpty()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        text=False,
    )
    os.close(slave_fd)

    output = bytearray()

    def read_available(max_wait: float) -> None:
        end = time.time() + max_wait
        while time.time() < end:
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if not r:
                if proc.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            output.extend(chunk)

    try:
        read_available(1.5)

        if proc.poll() is None:
            script = pty_script_for(args)
            try:
                os.write(master_fd, script)
            except OSError:
                # The PTY can close between poll and write when the child exits quickly.
                pass

        probe_window = min(timeout_s, max(probe_window_s, 1.0))
        deadline = time.time() + probe_window
        while time.time() < deadline:
            read_available(0.2)
            if proc.poll() is not None:
                break

        if proc.poll() is None:
            try:
                os.write(master_fd, b"\x03")
            except OSError:
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)

        read_available(0.5)
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass

    elapsed = int((time.time() - start) * 1000)
    text = output.decode("utf-8", errors="replace")
    return ExecResult(exit_code=proc.returncode if proc.returncode is not None else -9, output=text, elapsed_ms=elapsed)


def expand_args(args: List[str], env: Dict[str, str]) -> Tuple[List[str], List[str]]:
    missing: List[str] = []
    expanded: List[str] = []

    for token in args:
        local_missing: List[str] = []

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            value = env.get(key)
            if value is None or value == "":
                local_missing.append(key)
                return ""
            return value

        out = PLACEHOLDER_RE.sub(repl, token)
        expanded.append(out)
        missing.extend(local_missing)

    return expanded, sorted(set(missing))


def is_exit_ok(exit_code: int, policy: str) -> bool:
    if policy == "zero_only":
        return exit_code == 0
    # allow signal terminations from controlled probe stop
    return exit_code == 0 or exit_code < 0 or exit_code in {130, 143}


def check_requires(requires: List[str], env: Dict[str, str], blocked_requirements: List[str]) -> List[str]:
    blocked: List[str] = []
    blocked_set = set(blocked_requirements)
    for req in requires:
        if req in blocked_set:
            blocked.append(req)
            continue
        if req.startswith("env:"):
            key = req.split(":", 1)[1]
            if not env.get(key):
                blocked.append(req)
        elif req.startswith("file:"):
            key = req.split(":", 1)[1]
            path = env.get(key, "")
            if not path or not Path(path).exists():
                blocked.append(req)
        elif req.startswith("tool:"):
            tool = req.split(":", 1)[1]
            if shutil.which(tool, path=env.get("PATH")) is None:
                blocked.append(req)
    return blocked


def blocked_requirements_from_preflight(preflight_details: Dict[str, object]) -> List[str]:
    blocked: List[str] = []
    blocked.extend(f"env:{name}" for name in preflight_details.get("missing_env", []))
    blocked.extend(f"tool:{name}" for name in preflight_details.get("missing_tools", []))
    blocked.extend(f"net:{name}" for name in preflight_details.get("failed_connectivity", []))
    return sorted(set(blocked))


def coerce_preflight_details(raw: Dict[str, object]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "missing_env": raw.get("missing_env", []),
        "missing_tools": raw.get("missing_tools", []),
        "failed_connectivity": raw.get("failed_connectivity", []),
    }
    return result


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def write_reports(records: List[dict], blocked: List[dict], failed: List[dict], run_id: str) -> None:
    summary = {
        "run_id": run_id,
        "ran": len(records),
        "passed": len([r for r in records if r["status"] == "passed"]),
        "failed": len(failed),
        "blocked": len(blocked),
        "records": records,
        "failed_records": failed,
        "blocked_records": blocked,
    }

    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ABI CLI Full Matrix Report",
        "",
        f"- Ran: {summary['ran']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Blocked: {summary['blocked']}",
        "",
        "## Residual risk",
        "",
        "- External integrations can still fail due to remote service instability.",
        "- Interactive probes validate startup/exit paths, not full manual UX depth.",
    ]

    if failed:
        lines.extend(["", "## Failed", ""])
        for item in failed:
            lines.append(f"- `{item['id']}` rc={item['exit_code']} ({item['kind']})")

    if blocked:
        lines.extend(["", "## Blocked", ""])
        for item in blocked:
            lines.append(f"- `{item['id']}` blocked by: {', '.join(item.get('blocked_by', []))}")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()

    env_overrides: Dict[str, str] = {}
    if args.env_file:
        env_overrides = load_env_file(Path(args.env_file).expanduser())

    merged_env = dict(os.environ)
    merged_env.update(env_overrides)

    # Generate matrix from typed Zig manifest.
    try:
        run_checked(
            [
                "zig",
                "run",
                "--dep",
                "abi",
                "-Mmain=tools/cli/full_matrix_main.zig",
                "--dep",
                "build_options",
                "-Mabi=src/abi.zig",
                "-Mbuild_options=tools/cli/tests/build_options_stub.zig",
                "--",
                "--json-out",
                str(MATRIX_JSON),
            ],
            cwd=repo,
            env=merged_env,
            timeout=180,
        )
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "")[-4000:]
        stderr = (exc.stderr or "")[-4000:]
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        print("Matrix generation failed. Could not compile or emit CLI matrix JSON.", file=sys.stderr)
        return 1

    # Run preflight.
    preflight_cmd = [
        "zig",
        "run",
        "tools/scripts/cli_full_preflight.zig",
        "--",
        "--json-out",
        str(PRECHECK_JSON),
    ]
    if args.env_file:
        preflight_cmd.extend(["--env-file", str(Path(args.env_file).expanduser())])

    preflight = subprocess.run(
        preflight_cmd,
        cwd=str(repo),
        env=merged_env,
        text=True,
        capture_output=True,
    )
    preflight_details: Dict[str, object] = {}
    if PRECHECK_JSON.exists():
        try:
            raw = json.loads(PRECHECK_JSON.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                preflight_details = coerce_preflight_details(raw)
        except Exception:
            preflight_details = {}

    preflight_blocked_reqs = blocked_requirements_from_preflight(preflight_details)
    preflight_blocked_record: Dict[str, object] | None = None
    if preflight.returncode != 0:
        blocked_by: List[str] = ["preflight"]
        blocked_by.extend(preflight_blocked_reqs)
        preflight_blocked_record = {
            "id": "preflight",
            "status": "blocked",
            "blocked_by": sorted(set(blocked_by)),
            "missing_env": preflight_details.get("missing_env", []),
            "missing_tools": preflight_details.get("missing_tools", []),
            "failed_connectivity": preflight_details.get("failed_connectivity", []),
            "output_tail": (preflight.stdout + preflight.stderr)[-4000:],
        }

        print(preflight.stdout, end="")
        print(preflight.stderr, end="", file=sys.stderr)
        print(f"Preflight failed. See: {PRECHECK_JSON}")
        if not args.allow_blocked:
            write_reports([], [preflight_blocked_record], [], run_id="preflight-blocked")
            return 1

    matrix = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    if not isinstance(matrix, list):
        print("Matrix JSON was not a list of entries.", file=sys.stderr)
        return 1

    if args.id_prefix:
        requested_prefixes = [prefix for prefix in args.id_prefix if prefix]
        if not requested_prefixes:
            print("At least one non-empty --id-prefix is required when filtering.", file=sys.stderr)
            return 1
        matrix = [entry for entry in matrix if any(str(entry.get("id", "")).startswith(prefix) for prefix in requested_prefixes)]
        if not matrix:
            print(f"No matrix entries matched --id-prefix filters: {requested_prefixes}", file=sys.stderr)
            return 1

    run_id = f"run-{int(time.time())}"
    log_dir = LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    temp_root = Path(tempfile.mkdtemp(prefix="abi-cli-full-"))
    home = temp_root / "home"
    workspace = temp_root / "workspace"
    state_dir = temp_root / "state"
    data_dir = temp_root / "data"
    config_dir = temp_root / "config"
    tmp_dir = temp_root / "tmp"
    for p in (home, workspace, state_dir, data_dir, config_dir, tmp_dir):
        p.mkdir(parents=True, exist_ok=True)

    isolated_env = dict(merged_env)
    isolated_env["HOME"] = str(home)
    isolated_env["XDG_CONFIG_HOME"] = str(config_dir)
    isolated_env["XDG_DATA_HOME"] = str(data_dir)
    isolated_env["XDG_STATE_HOME"] = str(state_dir)
    isolated_env["TMPDIR"] = str(tmp_dir)

    repo_copy = temp_root / "repo"
    shutil.copytree(repo, repo_copy, symlinks=True, ignore=shutil.ignore_patterns(".git", "zig-out", ".zig-cache"))

    # Build once and run the resulting binary across all vectors.
    build_cmd = ["zig", "build"]
    try:
        subprocess.run(build_cmd, cwd=str(repo_copy), env=isolated_env, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stdout = (exc.stdout or "")[-4000:]
        stderr = (exc.stderr or "")[-4000:]
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        if args.keep_temp:
            print(f"Temp root kept: {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)
        print("Failed to build CLI binary for matrix run.", file=sys.stderr)
        return 1

    abi_bin = repo_copy / "zig-out" / "bin" / ("abi.exe" if os.name == "nt" else "abi")
    if not abi_bin.exists():
        raise FileNotFoundError(f"abi binary not found after build: {abi_bin}")

    records: List[dict] = []
    failed: List[dict] = []
    blocked: List[dict] = []
    if preflight_blocked_record is not None:
        blocked.append(preflight_blocked_record)


    for idx, entry in enumerate(matrix, start=1):
        entry_id = entry["id"]
        raw_args = entry["args"]
        kind = entry["kind"]
        timeout_ms = int(float(entry["timeout_ms"]) * args.timeout_scale)
        timeout_s = max(timeout_ms / 1000.0, 1.0)
        requires = entry.get("requires", [])
        cwd_mode = entry.get("cwd_mode", "temp_workspace")
        exit_policy = entry.get("exit_policy", "zero_only")

        expanded_args, missing_vars = expand_args(raw_args, isolated_env)
        missing_reqs = check_requires(requires, isolated_env, preflight_blocked_reqs)

        if missing_vars or missing_reqs:
            item = {
                "id": entry_id,
                "status": "blocked",
                "kind": kind,
                "args": raw_args,
                "blocked_by": sorted(set([f"env:{v}" for v in missing_vars] + missing_reqs)),
            }
            records.append(item)
            blocked.append(item)
            continue

        cwd = repo_copy if cwd_mode == "repo_copy" else workspace
        cmd = [str(abi_bin), *expanded_args]

        try:
            if kind == "oneshot":
                result = run_oneshot(cmd, cwd, isolated_env, timeout_s)
            elif kind == "serve_probe":
                result = run_probe(cmd, cwd, isolated_env, timeout_s, startup_s=4.0)
            elif kind == "long_running_probe":
                result = run_probe(cmd, cwd, isolated_env, timeout_s, startup_s=8.0)
            elif kind == "pty_session":
                result = run_pty(
                    cmd,
                    expanded_args,
                    cwd,
                    isolated_env,
                    timeout_s,
                    probe_window_s=args.pty_probe_window,
                )
            else:
                result = run_oneshot(cmd, cwd, isolated_env, timeout_s)
        except subprocess.TimeoutExpired as exc:
            result = ExecResult(
                exit_code=124,
                output=timeout_output(exc),
                elapsed_ms=timeout_ms,
            )

        if kind in {"serve_probe", "long_running_probe"} and not result.started:
            # Some probe commands can exit quickly and successfully in headless mode.
            status = "passed" if result.exit_code == 0 else "failed"
        else:
            status = "passed" if is_exit_ok(result.exit_code, exit_policy) else "failed"

        tail = result.output[-6000:]
        log_name = f"{idx:04d}-{sanitize_filename(entry_id)}.log"
        (log_dir / log_name).write_text(tail, encoding="utf-8")

        item = {
            "id": entry_id,
            "status": status,
            "kind": kind,
            "args": expanded_args,
            "elapsed_ms": result.elapsed_ms,
            "exit_code": result.exit_code,
            "probe_started": result.started,
            "exit_policy": exit_policy,
            "log": str(log_dir / log_name),
            "output_tail": tail,
        }
        records.append(item)

        if status != "passed":
            failed.append(item)

    write_reports(records, blocked, failed, run_id=run_id)

    print(f"Wrote report: {REPORT_JSON}")
    print(f"Wrote markdown: {REPORT_MD}")
    print(f"Logs: {log_dir}")
    if args.keep_temp or failed:
        print(f"Temp root kept: {temp_root}")
    else:
        shutil.rmtree(temp_root, ignore_errors=True)

    if args.allow_blocked:
        return 1 if failed else 0
    return 1 if failed or blocked else 0


if __name__ == "__main__":
    sys.exit(main())
