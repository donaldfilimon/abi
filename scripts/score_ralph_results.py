#!/usr/bin/env python3
"""Score ABI Ralph-loop outputs and produce PASS/FAIL summary."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Rule:
    name: str
    keywords: tuple[str, ...]
    min_len: int


RULES: tuple[Rule, ...] = (
    Rule("vnext_policy", ("compat", "legacy", "vnext", "release"), 80),
    Rule("toolchain_determinism", ("zvm", ".zigversion", "PATH", "zig version"), 80),
    Rule("mod_stub_drift", ("mod", "stub", "parity", "compile"), 80),
    Rule("split_parity_validation", ("parity", "test", "behavior", "module"), 80),
    Rule("migration_mapping", ("Framework", "App", "Config", "Capability"), 70),
)


def _contains_keywords(text: str, keywords: tuple[str, ...]) -> float:
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / max(len(keywords), 1)


def _item_score(item: dict[str, Any], rule: Rule, require_live: bool) -> tuple[float, list[str]]:
    reasons: list[str] = []

    output = str(item.get("output") or "")
    notes = str(item.get("notes") or "")

    live = "provider=openai" in notes and "placeholder" not in notes and "dry_run" not in notes
    if require_live and not live:
        reasons.append("not_live_openai_output")

    if len(output.strip()) < rule.min_len:
        reasons.append("output_too_short")

    kw_score = _contains_keywords(output, rule.keywords)
    score = kw_score
    if len(output.strip()) >= rule.min_len:
        score = (score + 1.0) / 2.0
    if require_live and not live:
        score *= 0.2

    return score, reasons


def _load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("results JSON must be a list")
    out: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def _format_summary(
    in_path: Path,
    score: float,
    passed: bool,
    require_live: bool,
    min_average: float,
    item_rows: list[str],
) -> str:
    status = "PASS" if passed else "FAIL"
    live_text = "required" if require_live else "optional"
    rows = "\n".join(item_rows)
    return (
        "# Ralph Upgrade Score\n\n"
        f"- Input: `{in_path}`\n"
        f"- Live OpenAI outputs: {live_text}\n"
        f"- Average score: {score:.3f}\n"
        f"- Threshold: {min_average:.3f}\n"
        f"- Result: **{status}**\n\n"
        "## Item scores\n"
        f"{rows}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Ralph-loop outputs for ABI upgrade gates.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input Ralph results JSON.")
    parser.add_argument("--out", dest="out_path", required=True, help="Output markdown summary path.")
    parser.add_argument("--min-average", type=float, default=0.75, help="Minimum average score to pass.")
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Require live OpenAI provider outputs (non-placeholder, non-dry-run).",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        raise FileNotFoundError(f"missing results file: {in_path}")

    results = _load_results(in_path)
    if not results:
        raise ValueError("results file is empty")

    rows: list[str] = []
    total = 0.0
    count = min(len(results), len(RULES))

    for idx in range(count):
        item = results[idx]
        rule = RULES[idx]
        item_score, reasons = _item_score(item, rule, args.require_live)
        total += item_score
        reasons_text = ", ".join(reasons) if reasons else "ok"
        rows.append(f"- `{rule.name}`: {item_score:.3f} ({reasons_text})")

    avg = total / max(count, 1)
    passed = avg >= args.min_average

    summary = _format_summary(
        in_path=in_path,
        score=avg,
        passed=passed,
        require_live=args.require_live,
        min_average=args.min_average,
        item_rows=rows,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary)
    print(summary)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
