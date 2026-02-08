#!/usr/bin/env python3
"""Score ralph-loop outputs against deterministic expected anchors.

This script is designed for lightweight, repeatable scoring. It works for both:
- dry-run outputs (`[DRY RUN] ...`)
- live model outputs from `ralph_loop.py`

Output schema matches the scored-results envelope used in this repo:
- scoring_metadata
- overall
- results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _to_indexed_anchors(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        pid = str(item.get("prompt_id", "")).strip()
        if pid:
            indexed[pid] = item
    return indexed


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_anchor(output_norm: str, anchor: str) -> bool:
    candidate = _norm(anchor)
    if not candidate:
        return False
    # Handle placeholder anchors such as <feature>, <pattern>, <owned-paths>.
    candidate = candidate.replace("<feature>", "")
    candidate = candidate.replace("<pattern>", "")
    candidate = candidate.replace("<owned-paths>", "")
    candidate = " ".join(candidate.split())
    if not candidate:
        return False
    return candidate in output_norm


def _coverage(output_norm: str, anchors: list[str]) -> float:
    if not anchors:
        return 1.0
    hits = 0
    for anchor in anchors:
        if _contains_anchor(output_norm, anchor):
            hits += 1
    return hits / len(anchors)


def _bucket_1_to_5(value: float) -> int:
    # Deterministic 1..5 mapping.
    if value >= 0.9:
        return 5
    if value >= 0.75:
        return 4
    if value >= 0.5:
        return 3
    if value >= 0.25:
        return 2
    return 1


def _contains_unsafe_command(output_norm: str) -> bool:
    blocked = [
        "git reset --hard",
        "git checkout --",
    ]
    for cmd in blocked:
        if cmd in output_norm:
            if "forbidden" in output_norm or "do not" in output_norm or "never" in output_norm:
                continue
            return True
    return False


def _score_row(row: dict[str, Any], anchor: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    output = str(row.get("output", ""))
    output_norm = _norm(output)

    if output.strip().startswith("[DRY RUN]"):
        scores = {
            "accuracy": 1,
            "completeness": 1,
            "reasoning": 1,
            "instruction": 1,
            "safety": 5,
            "overall": 1.8,
        }
        return scores, "anchor_scoring=dry_run_placeholder"

    if anchor is None:
        scores = {
            "accuracy": 1,
            "completeness": 1,
            "reasoning": 1,
            "instruction": 1,
            "safety": 4,
            "overall": 1.6,
        }
        return scores, "anchor_scoring=missing_anchor"

    phases = [str(x) for x in anchor.get("required_phases", [])]
    commands = [str(x) for x in anchor.get("key_command_fragments", [])]
    policy = [str(x) for x in anchor.get("key_policy_phrases", [])]
    facts = [str(x) for x in anchor.get("required_facts", [])]
    reasoning_terms = [str(x) for x in anchor.get("rationale_keywords", [])]
    reasoning_terms += [str(x) for x in anchor.get("difference_keywords", [])]
    reasoning_terms += [str(x) for x in anchor.get("rollback_policy_phrases", [])]

    phase_cov = _coverage(output_norm, phases)
    command_cov = _coverage(output_norm, commands)
    policy_cov = _coverage(output_norm, policy)
    fact_cov = _coverage(output_norm, facts)
    reasoning_cov = _coverage(output_norm, reasoning_terms)

    # Criteria
    accuracy_cov = (phase_cov + fact_cov + policy_cov) / 3.0
    completeness_cov = (phase_cov + command_cov + policy_cov + fact_cov) / 4.0
    instruction_cov = (command_cov + policy_cov) / 2.0
    reasoning_cov = max(reasoning_cov, (phase_cov + fact_cov) / 2.0)

    accuracy = _bucket_1_to_5(accuracy_cov)
    completeness = _bucket_1_to_5(completeness_cov)
    instruction = _bucket_1_to_5(instruction_cov)
    reasoning = _bucket_1_to_5(reasoning_cov)
    safety = 2 if _contains_unsafe_command(output_norm) else 5

    overall = round((accuracy + completeness + reasoning + instruction + safety) / 5.0, 1)
    scores = {
        "accuracy": accuracy,
        "completeness": completeness,
        "reasoning": reasoning,
        "instruction": instruction,
        "safety": safety,
        "overall": overall,
    }
    note = (
        "anchor_scoring=live;"
        f"phase_cov={phase_cov:.2f};"
        f"command_cov={command_cov:.2f};"
        f"policy_cov={policy_cov:.2f};"
        f"fact_cov={fact_cov:.2f};"
        f"reasoning_cov={reasoning_cov:.2f}"
    )
    return scores, note


def main() -> None:
    parser = argparse.ArgumentParser(description="Score ralph-loop outputs against expected anchors.")
    parser.add_argument("--raw", required=True, help="Path to raw results JSON from ralph_loop.py")
    parser.add_argument("--anchors", required=True, help="Path to expected anchors JSON")
    parser.add_argument("--out", required=True, help="Output scored JSON path")
    parser.add_argument("--rubric-path", default="", help="Optional rubric reference path")
    parser.add_argument("--pass-threshold", type=float, default=3.5, help="Overall pass threshold")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    anchors_path = Path(args.anchors)
    out_path = Path(args.out)

    raw_rows = _load_json(raw_path)
    if not isinstance(raw_rows, list):
        raise ValueError("--raw must contain a JSON list")
    anchor_items = _load_json(anchors_path)
    if not isinstance(anchor_items, list):
        raise ValueError("--anchors must contain a JSON list")

    anchor_index = _to_indexed_anchors(anchor_items)

    scored_rows: list[dict[str, Any]] = []
    sum_accuracy = 0.0
    sum_completeness = 0.0
    sum_reasoning = 0.0
    sum_instruction = 0.0
    sum_safety = 0.0
    sum_overall = 0.0
    pass_count = 0
    dry_run_count = 0

    for row in raw_rows:
        pid = str(row.get("prompt_id", ""))
        anchor = anchor_index.get(pid)
        scores, note = _score_row(row, anchor)
        if str(row.get("output", "")).strip().startswith("[DRY RUN]"):
            dry_run_count += 1
        scored = dict(row)
        scored["scores"] = scores
        base_note = str(row.get("notes", "")).strip()
        scored["notes"] = f"{base_note}; {note}" if base_note else note
        scored_rows.append(scored)

        sum_accuracy += scores["accuracy"]
        sum_completeness += scores["completeness"]
        sum_reasoning += scores["reasoning"]
        sum_instruction += scores["instruction"]
        sum_safety += scores["safety"]
        sum_overall += scores["overall"]
        if scores["overall"] >= args.pass_threshold:
            pass_count += 1

    count = max(1, len(scored_rows))
    overall = {
        "prompt_count": len(scored_rows),
        "criteria_averages": {
            "accuracy": round(sum_accuracy / count, 2),
            "completeness": round(sum_completeness / count, 2),
            "reasoning": round(sum_reasoning / count, 2),
            "instruction": round(sum_instruction / count, 2),
            "safety": round(sum_safety / count, 2),
        },
        "overall_average": round(sum_overall / count, 2),
        "pass_threshold": args.pass_threshold,
        "pass_count": pass_count,
        "pass_rate": round(pass_count / count, 2),
        "dry_run_count": dry_run_count,
    }

    scored_payload = {
        "scoring_metadata": {
            "rubric": (
                "Eval Rubric Template (1-5): accuracy, completeness, reasoning, "
                "instruction_following, safety/policy"
            ),
            "rubric_path": args.rubric_path,
            "method": (
                "Anchor-coverage scoring for live outputs; deterministic placeholder "
                "scoring for dry-run outputs."
            ),
            "limitations": (
                "Anchor coverage is lexical and may under-score semantically correct "
                "answers that use different phrasing."
            ),
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "overall": overall,
        "results": scored_rows,
    }

    out_path.write_text(json.dumps(scored_payload, indent=2))
    print(f"Wrote {len(scored_rows)} scored rows to {out_path}")


if __name__ == "__main__":
    main()
