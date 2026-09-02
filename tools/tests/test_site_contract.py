"""Dependency-free checks for ABI's published product contract."""

from datetime import date
from html import unescape
import json
import math
from pathlib import Path, PurePosixPath
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"


def _slice(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except IndexError as exc:
        raise AssertionError(f"missing bounded section: {start!r} .. {end!r}") from exc


def _site_catalog(page: str, heading: str) -> list[str]:
    match = re.search(
        rf"<h3>{re.escape(heading)}\s+.*?</h3>\s*<div class=\"chips\">(.*?)</div>",
        page,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing site catalog: {heading}")
    return re.findall(r'<span class="chip">([^<]+)</span>', match.group(1))


def _wdbx_path_dependencies() -> dict[str, PurePosixPath]:
    manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    dependencies = manifest["workspace"]["dependencies"]
    return {
        name: PurePosixPath(spec["path"])
        for name, spec in dependencies.items()
        if isinstance(spec, dict) and str(spec.get("path", "")).startswith("../wdbx/")
    }


class SiteContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.page = (SITE / "index.html").read_text(encoding="utf-8")

    def test_quick_start_resolves_every_manifest_wdbx_dependency(self) -> None:
        block = unescape(_slice(self.page, "build &amp; validate", "</pre>"))
        commands = re.sub(r"<[^>]+>", "", block)
        lines = [line.strip() for line in commands.splitlines() if line.strip()]
        expected = [
            "mkdir abi-workspace",
            "cd abi-workspace",
            "git clone https://github.com/donaldfilimon/wdbx",
            "git clone https://github.com/donaldfilimon/abi",
            "cd abi",
            "./tools/check.sh",
        ]
        positions = [lines.index(command) for command in expected]
        self.assertEqual(positions, sorted(positions))

        dependencies = _wdbx_path_dependencies()
        self.assertTrue(dependencies)
        abi_root = PurePosixPath("/quick-start/abi-workspace/abi")
        sibling_root = abi_root.parent / "wdbx"
        for name, relative in dependencies.items():
            resolved = abi_root.parent / relative.parts[1] / PurePosixPath(*relative.parts[2:])
            self.assertTrue(
                resolved.is_relative_to(sibling_root / "crates"),
                f"{name} does not resolve through the documented sibling: {relative}",
            )

    def test_cli_catalog_matches_executable_and_golden_sources_exactly(self) -> None:
        source = (ROOT / "crates/abi-cli/src/usage.rs").read_text(encoding="utf-8")
        table = _slice(source, "pub const COMMANDS: &[Command] = &[", "\n];")
        executable = re.findall(r'Command\s*\{\s*name:\s*"([^"]+)"', table)
        golden = json.loads((ROOT / "tests/golden/help.json").read_text(encoding="utf-8"))
        self.assertEqual(_site_catalog(self.page, "CLI commands"), executable)
        self.assertEqual(executable, [command["name"] for command in golden["commands"]])

    def test_mcp_catalog_matches_executable_and_golden_sources_exactly(self) -> None:
        source = (ROOT / "crates/abi-mcp/src/handlers.rs").read_text(encoding="utf-8")
        table = _slice(source, "const TOOLS: &[ToolDescriptor] = &[", "\n];")
        executable = re.findall(r'ToolDescriptor\s*\{\s*name:\s*"([^"]+)"', table)
        golden = json.loads(
            (ROOT / "tests/golden/mcp-tools-list.json").read_text(encoding="utf-8")
        )
        self.assertEqual(_site_catalog(self.page, "MCP tools"), executable)
        self.assertEqual(executable, [tool["name"] for tool in golden["result"]["tools"]])

    def test_benchmark_fixture_has_exact_schema_and_synthetic_provenance(self) -> None:
        records = json.loads((SITE / "data/sample_benchmarks.json").read_text(encoding="utf-8"))
        self.assertIsInstance(records, list)
        self.assertTrue(records)
        expected_keys = {"date", "p50", "p90", "p99", "throughput"}
        dates: list[date] = []
        for record in records:
            self.assertEqual(set(record), expected_keys)
            dates.append(date.fromisoformat(record["date"]))
            values = [record[name] for name in ("p50", "p90", "p99", "throughput")]
            self.assertTrue(all(type(value) in (int, float) for value in values))
            self.assertTrue(all(math.isfinite(value) and value >= 0 for value in values))
            self.assertLessEqual(record["p50"], record["p90"])
            self.assertLessEqual(record["p90"], record["p99"])
        self.assertEqual(dates, sorted(set(dates)))

        dashboard = (SITE / "benchmarks.html").read_text(encoding="utf-8")
        data_readme = (SITE / "data/README.md").read_text(encoding="utf-8")
        self.assertIn("synthetic placeholder", dashboard)
        self.assertIn("No number here was measured from a running ABI build.", dashboard)
        self.assertIn("**synthetic placeholder data**", data_readme)
        self.assertIn("It is not a measurement of ABI", data_readme)

    def test_dashboard_rejects_malformed_values_before_rendering(self) -> None:
        dashboard = (SITE / "benchmarks.html").read_text(encoding="utf-8")
        validation = _slice(dashboard, "function validIsoDate(value)", "async function main()")
        self.assertIn(r"/^(\d{4})-(\d{2})-(\d{2})$/", validation)
        self.assertIn("year < 1 || month < 1 || month > 12", validation)
        self.assertIn("year % 400 === 0", validation)
        self.assertIn("day >= 1 && day <= monthDays[month - 1]", validation)
        self.assertIn("Object.keys(record)", validation)
        self.assertIn("Number.isFinite(record[name])", validation)
        self.assertIn("record[name] >= 0", validation)
        self.assertIn("record.p50 <= record.p90", validation)
        self.assertIn("record.p90 <= record.p99", validation)
        self.assertIn("index === 0 || records[index - 1].date < record.date", validation)
        guard = "if (!validRecords(records))"
        self.assertIn(guard, dashboard)
        self.assertLess(dashboard.index(guard), dashboard.index("const labels = records.map"))


if __name__ == "__main__":
    unittest.main()
