import json
import tempfile
import unittest
from pathlib import Path

from tools.abbey_contracts import (
    ContractError,
    discover_artifacts,
    load_json_strict,
    validate_fixture,
    verify_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "contracts" / "abbey"


class CorpusBoundaryTests(unittest.TestCase):
    def test_duplicate_members_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "duplicate_member"):
                load_json_strict(path)

    def test_corpus_has_no_symlink_or_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside-contract.json"
            outside.write_text("{}\n", encoding="utf-8")
            (root / "escaped.json").symlink_to(outside)

            with self.assertRaisesRegex(ContractError, "symlink_forbidden"):
                discover_artifacts(root)

    def test_manifest_lists_every_normative_artifact_once(self) -> None:
        report = verify_manifest(CORPUS_ROOT)

        self.assertEqual(report.unlisted, ())
        self.assertEqual(report.missing, ())
        self.assertEqual(report.duplicates, ())

    def test_manifest_builder_is_not_implicitly_available_to_verification(self) -> None:
        manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertIsInstance(manifest["aggregate_digest"], str)
        self.assertEqual(len(manifest["aggregate_digest"]), 64)


class SchemaContractTests(unittest.TestCase):
    def assert_fixture(self, taxonomy: str, name: str, expected: str) -> None:
        outcome = validate_fixture(
            CORPUS_ROOT,
            CORPUS_ROOT / "v1" / "fixtures" / taxonomy / name,
        )
        self.assertEqual(outcome.code, expected)

    def test_principal_keeps_channel_and_subject_identity_separate(self) -> None:
        self.assert_fixture("valid", "identity-principal.json", "valid")

    def test_platform_scope_rejects_cross_guild_wildcards(self) -> None:
        self.assert_fixture("invalid", "identity-wildcard-scope.json", "schema_invalid")

    def test_raw_discord_snowflakes_are_not_opaque_contract_ids(self) -> None:
        self.assert_fixture("privacy", "identity-raw-snowflake.json", "forbidden_content")

    def test_delegation_rejects_a_ninth_hop_before_retention(self) -> None:
        self.assert_fixture("boundary", "identity-delegation-nine-hops.json", "schema_invalid")

    def test_delegation_rejects_repeated_principals(self) -> None:
        self.assert_fixture("invalid", "identity-delegation-cycle.json", "delegation_cycle")

    def test_delegation_allows_a_finite_connected_chain(self) -> None:
        self.assert_fixture("valid", "identity-delegation-chain.json", "valid")


if __name__ == "__main__":
    unittest.main()
