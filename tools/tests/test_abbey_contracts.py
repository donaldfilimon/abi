import json
import shutil
import subprocess
import sys
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
from tools.vendor_abbey_contracts import VendorError, vendor


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "contracts" / "abbey"


class CorpusBoundaryTests(unittest.TestCase):
    def test_current_corpus_is_major_two_and_retains_v1_authority_bytes(self) -> None:
        manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["contract_major"], 2)
        self.assertEqual(manifest["contract_revision"], 2)
        paths = {row["path"] for row in manifest["artifacts"]}
        self.assertIn("v1/schemas/authorization/grant.schema.json", paths)
        self.assertIn("v2/schemas/authorization/grant.schema.json", paths)

    def test_unsupported_contract_major_fails_before_inventory_is_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "corpus"
            shutil.copytree(CORPUS_ROOT, root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["contract_major"] = 3
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ContractError, "contract_major_unsupported"):
                verify_manifest(root)

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


class AuthorizationInvariantTests(SchemaContractTests):
    def test_low_risk_grant_binds_capability_scope_and_expiry(self) -> None:
        self.assert_fixture("valid", "authorization-grant.json", "valid")

    def test_prohibited_capability_is_decodable_but_ungrantable(self) -> None:
        self.assert_fixture("invalid", "authorization-prohibited-grant.json", "schema_invalid")

    def test_approval_cannot_be_issued_by_its_request_subject(self) -> None:
        self.assert_fixture("invalid", "authorization-self-approval.json", "self_approval")

    def test_authority_envelopes_reject_unknown_fields(self) -> None:
        self.assert_fixture("invalid", "authorization-unknown-field.json", "schema_invalid")

    def test_closed_error_rejects_embedded_free_form_cause(self) -> None:
        self.assert_fixture("privacy", "authorization-error-cause.json", "schema_invalid")

    def test_dependency_degradation_cannot_increase_authority(self) -> None:
        self.assert_fixture("degraded", "authorization-dependency-allow.json", "degraded_authority")

    def test_prohibited_candidate_package_remains_decodable_for_denial(self) -> None:
        self.assert_fixture("valid", "authorization-prohibited-package.json", "valid")

    def test_exact_single_use_approval_is_valid(self) -> None:
        self.assert_fixture("valid", "authorization-approval.json", "valid")

    def test_redacted_closed_error_is_valid(self) -> None:
        self.assert_fixture("valid", "authorization-error.json", "valid")


class AuthorizationV2InvariantTests(unittest.TestCase):
    def assert_v2_fixture(self, taxonomy: str, name: str, expected: str) -> None:
        outcome = validate_fixture(CORPUS_ROOT, CORPUS_ROOT / "v2" / "fixtures" / taxonomy / name)
        self.assertEqual(outcome.code, expected)

    def test_exact_grant_approval_decision_and_redacted_evidence_envelopes(self) -> None:
        for name in (
            "authorization-grant-v2.json", "authorization-approval-v2.json",
            "policy-decision-v2.json", "audit-record-v2.json", "outcome-receipt-v2.json",
            "credential-ref-v2.json",
        ):
            with self.subTest(name=name):
                self.assert_v2_fixture("valid", name, "valid")

    def test_v2_prohibited_grant_and_manager_self_approval_fail_closed(self) -> None:
        self.assert_v2_fixture("invalid", "authorization-prohibited-grant-v2.json", "schema_invalid")
        self.assert_v2_fixture("invalid", "authorization-self-approval-v2.json", "self_approval")

    def test_v2_raw_audit_content_is_rejected_before_retention(self) -> None:
        self.assert_v2_fixture("privacy", "audit-content-v2.json", "forbidden_content")

    def test_immutable_change_set_and_distinct_human_approval_are_closed(self) -> None:
        self.assert_v2_fixture("valid", "execution-change-set-v2.json", "valid")
        self.assert_v2_fixture("valid", "authorization-change-approval-v2.json", "valid")
        self.assert_v2_fixture(
            "invalid", "execution-human-authored-change-set-v2.json", "proposal_authority_invalid"
        )
        self.assert_v2_fixture(
            "invalid", "authorization-change-self-approval-v2.json", "self_approval"
        )
        self.assert_v2_fixture(
            "invalid", "authorization-change-under-authorized-v2.json", "approval_insufficient"
        )


class FederationV1InvariantTests(unittest.TestCase):
    def assert_v2_fixture(self, taxonomy: str, name: str, expected: str) -> None:
        outcome = validate_fixture(CORPUS_ROOT, CORPUS_ROOT / "v2" / "fixtures" / taxonomy / name)
        self.assertEqual(outcome.code, expected)

    def test_hello_request_and_sanitized_receipt_are_closed(self) -> None:
        self.assert_v2_fixture("valid", "federation-hello-v1.json", "valid")
        self.assert_v2_fixture("valid", "federation-request-v1.json", "valid")
        self.assert_v2_fixture("valid", "federation-receipt-v1.json", "valid")

    def test_revision_mismatch_fails_before_any_method_body_is_trusted(self) -> None:
        self.assert_v2_fixture(
            "invalid", "federation-revision-mismatch-v1.json", "schema_invalid"
        )

    def test_transport_policy_is_local_bounded_and_never_downgrades_authority(self) -> None:
        policy = json.loads(
            (CORPUS_ROOT / "v2" / "transport" / "abbey-v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(policy["service"], "abbey.v1")
        self.assertEqual(policy["transport"], "unix_domain_socket")
        self.assertEqual(policy["frame_length_encoding"], "uint32_big_endian")
        self.assertEqual(policy["maximum_frame_bytes"], 1024 * 1024)
        self.assertEqual(policy["maximum_json_container_depth"], 32)
        self.assertEqual(policy["authority_legacy_downgrade"], "forbidden")
        self.assertEqual(policy["authority_legacy_retry"], "forbidden")
        self.assertEqual(policy["authority_backend_fallback"], "forbidden")
        self.assertEqual(policy["c6_non_loopback_transport"], "unshipped")


class ExecutionLifecycleTests(SchemaContractTests):
    def test_effect_request_requires_idempotency_and_cancellation_references(self) -> None:
        self.assert_fixture("valid", "execution-request-envelope.json", "valid")
        self.assert_fixture("invalid", "execution-missing-idempotency.json", "idempotency_required")

    def test_response_has_one_closed_terminal_state(self) -> None:
        self.assert_fixture("invalid", "execution-nonterminal-response.json", "schema_invalid")

    def test_metadata_extensions_are_preserved_without_widening_authority(self) -> None:
        self.assert_fixture("unknown-field", "execution-metadata-extension.json", "valid")

    def test_actuator_cancellation_race_is_outcome_unresolved(self) -> None:
        self.assert_fixture("cancellation", "execution-actuator-race.json", "valid")

    def test_partial_rollback_counts_completed_reverted_and_unresolved(self) -> None:
        self.assert_fixture("degraded", "execution-partial-rollback.json", "valid")

    def test_stale_cancellation_reference_fails_closed(self) -> None:
        self.assert_fixture("cancellation", "execution-stale-cancellation.json", "cancellation_mismatch")

    def test_receipt_cannot_embed_content(self) -> None:
        self.assert_fixture("privacy", "execution-receipt-content.json", "forbidden_content")

    def test_execution_binds_the_approved_proposal_digest(self) -> None:
        self.assert_fixture("valid", "execution-approved-proposal.json", "valid")

    def test_recommendation_is_explicitly_non_authorizing(self) -> None:
        self.assert_fixture("valid", "execution-recommendation.json", "valid")

    def test_action_proposal_is_typed_expiring_and_verifiable(self) -> None:
        self.assert_fixture("valid", "execution-action-proposal.json", "valid")

    def test_completion_and_cancellation_are_closed_valid_terminals(self) -> None:
        self.assert_fixture("valid", "execution-complete-response.json", "valid")
        self.assert_fixture("cancellation", "execution-before-start.json", "valid")


class ConsentContractTests(SchemaContractTests):
    def test_epoch_opens_only_with_current_manager_and_unanimous_consent(self) -> None:
        self.assert_fixture("valid", "consent-open-transition.json", "valid")
        self.assert_fixture("invalid", "consent-open-without-manager.json", "consent_open_denied")

    def test_participant_change_closes_and_cancels_epoch_bound_stages(self) -> None:
        self.assert_fixture("cancellation", "consent-participant-change-close.json", "valid")
        self.assert_fixture("invalid", "consent-participant-change-stays-open.json", "consent_close_required")

    def test_barge_in_cancels_playback_without_closing_consent(self) -> None:
        self.assert_fixture("cancellation", "consent-barge-in.json", "valid")

    def test_operator_report_is_fixed_redacted_local_evidence(self) -> None:
        self.assert_fixture("valid", "consent-operator-flow.json", "valid")

    def test_operator_report_rejects_identity_and_media_content(self) -> None:
        self.assert_fixture("privacy", "consent-report-content.json", "forbidden_content")

    def test_connection_loss_closes_the_epoch(self) -> None:
        self.assert_fixture("degraded", "consent-connection-loss.json", "valid")


class EpisodeLearningTests(SchemaContractTests):
    def test_adapter_proposes_but_does_not_compute_canonical_episode_digest(self) -> None:
        self.assert_fixture("valid", "episode-proposal.json", "valid")
        self.assert_fixture("invalid", "episode-adapter-digest.json", "schema_invalid")

    def test_evidence_separates_integrity_provenance_semantics_and_truth(self) -> None:
        self.assert_fixture("valid", "episode-evidence.json", "valid")

    def test_claim_keeps_lifecycle_and_evidence_level_separate(self) -> None:
        self.assert_fixture("valid", "episode-claim.json", "valid")
        self.assert_fixture("invalid", "episode-overstated-claim.json", "evidence_overclaim")

    def test_tombstone_carries_retention_and_deletion_key(self) -> None:
        self.assert_fixture("valid", "episode-tombstone.json", "valid")

    def test_unset_and_explicit_disabled_both_deny_adaptive_updates(self) -> None:
        self.assert_fixture("invalid", "learning-unset-update.json", "learning_disabled")
        self.assert_fixture("invalid", "learning-disabled-update.json", "learning_disabled")

    def test_quiet_override_denies_unsolicited_action(self) -> None:
        self.assert_fixture("degraded", "learning-quiet-override.json", "valid")

    def test_learning_message_cannot_carry_authority_or_platform_writes(self) -> None:
        self.assert_fixture("privacy", "learning-authority-payload.json", "learning_authority_forbidden")

    def test_mandatory_incident_still_requires_minimization_and_deletion(self) -> None:
        self.assert_fixture("valid", "episode-mandatory-incident.json", "valid")
        self.assert_fixture("invalid", "episode-mandatory-unbounded.json", "mandatory_controls_missing")


class VendoringTests(unittest.TestCase):
    SOURCE_REVISION = "a67d6b47b7ff1c658e40164cb2cf81cff583cb4f"
    AGGREGATE_DIGEST = "3ffd487bdc497b7ce54b8c29978a3686dcbffdb66a85957a0ee4f99ba576cdfd"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="abbey-vendor-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "vendored"
        shutil.copytree(CORPUS_ROOT, self.source)

    def write_destination(self, revision: str | None = None):
        return vendor(
            self.source,
            self.destination,
            revision or self.SOURCE_REVISION,
            check=False,
        )

    def test_write_vendors_only_exact_manifest_committed_bytes(self) -> None:
        report = self.write_destination()
        manifest = json.loads((self.source / "manifest.json").read_text(encoding="utf-8"))
        expected_files = {"manifest.json", *(row["path"] for row in manifest["artifacts"])}
        actual_files = {
            path.relative_to(self.destination / "corpus").as_posix()
            for path in (self.destination / "corpus").rglob("*")
            if path.is_file()
        }

        self.assertTrue(report.wrote)
        self.assertEqual(report.aggregate_digest, self.AGGREGATE_DIGEST)
        self.assertEqual(report.artifact_count, 113)
        self.assertEqual(actual_files, expected_files)
        for relative in expected_files:
            self.assertEqual(
                (self.destination / "corpus" / relative).read_bytes(),
                (self.source / relative).read_bytes(),
                relative,
            )

        lock = json.loads(
            (self.destination / "abbey-contracts.lock.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            lock,
            {
                "source_repository": "https://github.com/donaldfilimon/abi",
                "source_revision": self.SOURCE_REVISION,
                "contract_major": 2,
                "contract_revision": 2,
                "aggregate_digest": self.AGGREGATE_DIGEST,
            },
        )

    def test_documented_cli_executes_from_the_repository_root(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "vendor_abbey_contracts.py"),
                "--source",
                str(self.source),
                "--destination",
                str(self.destination),
                "--source-revision",
                self.SOURCE_REVISION,
                "--write",
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.destination / "abbey-contracts.lock.json").is_file())

    def test_check_is_read_only_for_bytes_and_mtimes(self) -> None:
        self.write_destination()
        before = {
            path.relative_to(self.destination).as_posix(): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in self.destination.rglob("*")
            if path.is_file()
        }

        report = vendor(self.source, self.destination, self.SOURCE_REVISION, check=True)

        after = {
            path.relative_to(self.destination).as_posix(): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in self.destination.rglob("*")
            if path.is_file()
        }
        self.assertFalse(report.wrote)
        self.assertEqual(after, before)

    def test_source_with_unmanifested_bytes_is_refused(self) -> None:
        (self.source / "unmanaged.json").write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(VendorError, "source_inventory_mismatch"):
            self.write_destination()

        self.assertFalse(self.destination.exists())

    def test_source_manifest_traversal_is_refused(self) -> None:
        manifest_path = self.source / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"][0]["path"] = "../escape.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(VendorError, "source_corpus_invalid"):
            self.write_destination()

        self.assertFalse((self.root / "escape.json").exists())

    def test_destination_symlink_is_refused(self) -> None:
        target = self.root / "symlink-target"
        target.mkdir()
        self.destination.symlink_to(target, target_is_directory=True)

        with self.assertRaisesRegex(VendorError, "destination_symlink"):
            self.write_destination()

    def test_nonempty_unmanaged_destination_is_refused(self) -> None:
        self.destination.mkdir()
        sentinel = self.destination / "consumer-owned.txt"
        sentinel.write_text("keep\n", encoding="utf-8")

        with self.assertRaisesRegex(VendorError, "unmanaged_destination"):
            self.write_destination()

        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep\n")

    def test_mutable_or_noncanonical_revision_is_refused(self) -> None:
        for revision in ("main", "a" * 39, "A" * 40, "g" * 40, "a" * 41):
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(VendorError, "source_revision_invalid"):
                    vendor(self.source, self.destination, revision, check=False)

    def test_check_rejects_an_extra_destination_file(self) -> None:
        self.write_destination()
        extra = self.destination / "corpus" / "consumer-extra.json"
        extra.write_text("{}\n", encoding="utf-8")

        with self.assertRaisesRegex(VendorError, "destination_inventory_mismatch"):
            vendor(self.source, self.destination, self.SOURCE_REVISION, check=True)

    def test_check_rejects_a_destination_byte_mismatch(self) -> None:
        self.write_destination()
        readme = self.destination / "corpus" / "README.md"
        raw = bytearray(readme.read_bytes())
        raw[0] ^= 1
        readme.write_bytes(raw)

        with self.assertRaisesRegex(VendorError, "destination_corpus_invalid"):
            vendor(self.source, self.destination, self.SOURCE_REVISION, check=True)

    def test_write_replaces_only_a_valid_managed_destination(self) -> None:
        self.write_destination()
        replacement_revision = "b" * 40

        report = self.write_destination(replacement_revision)

        lock = json.loads(
            (self.destination / "abbey-contracts.lock.json").read_text(encoding="utf-8")
        )
        self.assertTrue(report.wrote)
        self.assertEqual(lock["source_revision"], replacement_revision)

    def test_write_preserves_a_destination_with_a_mutated_lock(self) -> None:
        self.write_destination()
        lock_path = self.destination / "abbey-contracts.lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["unexpected"] = True
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(VendorError, "destination_lock_invalid"):
            self.write_destination("b" * 40)

        self.assertTrue(lock_path.exists())
        self.assertIn("unexpected", json.loads(lock_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
