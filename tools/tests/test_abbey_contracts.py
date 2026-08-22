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


if __name__ == "__main__":
    unittest.main()
