from pathlib import Path
import unittest

from tools.ci_contract import validate_workflow


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


class PublicWdbxWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_repository_workflow_is_safe_and_credential_free(self) -> None:
        self.assertEqual(validate_workflow(self.workflow), ())

    def test_validator_catches_a_secret_checkout_regression(self) -> None:
        mutated = self.workflow.replace(
            "          path: wdbx",
            "          token: ${{ secrets.WDBX_CHECKOUT_TOKEN }}\n          path: wdbx",
            1,
        )
        self.assertIn("wdbx checkout must not use a secret", validate_workflow(mutated))

    def test_validator_catches_self_hosted_fork_exposure(self) -> None:
        mutated = self.workflow.replace(
            "github.event.pull_request.head.repo.full_name == github.repository",
            "github.event.pull_request.head.repo.full_name != github.repository",
            1,
        )
        self.assertIn(
            "trusted self-hosted job must require a same-repository pull request",
            validate_workflow(mutated),
        )

    def test_validator_catches_loss_of_the_hosted_fork_path(self) -> None:
        mutated = self.workflow.replace("runs-on: macos-latest", "runs-on: [self-hosted]", 1)
        self.assertIn(
            "fork pull requests must run on a GitHub-hosted runner",
            validate_workflow(mutated),
        )

    def test_validator_catches_a_mutable_or_stale_wdbx_revision(self) -> None:
        mutated = self.workflow.replace(
            "8ceca077e1d888c2955a0aa52bcbb278c01967a5",
            "main",
            1,
        )
        self.assertIn("WDBX revision must be the reviewed immutable commit", validate_workflow(mutated))


if __name__ == "__main__":
    unittest.main()
