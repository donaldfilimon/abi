from pathlib import Path
import re
import tempfile
import tomllib
import unittest

from tools.ci_contract import (
    sibling_dependency_requirements,
    validate_checkout_credentials,
    validate_workflow,
)


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
MANIFEST = ROOT / "Cargo.toml"


def _workflow_files(root: Path) -> list[Path]:
    return sorted((*root.glob("*.yml"), *root.glob("*.yaml")))


class PublicWdbxWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")
        self.manifest = MANIFEST.read_text(encoding="utf-8")

    def validate(self, workflow: str | None = None, manifest: str | None = None) -> tuple[str, ...]:
        return validate_workflow(workflow or self.workflow, manifest or self.manifest)

    def test_repository_workflow_is_safe_and_credential_free(self) -> None:
        self.assertEqual(self.validate(), ())

    def test_self_hosted_toolchain_install_uses_one_component_argument(self) -> None:
        command = re.search(
            r"rustup toolchain install nightly-2026-09-01[^\n]+",
            self.workflow,
        )
        self.assertIsNotNone(command)
        self.assertEqual(
            command.group(0).strip(),
            (
                "rustup toolchain install nightly-2026-09-01 --profile minimal "
                "--component rustfmt,clippy,rust-src"
            ),
        )

    def test_every_repository_workflow_disables_checkout_credentials(self) -> None:
        workflows = _workflow_files(ROOT / ".github" / "workflows")
        self.assertTrue(workflows)
        for workflow in workflows:
            with self.subTest(workflow=workflow.name):
                self.assertEqual(
                    validate_checkout_credentials(workflow.read_text(encoding="utf-8")),
                    (),
                )

    def test_repository_workflow_discovery_includes_both_yaml_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.yml").write_text("name: one\n", encoding="utf-8")
            (root / "two.yaml").write_text("name: two\n", encoding="utf-8")
            (root / "ignored.txt").write_text("name: ignored\n", encoding="utf-8")
            self.assertEqual(
                [path.name for path in _workflow_files(root)],
                ["one.yml", "two.yaml"],
            )

    def test_sibling_requirements_come_from_the_workspace_manifest(self) -> None:
        requirements = sibling_dependency_requirements(self.manifest)
        expected = {
            spec["path"]
            for spec in tomllib.loads(self.manifest)["workspace"]["dependencies"].values()
            if isinstance(spec, dict) and spec.get("path", "").startswith("../wdbx/")
        }
        self.assertEqual(set(requirements), {"wdbx"})
        self.assertEqual(set(requirements["wdbx"]), expected)

    def test_validator_catches_a_secret_checkout_regression(self) -> None:
        mutated = self.workflow.replace(
            "          path: wdbx",
            "          token: ${{ secrets.WDBX_CHECKOUT_TOKEN }}\n          path: wdbx",
            1,
        )
        self.assertIn("wdbx checkout must not use a secret", self.validate(mutated))

    def test_validator_catches_self_hosted_fork_exposure(self) -> None:
        mutated = self.workflow.replace(
            "github.event.pull_request.head.repo.full_name == github.repository",
            "github.event.pull_request.head.repo.full_name != github.repository",
            1,
        )
        self.assertIn(
            "trusted self-hosted job must require a same-repository pull request",
            self.validate(mutated),
        )

    def test_validator_catches_loss_of_the_hosted_fork_path(self) -> None:
        mutated = self.workflow.replace("runs-on: macos-latest", "runs-on: [self-hosted]", 1)
        self.assertIn(
            "fork pull requests must run on a GitHub-hosted runner",
            self.validate(mutated),
        )

    def test_validator_catches_a_mutable_or_stale_wdbx_revision(self) -> None:
        mutated, count = re.subn(
            r"(?m)^(  WDBX_REVISION:) [0-9a-f]{40}$", r"\1 main", self.workflow, count=1
        )
        self.assertEqual(count, 1)
        self.assertIn("wdbx revision must be an immutable commit", self.validate(mutated))

    def test_validator_catches_a_missing_manifest_required_sibling_checkout(self) -> None:
        mutated_manifest = self.manifest.replace(
            'abi-wdbx = { path = "../wdbx/crates/abi-wdbx" }',
            'abi-wdbx = { path = "../substrate/crates/abi-wdbx" }',
        )
        self.assertIn(
            "every ABI CI job must check out the required substrate repository once",
            self.validate(manifest=mutated_manifest),
        )

    def test_validator_catches_a_wrong_sibling_checkout_path(self) -> None:
        mutated = self.workflow.replace("          path: wdbx", "          path: vendor/wdbx", 1)
        self.assertIn("check must place wdbx at the sibling path", self.validate(mutated))

    def test_validator_catches_persisted_checkout_credentials(self) -> None:
        mutated = self.workflow.replace(
            "          persist-credentials: false",
            "          persist-credentials: true",
            1,
        )
        self.assertIn("every checkout must disable persisted credentials", self.validate(mutated))

    def test_validator_catches_a_checkout_with_no_with_mapping(self) -> None:
        mutated = self.workflow.replace(
            "        with:\n          path: abi\n          persist-credentials: false\n",
            "",
            1,
        )
        self.assertIn("every checkout must disable persisted credentials", self.validate(mutated))

    def test_validator_catches_a_checkout_with_no_persist_credentials_key(self) -> None:
        mutated = self.workflow.replace("          persist-credentials: false\n", "", 1)
        self.assertIn("every checkout must disable persisted credentials", self.validate(mutated))

    def test_validator_accepts_quoted_false(self) -> None:
        for value in ('"false"', "'false'"):
            with self.subTest(value=value):
                mutated = self.workflow.replace(
                    "          persist-credentials: false",
                    f"          persist-credentials: {value}",
                )
                self.assertEqual(self.validate(mutated), ())

    def test_unrelated_nested_key_cannot_satisfy_a_checkout(self) -> None:
        mutated = self.workflow.replace("          persist-credentials: false\n", "", 1)
        mutated = mutated.replace(
            "    steps:\n",
            "    env:\n      persist-credentials: false\n    steps:\n",
            1,
        )
        self.assertIn("every checkout must disable persisted credentials", self.validate(mutated))

    def test_every_checkout_step_is_checked(self) -> None:
        self.assertGreater(self.workflow.count("uses: actions/checkout@"), 1)
        mutated = self.workflow.replace(
            "          persist-credentials: false",
            "          persist-credentials: true",
            self.workflow.count("uses: actions/checkout@"),
        )
        self.assertIn("every checkout must disable persisted credentials", self.validate(mutated))

    def test_quoted_or_case_varied_checkout_cannot_evade_the_policy(self) -> None:
        action = re.search(r"uses: (actions/checkout@[^\s]+)", self.workflow)
        self.assertIsNotNone(action)
        checkout = action.group(1)
        values = (
            f'"{checkout}"',
            f"'{checkout}'",
            checkout.replace("actions/checkout", "Actions/Checkout"),
        )
        for value in values:
            with self.subTest(value=value):
                mutated = self.workflow.replace(
                    "    steps:\n",
                    f"    steps:\n      - uses: {value}\n",
                    1,
                )
                expected = ("every checkout must disable persisted credentials",)
                self.assertEqual(validate_checkout_credentials(mutated), expected)
                self.assertIn(expected[0], self.validate(mutated))

    def test_folded_flow_and_anchored_checkout_forms_fail_closed(self) -> None:
        action = re.search(r"uses: (actions/checkout@[^\s]+)", self.workflow)
        self.assertIsNotNone(action)
        checkout = action.group(1)
        steps = (
            f"      - uses: >-\n          {checkout}\n",
            f"      - {{uses: {checkout}}}\n",
            f"      - uses: &checkout {checkout}\n",
            f"      -\n        uses: {checkout}\n",
        )
        expected = ("every checkout must disable persisted credentials",)
        for step in steps:
            with self.subTest(step=step):
                mutated = self.workflow.replace("    steps:\n", f"    steps:\n{step}", 1)
                self.assertEqual(validate_checkout_credentials(mutated), expected)
                self.assertIn(expected[0], self.validate(mutated))

    def test_flow_sequence_checkout_cannot_evade_step_enumeration(self) -> None:
        action = re.search(r"uses: (actions/checkout@[^\s]+)", self.workflow)
        self.assertIsNotNone(action)
        workflow = f"jobs:\n  check:\n    steps: [{{uses: {action.group(1)}}}]\n"
        self.assertEqual(
            validate_checkout_credentials(workflow),
            ("every checkout must disable persisted credentials",),
        )

    def test_yaml_aliases_and_escaped_uses_scalars_fail_closed(self) -> None:
        expected = ("every checkout must disable persisted credentials",)
        workflows = (
            "jobs:\n  check:\n    steps:\n      - uses: *checkout\n",
            (
                "env:\n  ACTION: &1 actions/checkout@deadbeef\n"
                "jobs:\n  check:\n    steps:\n      - uses: *1\n"
            ),
            'jobs:\n  check:\n    steps:\n      - uses: "actions/check\\u006fut@deadbeef"\n',
            (
                'jobs:\n  check:\n    steps:\n      - "\\u0075ses": '
                '"actions/check\\u006fut@deadbeef"\n'
            ),
        )
        for workflow in workflows:
            with self.subTest(workflow=workflow):
                self.assertEqual(validate_checkout_credentials(workflow), expected)

    def test_a_workflow_with_no_checkout_is_vacuously_credential_safe(self) -> None:
        workflow = """name: docs
jobs:
  build:
    steps:
      - run: echo ok
"""
        self.assertEqual(validate_checkout_credentials(workflow), ())

    def test_repository_policy_rejects_a_missing_with_block_outside_ci(self) -> None:
        dependency_scan = (
            ROOT / ".github" / "workflows" / "dependency-scan.yml"
        ).read_text(encoding="utf-8")
        mutated = dependency_scan.replace(
            "        with:\n          persist-credentials: false\n",
            "",
            1,
        )
        self.assertEqual(
            validate_checkout_credentials(mutated),
            ("every checkout must disable persisted credentials",),
        )


if __name__ == "__main__":
    unittest.main()
