"""Narrow source-linked documentation policy checks."""

from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]


class DurableStoreDocumentationPolicyTests(unittest.TestCase):
    def test_current_writer_lock_retry_budget_matches_wdbx_source(self) -> None:
        manifest = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
        durable_path = manifest["workspace"]["dependencies"]["abi-wdbx"]["path"]
        source = (ROOT / durable_path / "src/durable.rs").resolve().read_text(encoding="utf-8")
        source_match = re.search(
            r"WRITER_LOCK_RETRY_BUDGET[^=]*=\s*[^;]*from_millis\((\d+)\)", source
        )
        self.assertIsNotNone(source_match, "WDBX retry budget constant not found")

        current_docs = (ROOT / "tasks/todo.md").read_text(encoding="utf-8")
        docs_match = re.search(r"WouldBlock.*?\((\d+) ms budget,\s*\d+ ms steps\)", current_docs)
        self.assertIsNotNone(docs_match, "current retry-budget documentation not found")
        self.assertEqual(docs_match.group(1), source_match.group(1))

    def test_completion_persistence_claims_match_the_application_wrappers(self) -> None:
        util = (ROOT / "crates/abi-cli/src/util.rs").read_text(encoding="utf-8")
        completion = (ROOT / "crates/abi-cli/src/complete.rs").read_text(encoding="utf-8")
        agent = (ROOT / "crates/abi-cli/src/agent.rs").read_text(encoding="utf-8")
        top_level_train = (ROOT / "crates/abi-cli/src/train.rs").read_text(encoding="utf-8")
        mcp_ai_tools = (ROOT / "crates/abi-mcp/src/ai_tools.rs").read_text(encoding="utf-8")
        self.assertIn('format!("{home}/.abi/wdbx")', util)
        self.assertIn("let mut store = util::open_store();", completion)
        self.assertIn("let mut store = util::open_store();", agent)
        self.assertNotIn("open_store(", top_level_train)
        self.assertIn("pub fn run(input:", mcp_ai_tools)
        self.assertIn("pub fn run_train(", mcp_ai_tools)
        self.assertIn("pub fn run_learn(", mcp_ai_tools)
        self.assertGreaterEqual(mcp_ai_tools.count("open_wdbx_store()?"), 3)

        public_docs = (
            ROOT / "README.md",
            ROOT / "docs/contracts/external-claims-audit.mdx",
            ROOT / "docs/contracts/public-api.mdx",
            ROOT / "docs/spec/abbey-core-identity.mdx",
        )
        for path in public_docs:
            with self.subTest(path=path.relative_to(ROOT)):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("CompletionRequest.store_result", text)
                self.assertNotIn("CLI/MCP completion and training", text)
                self.assertNotIn("accepted=false", text)
                self.assertNotIn("empty input before touching WDBX", text)
                self.assertIn("ABI_WDBX_PERSIST=0", text)
                self.assertIn("ABI_WDBX_PATH=:memory:", text)
                self.assertIn("abi complete", text)
                self.assertIn("abi agent train", text)
                self.assertIn("top-level `abi train`", text)
                self.assertIn("ai_complete", text)
                self.assertIn("ai_train", text)
                self.assertIn("ai_learn", text)


if __name__ == "__main__":
    unittest.main()
