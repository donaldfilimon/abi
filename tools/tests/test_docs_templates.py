"""Regression guards for documentation renderers used by hosted Pages."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


class DocumentationTemplateTests(unittest.TestCase):
    def test_docs_do_not_embed_github_expression_openers(self) -> None:
        offenders: list[str] = []
        for path in sorted(DOCS.rglob("*")):
            if path.suffix not in {".md", ".mdx"}:
                continue
            text = path.read_text(encoding="utf-8")
            if "${{" in text:
                offenders.append(path.relative_to(ROOT).as_posix())

        self.assertEqual(
            offenders,
            [],
            "literal GitHub expression openers are parsed as unterminated Liquid "
            "variables by GitHub Pages; construct or escape the braces instead",
        )


if __name__ == "__main__":
    unittest.main()
