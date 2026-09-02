"""Dependency-free checks for the exact GitHub Pages artifact and workflow."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re
import unittest

from tools.ci_contract import validate_checkout_credentials


ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
WORKFLOW = ROOT / ".github/workflows/benchmarks-gh-pages.yml"


class _PageReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.references: list[tuple[str, str, str, dict[str, str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        names = [name for name, _ in attrs]
        if len(names) != len(set(names)):
            raise AssertionError(f"duplicate attributes are forbidden on <{tag}>")
        fields = {name: value or "" for name, value in attrs}
        if fields.get("id"):
            self.ids.add(fields["id"])
        for attribute in ("href", "src"):
            if fields.get(attribute):
                self.references.append((tag, attribute, fields[attribute], fields))
        if fields.get("srcset"):
            if re.search(r"(?:^|,\s*)data:", fields["srcset"], re.IGNORECASE):
                raise AssertionError("data URLs are forbidden in srcset")
            for candidate in fields["srcset"].split(","):
                target = candidate.strip().split(maxsplit=1)[0]
                if target:
                    self.references.append((tag, "srcset", target, fields))


def _parse_page(path: Path) -> _PageReferences:
    parser = _PageReferences()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser


def _resolve_local_reference(page: Path, target: str) -> tuple[Path, str]:
    """Resolve one local URL while proving it stays inside the Pages artifact."""

    url = urlsplit(target)
    if url.scheme:
        raise AssertionError(f"absolute URL cannot satisfy a local reference: {target}")
    if not url.scheme and (url.netloc or target.startswith("//")):
        raise AssertionError(f"protocol-relative URL is forbidden: {target}")
    local_path = unquote(url.path)
    if "\\" in local_path:
        raise AssertionError(f"backslash is forbidden in a published URL: {target}")
    if page.name == "404.html" and local_path.startswith("/abi/"):
        local_path = local_path.removeprefix("/abi/")
        base = SITE
    elif local_path.startswith("/"):
        raise AssertionError(f"normal page asset/link must be relative: {page.name}: {target}")
    else:
        base = page.parent
    resolved = (base / local_path).resolve()
    if not resolved.is_relative_to(SITE.resolve()):
        raise AssertionError(f"local reference escapes the site root: {page.name}: {target}")
    return resolved, url.fragment


def _reference_is_local(
    tag: str,
    attribute: str,
    target: str,
    attrs: dict[str, str],
) -> bool:
    """Classify a reference, rejecting origins that can redirect artifact loads."""

    if tag == "base" and attribute == "href":
        raise AssertionError("base href is forbidden in the published artifact")
    url = urlsplit(target)
    if not url.scheme and (url.netloc or target.startswith("//")):
        raise AssertionError(f"protocol-relative URL is forbidden: {target}")
    if url.scheme.lower() == "data":
        if attribute == "srcset":
            raise AssertionError("data URLs are forbidden in srcset")
        return False
    if not url.scheme:
        return True

    rel = set(attrs.get("rel", "").lower().split())
    local_asset = attribute in {"src", "srcset"} or (
        tag == "link" and bool(rel & {"stylesheet", "icon", "preload", "modulepreload"})
    )
    if local_asset:
        raise AssertionError(f"published asset must be local: {target}")
    return False


class PagesContractTests(unittest.TestCase):
    def test_artifact_has_nojekyll_pages_and_all_referenced_local_files(self) -> None:
        self.assertTrue((SITE / ".nojekyll").is_file())
        pages = sorted(SITE.rglob("*.html"))
        self.assertEqual([page.name for page in pages], ["404.html", "benchmarks.html", "index.html"])
        self.assertFalse(any(path.is_symlink() for path in SITE.rglob("*")))

        parsed = {page.resolve(): _parse_page(page) for page in pages}
        for page in pages:
            for tag, attribute, target, attrs in parsed[page.resolve()].references:
                try:
                    is_local = _reference_is_local(tag, attribute, target, attrs)
                except AssertionError as exc:
                    self.fail(f"{page.name}: {exc}")
                if not is_local:
                    continue
                url = urlsplit(target)
                if target.startswith("#"):
                    self.assertIn(
                        url.fragment,
                        parsed[page.resolve()].ids,
                        f"missing anchor in {page.name}: {target}",
                    )
                    continue
                resolved, fragment = _resolve_local_reference(page, target)
                if unquote(url.path).endswith("/") or not resolved.suffix:
                    resolved /= "index.html"
                self.assertTrue(resolved.exists(), f"missing local target in {page.name}: {target}")
                if fragment and resolved.suffix == ".html":
                    target_parser = parsed.get(resolved.resolve()) or _parse_page(resolved)
                    self.assertIn(fragment, target_parser.ids, f"missing target anchor: {target}")

        benchmark = (SITE / "benchmarks.html").read_text(encoding="utf-8")
        for target in re.findall(r'(?:from|fetch\()\s*["\']([^"\']+)', benchmark):
            self.assertFalse(target.startswith(("http://", "https://", "//")))
            resolved, _ = _resolve_local_reference(SITE / "benchmarks.html", target)
            self.assertTrue(resolved.exists(), target)

    def test_local_reference_resolution_rejects_escape_external_and_backslash_urls(self) -> None:
        page = SITE / "index.html"
        for target in (
            "https://example.invalid/asset.js",
            "//example.invalid/asset.js",
            "../Cargo.toml",
            "%2e%2e/Cargo.toml",
            "..\\Cargo.toml",
            "%2e%2e%5cCargo.toml",
            "\\\\example.invalid\\asset.js",
        ):
            with self.subTest(target=target), self.assertRaises(AssertionError):
                _resolve_local_reference(page, target)

    def test_local_reference_resolution_accepts_nested_and_root_404_assets(self) -> None:
        nested, _ = _resolve_local_reference(SITE / "index.html", "assets/styles.css")
        rooted, _ = _resolve_local_reference(SITE / "404.html", "/abi/assets/styles.css")
        self.assertEqual(nested, rooted)
        self.assertTrue(nested.is_relative_to(SITE.resolve()))

    def test_nested_page_references_resolve_from_the_containing_directory(self) -> None:
        nested_page = SITE / "nested" / "page.html"
        resolved, _ = _resolve_local_reference(nested_page, "../assets/styles.css")
        self.assertEqual(resolved, (SITE / "assets/styles.css").resolve())

    def test_base_and_external_artifact_references_are_rejected(self) -> None:
        for markup in (
            '<base href="https://example.invalid/">',
            '<base href="assets/">',
            '<img src="https://example.invalid/pixel.png">',
            '<link rel="stylesheet" href="https://example.invalid/site.css">',
            '<source srcset="assets/pixel.png 1x, https://example.invalid/pixel.png 2x">',
        ):
            with self.subTest(markup=markup):
                parser = _PageReferences()
                parser.feed(markup)
                rejected = []
                for tag, attribute, target, attrs in parser.references:
                    try:
                        _reference_is_local(tag, attribute, target, attrs)
                    except AssertionError as exc:
                        rejected.append(str(exc))
                self.assertTrue(rejected, markup)

    def test_inline_src_is_allowed_but_data_srcset_is_rejected(self) -> None:
        parser = _PageReferences()
        parser.feed('<img src="data:image/svg+xml,%3Csvg%3E">')
        tag, attribute, target, attrs = parser.references[0]
        self.assertFalse(_reference_is_local(tag, attribute, target, attrs))

        with self.assertRaisesRegex(AssertionError, "data URLs are forbidden in srcset"):
            _PageReferences().feed(
                '<source srcset="data:image/svg+xml,%3Csvg%3E 1x, assets/pixel.png 2x">'
            )

    def test_duplicate_attributes_cannot_hide_the_browser_selected_reference(self) -> None:
        with self.assertRaisesRegex(AssertionError, "duplicate attributes are forbidden"):
            _PageReferences().feed(
                '<script src="https://example.invalid/app.js" src="assets/app.js"></script>'
            )

    def test_only_404_uses_the_abi_absolute_path_exception(self) -> None:
        for page_name in ("index.html", "benchmarks.html"):
            for _, _, target, _ in _parse_page(SITE / page_name).references:
                self.assertFalse(target.startswith("/abi/"), f"{page_name}: {target}")
        references = [target for _, _, target, _ in _parse_page(SITE / "404.html").references]
        self.assertTrue(any(target.startswith("/abi/") for target in references))
        self.assertTrue(all(not target.startswith("/") or target.startswith("/abi/") for target in references))

    def test_pages_load_no_external_runtime_scripts_or_styles(self) -> None:
        for page in SITE.rglob("*.html"):
            for tag, _, target, attrs in _parse_page(page).references:
                runtime = tag == "script" or (
                    tag == "link" and "stylesheet" in attrs.get("rel", "").lower().split()
                )
                if runtime:
                    url = urlsplit(target)
                    self.assertFalse(url.scheme or url.netloc, f"{page.name}: {target}")
                    _resolve_local_reference(page, target)

    def test_workflow_has_exact_pages_permissions_and_artifact_contract(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        permissions = workflow.split("permissions:\n", 1)[1].split("\n\n", 1)[0]
        self.assertEqual(permissions, "  contents: read\n  pages: write\n  id-token: write")
        upload = workflow.split("uses: actions/upload-pages-artifact@", 1)[1].split("\n\n", 1)[0]
        self.assertIn("path: ./site\n          include-hidden-files: true", upload)

        actions = re.findall(r"uses: ([^@\s]+)@([^\s]+)", workflow)
        self.assertEqual(
            [action for action, _ in actions],
            ["actions/checkout", "actions/configure-pages", "actions/upload-pages-artifact", "actions/deploy-pages"],
        )
        for action, revision in actions:
            self.assertRegex(revision, r"^[0-9a-f]{40}$", action)
        self.assertEqual(validate_checkout_credentials(workflow), ())
        self.assertNotIn("${{ secrets.", workflow)
        self.assertNotRegex(workflow, r"(?m)^\s*(?:git\s+push|branch:\s*gh-pages|target_branch:)")


if __name__ == "__main__":
    unittest.main()
