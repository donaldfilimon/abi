#!/bin/bash
# Documentation Generation Verification Script

set -e

echo "🔍 Verifying documentation generation system..."
echo ""

# Check if Zig is available
if ! command -v zig &> /dev/null; then
    echo "❌ Error: Zig compiler not found"
    echo "   Please install Zig $(cat .zigversion) to continue"
    exit 1
fi

echo "✅ Zig compiler found: $(zig version)"
echo ""

# Clean old docs
echo "🧹 Cleaning old documentation..."
rm -rf docs/generated docs/assets docs/_layouts docs/_data docs/.nojekyll \
       docs/_config.yml docs/index.html docs/sitemap.xml docs/robots.txt \
       docs/README.md docs/zig-docs .github/workflows/deploy_docs.yml

echo "✅ Old documentation removed"
echo ""

# Build documentation generator
echo "🔨 Building documentation generator..."
zig build docs

echo ""
echo "✅ Documentation generation complete!"
echo ""

# Verify generated files exist
echo "🔍 Verifying generated files..."

EXPECTED_FILES=(
    "docs/.nojekyll"
    "docs/_config.yml"
    "docs/_layouts/documentation.html"
    "docs/_data/navigation.yml"
    "docs/sitemap.xml"
    "docs/robots.txt"
    "docs/generated/MODULE_REFERENCE.md"
    "docs/generated/API_REFERENCE.md"
    "docs/generated/EXAMPLES.md"
    "docs/generated/PERFORMANCE_GUIDE.md"
    "docs/generated/DEFINITIONS_REFERENCE.md"
    "docs/generated/CODE_API_INDEX.md"
    "docs/generated/search_index.json"
    "docs/assets/css/documentation.css"
    "docs/assets/js/documentation.js"
    "docs/assets/js/search.js"
    "docs/index.html"
    "docs/README.md"
    ".github/workflows/deploy_docs.yml"
    "docs/zig-docs/index.html"
)

MISSING_FILES=()
for file in "${EXPECTED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
        echo "  ❌ Missing: $file"
    else
        echo "  ✅ Found: $file"
    fi
done

echo ""

if [ ${#MISSING_FILES[@]} -eq 0 ]; then
    echo "✅ All expected documentation files were generated!"
    echo ""
    echo "📝 Generated documentation structure:"
    tree -L 3 docs/ 2>/dev/null || find docs/ -type f | sort
    echo ""
    echo "🚀 Documentation is ready for deployment!"
    echo "   To preview locally, run: python3 -m http.server --directory docs 8000"
    exit 0
else
    echo "❌ Documentation generation incomplete!"
    echo "   ${#MISSING_FILES[@]} file(s) missing"
    exit 1
fi
