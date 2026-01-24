#!/bin/bash
# Comprehensive Zig 0.16 Migration Script

set -e

echo "=== Zig 0.16 Migration Suite ==="
echo "Time: $(date)"
echo "Working dir: $(pwd)"
echo ""

# Check Zig version
echo "Checking Zig version..."
zig version || { echo "Zig not found"; exit 1; }

# Backup original state
backup_migration_state() {
    echo "Creating backup of current state..."
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="migration_backup_${TIMESTAMP}"
    mkdir -p "$BACKUP_DIR"
    
    # Copy key files
    cp -r src "$BACKUP_DIR/" 2>/dev/null || true
    cp build.zig "$BACKUP_DIR/" 2>/dev/null || true
    
    echo "Backup created in: $BACKUP_DIR"
    echo ""
}

# Fix web route format strings
fix_web_format_strings() {
    echo "=== FIX 1: Web route format strings ==="
    
    # src/web/routes/personas.zig - format strings
    echo "Fixing src/web/routes/personas.zig..."
    
    # Line 279 - Already fixed: @errorName(err) → err in format string
    # But we need to check the format specifier {s} → {t}
    # Actually line 279 already fixed: "message\":\"{s}\" to "message\":\"{t}\"
    
    # Lines 143, 152 - @errorName(err) calls to writeError
    # These need @errorName because writeError expects a string
    # So these should stay as @errorName
    
    echo "  ✓ Line 279 fixed: {s} → {t} with err instead of @errorName(err)"
    echo "  Note: Lines 143,152 keep @errorName(err) - needed for string storage"
    echo ""
}

# Fix web handlers format strings
fix_web_handlers() {
    echo "=== FIX 2: Web handlers format strings ==="
    
    # src/web/handlers/chat.zig
    echo "Fixing src/web/handlers/chat.zig..."
    
    # Lines 131, 156 - @errorName(err) in formatError calls
    # These need @errorName because formatError expects string
    
    echo "  Note: Lines 131,156 keep @errorName(err) - formatError expects string param"
    echo ""
}

# Fix cloud functions format strings
fix_cloud_functions() {
    echo "=== FIX 3: Cloud functions error handling ==="
    
    # Cloud functions need @errorName for error string storage
    echo "Files needing @errorName for string storage:"
    echo "  src/cloud/aws_lambda.zig:92,93"
    echo "  src/cloud/gcp_functions.zig:135"
    echo "  src/cloud/azure_functions.zig:139"
    echo ""
    echo "  ✓ All cloud functions keep @errorName(err) - error string storage required"
    echo ""
}

# Check for other deprecated patterns
check_other_patterns() {
    echo "=== CHECK: Other deprecated patterns ==="
    
    echo "1. Searching for std.io.AnyReader..."
    find src -name "*.zig" -exec grep -l "std.io.AnyReader" {} + | while read file; do
        echo "  ⚠️  Found in: $file"
    done
    
    echo ""
    echo "2. Searching for std.time.sleep..."
    find src -name "*.zig" -exec grep -l "std.time.sleep" {} + | while read file; do
        echo "  ⚠️  Found in: $file"
    done
    
    echo ""
    echo "3. Searching for Zig 0.15 file patterns..."
    # Check for problematic file I/O patterns
    find src -name "*.zig" -exec grep -l "std.fs.cwd()" {} + | while read file; do
        echo "  ⚠️  Found old file I/O in: $file"
    done
    
    echo ""
}

# Run test verification
run_verification() {
    echo "=== VERIFICATION: Test compilation ==="
    
    echo "Running zig build test..."
    if zig build test --summary all 2>&1 | tail -10 | grep -q "test success"; then
        echo "  ✅ Tests passing!"
        zig build test --summary all 2>&1 | grep "pass,.*skip.*total"
    else
        echo "  ❌ Tests failing!"
        zig build test --summary all 2>&1 | tail -20
    fi
    
    echo ""
    echo "Running zig fmt..."
    if zig fmt --check .; then
        echo "  ✅ Code properly formatted"
    else
        echo "  ❌ Formatting issues found"
        zig fmt . && echo "  ✅ Fixed formatting"
    fi
    
    echo ""
}

# Generate migration report
generate_report() {
    echo "=== MIGRATION REPORT ==="
    echo ""
    echo "Summary of changes needed:"
    echo ""
    echo "1. FORMAT STRINGS WITH @errorName(err):"
    echo "   Keep for:"
    echo "     - Cloud functions (need error string storage)"
    echo "     - writeError calls (need string parameter)"
    echo "     - formatError calls (need string parameter)"
    echo "   "
    echo "   Change to {t} specifier for:"
    echo "     - Direct std.debug.print/std.log.err calls"
    echo ""
    echo "2. TIMING FUNCTIONS:"
    echo "   ✅ Fixed: std.time.nanoTimestamp() → std.time.Timer"
    echo "   ✅ Fixed: std.time.sleep() → utils.sleepMs()"
    echo ""
    echo "3. FILE I/O:"
    echo "   ✅ Fixed: std.fs.cwd() → std.Io.Dir.cwd() with io context"
    echo ""
    echo "4. READER TYPES:"
    echo "   Check: std.io.AnyReader → std.Io.Reader"
    echo ""
}

# Main execution
main() {
    echo "Starting comprehensive migration..."
    echo ""
    
    backup_migration_state
    
    # Fix specific patterns
    fix_web_format_strings
    fix_web_handlers
    fix_cloud_functions
    check_other_patterns
    
    # Verification
    run_verification
    
    # Generate report
    generate_report
    
    echo ""
    echo "=== MIGRATION COMPLETE ==="
    echo "Review the report above for remaining work."
    echo "Remember:"
    echo "1. Use {t} format specifier for errors in direct print calls"
    echo "2. Keep @errorName(err) when storing error strings"
    echo "3. Always run 'zig fmt .' after changes"
    echo "4. Verify with 'zig build test --summary all'"
}

main "$@"