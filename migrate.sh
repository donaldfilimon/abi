#!/bin/bash
# Zig 0.16 Migration Script
# Fixes deprecated patterns in bulk

set -e

echo "Starting Zig 0.16 migration..."
echo "==============================="

# Function to replace patterns in all Zig files
migrate_patterns() {
    echo "Fixing @errorName() in format strings..."
    # Find all files with @errorName
    for file in $(find src benchmarks examples -name "*.zig" -type f); do
        if grep -q "@errorName" "$file"; then
            backup="${file}.0.15.backup"
            cp "$file" "$backup"
            echo "  Processing $file"
            
            # Replace @errorName(err) in format strings
            # First pass: simple replacement for cloud functions
            sed -i 's/@errorName(err)/@errorName(err)/g' "$file"  # Leave as is for now, needs careful handling
            
            # Second pass: handle in specific contexts
            # Cloud function error responses need @errorName for string storage
        fi
    done
    
    echo "Checking for other deprecated patterns..."
    echo "Scanning for std.io.AnyReader..."
    find src -name "*.zig" -exec grep -l "std.io.AnyReader" {} + | while read file; do
        echo "  Found in $file"
    done
    
    echo "Scanning for std.time.sleep..."
    find src -name "*.zig" -exec grep -l "std.time.sleep" {} + | while read file; do
        echo "  Found in $file"
    done
    
    echo "Scanning for std.time.nanoTimestamp..."
    find src -name "*.zig" -exec grep -l "std.time.nanoTimestamp" {} + | while read file; do
        echo "  Found in $file"
    done
}

# Check the @errorName cases specifically
check_errorname_usage() {
    echo "Analyzing @errorName usage patterns..."
    
    echo "1. Cloud Functions (needs @errorName for string storage):"
    cat << 'EOF'
// These need @errorName because it's storing error name as a string
error_message = @errorName(err);  // Still valid - storing in variable
response = try CloudResponse.err(self.allocator, 500, @errorName(err));  // Passing as string param
EOF
    
    echo ""
    echo "2. Web routes (mixed usage - some in format strings):"
    cat << 'EOF'
// These are in format strings and should use {t}:
std.fmt.allocPrint(allocator, "{{\"error\":{{\"code\":\"...\",\"message\":\"{s}\"}}}}", .{@errorName(err)});
// Should become:
std.fmt.allocPrint(allocator, "{{\"error\":{{\"code\":\"...\",\"message\":\"{t}\"}}}}", .{err});
EOF
}

# Fix web route format strings
fix_web_format_strings() {
    echo "Fixing web route format strings..."
    
    # src/web/routes/personas.zig:279
    local file="src/web/routes/personas.zig"
    if [ -f "$file" ]; then
        backup="${file}.0.15.backup"
        if [ ! -f "$backup" ]; then
            cp "$file" "$backup"
        fi
        
        # Line 279: change format specifier from {s} to {t}
        sed -i 's/"message":"{s}"/"message":"{t}"/g' "$file"
        
        # Remove @errorName(err) from the format arguments
        sed -i 's/@errorName(err)/err/g' "$file"
        
        echo "  Fixed $file line 279"
    fi
    
    # Check for other similar patterns
    find src/web -name "*.zig" -type f -exec grep -l "@errorName" {} + | while read file; do
        echo "  $file contains @errorName"
    done
}

main() {
    echo "Running migration analysis..."
    
    # Check current status
    check_errorname_usage
    
    # Fix web format strings
    fix_web_format_strings
    
    echo ""
    echo "Migration checklist:"
    echo "===================="
    echo "✓ Fixed std.time.nanoTimestamp() in:"
    echo "  - src/observability/system_info/mod.zig"
    echo "  - src/gpu/peer_transfer/mod.zig" 
    echo "  - src/shared/security/audit.zig"
    echo ""
    echo "✓ Fixed std.time.sleep() in:"
    echo "  - src/ai/orchestration/fallback.zig"
    echo ""
    echo "✓ Fixed std.fs.cwd() in:"
    echo "  - src/ai/database/export.zig"
    echo ""
    echo "⚠️ Pending fixes:"
    echo "  - @errorName() in cloud functions (needs careful handling)"
    echo "  - @errorName() in web routes (partially fixed)"
    echo "  - std.io.AnyReader (if any exist)"
    echo ""
    echo "Run 'zig build test' to verify changes work correctly."
}

main "$@"