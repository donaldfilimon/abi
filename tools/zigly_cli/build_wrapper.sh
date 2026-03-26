#!/bin/bash
set -euo pipefail

ZIG2="$(readlink -f "$HOME/.local/bin/zig")"
if [ -z "$ZIG2" ]; then ZIG2="$HOME/.local/bin/zig"; fi

SYSROOT="$(xcrun --show-sdk-path 2>/dev/null || echo "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")"
MACOS_VER="$(sw_vers -productVersion 2>/dev/null || echo "26.4")"
ZIG_LIB="$(dirname "$(dirname "$ZIG2")")/lib"

echo "ZIG2 is: $ZIG2"
echo "ZIG_LIB is: $ZIG_LIB"

# Compile build.zig to object
"$ZIG2" build -Dtarget=aarch64-macos-gnu 2>build_err.log || true

OBJ="$(ls -t .zig-cache/o/*/build_zcu.o 2>/dev/null | head -1)"
if [ -n "$OBJ" ] && [ -f "$OBJ" ]; then
    BUILD_BIN="${OBJ%_zcu.o}"
    CRT="$(ls -t "$HOME/.cache/zig/o/"*/libcompiler_rt.a 2>/dev/null | head -1)"
    
    echo "Relinking build runner..."
    /usr/bin/ld -dynamic -platform_version macos "$MACOS_VER" "$MACOS_VER" \
        -syslibroot "$SYSROOT" -e _main -o "$BUILD_BIN" "$OBJ" "$CRT" "$SYSROOT/usr/lib/libSystem.B.tbd" "$SYSROOT/usr/lib/libc++.tbd"
    
    if [ -x "$BUILD_BIN" ]; then
        echo "Running relinked build runner..."
        "$BUILD_BIN" "$ZIG2" "$ZIG_LIB" "$(pwd)" ".zig-cache" "$HOME/.cache/zig" 2>target_err.log || true
        
        # Now relink the actual zigly_cli binary
        TARGET_OBJ="$(ls -t .zig-cache/o/*/zigly_zcu.o 2>/dev/null | head -1)"
        if [ -n "$TARGET_OBJ" ] && [ -f "$TARGET_OBJ" ]; then
            TARGET_BIN="${TARGET_OBJ%_zcu.o}"
            echo "Relinking zigly native..."
            /usr/bin/ld -dynamic -platform_version macos "$MACOS_VER" "$MACOS_VER" \
                -syslibroot "$SYSROOT" -e _main -o "$TARGET_BIN" "$TARGET_OBJ" "$CRT" "$SYSROOT/usr/lib/libSystem.B.tbd" "$SYSROOT/usr/lib/libc++.tbd"
            
            mkdir -p zig-out/bin
            cp "$TARGET_BIN" zig-out/bin/zigly
            echo "Success! Binary is at zig-out/bin/zigly"
        else
            echo "Failed to find zigly object file"
            cat target_err.log
        fi
    fi
fi
