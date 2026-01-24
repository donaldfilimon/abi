@echo off
rem Full verification script for the ABI repository.

rem Formatting check
zig fmt --check .
if errorlevel 1 exit /b 1

rem Unit tests
zig build test
if errorlevel 1 exit /b 1

rem CLI smoke tests
zig build cli-tests
if errorlevel 1 exit /b 1

rem Benchmarks (non‑critical)
zig build bench-all
rem Continue even if benchmarks fail

exit /b 0
