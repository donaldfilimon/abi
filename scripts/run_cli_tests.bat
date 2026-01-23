@echo off
setlocal

set "ABI=zig build run --"
set "FAIL=0"

echo Starting CLI Smoke Tests...

call :test_cmd %ABI% version
call :test_cmd %ABI% help
call :test_cmd %ABI% system-info
call :test_cmd %ABI% db stats
call :test_cmd %ABI% gpu backends
call :test_cmd %ABI% network list
call :test_cmd %ABI% train info
call :test_cmd %ABI% convert help
call :test_cmd %ABI% llm list

echo.
if "%FAIL%"=="0" (
    echo [PASS] All CLI tests passed.
    exit /b 0
) else (
    echo [FAIL] Some CLI tests failed.
    exit /b 1
)

:test_cmd
    set "CMD=%*"
    echo.
    echo [TEST] %CMD%
    %CMD%
    if errorlevel 1 (
        echo [ERROR] Command failed with exit code %errorlevel%
        set "FAIL=1"
    ) else (
        echo [OK]
    )
    exit /b 0