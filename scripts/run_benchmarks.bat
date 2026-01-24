@echo off
rem Run all benchmark suites and report any failures.

set "FAIL=0"

rem Function to run a command and capture error level
goto :eof

:run
    set "CMD=%~1"
    echo =============================
    echo Running %CMD% ...
    %CMD%
    if errorlevel 1 (
        echo ERROR: %CMD% failed with exit code %errorlevel%
        set FAIL=1
        exit /b 1
    ) else (
        echo %CMD% succeeded
    )
    exit /b 0

rem Execute benchmarks
call :run "zig build bench-all"

if "%FAIL%"=="1" (
    echo One or more benchmarks failed.
    exit /b 1
) else (
    echo All benchmarks completed successfully.
    exit /b 0
)
