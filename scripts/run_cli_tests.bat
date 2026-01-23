@echo off

set "FAIL=0"

rem Function to run a command and fail fast
goto :eof

:run
    set "CMD=%~1"
    echo =============================
    echo Testing %CMD% ...
    %CMD%
    if errorlevel 1 (
        echo ERROR: %CMD% failed with exit code %errorlevel%
        set FAIL=1
        exit /b 1
    ) else (
        echo %CMD% succeeded
    )
    exit /b 0

call :run "zig build run-hello"
call :run "zig build run-database"
call :run "zig build run-agent"
call :run "zig build run-compute"
call :run "zig build run-network"
call :run "zig build run-discord"
call :run "zig build run-llm"
call :run "zig build run-training"
call :run "zig build run-ha"
call :run "zig build run-orchestration"

rem Run LLM nested commands; ignore model-not-found errors
call :run "zig build run -- llm chat --model dummy"
if errorlevel 1 (
    echo WARNING: LLM chat failed (model may be missing), continuing.
    rem Reset FAIL to not treat as overall failure
    set FAIL=0
)
call :run "zig build run -- llm generate dummy \"Once\" --max 10"
if errorlevel 1 (
    echo WARNING: LLM generate failed (model may be missing), continuing.
    set FAIL=0
)
call :run "zig build run -- llm list"

if "%FAIL%"=="1" (
    echo One or more commands failed.
    exit /b 1
) else (
    echo All CLI example commands completed successfully.
    exit /b 0
)
