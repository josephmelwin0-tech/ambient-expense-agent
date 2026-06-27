@echo off
if "%1"=="install" (
    uv sync
) else if "%1"=="playground" (
    uv run python -m expense_agent.fast_api_app
) else if "%1"=="generate-traces" (
    uv run python tests/eval/generate_traces.py
) else if "%1"=="grade" (
    set GOOGLE_APPLICATION_CREDENTIALS=tests/eval/dummy_creds.json
    agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml
) else (
    echo Unknown target: %1
    exit /b 1
)
