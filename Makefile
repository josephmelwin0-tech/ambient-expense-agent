# Makefile for ambient-expense-agent

.PHONY: install playground test lint generate-traces grade

install:
	uv sync

playground:
	uv run python -m expense_agent.fast_api_app

generate-traces:
	uv run python tests/eval/generate_traces.py

grade:
	GOOGLE_APPLICATION_CREDENTIALS=tests/eval/dummy_creds.json agents-cli eval grade --traces artifacts/traces/generated_traces.json --config tests/eval/eval_config.yaml
