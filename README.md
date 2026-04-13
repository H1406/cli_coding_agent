# CLI Coding Agent

Starter repository for building and deploying a local CLI coding agent.

## What is included

- `src/cli_coding_agent/`: application source
- `tests/`: test scaffolding
- `docs/`: architecture and deployment notes
- `scripts/`: local developer helpers
- `examples/`: example configuration
- `.github/workflows/`: CI starter
- `Dockerfile`: container packaging baseline

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
agent --help
pytest
```

## Suggested next steps

1. Replace the stubbed agent loop in `src/cli_coding_agent/agent.py` with your model orchestration.
2. Expand `src/cli_coding_agent/tools.py` with the tools your agent is allowed to call.
3. Add provider-specific environment variables in `.env.example`.
4. Update `docs/deployment.md` once you choose your runtime target.

## Project layout

```text
.
├── .github/workflows/ci.yml
├── Dockerfile
├── Makefile
├── docs/
├── examples/
├── scripts/
├── src/cli_coding_agent/
└── tests/
```
